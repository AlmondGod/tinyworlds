from pathlib import Path
import time
import glob
import subprocess
import os
import re
from typing import Optional

from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    StateDictOptions,
)
from torch.distributed.fsdp import FSDPModule
from torch.nn.parallel import DistributedDataParallel as DDP

MODEL_CHECKPOINT = "model_state_dict.pt"
OPTIMIZER_CHECKPOINT = "optim_state_dict.pt"
STATE = "state.pt"

def readable_timestamp():
    """Generate a sortable timestamp for filenames (no weekday)."""
    return time.strftime("%Y_%m_%d_%H_%M_%S")

def find_latest_checkpoint(base_dir, model_name, run_root_dir: Optional[str] = None, stage_name: Optional[str] = None):
    """Find latest checkpoint.
    If run_root_dir (and optional stage_name) are provided, search only under
    <run_root_dir>/<stage_name>/checkpoints (or <run_root_dir>/**/checkpoints if stage_name None).
    Otherwise, fall back to project-wide model type-based search.
    Newest run dir first, then highest step within it.
    If the newest run root has none for this model, keep searching older runs.
    """
    def collect_checkpoint_paths(roots, model_name):
        alias_map = {
            'video_tokenizer': ['video_tokenizer'],
            'latent_actions': ['latent_actions', 'lam', 'actions', 'action_tokenizer'],
            'dynamics': ['dynamics'],
        }
        aliases = alias_map.get(model_name, [model_name])
        candidates = []
        seen = set()

        def add_candidate(path: str) -> None:
            norm = os.path.normpath(path)
            if norm in seen:
                return
            if os.path.isdir(norm):
                # require at least state or model file
                state_file = os.path.join(norm, STATE)
                model_file = os.path.join(norm, MODEL_CHECKPOINT)
                if os.path.isfile(state_file) or os.path.isfile(model_file):
                    candidates.append(norm)
                    seen.add(norm)
            else:
                _, ext = os.path.splitext(norm)
                if ext in ('.pt', '.pth'):
                    candidates.append(norm)
                    seen.add(norm)

        patterns = []
        for alias in aliases:
            patterns.append(f"*{alias}_step_*")
            patterns.append(f"*{alias}_checkpoint_*")

        for root in roots:
            for pat in patterns:
                search_pattern = os.path.join(root, "**", pat)
                for match in glob.glob(search_pattern, recursive=True):
                    add_candidate(match)
        return candidates

    def run_dir_of(path: str) -> str:
        # Walk up until we reach the 'checkpoints' directory, then return its parent (the stage dir)
        d = path if os.path.isdir(path) else os.path.dirname(path)
        while d and os.path.basename(d) != 'checkpoints':
            parent = os.path.dirname(d)
            if parent == d:
                break
            d = parent
        # If we found 'checkpoints', return its parent; otherwise, fallback to two-level up
        if d and os.path.basename(d) == 'checkpoints':
            return os.path.dirname(d)
        # Fallback: use the directory containing the checkpoint path (or its parent)
        candidate_dir = path if os.path.isdir(path) else os.path.dirname(path)
        return candidate_dir

    def project_wide_search():
        # Fallback to model_type/results search in repository
        # Allow multiple possible directory names per model
        model_type_dirs = {
            'video_tokenizer': ['video_tokenizer'],
            'latent_actions': ['latent_actions', 'lam', 'actions', 'action_tokenizer'],
            'dynamics': ['dynamics',],
        }
        dirs = model_type_dirs.get(model_name)
        if not dirs:
            roots = [os.path.join(base_dir, 'results', '**', 'checkpoints')]
        else:
            roots = [os.path.join(base_dir, 'results', '**', d, 'checkpoints') for d in dirs]
        files = collect_checkpoint_paths(roots, model_name)
        if not files:
            # Generic fallback: search all checkpoints regardless of stage dir name
            generic_roots = [os.path.join(base_dir, 'results', '**', 'checkpoints')]
            files = collect_checkpoint_paths(generic_roots, model_name)
        if not files:
            raise Exception(f"No checkpoint files found for {model_name}")
        run_dir_to_files = {}
        for p in files:
            rd = run_dir_of(p)
            run_dir_to_files.setdefault(rd, []).append(p)
        newest_run_dir = max(run_dir_to_files.keys(), key=lambda d: os.path.getctime(d))
        candidate_files = run_dir_to_files[newest_run_dir]
        return candidate_files

    if run_root_dir is not None:
        if stage_name:
            roots = [os.path.join(run_root_dir, stage_name, 'checkpoints')]
        else:
            roots = [os.path.join(run_root_dir, '**', 'checkpoints')]
        files = collect_checkpoint_paths(roots, model_name)
        if not files:
            # Fallback: search project-wide older runs until found
            candidate_files = project_wide_search()
        else:
            # Group by run dir within the provided root and choose newest run dir
            run_dir_to_files = {}
            for p in files:
                rd = run_dir_of(p)
                run_dir_to_files.setdefault(rd, []).append(p)
            newest_run_dir = max(run_dir_to_files.keys(), key=lambda d: os.path.getctime(d))
            candidate_files = run_dir_to_files[newest_run_dir]
    else:
        candidate_files = project_wide_search()

    def extract_step(path: str) -> int:
        fname = os.path.basename(path)
        m = re.search(r"_step_(\d+)", fname)
        return int(m.group(1)) if m else -1

    candidate_files.sort(key=lambda p: (extract_step(p), os.path.getctime(p)))
    return candidate_files[-1]

def run_command(cmd, description):
    # empirical max dataLoader throughput settings (I used 1-6 H100s)
    env = os.environ.copy()
    env.setdefault("NG_NUM_WORKERS", str(max(2, (os.cpu_count() or 4) - 2)))
    env.setdefault("NG_PREFETCH_FACTOR", "4")
    env.setdefault("NG_PIN_MEMORY", "1")
    env["NG_PERSISTENT_WORKERS"] = "0"
    env.setdefault("TORCH_CUDNN_V8_API_ENABLED", "1")

    try:
        result = subprocess.run(cmd, check=True, capture_output=False, env=env)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e.stderr}")
        return False
    except KeyboardInterrupt:
        return False

def save_training_state(model, optimizer, scheduler, config, checkpoints_dir, prefix, step):
    """Save a checkpoint with model/optimizer/scheduler and the exact config.
    The filename includes the global step and a timestamp for uniqueness.
    """
    import torch
    ts = readable_timestamp()
    if isinstance(model, (FSDPModule, DDP)):
        state_dict = get_model_state_dict(
            model=model,
            options=StateDictOptions(
                full_state_dict=True,
                cpu_offload=True,
            ),
        )
        optimizer_state_dict = get_optimizer_state_dict(
            model=model,
            optimizers=optimizer,
            options=StateDictOptions(
                full_state_dict=True,
                cpu_offload=True,
            ),
        )
    else:
        # Avoid saving the model with _orig_mod prefix if it's compiled
        state_dict = getattr(model, '_orig_mod', model).state_dict()
        optimizer_state_dict = optimizer.state_dict()
    state = {
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'config': config,
        'step': int(step) if step is not None else None,
        'timestamp': ts,
    }
    os.makedirs(checkpoints_dir, exist_ok=True)
    ckpt_path = os.path.join(checkpoints_dir, f"{prefix}_step_{int(step) if step is not None else 0}")
    os.makedirs(ckpt_path, exist_ok=True)
    torch.save(state_dict, Path(ckpt_path) / MODEL_CHECKPOINT)
    torch.save(optimizer_state_dict, Path(ckpt_path) / OPTIMIZER_CHECKPOINT)
    torch.save(state, Path(ckpt_path) / STATE)
    return ckpt_path


def load_videotokenizer_from_checkpoint(checkpoint_path, device, model = None, is_distributed = False):
    """Instantiate VideoTokenizer from a checkpoint's saved config and load weights."""
    import torch
    from models.video_tokenizer import VideoTokenizer
    model_sd = torch.load(Path(checkpoint_path) / MODEL_CHECKPOINT, map_location='cpu', weights_only=True)
    state_cfg = torch.load(Path(checkpoint_path) / STATE, map_location='cpu', weights_only=False)
    cfg = state_cfg.get('config', {}) or {}
    frame_size = cfg.get('frame_size', 128)
    kwargs = {
        'frame_size': (frame_size, frame_size),
        'patch_size': cfg.get('patch_size', 8),
        'embed_dim': cfg.get('embed_dim', 128),
        'num_heads': cfg.get('num_heads', 8),
        'hidden_dim': cfg.get('hidden_dim', 256),
        'num_blocks': cfg.get('num_blocks', 4),
        'latent_dim': cfg.get('latent_dim', 6),
        'num_bins': cfg.get('num_bins', 4),
    }
    if model is None:
        model = VideoTokenizer(**kwargs)
    set_model_state_dict(
        model=model,
        model_state_dict=model_sd,
        options=StateDictOptions(
            full_state_dict=True,
            broadcast_from_rank0=is_distributed,
        ),
    )
    model = model.to(device)
    return model, state_cfg


def load_latent_actions_from_checkpoint(checkpoint_path, device, model = None, is_distributed = False):
    """Instantiate LatentActionModel from a checkpoint's saved config and load weights."""
    import torch
    from models.latent_actions import LatentActionModel
    model_sd = torch.load(Path(checkpoint_path) / MODEL_CHECKPOINT, map_location='cpu', weights_only=True)
    state_cfg = torch.load(Path(checkpoint_path) / STATE, map_location='cpu', weights_only=False)
    cfg = state_cfg.get('config', {}) or {}
    frame_size = cfg.get('frame_size', 128)
    kwargs = {
        'frame_size': (frame_size, frame_size),
        'n_actions': cfg.get('n_actions', 8),
        'patch_size': cfg.get('patch_size', 8),
        'embed_dim': cfg.get('embed_dim', 128),
        'num_heads': cfg.get('num_heads', 8),
        'hidden_dim': cfg.get('hidden_dim', 256),
        'num_blocks': cfg.get('num_blocks', 4),
    }
    if model is None:
        model = LatentActionModel(**kwargs)
    set_model_state_dict(
        model=model,
        model_state_dict=model_sd,
        options=StateDictOptions(
            full_state_dict=True,
            broadcast_from_rank0=is_distributed,
        ),
    )
    model = model.to(device)
    return model, state_cfg


def load_dynamics_from_checkpoint(checkpoint_path, device, model = None, is_distributed = False):
    """Instantiate DynamicsModel from a checkpoint's saved config and load weights."""
    import torch
    from models.dynamics import DynamicsModel
    model_sd = torch.load(Path(checkpoint_path) / MODEL_CHECKPOINT, map_location='cpu', weights_only=True)
    state_cfg = torch.load(Path(checkpoint_path) / STATE, map_location='cpu', weights_only=False)
    cfg = state_cfg.get('config', {}) or {}
    frame_size = cfg.get('frame_size', 128)
    # Infer conditioning_dim from checkpoint if missing
    conditioning_dim = cfg.get('conditioning_dim', None)
    if conditioning_dim is None:
        cond_inferred = None
        for k, v in model_sd.get('model', {}).items():
            # Linear weight shape: [out_features, in_features]; in_features is conditioning dim
            if k.endswith('to_gamma_beta.1.weight'):
                cond_inferred = int(v.shape[1])
                break
        conditioning_dim = cond_inferred if cond_inferred is not None else 3
    kwargs = {
        'frame_size': (frame_size, frame_size),
        'patch_size': cfg.get('patch_size', 8),
        'embed_dim': cfg.get('embed_dim', 128),
        'num_heads': cfg.get('num_heads', 8),
        'hidden_dim': cfg.get('hidden_dim', 256),
        'num_blocks': cfg.get('num_blocks', 4),
        'conditioning_dim': conditioning_dim,
        'latent_dim': cfg.get('latent_dim', 6),
        'num_bins': cfg.get('num_bins', 4),
    }
    if model is None:
        model = DynamicsModel(**kwargs)
    set_model_state_dict(
        model=model,
        model_state_dict=model_sd,
        options=StateDictOptions(
            full_state_dict=True,
            broadcast_from_rank0=is_distributed,
        )
    )
    model = model.to(device)
    return model, state_cfg

def prepare_pipeline_run_root(run_name: Optional[str] = None, base_cwd: Optional[str] = None):
    """Create a top-level run root directory results/<timestamp_or_name>"""
    cwd = base_cwd or os.getcwd()
    ts = readable_timestamp()
    name = run_name or ts
    run_root = os.path.join(cwd, 'results', name)
    os.makedirs(run_root, exist_ok=True)
    return run_root, name


def prepare_stage_dirs(run_root_dir: str, stage_name: str):
    """Create stage subdirectories under the given run root.

    Structure:
      <run_root_dir>/<stage_name>/checkpoints
      <run_root_dir>/<stage_name>/visualizations

    Returns (stage_dir, checkpoints_dir, visualizations_dir).
    """
    stage_dir = os.path.join(run_root_dir, stage_name)
    checkpoints_dir = os.path.join(stage_dir, 'checkpoints')
    visualizations_dir = os.path.join(stage_dir, 'visualizations')
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(visualizations_dir, exist_ok=True)
    return stage_dir, checkpoints_dir, visualizations_dir
