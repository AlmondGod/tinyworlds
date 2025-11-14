from contextlib import nullcontext
import torch
import torch.optim as optim
import os
from models.latent_actions import LatentActionModel
from datasets.data_utils import load_data_and_data_loaders, visualize_reconstruction
from utils.scheduler_utils import create_cosine_scheduler
from tqdm import tqdm
import wandb
from utils.utils import readable_timestamp, save_training_state, prepare_stage_dirs, prepare_pipeline_run_root
from utils.config import LatentActionsConfig, load_stage_config_merged
from utils.utils import save_training_state, load_latent_actions_from_checkpoint
from utils.wandb_utils import init_wandb, log_system_metrics, finish_wandb, log_action_distribution, log_learning_rate
from dataclasses import asdict
from utils.distributed import init_distributed_from_env, prepare_model_for_distributed, unwrap_model, print_param_count_if_main, cleanup_distributed
from torch.distributed.fsdp import FSDPModule

def main():
    # latent actions config merged with training_config.yaml (training takes priority), plus CLI overrides
    args: LatentActionsConfig = load_stage_config_merged(LatentActionsConfig, default_config_path=os.path.join(os.getcwd(), 'configs', 'latent_actions.yaml'))

    # DDP setup
    dist_setup = init_distributed_from_env()

    # run save dir if it doesn't exist (running not from full train)
    timestamp = readable_timestamp()
    run_root = os.environ.get('NG_RUN_ROOT_DIR')
    if not run_root:
        run_root, _ = prepare_pipeline_run_root(base_cwd=os.getcwd())
    is_main = dist_setup['is_main']
    stage_dir, checkpoints_dir, visualizations_dir = prepare_stage_dirs(run_root, 'latent_actions')
    if is_main:
        print(f"Latent Actions Training")
        print(f'Results will be saved in {stage_dir}')

    # dataloader
    data_overrides = {}
    if hasattr(args, 'fps') and args.fps is not None:
        data_overrides['fps'] = args.fps
    if hasattr(args, 'preload_ratio') and args.preload_ratio is not None:
        data_overrides['preload_ratio'] = args.preload_ratio
    training_data, validation_data, training_loader, validation_loader, x_train_var = load_data_and_data_loaders(
        dataset=args.dataset,
        batch_size=args.batch_size_per_gpu,
        num_frames=args.context_length,
        distributed=dist_setup['is_distributed'],
        rank=dist_setup['device_mesh'].get_rank() if dist_setup['device_mesh'] is not None else 0,
        world_size=dist_setup['world_size'],
        **data_overrides,
    )

    # init model and optional ckpt load
    model = LatentActionModel(
        frame_size=(args.frame_size, args.frame_size),
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        hidden_dim=args.hidden_dim,
        num_blocks=args.num_blocks,
        n_actions=args.n_actions,
    ).to(args.device)
    if args.checkpoint:
        model, _ = load_latent_actions_from_checkpoint(
            args.checkpoint, 
            args.device,
            model,
            dist_setup['is_distributed'],
        )

    # optional DDP, compile, param count, tf32
    print_param_count_if_main(model, "LatentActionModel", is_main)
    if args.compile:
        model = torch.compile(model, mode="reduce-overhead", fullgraph=False, dynamic=True)
    model = prepare_model_for_distributed(
        model, 
        args.distributed, 
        model_type=model.model_type, 
        device_mesh=dist_setup['device_mesh'],
    )
    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # param groups to avoid weight decay on biases and norm layers
    decay = []
    no_decay = []
    for name, param in unwrap_model(model).named_parameters():
        if param.requires_grad:
            if len(param.shape) == 1 or name.endswith(".bias") or "norm" in name:
                no_decay.append(param)
            else:
                decay.append(param)

    # fused AdamW
    optimizer = optim.AdamW([
        {'params': decay, 'weight_decay': 0.01},
        {'params': no_decay, 'weight_decay': 0}
    ], lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-8, fused=True)

    # cosine scheduler for lr warmup and AMP
    scheduler = create_cosine_scheduler(optimizer, args.n_updates)
    train_ctx = torch.amp.autocast(args.device, enabled=True, dtype=torch.bfloat16) if args.amp and not args.distributed.use_fsdp else nullcontext()

    results = {
        'n_updates': 0,
        'loss_vals': [],
    }

    # init wandb
    if args.use_wandb and is_main:
        cfg = asdict(args)
        cfg.update({'timestamp': timestamp})
        run_name = f"latent_actions_{timestamp}"
        init_wandb(args.wandb_project, cfg, run_name)

    unwrap_model(model).train()

    train_iter = iter(training_loader)
    for i in tqdm(range(args.n_updates), disable=not is_main):
        optimizer.zero_grad(set_to_none=True)
        if isinstance(model, FSDPModule):
            model.set_requires_gradient_sync(False)
        if args.compile:
            torch.compiler.cudagraph_mark_step_begin()
        for micro_batch in range(args.gradient_accumulation_steps):
            try:
                (x, _) = next(train_iter)
            except StopIteration:
                train_iter = iter(training_loader)
                (x, _) = next(train_iter)

            x = x.to(args.device, non_blocking=True)

            with train_ctx:
                loss, pred_frames = model(x)
                loss /= args.gradient_accumulation_steps
                if isinstance(model, FSDPModule):
                    if (micro_batch + 1) % args.gradient_accumulation_steps == 0:
                        model.set_requires_gradient_sync(True)
                loss.backward()

        torch.nn.utils.clip_grad_norm_(unwrap_model(model).parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        results['n_updates'] = i
        results['loss_vals'].append(loss.detach().cpu())

        if args.use_wandb and is_main:
            wandb.log({
                'train/loss': loss.item(),
            }, step=i)
            log_system_metrics(i)
            log_learning_rate(optimizer, i)
  
        # save model and visualize results
        if i % args.log_interval == 0:
            if args.use_wandb:
                with torch.no_grad():
                    actions = unwrap_model(model).encoder(x)
                    actions_quantized = unwrap_model(model).quantizer(actions)
                    idx = unwrap_model(model).quantizer.get_indices_from_latents(actions_quantized)
                    codebook_usage = idx.unique().numel() / unwrap_model(model).quantizer.codebook_size
                    z_e_var = actions.var(dim=0, unbiased=False).mean().item()
                    pred_frames_var = pred_frames.var(dim=0, unbiased=False).mean().item()

            if args.use_wandb and is_main:
                wandb.log({
                    "latent_actions/codebook_usage": codebook_usage,
                    "latent_actions/encoder_variance": z_e_var,
                    "latent_actions/decoder_variance": pred_frames_var,
                }, step=i)
                log_action_distribution(idx, i, args.n_actions)

            hyperparameters = vars(args)
            save_training_state(model, optimizer, None, hyperparameters, checkpoints_dir, prefix='latent_actions', step=i)
            if is_main:
                save_path = os.path.join(visualizations_dir, f'reconstructions_latent_actions_step_{i}.png')
                visualize_reconstruction(x, pred_frames, save_path)
            
                print('\n Step', i, 'Loss:', loss.item(), 'Codebook Usage:', codebook_usage, 'Encoder Variance:', z_e_var, 'Decoder Variance:', pred_frames_var)

    # finish wandb
    if args.use_wandb and is_main:
        finish_wandb()
    cleanup_distributed(dist_setup['is_distributed'])

if __name__ == "__main__":
    main()
