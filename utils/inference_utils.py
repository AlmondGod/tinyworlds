import torch
import time
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
from utils.utils import load_videotokenizer_from_checkpoint, load_latent_actions_from_checkpoint, load_dynamics_from_checkpoint
from einops import repeat

def load_models(video_tokenizer_path, latent_actions_path, dynamics_path, device, use_actions=True, load_dynamics=True):
    # Load tokenizer and optionally dynamics/latent actions.
    video_tokenizer, _vt_ckpt = load_videotokenizer_from_checkpoint(video_tokenizer_path, device)
    video_tokenizer.eval()
    latent_action_model = None
    if use_actions:
        latent_action_model, _latent_action_ckpt = load_latent_actions_from_checkpoint(latent_actions_path, device)
        latent_action_model.eval()
    dynamics_model = None
    if load_dynamics:
        dynamics_model, _dyn_ckpt = load_dynamics_from_checkpoint(dynamics_path, device)
        dynamics_model.eval()
    return video_tokenizer, latent_action_model, dynamics_model


def reconstruct_frames(video_tokenizer, frames):
    indices = video_tokenizer.tokenize(frames)
    latents = video_tokenizer.quantizer.get_latents_from_indices(indices)
    return video_tokenizer.detokenize(latents)


def compute_frame_mse(predicted_frames, target_frames):
    predicted = ((predicted_frames.detach().to(torch.float32) + 1) / 2).clamp(0, 1)
    target = ((target_frames.detach().to(torch.float32) + 1) / 2).clamp(0, 1)
    return torch.mean((predicted - target) ** 2).item()


def visualize_inference(predicted_frames, ground_truth_frames, inferred_actions, fps, use_actions=True):
    # Move to CPU and convert to numpy
    predicted_frames = predicted_frames.detach().cpu()
    ground_truth_frames = ground_truth_frames.detach().cpu()
    
    # Denormalize frames from [-1, 1] to [0, 1]
    predicted_frames = (predicted_frames + 1) / 2
    predicted_frames = torch.clamp(predicted_frames, 0, 1)
    ground_truth_frames = (ground_truth_frames + 1) / 2
    ground_truth_frames = torch.clamp(ground_truth_frames, 0, 1)

    # Get dimensions
    B, T, C, H, W = predicted_frames.shape

    _, num_gt_frames, _, _, _ = ground_truth_frames.shape
    
    # Create figure with ground truth and predictions side by side
    fig, axes = plt.subplots(2, T, figsize=(4 * T, 8))
    
    # Handle single subplot case
    if T == 1:
        axes = axes.reshape(2, 1)

    # Plot ground truth frames (top row)
    for i in range(num_gt_frames):
        frame = ground_truth_frames[0, i].permute(1, 2, 0).numpy()  # [H, W, C]
        axes[0, i].imshow(frame)
        axes[0, i].set_title(f'Ground Truth {i+1}', fontsize=12, color='green')
        axes[0, i].axis('off')

    # Plot predicted frames (bottom row)
    for i in range(T):
        frame = predicted_frames[0, i].permute(1, 2, 0).numpy()  # [H, W, C]
        axes[1, i].imshow(frame)
        title = f'Predicted {i+1}'
        if use_actions and i < len(inferred_actions):
            title += f'\nAction {inferred_actions[i].item()}' if i < len(inferred_actions) else ''
        axes[1, i].set_title(title, fontsize=12, color='red')
        axes[1, i].axis('off')
    
    plt.suptitle('Ground Truth vs Predicted Frames', fontsize=16, fontweight='bold')
    
    # Save the visualization
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_dir = "inference_results"
    os.makedirs(save_dir, exist_ok=True)
    
    if use_actions:
        save_path = os.path.join(save_dir, f"inference_results_gt_vs_pred_{timestamp}.png")
        mp4_path = os.path.join(save_dir, f"inference_video_{timestamp}.mp4")
    else:
        save_path = os.path.join(save_dir, f"inference_results_gt_vs_pred_no_actions_{timestamp}.png")
        mp4_path = os.path.join(save_dir, f"inference_video_no_actions_{timestamp}.mp4")
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to: {save_path}")

    all_frames = torch.cat([ground_truth_frames, predicted_frames], dim=1)
    save_frames_as_mp4(all_frames, mp4_path, fps)
    
    # Calculate and display reconstruction error
    mse_error = torch.mean((predicted_frames - ground_truth_frames) ** 2).item()
    print(f"\nInference stats:")
    print(f"Total frames generated: {T}")
    print(f"Mean Squared Error (GT vs Pred): {mse_error:.6f}")
    if use_actions:
        print(f"Actions used: {[action.item() for action in inferred_actions]}")
    else:
        print(f"No actions used.")


# TODO: get working mp4
def save_frames_as_mp4(frames, output_path, fps=2):
    B, T, C, H, W = frames.shape

    # OpenCV expects (W, H)
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_path, fourcc, fps, (W, H))

    for i in range(T):
        frame = frames[0, i].detach().cpu().permute(1, 2, 0).numpy()  # [H, W, C]
        # Ensure float32
        frame = frame.astype(np.float32)
        # Clamp and scale
        frame = np.clip(frame, 0, 1)
        frame = (frame * 255).astype(np.uint8)
        # If grayscale, convert to 3 channels
        if frame.shape[2] == 1:
            frame = np.repeat(frame, 3, axis=2)
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)

    out.release()
    print(f"MP4 video saved to: {output_path}")


def sample_random_action(n_actions):
    random_action = torch.randint(0, n_actions, (1,))
    return random_action


def get_action_latent(args, inferred_actions, n_actions, context_frames, latent_action_model, step):
    if args.use_interactive_mode: # let user input actions
        print("using interactive mode")
        user_input = input(f"Enter action id [0..{n_actions-1}] for step {step+1}: ").strip()

        assert user_input.isdigit() and 0 <= int(user_input) < n_actions, f"Invalid input. Please enter an integer in [0,{n_actions-1}]"
        val = int(user_input)
        sampled_action_index = torch.tensor([val], device=args.device)

        inferred_actions.append(sampled_action_index)
        recent = inferred_actions[-args.context_window:] if len(inferred_actions) > args.context_window else inferred_actions
        recent_tensor = repeat(torch.tensor(recent, device=args.device), 'i -> 1 i') # [1, i or T_ctx]
        action_latent = latent_action_model.quantizer.get_latents_from_indices(recent_tensor)
        
        if args.prediction_horizon > 1:
            action_latent = repeat(action_latent, 'b 1 a -> b ph a', ph=args.prediction_horizon)
        if len(recent) < args.context_window:
            gt_pad_actions = latent_action_model.encode(context_frames[:, :args.context_window - len(recent) + 1])
            quantized_gt_pad_actions = latent_action_model.quantizer(gt_pad_actions)
            action_latent = torch.cat([quantized_gt_pad_actions, action_latent], dim=1)
    elif args.use_gt_actions: # use action tokenizer actions
        print("using gt actions")
        gt_action_latents = latent_action_model.encode(context_frames) # [1, T - 1, A]
        sampled_action_index = sample_random_action(n_actions) # [1]
        inferred_actions.append(sampled_action_index) # [i]
        sampled_action_index_tensor = repeat(torch.tensor(sampled_action_index, device=args.device), 'i -> 1 i') # [1, i]
        sampled_action_latent = latent_action_model.quantizer.get_latents_from_indices(sampled_action_index_tensor) # [1, i, A]
        action_latent = torch.cat([gt_action_latents, sampled_action_latent], dim=1) # [1, T, A]
    elif args.use_actions: # use random actions
        print(f"using random actions")
        sampled_action_index = sample_random_action(n_actions) # [1]
        inferred_actions.append(sampled_action_index) # [i]
        recent_inferred_actions = inferred_actions[-args.context_window:] if len(inferred_actions) > args.context_window else inferred_actions # [i or T_ctx]
        recent_inferred_actions_tensor = repeat(torch.tensor(recent_inferred_actions, device=args.device), 'i -> 1 i') # [1, i or T_ctx]

        action_latent = latent_action_model.quantizer.get_latents_from_indices(recent_inferred_actions_tensor) # [1, i or T_ctx, A]
        if args.prediction_horizon > 1:
            action_latent = repeat(action_latent, 'b 1 a -> b ph a', ph=args.prediction_horizon)

        if len(recent_inferred_actions) < args.context_window:
            # if we dont have enough inferred actions (in the beginning) add enough gt to fill the sequence
            gt_pad_actions = latent_action_model.encode(context_frames[:, :args.context_window - len(recent_inferred_actions) + 1])  # [1, context_window - len(inferred_actions), A]
            quantized_gt_pad_actions = latent_action_model.quantizer(gt_pad_actions) # [1, context_window - len(inferred_actions), A]
            action_latent = torch.cat([quantized_gt_pad_actions, action_latent], dim=1) # [1, S, A]
    else:
        sampled_action_index = None
        action_latent = None

    return sampled_action_index, action_latent
