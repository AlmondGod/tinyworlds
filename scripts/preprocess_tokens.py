"""Pre-tokenize a video dataset using a trained VideoTokenizer and save indices to HDF5.

The output HDF5 contains:
  tokens: [N, P] int32  -- one row of P patch-token indices per frame
  metadata: attrs with frame_size, patch_size, latent_dim, num_bins

Usage:
  python scripts/preprocess_tokens.py \
      --video_tokenizer_path runs/.../video_tokenizer \
      --dataset PONG \
      --output_path data/pong_tokens.h5 \
      --device cuda \
      --batch_size 64
"""
import os
import argparse
import torch
import h5py
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.utils import load_videotokenizer_from_checkpoint
from datasets.data_utils import load_data_and_data_loaders


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_tokenizer_path", required=True, help="Path to video tokenizer checkpoint dir")
    parser.add_argument("--dataset", required=True, help="Dataset name (PONG, SONIC, ZELDA, ...)")
    parser.add_argument("--output_path", required=True, help="Output HDF5 path for token indices")
    parser.add_argument("--device", default="cuda", help="Device to run tokenizer on")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_frames", type=int, default=1, help="Frames per sample (use 1 to tokenize each frame independently)")
    parser.add_argument("--preload_ratio", type=float, default=1.0)
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()

    print(f"Loading video tokenizer from {args.video_tokenizer_path}")
    video_tokenizer, _ = load_videotokenizer_from_checkpoint(
        checkpoint_path=args.video_tokenizer_path,
        device=args.device,
    )
    video_tokenizer.eval()
    video_tokenizer.to(args.device)

    print(f"Loading dataset {args.dataset}")
    train_data, val_data, _, _, _ = load_data_and_data_loaders(
        dataset=args.dataset,
        batch_size=args.batch_size,
        num_frames=args.num_frames,
        preload_ratio=args.preload_ratio,
    )

    all_tokens = []
    for split_name, split_data in [("train", train_data), ("val", val_data)]:
        loader = DataLoader(split_data, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=2)
        print(f"Tokenizing {split_name} split ({len(split_data)} samples)...")
        for frames, _ in tqdm(loader):
            frames = frames.to(args.device)  # [B, T, C, H, W]
            indices = video_tokenizer.tokenize(frames)  # [B, T, P]
            B, T, P = indices.shape
            indices = indices.reshape(B * T, P)  # flatten T into N
            all_tokens.append(indices.cpu().numpy().astype(np.int32))

    all_tokens = np.concatenate(all_tokens, axis=0)  # [N, P]
    print(f"Total frames tokenized: {all_tokens.shape[0]}, patches per frame: {all_tokens.shape[1]}")

    # save to HDF5
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    with h5py.File(args.output_path, 'w') as f:
        f.create_dataset('tokens', data=all_tokens, compression='lzf')
        # store metadata so consumers can reconstruct quantizer params
        vt = video_tokenizer
        f.attrs['patch_size'] = vt.encoder.patch_embed.patch_size if hasattr(vt.encoder.patch_embed, 'patch_size') else -1
        f.attrs['latent_dim'] = vt.quantizer.latent_dim
        f.attrs['num_bins'] = vt.quantizer.num_bins
        f.attrs['codebook_size'] = vt.codebook_size
        f.attrs['patches_per_frame'] = int(all_tokens.shape[1])
    print(f"Saved tokenized dataset to {args.output_path}")


if __name__ == "__main__":
    main()
