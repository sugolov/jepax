#!/usr/bin/env python3
import os
import argparse
from pathlib import Path
from tqdm import tqdm
from PIL import Image


def parse_args():
    p = argparse.ArgumentParser(description="Download ImageNet from HuggingFace")
    p.add_argument("--data_dir", type=str, default="~/data/imagenet",
                   help="Directory to save ImageNet")
    p.add_argument("--num_proc", type=int, default=8,
                   help="Number of processes for downloading")
    p.add_argument("--train_only", action="store_true",
                   help="Only download training set")
    p.add_argument("--val_only", action="store_true",
                   help="Only download validation set")
    return p.parse_args()


def save_split(dataset, split_dir, split_name):
    """Save a dataset split to ImageFolder structure."""
    split_dir = Path(split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving {split_name} split to {split_dir}...")
    
    for idx, sample in enumerate(tqdm(dataset, desc=f"Processing {split_name}")):
        img = sample["image"]
        label = sample["label"]
        
        # Get synset id (class folder name)
        # HF imagenet-1k uses integer labels, need to map to synset
        synset = dataset.features["label"].int2str(label)
        
        # Create class directory
        class_dir = split_dir / synset
        class_dir.mkdir(exist_ok=True)
        
        # Save image
        img_path = class_dir / f"{idx:08d}.JPEG"
        if not img_path.exists():
            # Convert to RGB if necessary (some images are grayscale)
            if img.mode != "RGB":
                img = img.convert("RGB")
            img.save(img_path, "JPEG")
    
    print(f"Finished saving {split_name} to {split_dir}")


def main():
    args = parse_args()
    
    # Expand user path
    data_dir = Path(args.data_dir).expanduser()
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading ImageNet to {data_dir}")
    print("Note: You must have accepted the license at https://huggingface.co/datasets/ILSVRC/imagenet-1k")
    print()
    
    # Import here to give better error message if not installed
    from datasets import load_dataset
    
    # Determine which splits to download
    splits_to_download = []
    if args.val_only:
        splits_to_download = ["validation"]
    elif args.train_only:
        splits_to_download = ["train"]
    else:
        splits_to_download = ["train", "validation"]
    
    # Download and process each split
    for split in splits_to_download:
        print(f"\nLoading {split} split from HuggingFace...")
        
        # Load dataset (this downloads it)
        dataset = load_dataset(
            "ILSVRC/imagenet-1k",
            split=split,
            trust_remote_code=True,
            num_proc=args.num_proc,
        )
        
        # Map split name to folder name
        folder_name = "train" if split == "train" else "val"
        split_dir = data_dir / folder_name
        
        # Save to ImageFolder structure
        save_split(dataset, split_dir, split)
    
    print(f"\nDone! ImageNet saved to {data_dir}")

if __name__ == "__main__":
    main()