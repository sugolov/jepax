#!/usr/bin/env python3
"""
Download ImageNet from HuggingFace with retry logic for unstable networks.

Usage:
    python download_imagenet.py --data_dir ~/data/imagenet
    python download_imagenet.py --data_dir ~/data/imagenet --split validation
"""

import argparse
import time
from pathlib import Path

from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser(description="Download ImageNet from HuggingFace")
    p.add_argument("--data_dir", type=str, default="~/data/imagenet")
    p.add_argument(
        "--split", type=str, default="both", choices=["train", "validation", "both"]
    )
    p.add_argument("--max_retries", type=int, default=5)
    return p.parse_args()


def download_with_retry(split, max_retries=5):
    """Download a split with retry logic."""
    from datasets import load_dataset

    for attempt in range(max_retries):
        try:
            print(f"Downloading {split} (attempt {attempt + 1}/{max_retries})...")
            dataset = load_dataset(
                "ILSVRC/imagenet-1k",
                split=split,
                trust_remote_code=True,
                num_proc=1,  # avoid multiprocessing issues on unstable networks
            )
            return dataset
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                wait = 2**attempt
                print(f"Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise


def save_split(dataset, split_dir, split_name):
    """Save dataset to ImageFolder structure."""
    split_dir = Path(split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving {split_name} to {split_dir}...")

    for idx, sample in enumerate(tqdm(dataset, desc=f"Saving {split_name}")):
        img = sample["image"]
        label = sample["label"]

        synset = dataset.features["label"].int2str(label)
        class_dir = split_dir / synset
        class_dir.mkdir(exist_ok=True)

        img_path = class_dir / f"{idx:08d}.JPEG"
        if not img_path.exists():
            if img.mode != "RGB":
                img = img.convert("RGB")
            img.save(img_path, "JPEG")

    print(f"Done: {split_dir}")


def main():
    args = parse_args()
    data_dir = Path(args.data_dir).expanduser()
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading ImageNet to {data_dir}")
    print(
        "Note: Accept license at https://huggingface.co/datasets/ILSVRC/imagenet-1k\n"
    )

    splits = []
    if args.split == "both":
        splits = [("train", "train"), ("validation", "val")]
    elif args.split == "train":
        splits = [("train", "train")]
    else:
        splits = [("validation", "val")]

    for hf_split, folder_name in splits:
        dataset = download_with_retry(hf_split, args.max_retries)
        save_split(dataset, data_dir / folder_name, hf_split)
        del dataset  # free memory

    print(f"\nDone! ImageNet saved to {data_dir}")


if __name__ == "__main__":
    main()
