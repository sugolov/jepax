import os

import numpy as np

import torch
from torch.utils.data import DataLoader, Subset

from torchvision import datasets, transforms


def numpy_collate(batch):
    """Collate function to convert batch to numpy arrays in BHWC format"""
    images, labels = zip(*batch)
    images = torch.stack(images).numpy()  # (B, C, H, W)
    images = np.ascontiguousarray(np.transpose(images, (0, 2, 3, 1)))  # (B, H, W, C)
    # Handle both int labels and tensor labels (CelebA attributes)
    if isinstance(labels[0], torch.Tensor):
        labels = torch.stack(labels).numpy()
    else:
        labels = np.array(labels)
    return images, labels


# from IJEPA codebase
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_dataset(
    dataset_name,
    data_dir,
    batch_size=32,
    is_train=True,
    num_workers=4,
    img_size=None,
    max_samples=None,
    crop_scale=(0.3, 1.0),
    horizontal_flip=False,
    normalize=True,
):
    dataset_name = dataset_name.upper()

    default_sizes = {
        "CIFAR10": 32,
        "CIFAR": 32,
        "CIFAR100": 32,
        "IMAGENET": 224,
        "IMNET": 224,
        "CELEBA": 64,
    }
    if dataset_name not in default_sizes:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    image_size = img_size if img_size is not None else default_sizes[dataset_name]

    if is_train:
        transform_list = [
            transforms.RandomResizedCrop(image_size, scale=crop_scale),
        ]
        if horizontal_flip:
            transform_list.append(transforms.RandomHorizontalFlip())
        transform_list.append(transforms.ToTensor())
        if normalize:
            transform_list.append(transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD))
        transform = transforms.Compose(transform_list)
    else:
        transform_list = [
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ]
        if normalize:
            transform_list.append(transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD))
        transform = transforms.Compose(transform_list)

    if dataset_name == "CIFAR10":
        dataset = datasets.CIFAR10(
            data_dir, train=is_train, transform=transform, download=True
        )
        num_classes = 10
    elif dataset_name in ["CIFAR", "CIFAR100"]:
        dataset = datasets.CIFAR100(
            data_dir, train=is_train, transform=transform, download=True
        )
        num_classes = 100
    elif dataset_name in ["IMAGENET", "IMNET"]:
        root = os.path.join(data_dir, "train" if is_train else "val")
        dataset = datasets.ImageFolder(root, transform=transform)
        num_classes = 1000
    elif dataset_name == "CELEBA":
        split = "train" if is_train else "valid"
        dataset = datasets.CelebA(
            data_dir,
            split=split,
            target_type="attr",
            transform=transform,
            download=True,
        )
        num_classes = 40

    n_samples = len(dataset)
    if max_samples is not None and max_samples < n_samples:
        dataset = Subset(dataset, range(max_samples))
        n_samples = max_samples

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=is_train,
        collate_fn=numpy_collate,
        persistent_workers=num_workers > 0,
    )

    return dataloader, num_classes, n_samples, image_size
