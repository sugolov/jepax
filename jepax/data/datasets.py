import os
import json

import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset

from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader


def numpy_collate(batch):
    """Collate function to convert batch to numpy arrays in BHWC format"""
    images, labels = zip(*batch)
    images = torch.stack(images).numpy()  # (B, C, H, W)
    images = np.ascontiguousarray(np.transpose(images, (0, 2, 3, 1)))  # (B, H, W, C)
    labels = np.array(labels)
    return images, labels


def build_dataset(
    dataset_name,
    data_dir,
    batch_size=32,
    is_train=True,
    num_workers=4,
):
    dataset_name = dataset_name.upper()
    
    if dataset_name in ['CIFAR10', 'CIFAR', 'CIFAR100']:
        image_size = 32
    elif dataset_name in ['IMAGENET', 'IMNET']:
        image_size = 224
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    if is_train:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ])
    
    if dataset_name == 'CIFAR10':
        dataset = datasets.CIFAR10(data_dir, train=is_train, transform=transform, download=True)
        num_classes = 10
    elif dataset_name in ['CIFAR', 'CIFAR100']:
        dataset = datasets.CIFAR100(data_dir, train=is_train, transform=transform, download=True)
        num_classes = 100
    elif dataset_name in ['IMAGENET', 'IMNET']:
        root = os.path.join(data_dir, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        num_classes = 1000
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=is_train,
        collate_fn=numpy_collate
    )
    
    return dataloader, num_classes, len(dataset), image_size