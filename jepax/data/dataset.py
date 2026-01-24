import os
import json

import jax
import grain.python as grain
import numpy as np
from PIL import Image

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader

def _worker_init_fn(_):
    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)

def numpy_collate(batch):
    """Collate function to convert batch to numpy arrays in BHWC format"""
    images, labels = zip(*batch)
    images = torch.stack(images).numpy()  # (B, C, H, W)
    images = np.ascontiguousarray(np.transpose(images, (0, 2, 3, 1)))  # (B, H, W, C)
    labels = np.array(labels)
    return images, labels

class TorchDataSource(grain.RandomAccessDataSource):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img, label = self.data[idx]
        #img = np.array(img)                 # (C, H, W)
        #img = np.transpose(img, (1, 2, 0))  # (H, W, C)
        return {"image": img, "label": label}
    

class RandomResizedCrop(grain.MapTransform):
    def __init__(self, size, scale=(0.2, 1.0), ratio=(3/4, 4/3)):
        self.size = size
        self.scale = scale
        self.ratio = ratio
    
    def map(self, x):
        img = x["image"]
        w, h = img.size
        area = w * h
        
        # Try up to 10 times to find valid crop
        for _ in range(10):
            target_area = area * np.random.uniform(*self.scale)
            aspect = np.exp(np.random.uniform(np.log(self.ratio[0]), np.log(self.ratio[1])))
            
            crop_w = int(round(np.sqrt(target_area * aspect)))
            crop_h = int(round(np.sqrt(target_area / aspect)))
            
            if 0 < crop_w <= w and 0 < crop_h <= h:
                x1 = np.random.randint(0, w - crop_w + 1)
                y1 = np.random.randint(0, h - crop_h + 1)
                img = img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
                x["image"] = img.resize((self.size, self.size), Image.BILINEAR)
                return x
        
        # Fallback: center crop
        scale = min(w, h) / self.size
        crop_size = int(self.size * scale)
        x1 = (w - crop_size) // 2
        y1 = (h - crop_size) // 2
        img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        x["image"] = img.resize((self.size, self.size), Image.BILINEAR)
        return x

class Resize(grain.MapTransform):
    def __init__(self, size):
        self.size = size
    
    def map(self, x):
        img = x["image"]
        w, h = img.size
        if w < h:
            new_w, new_h = self.size, int(h * self.size / w)
        else:
            new_w, new_h = int(w * self.size / h), self.size
        x["image"] = img.resize((new_w, new_h), Image.BILINEAR)
        return x

class CenterCrop(grain.MapTransform):
    def __init__(self, size):
        self.size = size
    
    def map(self, x):
        img = x["image"]
        w, h = img.size
        x1 = (w - self.size) // 2
        y1 = (h - self.size) // 2
        x["image"] = img.crop((x1, y1, x1 + self.size, y1 + self.size))
        return x

class RandomHorizontalFlip(grain.MapTransform):
    def __init__(self, p=0.5):
        self.p = p
    def map(self, x):
        if np.random.random() < self.p:
            x["image"] = x["image"].transpose(Image.FLIP_LEFT_RIGHT)
        return x


class ToNumpyFloat32(grain.MapTransform):
    """PIL -> numpy float32 HWC in [0, 1]"""
    def map(self, x):
        img = np.array(x["image"], dtype=np.float32) / 255.0
        x["image"] = img
        return x

def build_dataloader(
    dataset_name,
    data_dir,
    batch_size=32,
    is_train=True,
    num_workers=4,
    shuffle=False,
    prefetch_factor=4,
    seed=0,
    sharding=None
):
    dataset_name = dataset_name.upper()
    
    if dataset_name in ['CIFAR10', 'CIFAR', 'CIFAR100']:
        image_size = 32
        transforms = [
                RandomHorizontalFlip(p=0.5),
                ToNumpyFloat32()
            ]
    elif dataset_name in ['IMAGENET', 'IMNET']:
        image_size = 224
        if is_train:
            transforms = [
                RandomResizedCrop(image_size),
                RandomHorizontalFlip(p=0.5),
                ToNumpyFloat32(),
            ]
        else:
            transforms = [
                Resize(256),
                CenterCrop(image_size),
                ToNumpyFloat32(),
            ]
    else:
        transforms = [ToNumpyFloat32()]
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    if dataset_name == 'CIFAR10':
        dataset = datasets.CIFAR10(data_dir, train=is_train, download=True)
        num_classes = 10
    elif dataset_name in ['CIFAR', 'CIFAR100']:
        dataset = datasets.CIFAR100(data_dir, train=is_train, download=True)
        num_classes = 100
    elif dataset_name in ['IMAGENET', 'IMNET']:
        root = os.path.join(data_dir, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root)
        num_classes = 1000
    
    torch_source = TorchDataSource(dataset)

    if sharding:
        num_processes = jax.process_count()
        print(f"num processes: {num_processes}")
        assert batch_size % num_processes == 0, "The batch size must divide number "\
            "of processes"
        batch_size_loader = batch_size // num_processes
        print(f"batch size loader: {batch_size_loader}")
    else:
        batch_size_loader = batch_size

    dataloader = grain.DataLoader(
        data_source=torch_source,
        sampler=grain.IndexSampler(
            len(torch_source), 
            shuffle=shuffle, 
            shard_options=grain.ShardByJaxProcess() if sharding else grain.NoSharding(),
            seed=seed,
        ),
        operations=transforms + [grain.Batch(batch_size=batch_size_loader, drop_remainder=is_train)],
        worker_count=num_workers,
        worker_buffer_size=prefetch_factor
    )
    steps_per_epoch = len(torch_source) // batch_size
    return dataloader, num_classes, steps_per_epoch, image_size

def build_torch_dataloader(
    dataset_name,
    data_dir,
    batch_size=32,
    is_train=True,
    num_workers=4,
    shuffle=False,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4,
    sharding=None
):
    dataset_name = dataset_name.upper()
    
    if dataset_name in ['CIFAR10', 'CIFAR', 'CIFAR100']:
        image_size = 32
        transform = transforms.Compose([transforms.ToTensor()])
    elif dataset_name in ['IMAGENET', 'IMNET']:
        image_size = 224
        if is_train:
            transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
            ])
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
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
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=is_train,
        collate_fn=numpy_collate,
        prefetch_factor=prefetch_factor,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        worker_init_fn=_worker_init_fn
    )
    
    return dataloader, num_classes, len(dataloader), image_size