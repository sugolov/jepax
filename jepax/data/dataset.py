import os

import grain.python as grain
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

DATASET_STATS = {
    "CIFAR10": ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    "CIFAR100": ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    "IMAGENET": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
}


def get_normalize_stats(dataset_name: str):
    return DATASET_STATS.get(dataset_name.upper())


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
        return {"image": img, "label": label}


class RandomResizedCrop(grain.MapTransform):
    def __init__(self, size, scale=(0.2, 1.0), ratio=(3 / 4, 4 / 3)):
        self.size = size
        self.scale = scale
        self.ratio = ratio

    def map(self, x):
        img = x["image"]
        w, h = img.size
        area = w * h

        for _ in range(10):
            target_area = area * np.random.uniform(*self.scale)
            aspect = np.exp(
                np.random.uniform(np.log(self.ratio[0]), np.log(self.ratio[1]))
            )

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
    """PIL -> numpy float32 CHW in [0, 1]"""

    def map(self, x):
        img = np.array(x["image"], dtype=np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
        x["image"] = img
        return x


class Normalize(grain.MapTransform):
    """Normalize CHW numpy images."""

    def __init__(self, mean, std):
        self.mean = np.asarray(mean, dtype=np.float32)[:, None, None]
        self.std = np.asarray(std, dtype=np.float32)[:, None, None]

    def map(self, x):
        x["image"] = (x["image"] - self.mean) / self.std
        return x


class TwoViewRandomResizedCrop(grain.MapTransform):
    """Create two HWC float32 random-resized-crop views."""

    def __init__(self, size, scale=(0.2, 1.0)):
        self.crop = RandomResizedCrop(size, scale=scale)

    def map(self, x):
        img = x["image"]
        view1 = self.crop.map({"image": img})["image"]
        view2 = self.crop.map({"image": img})["image"]
        x["view1"] = np.array(view1, dtype=np.float32) / 255.0
        x["view2"] = np.array(view2, dtype=np.float32) / 255.0
        del x["image"]
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
    normalize=False,
):
    dataset_name = dataset_name.upper()

    if dataset_name in ["CIFAR10", "CIFAR", "CIFAR100"]:
        image_size = 32
        transforms = [RandomHorizontalFlip(p=0.5), ToNumpyFloat32()]
    elif dataset_name in ["IMAGENET", "IMNET"]:
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

    if dataset_name == "CIFAR10":
        dataset = datasets.CIFAR10(data_dir, train=is_train, download=True)
        num_classes = 10
    elif dataset_name in ["CIFAR", "CIFAR100"]:
        dataset = datasets.CIFAR100(data_dir, train=is_train, download=True)
        num_classes = 100
    elif dataset_name in ["IMAGENET", "IMNET"]:
        root = os.path.join(data_dir, "train" if is_train else "val")
        dataset = datasets.ImageFolder(root)
        num_classes = 1000

    norm_key = dataset_name if dataset_name in DATASET_STATS else None
    if norm_key is None and dataset_name in ["CIFAR", "CIFAR100"]:
        norm_key = "CIFAR100"
    elif norm_key is None and dataset_name in ["IMAGENET", "IMNET"]:
        norm_key = "IMAGENET"
    if normalize and norm_key is not None:
        transforms.append(Normalize(*DATASET_STATS[norm_key]))

    torch_source = TorchDataSource(dataset)

    dataloader = grain.DataLoader(
        data_source=torch_source,
        sampler=grain.IndexSampler(
            len(torch_source),
            shuffle=shuffle,
            shard_options=grain.NoSharding(),
            seed=seed,
            num_epochs=1,
        ),
        operations=transforms
        + [grain.Batch(batch_size=batch_size, drop_remainder=is_train)],
        worker_count=num_workers,
        worker_buffer_size=prefetch_factor,
    )
    # for drop_remainder=is_train
    if is_train:
        steps_per_epoch = len(torch_source) // batch_size
    else:
        steps_per_epoch = (len(torch_source) + batch_size - 1) // batch_size
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
):
    dataset_name = dataset_name.upper()

    norm_key = dataset_name if dataset_name in DATASET_STATS else None
    if norm_key is None and dataset_name in ["CIFAR", "CIFAR100"]:
        norm_key = "CIFAR100"
    elif norm_key is None and dataset_name in ["IMAGENET", "IMNET"]:
        norm_key = "IMAGENET"
    norm_transform = (
        [transforms.Normalize(*DATASET_STATS[norm_key])] if norm_key else []
    )

    if dataset_name in ["CIFAR10", "CIFAR", "CIFAR100"]:
        image_size = 32
        transform = transforms.Compose([transforms.ToTensor()] + norm_transform)
    elif dataset_name in ["IMAGENET", "IMNET"]:
        image_size = 224
        if is_train:
            transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(image_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]
                + norm_transform
            )
        else:
            transform = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(image_size),
                    transforms.ToTensor(),
                ]
                + norm_transform
            )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

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
        worker_init_fn=_worker_init_fn,
    )

    return dataloader, num_classes, len(dataloader), image_size


def build_two_view_dataloader(
    dataset_name,
    data_dir,
    batch_size=256,
    is_train=True,
    num_workers=4,
    shuffle=True,
    prefetch_factor=4,
    crop_scale=(0.2, 1.0),
    seed=0,
):
    """Build a two-view dataloader for contrastive SSL."""
    dataset_name = dataset_name.upper()

    if dataset_name in ["CIFAR10", "CIFAR", "CIFAR100"]:
        image_size = 32
    elif dataset_name in ["IMAGENET", "IMNET"]:
        image_size = 224
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    if dataset_name == "CIFAR10":
        base_dataset = datasets.CIFAR10(data_dir, train=is_train, download=True)
        num_classes = 10
    elif dataset_name in ["CIFAR", "CIFAR100"]:
        base_dataset = datasets.CIFAR100(data_dir, train=is_train, download=True)
        num_classes = 100
    elif dataset_name in ["IMAGENET", "IMNET"]:
        root = os.path.join(data_dir, "train" if is_train else "val")
        base_dataset = datasets.ImageFolder(root)
        num_classes = 1000

    torch_source = TorchDataSource(base_dataset)
    dataloader = grain.DataLoader(
        data_source=torch_source,
        sampler=grain.IndexSampler(
            len(torch_source),
            shuffle=shuffle,
            shard_options=grain.NoSharding(),
            seed=seed,
            num_epochs=1,
        ),
        operations=[
            TwoViewRandomResizedCrop(image_size, scale=crop_scale),
            grain.Batch(batch_size=batch_size, drop_remainder=is_train),
        ],
        worker_count=num_workers,
        worker_buffer_size=prefetch_factor,
    )

    # for drop_remainder
    if is_train:
        steps_per_epoch = len(base_dataset) // batch_size
    else:
        steps_per_epoch = (len(base_dataset) + batch_size - 1) // batch_size
    return dataloader, num_classes, steps_per_epoch, image_size
