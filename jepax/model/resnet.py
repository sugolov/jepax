"""ResNet backbone for EB-JEPA SSL."""

from typing import Callable, Optional, Sequence, Type, Union

import equinox as eqx
import jax
from jax import numpy as jnp
from jaxtyping import Array, Key


# from eqxvision


def _convnxn(
    in_planes: int,
    out_planes: int,
    kernel_size: int,
    stride: Union[int, Sequence[int]] = 1,
    groups: int = 1,
    dilation: Union[int, Sequence[int]] = 1,
    key: Optional[Key[Array, ""]] = None,
) -> eqx.nn.Conv2d:
    if key is None:
        raise ValueError("key cannot be None")
    return eqx.nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=kernel_size,
        stride=stride,
        padding=dilation,
        groups=groups,
        use_bias=False,
        dilation=dilation,
        key=key,
    )


def _convnxn_no_pad(
    in_planes: int,
    out_planes: int,
    kernel_size: int = 1,
    stride: Union[int, Sequence[int]] = 1,
    key: Optional[Key[Array, ""]] = None,
    use_bias: bool = False,
) -> eqx.nn.Conv2d:
    if key is None:
        raise ValueError("key cannot be None")
    return eqx.nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=kernel_size,
        stride=stride,
        use_bias=use_bias,
        key=key,
    )


class _BasicBlock(eqx.Module):
    conv1: eqx.nn.Conv2d
    norm1: eqx.Module
    relu: Callable
    conv2: eqx.nn.Conv2d
    norm2: eqx.Module
    downsample: eqx.Module
    stride: int

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[eqx.Module] = None,
        groups: int = 1,
        dilation: int = 1,
        norm_layer: Optional[Callable] = None,
        key: Optional[Key[Array, ""]] = None,
    ) -> None:
        if key is None:
            raise ValueError("key must be specified")
        if norm_layer is not None:
            self.norm1 = norm_layer(planes)
            self.norm2 = norm_layer(planes)
        else:
            self.norm1 = eqx.nn.Identity()
            self.norm2 = eqx.nn.Identity()
        if groups != 1:
            raise ValueError("Groups must equal 1")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        keys = jax.random.split(key, 2)
        self.conv1 = _convnxn(inplanes, planes, 3, stride, key=keys[0])
        self.relu = jax.nn.relu
        self.conv2 = _convnxn(planes, planes, 3, key=keys[1])
        self.downsample = downsample if downsample else eqx.nn.Identity()
        self.stride = stride

    def __call__(self, x, key=None):
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNet(eqx.Module):
    inplanes: int
    dilation: int
    groups: int
    conv1: eqx.nn.Conv2d
    norm1: eqx.Module
    relu: Callable
    maxpool: eqx.nn.MaxPool2d
    layer1: eqx.nn.Sequential
    layer2: eqx.nn.Sequential
    layer3: eqx.nn.Sequential
    layer4: eqx.nn.Sequential
    avgpool: eqx.nn.AdaptiveAvgPool2d
    fc: eqx.nn.Linear

    def __init__(
        self,
        block: Type[_BasicBlock],
        layers: list[int],
        num_classes: int = 1,
        groups: int = 1,
        norm_layer: Optional[Callable] = None,
        key: Optional[Key[Array, ""]] = None,
    ) -> None:
        if key is None:
            raise TypeError("key cannot be None.")
        keys = jax.random.split(key, 6)
        self.inplanes = 64
        self.dilation = 1
        self.groups = groups
        self.conv1 = eqx.nn.Conv2d(
            3,
            self.inplanes,
            kernel_size=7,
            stride=2,
            padding=3,
            use_bias=False,
            key=keys[0],
        )
        if norm_layer is None:
            norm_layer = eqx.nn.Identity
        self.norm1 = norm_layer(self.inplanes)
        self.relu = jax.nn.relu
        self.maxpool = eqx.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], norm_layer, key=keys[1])
        self.layer2 = self._make_layer(
            block, 128, layers[1], norm_layer, stride=2, key=keys[2]
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], norm_layer, stride=2, key=keys[3]
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], norm_layer, stride=2, key=keys[4]
        )
        self.avgpool = eqx.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = eqx.nn.Linear(512, num_classes, key=keys[5])

    def _make_layer(self, block, planes, blocks, norm_layer, stride=1, key=None):
        if key is None:
            raise ValueError("key must be specified")
        keys = jax.random.split(key, blocks + 1)
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = eqx.nn.Sequential(
                [
                    _convnxn_no_pad(self.inplanes, planes, 1, stride, key=keys[0]),
                    norm_layer(planes),
                ]
            )
        layers = [
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.dilation,
                norm_layer,
                key=keys[1],
            )
        ]
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                    key=keys[i + 1],
                )
            )
        return eqx.nn.Sequential(layers)

    def forward_features(self, x, key=None):
        if key is None:
            keys = [None] * 5
        else:
            keys = jax.random.split(key, 5)
        x = self.conv1(x, key=keys[0])
        x = self.norm1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x, key=keys[1])
        x = self.layer2(x, key=keys[2])
        x = self.layer3(x, key=keys[3])
        x = self.layer4(x, key=keys[4])
        x = self.avgpool(x)
        return jnp.ravel(x)

    def __call__(self, x, key=None):
        features = self.forward_features(x, key=key)
        return self.fc(features)


def resnet18(**kwargs) -> ResNet:
    return ResNet(_BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet34(**kwargs) -> ResNet:
    return ResNet(_BasicBlock, [3, 4, 6, 3], **kwargs)


class ResNetBackbone(eqx.Module):
    resnet: ResNet

    def __call__(self, key, x, mask=None, train=True, get_intermediates=False):
        features = self.resnet.forward_features(x, key=key)
        out = features[None, :]  # [1, 512]
        if get_intermediates:
            return out, [], jnp.array([0]), 1
        return out, jnp.array([0]), 1


def build_resnet_backbone(
    variant: str = "resnet18",
    *,
    key: Key[Array, ""],
) -> tuple[ResNetBackbone, int]:
    if variant == "resnet18":
        make_fn = resnet18
    elif variant == "resnet34":
        make_fn = resnet34
    else:
        raise ValueError(f"Unknown resnet variant: {variant}")

    def gn(channels):
        return eqx.nn.GroupNorm(min(32, channels), channels)

    resnet = make_fn(num_classes=1, norm_layer=gn, key=key)
    return ResNetBackbone(resnet=resnet), 512
