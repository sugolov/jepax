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


def _apply_norm(norm, x, state):
    if isinstance(norm, eqx.nn.BatchNorm):
        return norm(x, state)
    if isinstance(norm, eqx.nn.Identity):
        return x, state
    return norm(x), state


class _BasicBlock(eqx.Module):
    conv1: eqx.nn.Conv2d
    norm1: eqx.Module
    conv2: eqx.nn.Conv2d
    norm2: eqx.Module
    has_downsample: bool = eqx.field(static=True)
    ds_conv: eqx.nn.Conv2d | None
    ds_norm: eqx.Module | None
    stride: int

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample_conv: Optional[eqx.nn.Conv2d] = None,
        downsample_norm: Optional[eqx.Module] = None,
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
        self.conv2 = _convnxn(planes, planes, 3, key=keys[1])
        self.has_downsample = downsample_conv is not None
        self.ds_conv = downsample_conv
        self.ds_norm = downsample_norm
        self.stride = stride

    def __call__(self, x, state):
        out = self.conv1(x)
        out, state = _apply_norm(self.norm1, out, state)
        out = jax.nn.relu(out)
        out = self.conv2(out)
        out, state = _apply_norm(self.norm2, out, state)
        if self.has_downsample:
            identity = self.ds_conv(x)
            identity, state = _apply_norm(self.ds_norm, identity, state)
        else:
            identity = x
        out += identity
        out = jax.nn.relu(out)
        return out, state


class ResNet(eqx.Module):
    inplanes: int
    dilation: int
    groups: int
    conv1: eqx.nn.Conv2d
    norm1: eqx.Module
    maxpool: eqx.Module
    layer1: tuple
    layer2: tuple
    layer3: tuple
    layer4: tuple
    avgpool: eqx.nn.AdaptiveAvgPool2d
    fc: eqx.nn.Linear

    def __init__(
        self,
        block: Type[_BasicBlock],
        layers: list[int],
        num_classes: int = 1,
        groups: int = 1,
        norm_layer: Optional[Callable] = None,
        small_input: bool = False,
        key: Optional[Key[Array, ""]] = None,
    ) -> None:
        if key is None:
            raise TypeError("key cannot be None.")
        keys = jax.random.split(key, 6)
        self.inplanes = 64
        self.dilation = 1
        self.groups = groups
        if small_input:
            self.conv1 = eqx.nn.Conv2d(
                3, self.inplanes, kernel_size=3, stride=1, padding=2,
                use_bias=False, key=keys[0],
            )
            self.maxpool = eqx.nn.Identity()
        else:
            self.conv1 = eqx.nn.Conv2d(
                3, self.inplanes, kernel_size=7, stride=2, padding=3,
                use_bias=False, key=keys[0],
            )
            self.maxpool = eqx.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        if norm_layer is None:
            norm_layer = eqx.nn.Identity
        self.norm1 = norm_layer(self.inplanes)
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
        ds_conv = None
        ds_norm = None
        if stride != 1 or self.inplanes != planes:
            ds_conv = _convnxn_no_pad(self.inplanes, planes, 1, stride, key=keys[0])
            ds_norm = norm_layer(planes)
        layer_list = [
            block(
                self.inplanes, planes, stride,
                ds_conv, ds_norm,
                self.groups, self.dilation, norm_layer,
                key=keys[1],
            )
        ]
        self.inplanes = planes
        for i in range(1, blocks):
            layer_list.append(
                block(
                    self.inplanes, planes,
                    groups=self.groups, dilation=self.dilation,
                    norm_layer=norm_layer, key=keys[i + 1],
                )
            )
        return tuple(layer_list)

    def forward_features(self, x, state):
        x = self.conv1(x)
        x, state = _apply_norm(self.norm1, x, state)
        x = jax.nn.relu(x)
        x = self.maxpool(x)
        for block in self.layer1:
            x, state = block(x, state)
        for block in self.layer2:
            x, state = block(x, state)
        for block in self.layer3:
            x, state = block(x, state)
        for block in self.layer4:
            x, state = block(x, state)
        x = self.avgpool(x)
        return jnp.ravel(x), state

    def __call__(self, x, state):
        features, state = self.forward_features(x, state)
        return self.fc(features), state


def resnet18(**kwargs) -> ResNet:
    return ResNet(_BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet34(**kwargs) -> ResNet:
    return ResNet(_BasicBlock, [3, 4, 6, 3], **kwargs)


class ResNetBackbone(eqx.Module):
    resnet: ResNet

    def __call__(self, key, x, state):
        features, state = self.resnet.forward_features(x, state)
        out = features[None, :]  # [1, 512]
        return out, state


class InferenceResNet(eqx.Module):
    backbone: ResNetBackbone
    state: eqx.nn.State

    def __call__(self, key, x, mask=None, train=False):
        out, _ = self.backbone(key, x, self.state)
        return out, None, None


def build_resnet_backbone(
    variant: str = "resnet18",
    *,
    key: Key[Array, ""],
    small_input: bool = False,
) -> tuple[ResNetBackbone, int]:
    if variant == "resnet18":
        make_fn = resnet18
    elif variant == "resnet34":
        make_fn = resnet34
    else:
        raise ValueError(f"Unknown resnet variant: {variant}")

    def bn(channels):
        return eqx.nn.BatchNorm(channels, axis_name="batch", mode="batch", momentum=0.9)

    resnet = make_fn(num_classes=1, norm_layer=bn, small_input=small_input, key=key)
    return ResNetBackbone(resnet=resnet), 512
