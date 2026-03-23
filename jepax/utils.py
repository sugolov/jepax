import functools as ft
from collections.abc import Callable, Hashable
from typing import Any, overload

import equinox as eqx
import jax
import jax.sharding
from equinox._custom_types import sentinel
from equinox._module import Static
from jaxtyping import PyTree


class _ShardMapWrapper(eqx.Module):
    _fun: Callable
    _out_specs: PyTree[jax.sharding.PartitionSpec]
    _in_specs: PyTree[jax.sharding.PartitionSpec]
    _mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh | None
    _axis_names: frozenset[Hashable]
    _check_vma: bool

    @property
    def __wrapped__(self):
        return self._fun

    def __call__(self, /, *args, **kwargs):
        if len(kwargs) != 0:
            raise RuntimeError(
                "keyword arguments cannot be used with functions wrapped with "
                "`filter_shard_map`"
            )
        del kwargs

        dynamic_args, static_args = eqx.partition(args, eqx.is_array)

        def _fun_wrapper(_dynamic_args):
            _args = eqx.combine(_dynamic_args, static_args)
            _out = self._fun(*_args)
            _dynamic_out, _static_out = eqx.partition(_out, eqx.is_array)
            return _dynamic_out, Static(_static_out)

        # for type ignore see: https://github.com/jax-ml/jax/issues/35101
        dynamic_out, static_out = jax.shard_map(  # pyright: ignore
            _fun_wrapper,
            mesh=self._mesh,
            in_specs=(self._in_specs,),
            out_specs=(self._out_specs, jax.sharding.PartitionSpec()),
            axis_names=self._axis_names,
            check_vma=self._check_vma,
        )(dynamic_args)

        return eqx.combine(dynamic_out, static_out.value)

    def __get__(self, instance, owner):
        del owner
        if instance is None:
            return self
        return eqx.Partial(self, instance)


@overload
def filter_shard_map(
    *,
    out_specs: PyTree[jax.sharding.PartitionSpec],
    in_specs: PyTree[jax.sharding.PartitionSpec],
    mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh | None = None,
    axis_names: frozenset[Hashable] = frozenset(),
    check_vma: bool = True,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    pass


@overload
def filter_shard_map(
    fun: Callable[..., Any],
    *,
    out_specs: PyTree[jax.sharding.PartitionSpec],
    in_specs: PyTree[jax.sharding.PartitionSpec],
    mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh | None = None,
    axis_names: frozenset[Hashable] = frozenset(),
    check_vma: bool = True,
) -> Callable[..., Any]:
    pass


def filter_shard_map(
    fun=sentinel,
    *,
    out_specs: PyTree[jax.sharding.PartitionSpec],
    in_specs: PyTree[jax.sharding.PartitionSpec],
    mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh | None = None,
    axis_names: frozenset[Hashable] = frozenset(),
    check_vma: bool = True,
):
    """Map a function over shards of data using a mesh of devices.

    **Arguments:**

    - `fun` is a callable to be mapped. Each application of `fun` takes as
        input a shard of the mapped-over arguments and produces a shard of the
        output.
    - `out_specs` is a PyTree of `jax.sharding.PartitionSpec`, whose structure
        should be a tree prefix of the output of `fun`. Each `PartitionSpec`
        represents how the corresponding output shards should be concatenated.
        Mentioning a mesh axis name at a position expresses concatenation along
        that axis; not mentioning an axis name expresses a promise that the
        output values are equal along that axis.
    - `in_specs` is a PyTree of `jax.sharding.PartitionSpec`, whose structure
        should be a tree prefix of the input `args` tuple. Each `PartitionSpec`
        represents how the corresponding argument should be sharded along the
        named axes of `mesh`. Mentioning a mesh axis name at a position
        expresses sharding along that axis; not mentioning an axis name
        expresses replication. Non-array leaves are automatically passed
        through unchanged.
    - `mesh` represents the array of devices over which to shard. If `None`,
        inferred from context (e.g. `jax.set_mesh`).
    - `axis_names` is a set of axis names from `mesh` over which `fun` is
        manual. If empty, `fun` is manual over all mesh axes.
    - `check_vma` is a boolean (default `True`) representing whether to enable
        additional validity checks and automatic differentiation optimizations.

    **Returns:**

    A callable representing a mapped version of `fun`, which accepts positional
    arguments corresponding to those of `fun` and produces output corresponding
    to that of `fun`.
    """

    if fun is sentinel:
        return ft.partial(
            filter_shard_map,
            out_specs=out_specs,
            in_specs=in_specs,
            mesh=mesh,
            axis_names=axis_names,
            check_vma=check_vma,
        )

    wrapper = _ShardMapWrapper(
        _fun=fun,
        _out_specs=out_specs,
        _in_specs=in_specs,
        _mesh=mesh,
        _axis_names=axis_names,
        _check_vma=check_vma,
    )
    return eqx.module_update_wrapper(wrapper)
