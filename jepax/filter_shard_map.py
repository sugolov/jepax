"""filter_shard_map: Equinox-aware wrapper around jax.shard_map.

Automatically partitions Equinox models into arrays (passed through
shard_map) and static leaves (closed over), so you don't need manual
eqx.partition/combine.
"""

import functools as ft
from collections.abc import Callable, Hashable
from typing import Any, overload

import equinox as eqx
import jax
import jax.sharding
from jaxtyping import PyTree


_sentinel = object()


class _Static(eqx.Module):
    value: Any = eqx.field(static=True)


class _ShardMapWrapper(eqx.Module):
    _fun: Callable
    _out_specs: PyTree[jax.sharding.PartitionSpec]
    _in_specs: PyTree[jax.sharding.PartitionSpec]
    _mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh | None
    _check_vma: bool

    @property
    def __wrapped__(self):
        return self._fun

    def __call__(self, /, *args, **kwargs):
        if len(kwargs) != 0:
            raise RuntimeError(
                "keyword arguments cannot be used with "
                "functions wrapped with filter_shard_map"
            )

        dynamic_args, static_args = eqx.partition(args, eqx.is_array)

        def _fun_wrapper(_dynamic_args):
            _args = eqx.combine(_dynamic_args, static_args)
            _out = self._fun(*_args)
            _dynamic_out, _static_out = eqx.partition(
                _out, eqx.is_array
            )
            return _dynamic_out, _Static(_static_out)

        dynamic_out, static_out = jax.shard_map(
            _fun_wrapper,
            mesh=self._mesh,
            in_specs=(self._in_specs,),
            out_specs=(
                self._out_specs,
                jax.sharding.PartitionSpec(),
            ),
            check_vma=self._check_vma,
        )(dynamic_args)

        return eqx.combine(dynamic_out, static_out.value)


@overload
def filter_shard_map(
    *,
    out_specs: PyTree[jax.sharding.PartitionSpec],
    in_specs: PyTree[jax.sharding.PartitionSpec],
    mesh: jax.sharding.Mesh
    | jax.sharding.AbstractMesh
    | None = None,
    check_vma: bool = True,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]: ...


@overload
def filter_shard_map(
    fun: Callable[..., Any],
    *,
    out_specs: PyTree[jax.sharding.PartitionSpec],
    in_specs: PyTree[jax.sharding.PartitionSpec],
    mesh: jax.sharding.Mesh
    | jax.sharding.AbstractMesh
    | None = None,
    check_vma: bool = True,
) -> Callable[..., Any]: ...


def filter_shard_map(
    fun=_sentinel,
    *,
    out_specs: PyTree[jax.sharding.PartitionSpec],
    in_specs: PyTree[jax.sharding.PartitionSpec],
    mesh: jax.sharding.Mesh
    | jax.sharding.AbstractMesh
    | None = None,
    check_vma: bool = True,
):
    """Equinox-aware ``jax.shard_map``.

    Non-array leaves (ints, bools, static fields) are automatically
    closed over rather than passed through shard_map.

    **Arguments:**

    - ``fun``: function to shard-map.
    - ``in_specs``: PartitionSpec pytree matching positional args.
    - ``out_specs``: PartitionSpec pytree matching outputs.
    - ``mesh``: device mesh (or None to infer from context).
    - ``check_vma``: enable validity checks (default True).
    """
    if fun is _sentinel:
        return ft.partial(
            filter_shard_map,
            out_specs=out_specs,
            in_specs=in_specs,
            mesh=mesh,
            check_vma=check_vma,
        )

    return _ShardMapWrapper(
        _fun=fun,
        _out_specs=out_specs,
        _in_specs=in_specs,
        _mesh=mesh,
        _check_vma=check_vma,
    )
