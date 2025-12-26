<h1 align='center'>jepax</h1>
<h2 align='center'>JEPA models in JAX.</h2>

jepax is a [JAX](https://github.com/google/jax)/[Equinox](https://github.com/patrick-kidger/equinox) implementation of Joint-Embedding Predictive Architecture (JEPA) models and related self-supervised learning methods. 


## Installation

```bash
git clone https://github.com/sugolov/jepax.git
cd jepax
pip install -e .
```

Requires Python >= 3.10.

## Future Development

- [ ] Reproduce ImageNet results from IJEPA
- [ ] [RCDM Visualization](https://arxiv.org/abs/2112.09164)
- [ ] Multigpu (single node) training (easy)
- [ ] Multinode training (harder)
- [ ] LeJEPA
- [ ] V-JEPA
- [ ] Update tests
- [ ] Pre-trained model weights
- [ ] Benchmarks against PyTorch implementation

## Other Resources

- [Awesome JEPA (list of JEPA papers/code)](https://github.com/lockwo/awesome-jepa)
- [I-JEPA paper](https://arxiv.org/abs/2301.08243)
- [V-JEPA paper](https://arxiv.org/abs/2402.03406)
- [Original PyTorch I-JEPA](https://github.com/facebookresearch/ijepa)
- [Original PyTorch V-JEPA](https://github.com/facebookresearch/jepa)
- [Yann LeCun's position paper](https://openreview.net/pdf?id=BZ5a1r-kVsf)

## See Also

Other JAX libraries: [Awesome JAX](https://github.com/lockwo/awesome-jax).
