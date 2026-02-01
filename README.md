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

### Dependencies

Install `opencv`, required for `ffcv`: https://github.com/libffcv/ffcv-imagenet

**macOS:**
```bash
brew install opencv pkg-config
```

**Linux:**
```bash
sudo apt update
sudo apt install -y libopencv-dev pkg-config libturbojpeg0-dev
```

### Dataset

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxx
python jepax/data/download_imagenet.py --data_dir ~/your/data/dir
```

## TODO

- fix bool masks to indices to take less memory with masker
- smarter cached masking
- compute whether training is reasonable for wandb run
- fix lr log
- make sure eval is correctly sharded and is not OOMing
- think about sharding predictor to a different gpu
- initialize vit weights correctly
- triple check configs with ijepa paper
- gpu profile
- compute mfu
- add imnet22k download option
- increase wd linearly
- increase momentum linearly
- DOUBLE CHECK loss and implementation
- linear eval on last 4 layers and last layer

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

## Cites

- [MPX](https://arxiv.org/pdf/2507.03312)
