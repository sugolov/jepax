# jepax
Implementation of JEPA training in JAX + Equinox 

## Environment

### Dependencies 
--- 
```
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade jax[cuda12] jaxlib equinox einops flax optax pytest torch torchvision wandb numpy matplotlib tqdm aim huggingface_hub datasets
```

### Dataset
```
export HF_TOKEN=hf_xxxxxxxxxxxxx
python jepax/data/download_imagenet.py --data_dir ~/your/data/dir
```


### TODOs
- tokens for predictor
- loss functions