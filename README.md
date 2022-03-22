# AdaIN - PyTorch Implementation

I implemented [Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization](https://arxiv.org/abs/1703.06868)


## Install

```
pip install -e . -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

## Train

```
python src/train.py
```

## Visualize
```
tensorboar --logdir .
```