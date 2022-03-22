# AdaIN - PyTorch Implementation


<img src=images/sample.png height="300">    <img src=images/adain.png height="300">

Implementation of [Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization](https://arxiv.org/abs/1703.06868)


## Install

```
git clone https://github.com/nviolante25/adain.git
pip install -e . -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

## Train
Make sure the training data directories in `train.py` are correct
```
python src/train.py
```

## Visualize
```
tensorboard --logdir .
```