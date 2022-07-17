# AdaIN - PyTorch Implementation


<img src=images/sample.png height="300">    <img src=images/adain.png height="300">

Implementation of [Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization](https://arxiv.org/abs/1703.06868)


## Install

```
git clone https://github.com/nviolante25/adain.git
pip install -e . -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

## Train

```
python src/train.py --source-style=<style-dataset-path> --source-content=<content-dataset-path> --dest=<output-path>
```

## Visualize

```
tensorboard --logdir=<output-path>
```
Includes style transfer visualisation to monitor the quality of training

<img src=images/training.png height="300"> 