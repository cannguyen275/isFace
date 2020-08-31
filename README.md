# isFace

![apm](https://img.shields.io/apm/l/vim-mode.svg)

This repository mainly is for face classification, to find a image is a face or non-face with a light CNN model.

This simple idea help to pass a problem of false detection in my face detection system. Use this as an ensemble model with the output of your face detector can remove many false detection.

## Performance

|Accuracy|MegaFace (face cropped)|Download|
|---|---|---|
|ours|97.95%|[Link](https://github.com/cannguyen275/isFace/releases/download/v1.0/checkpoint_149_0.020007662697025808.tar)|

## Dataset
### Introduction

Dataset is self-collected contained. This is how I create a data:

- Collect from MegaFace: crop faces from train set for face class, random a non-face location for non-face class.
- Collect from COCO dataset: which each class of COCO dataset, randomly a 400 images for each class. I labels it as non-face class.

Finally I have a dataset contained 8,855 images for face and 20,964 images for non-face in the training set. For validation, I have 4,000 images for the evaluate.

All images is resize to 112x112. Face is crop and resize by using Face-Alignment.

## Dependencies
- Python 3.6.8
- PyTorch 1.5

## Usage

### Data preprocess
Get data:
```bash
wget https://github.com/cannguyen275/isFace/releases/download/v1.0/data.zip
```
Then extract and put it in dataset folder following this stucture.
```
|-dataset
| |- train
|    |- face
|    |- nonface
| |- val
|    |- face
|    |- nonface
```
### Train
```bash
$ python main.py
```

To visualize the training process：
```bash
$ tensorboard --logdir=runs
```



