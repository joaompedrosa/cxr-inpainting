# cxr-inpainting
A PyTorch implementation for AnaCattNet-AR from "Anatomically-Guided Chest Radiography Image Inpainting: Synthetizing Constrastive Examples"
Based on the PyTorch implementation (https://github.com/daa233/generative-inpainting-pytorch) of the paper [Generative Image Inpainting with Contextual Attention](https://arxiv.org/abs/1801.07892) according to the author's [TensorFlow implementation](https://github.com/JiahuiYu/generative_inpainting).

## Prerequisites
This code has been tested on Windows 10 and Python 3.6 and the following are the main requirements that need to be installed:
matplotlib          3.3.4
numpy               1.19.5
pandas              1.1.5
Pillow              8.4.0
PyYAML              5.4.1
torch               1.10.2+cu113
torchvision         0.11.3+cu113

## Run anterior rib segmentation
Run segment_dataset.py.

## Run inpainting
Run test_dataset.py. Note that anterior rib segmentations for the chosen data (media\\test) must be available at media\\test_antribsegm.
