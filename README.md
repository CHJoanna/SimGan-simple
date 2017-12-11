# SimGAN

## Project Description 
Title: **Make Virtual World Real**

This project is motivated by[SimGAN](https://arxiv.org/pdf/1612.07828.pdf). 
Detail is described in the report.

models.py includes SimGAN, CycleGAN and model from this work.

## Example Results
From left to right: virtual, refined using SimGAN, refined using model from this work.
<img src="./demo/virtual.gif" alt="virtual" width="256"/> <img src="./demo/SimGAN.gif" alt="SimGAN" width="256"/>  <img src="./demo/ThisWork.gif" alt="ThisWork" width="256"/>

<img src=".demo/RefinedImg.png"/>

## Acknowledgments
Code modify from [CycleGAN-Tensorflow-PyTorch-Simple](https://github.com/LynnHo/CycleGAN-Tensorflow-PyTorch-Simple) and
 [simulated-unsupervised-tensorflow](https://github.com/carpedm20/simulated-unsupervised-tensorflow).
