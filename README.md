# SimGAN
## Project Description 
Title: **Make Virtual World Real**
* This project is motivated by[SimGAN](https://arxiv.org/pdf/1612.07828.pdf). 
Detail is described in the [report](./demo/report.pdf).
* models.py includes SimGAN, CycleGAN and model from this work.

## Example Results
From left to right: virtual, refined using SimGAN, refined using model from this work.
<img src="./demo/virtual.gif" alt="virtual" width="285"/> <img src="./demo/SimGAN.gif" alt="SimGAN" width="285"/>  <img src="./demo/ThisWork.gif" alt="ThisWork" width="285"/>

<img src="./demo/RefinedImg.png" alt="sample output"/>

  
    
I also try to do using CycleGAN, see the [exapmle1](./demo/CycleGAN1.jpg) and [example2](./demo/CycleGAN2.jpg).

## Data
Road scene dataset:
1. Synthetic images: [Virtual KITTI](http://www.europe.naverlabs.com/Research/Computer-Vision/Proxy-Virtual-Worlds)
2. Real images: [KITTI Object Detection](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d)


Data directory:
* Training data:  
    * virtual images under ```./datasets/road/trainA```
    * real images under ```./datasets/road/trainB```
* Test data:
    * modify test.py for the test data directory: ```x_list = glob('./datasets/' + dataset + '/vkitti_1.3.1_rgb/0018/morning/*.png')```
    * the refined images will be saved under ```./test_predictions/```


In addition, you could also download datasets to run CycleGAN or SimGAN, e.g. ```sh ./download_dataset.sh horse2zebra```.
* [Example output](./demo/horse2zebra.png) of horse2zebra run using SimGAN after 20 epochs with lambda_=0.1
* [Example outpu](./demo/hand.jpg) of [NYU hand dataset](https://cims.nyu.edu/~tompson/NYU_Hand_Pose_Dataset.htm) run using SimGAN after 20 epochs with lambda_=10.0

## Train
```
python train.py --dataset=road --channel=3 --ratio=2 --lambda_=10.0
```

## Test
```
python test.py --dataset=road --channel=3 --ratio=2 --lambda_=10.0
```

## Note
* I use least square GAN instead of negative log likelihood objective.
* For tensorboard run: ```tensorboard --logdir=summaries```

## Acknowledgments
Code modify from [CycleGAN-Tensorflow-PyTorch-Simple](https://github.com/LynnHo/CycleGAN-Tensorflow-PyTorch-Simple) and
 [simulated-unsupervised-tensorflow](https://github.com/carpedm20/simulated-unsupervised-tensorflow).
