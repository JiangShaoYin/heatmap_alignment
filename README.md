# FACE LANDMARK PREDICTOR BASE ON HEATMAP

## 1. Introduction

​		This is a simple baseline method on predicting face landmarks, which generates 5 heatmaps to indicate landmark intensions. 

​		![](resources/result_screenshot_23.07.2019.png)

​		This code is modified from [[microsoft human pose estimation]](https://github.com/microsoft/human-pose-estimation.pytorch), which predicts 17 human pose landmarks. 

## 2. Train

### 		dataset annotation structure

```shell
img1_path  pt0.x  pt0.y  pt1.x  pt1.y  pt2.x  pt2.y ....
img2_path  pt0.x  pt0.y  pt1.x  pt1.y  pt2.x  pt2.y ....
img3_path  pt0.x  pt0.y  pt1.x  pt1.y  pt2.x  pt2.y ....
...
```

The landmarks order is shown above. 

### training cmd

```shell
python pose_estimation/train.py --cfg experiments/hm.yaml
```

- **Default landmark number**:  5
- **Default model**:  Resnet 18
- **Default learning rete**:  1e-4

​			

