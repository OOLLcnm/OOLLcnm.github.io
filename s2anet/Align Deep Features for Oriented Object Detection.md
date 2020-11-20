# Align Deep Features for Oriented Object Detection

## Abstract



## I. Introduction



## II. Related Works



## III. Proposed Method

**Baseline**: RetinaNet enable for oriented object detection

### A. RetinaNet as Baseline

- representative single-shot detector
- It consists of a backbone network and two task-specific subnetworks.
- Feature pyramid network is adopted as the backbone Classification and regression subnetworks are **fully convolutional networks**.
- Focal loss is proposed to address the extreme foreground-background class imbalance

> Feature pyramid network (FPN)
>
> T.-Y. Lin, P. Dollár, R. Girshick, K. He, B. Hariharan, and S. Belongie,
> “Feature pyramid networks for object detection,” in CVPR, 2017, pp.
> 2117–2125.
>
> fully convolutional networks (FCN)
>
> [Fully Convolutional Networks for Semantic Segmentation](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf)

RetinaNet is designed for generic object detection, outputting horizontal bounding box.

hbbox: ${(x, y, w, h)}$

obbox: ${(x, y, w, h, \theta)}, where \theta \in [-\frac{\pi}{4}, \frac{3\pi}{4}]$

![2020-11-19 15-49-47 的屏幕截图](/home/alex/typora笔记/论文/s2anet/2020-11-19 15-49-47 的屏幕截图.png)



### B. Alignment Convolution

![2020-11-19 15-54-48 的屏幕截图](/home/alex/typora笔记/论文/s2anet/2020-11-19 15-54-48 的屏幕截图.png)

- **标准二维卷积**
  $$
  for \ each \ location \ p\inΩ \ the \ output \ feature \ map \ Y \ is \\
  Y(p) = \sum_{r\in R}W(r)\cdot X(p+r) 
  \\ where\ X{\in}{Ω=\{0 , 1, ..., H-1\}}\times{\{0,1,..., W-1\}}\\ R={\{(r_x, r_y)\}}=\{(-1,-1),(-1,0),...,(0,1),(1,1)\}
  $$
  ![2020-11-19 16-12-36 的屏幕截图](/home/alex/typora笔记/论文/s2anet/2020-11-19 16-12-36 的屏幕截图.png)

- AlignConvs adds an additional offset field $O$ for each location **$p$**

$$
Y(p) = \sum_{r\in R;o \in O}W(r)\cdot X(p+r+o) 
\\ where\ X{\in}{Ω=\{0 , 1, ..., H-1\}}\times{\{0,1,..., W-1\}}\\ R={\{(r_x, r_y)\}}
$$

the offset field O is calculated as the difference between anchor-based sampling locations and regular sampling locations (i.e., p + r).

![2020-11-19 16-13-21 的屏幕截图](/home/alex/typora笔记/论文/s2anet/2020-11-19 16-13-21 的屏幕截图.png)

Let $(x, w, h, θ)$ represent the corresponding anchor box at location $p$.

For each $r\in R$ the anchor-based sampling location $L^r_p$ can be defined as
$$
L^r_p = {1\over S}((x, y)+{1\over k}(w, h)\cdot r)R{^T(\theta)}
$$
where k indicates the kernel size, S denotes the stride of the feature map, and $R(\theta)=(\begin{matrix} cos\theta , -sin\theta \\sin \theta ,cos\theta\end{matrix})_{2\times1}$is the rotation matrix, respectively.

The offset field O at location p is
$$
O = \sum_{r\in R}(L^r_p -p-r)
$$

- **Comparisons with other convolutions.**
  1. standard convolution samples over the feature map by a regular
     grid.
  2. DeformConv learns an offset field to augment the spatial
     sampling locations.
  3. AlignConv extracts grid-distributed
     features with the guide of anchor boxes by adding an additional
     offset field.

