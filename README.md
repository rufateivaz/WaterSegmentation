# Master Thesis - Water Segmentation for Autonomous Maritime Vehicles

### Keywords: Supervised Learning, Semantic Segmentation, Unsupervised Domain Adaptation, Adversarial Learning, PSPNet, UNet.

## Idea: 
Currently, there is no any autonomous maritime vehicle that can operate on the water surface without human supervision. To provide secure navigation for the autonomous maritime vehicles, one important task might be being able to detect and recognize large variety of objects (e.g., fishes, maritime signs, different types of boats, etc) on the water surface. 

The main address of this thesis is **water segmentation** for autonomous maritime vehicles, which can be considered as a secure task. It simply allows discriminating water from all other non-water objects. Consequently, the boat can move forward in open water case through planned route and stop/slow down in all other cases to reduce the risk of dangerous accident.


## Definition
In Semantic Segmentation tasks, providing datasets for the target domain is expensive (high pixel-level annotation cost). To reduce the annotation/labeling cost, one possible approach should be training the network with another but similar and already labeled dataset(s). However, in this case, the performance of the model in the target domain will more likely be poor because of the environmental distinctions of two domains. This is called **Domain Shift** problem. This problem can be tackled by **Domain Adaptation** techniques. These techniques have been applied to road scene segmentation tasks, but none of them has been implemented to reduce the annotation cost in water segmentation tasks. 

On the other hand, the our master thesis primarily focuses on "if an Unsupervised Domain Adaptation with Adversarial Learning method can be applied to our Water Segmentation task". This is the main contribution of the thesis.  


## Pipeline
The operations, followed in the master thesis are listed as below:
1) We select publicly available Tampere-WaterSeg dataset as our use case; i.e., it is the **target domain**. This dataset includes the images of three distinct scenarios: **open** - the boat operates on the open water surface of a lake in Nordic environment, **dock** - the boat operates on the lake but very close to the coastline and **channel** - the boat operates on a channel, where the environment is different than that of the lake.
2) We analyze which scenarios are difficult for our water segmentation task. For this, we train the segmentation models (UNet and PSPNet) with the Tampere-WaterSeg dataset and evaluate their performances accordingly. Based on the obtained results, we show that the **open** is the easiest and **dock** is the most challenging scenarios for segmenting water pixels from non-water ones. During our analyzes, we acquire a good segmentation score (0.9878) with the Supervised Learning method. However, this score is achieved with high annotation cost. Therefore, we aim to apply an established Unsupervised Domain Adaptation with adversarial learning method to our water segmentation task. The applied adversarial learning focuses on the output space rather than the future space, as the segmentation network can generate more similarities in the output space compare to the future space.
6) Firstly, we train our network with MaSTr-1325 dataset and adapt the obtained knowledge to the target domain (Tampere-WaterSeg) during training. We call this approach a single-source domain adaptation: **Single-UDA**. 
7) Secondly, we train our network with MaSTr-1325 and WaterDataset datasets and adapt the obtained knowledge to the target domain (Tampere-WaterSeg) during training. We call this approach a multi-source domain adaptation: **Multi-UDA**.
8) Finally, we compare the acquired results with three distinct approaches: Supervised Learning, Single-UDA, Multi-UDA. Note that **NoDA** means "no domain adaptation is performed".

## Comparison of the performance results acquired with Supervised Learning and Unsupervised Domain Adaptation
<img src="https://user-images.githubusercontent.com/25903137/119140414-6af64900-ba44-11eb-831c-aa4d35c51337.png"/>

## Example results of Supervised Learning, Unsupervised Domain Adaptation (single-source and multi-source) and no adaptation.
<p align="center">
  <img width="800" height="1200" src="https://user-images.githubusercontent.com/25903137/117814527-b64e7180-b264-11eb-8209-3271850e701e.jpg">
</p>
  
