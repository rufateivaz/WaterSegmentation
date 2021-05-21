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
1) We select publicly available Tampere-WaterSeg dataset as our use case; i.e., it is the **target domain**. This dataset includes the data of three distinct scenarios: **open**- the boat operates on the open water surface of a lake in Nordic environment, **dock** - the boat operates on the lake but very close to the coastline and **channel** - the boat operates on a channel, where the environment is different than that of the lake.
2) We analyze which scenarios are difficult for our water segmentation task. For this, we train the models (UNet and PSPNet) with the Tampere-WaterSeg dataset and evaluate their performances accordingly. Based on the obtained results, we decide that the **open** is the easiest and **dock** is the most challenging scenarios for segmenting water pixels from non-water ones. During our analyzes, we also not that we achieve a good segmentation score with the Supervised Learning method, which is more than 0.98
3) 


with the Supervised Learning method, the Tampere-WaterSeg dataset is used to train the segmentation models (UNet and PSPNet). Then, based on the evaluated performances, we decide that **open** is the easiest and the **docking** is the hardest scenarios for segmenting water from non-water objects. 
5) We show that the performance 
6) By observing the performances of the models on the distinct scenarios, we show that open scenario is the easiest scenario, while docking scenario is the hardest one.


## Analyzing how effective is apply an establisehd Unsupervised Domain Adaptation with Adversarial Learning method to the water segmentation task.  
In the source domain, we use two publicly available datasets: MaSTr-1325 Dataset and WaterDataset. 
MaSTr-1325 dataset is used as the source domain. Note that there are four labels (water, sky, environment and others) in this dataset, but we change the number of labeles into two (water and non-water), because our task is binary pixel-level classification.

## Comparison of the performance results acquired with Supervised Learning and Unsupervised Domain Adaptation
<img src="https://user-images.githubusercontent.com/25903137/119140414-6af64900-ba44-11eb-831c-aa4d35c51337.png"/>

## Example results of Supervised Learning, Unsupervised Domain Adaptation (single-source and multi-source) and no adaptation.
<p align="center">
  <img width="800" height="1200" src="https://user-images.githubusercontent.com/25903137/117814527-b64e7180-b264-11eb-8209-3271850e701e.jpg">
</p>
  
