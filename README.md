# Master Thesis - Water Segmentation for Autonomous Maritime Vehicles

### Keywords: Supervised Learning, Semantic Segmentation, Unsupervised Domain Adaptation, Adversarial Learning, PSPNet, UNet.

## Idea: 
If a maritime vehicle has a perception abilitiy such that, it can distinguish water objects from non-water ones, then they can autonomously operate on the water surface while there is no any other obstacle around (stop/slow down otherwise).

## About
In Semantic Segmentation tasks, providing datasets for the target domain is expensive (high pixel-level annotation cost). To reduce the annotation/labeling cost, one possible approach should be training the network with another but similar and already labeled dataset(s). However, in this case, the performance of the model in the target domain will more likely be poor because of the environmental distinctions of two domains. This is called <b>Domain Shift</b> problem. This problem can be tackled by <b>Domain Adaptation</b> techniques. These techniques have been applied to road scene segmentation tasks, but none of them has been implemented to reduce the annotation cost in water segmentation tasks. 

On the other hand, the our master thesis primarily focuses on "if an Unsupervised Domain Adaptation with Adversarial Learning method can be applied to our Water Segmentation task". This is the main contribution of the thesis.  


## Operations
The operations, followed in the master thesis are listed as below:
1) We select publicly available Tampere-WaterSeg dataset as our use case; i.e., it is the target domain. This dataset includes the data of three distinct scenarios: <b>open</b> - the boat operates on the open water surface of a Lake in Nordic environment, <b>dock</b> - the boat operates on the Lake but very close to the coastline and <b>channel</b> - the boat operates on a channel, where the environment is different than that of the lake.
2) Firstly, with the Supervised Learning method, the Tampere-WaterSeg dataset is used to train the segmentation models (UNet and PSPNet). Then, based on the evaluated performances, we decide that <b>open</b> is the easiest and the <b>docking</b> is the hardest scenarios for segmenting water from non-water objects. 
3) We show that the performance 
4) By observing the performances of the models on the distinct scenarios, we show that open scenario is the easiest scenario, while docking scenario is the hardest one.


## Analyzing how effective is apply an establisehd Unsupervised Domain Adaptation with Adversarial Learning method to the water segmentation task.  
In the source domain, we use two publicly available datasets: MaSTr-1325 Dataset and WaterDataset. 
MaSTr-1325 dataset is used as the source domain. Note that there are four labels (water, sky, environment and others) in this dataset, but we change the number of labeles into two (water and non-water), because our task is binary pixel-level classification.

## Source Domain
1)  
2) Multi-source Domain 
We use Mastr-1325 dataset 

## Unsupervised Domain Adaptation with Adversarial Learning Method


## Comparison of the performance results acquired with Supervised Learning and Unsupervised Domain Adaptation
<img src="https://user-images.githubusercontent.com/25903137/119140414-6af64900-ba44-11eb-831c-aa4d35c51337.png"/>

## Visual comparison of the performance results acquired with Supervised Learning and Unsupervised Domain Adaptation
<img src="https://user-images.githubusercontent.com/25903137/117814527-b64e7180-b264-11eb-8209-3271850e701e.jpg" width="600" height="800"/>

  
