# Master Thesis - Water Segmentation for Autonomous Maritime Vehicles

## Keywords: Supervised Learning, Semantic Segmentation, Unsupervised Domain Adaptation, Adversarial Learning, PSPNet, UNet.

## Idea: 
We believe that if a maritime vehicle has a perception abilitiy such that, it can distinguish water objects from non-water ones, then they can autonomously operate on the water surface while there is no any other obstacle around (stop/slow down otherwise).

## About
In Semantic Segmentation tasks, providing datasets for the target domain is expensive (high pixel-level annotation cost). To reduce the annotation/labeling cost, one possible approach should be training the network with another but similar and already labeled dataset(s). However, in this case, the performance of the model in the target domain will more likely be poor because of the environmental distinctions of two domains. This is called <b>Domain Shift</b> problem. This problem can be tackled by <b>Domain Adaptation</b> techniques. These techniques have been applied to road scene segmentation tasks, but none of them has been implemented to reduce the annotation cost in water segmentation tasks. 

On the other hand, we contribute our master thesis that primarily focuses on "if an Unsupervised Domain Adaptation with Adversarial Learning method can be applied to our Water Segmentation task". 

## Target Domain
We use publicly available Tampere-WaterSeg dataset as our use case; i.e., it is the target domain. This dataset includes the data of three distinct scenarios: <b>open</b> - the boat operates on the open water surface of a Lake in Nordic environment, <b>dock</b> - the boat operates on the Lake but very close to the coastline and <b>channel</b> - the boat operates on a channel, where the environment is different than that of the lake. 

## Supervised Learning
Using the Tampere-WaterSeg dataset, we train the segmentation networks (UNet and PSPNet) and evaluate their performances; we want to understand which scanario (open, dock or channel) is challenging/easiest for the water segmentation tasks. We show that the open scenario is the easiest scenario for segmenting water from non-water objects, while the docking scenario is the hardest, followed by the channel scenario.  

## Source Domain
In the source domain, we use two publicly available datasets: MaSTr-1325 Dataset and WaterDataset. 
MaSTr-1325 dataset is used as the source domain. Note that there are four labels (water, sky, environment and others) in this dataset, but we change the number of labeles into two (water and non-water), because our task is binary pixel-level classification.

## Source Domain
1)  
2) Multi-source Domain 
We use Mastr-1325 dataset 

## Unsupervised Domain Adaptation with Adversarial Learning Method


## Source Domain
We use MaSTr-1325 dataset alone as the source domain and we appyly Single-source unsupervised domain

<img src="https://user-images.githubusercontent.com/25903137/117814527-b64e7180-b264-11eb-8209-3271850e701e.jpg" width="600" height="800"/>

<img src="https://user-images.githubusercontent.com/25903137/119140414-6af64900-ba44-11eb-831c-aa4d35c51337.png"/>

 ## Note: 
 1) Supervised Learning method is more accurate but very costly (annotation cost). Therefore, an unsupervised domain adaptation technique can be used to train the network with another but similar datasets (source domain) and the obtained knowledge can be adapted to the target domain during training.
 2) Tampere-WaterSeg is our use case (or target domain), while the source domain includes another but similar and already labeled datasets (e.g., WaterDataset, MaSTr-1325, etc).

## The explanation of the results given above: 
  1) Supervised: The segmentation model is trained and tested with Tampere-WaterSeg (costly; annotation cost).
  2) Single-NoDA: No domain adaptation method is applied. The segmentation model is trained with MaSTr-1325 dataset and evaluated with Tampere-WaterSeg dataset.
  3) Single-UDA: Single source unsupervised domain adaptation: The model is trained with MaSTr-1325 dataset (source domain) and the acquired knowledge is adapted to the target domain (Tampere-WaterSeg) during training.
  4) Multi-NoDA: No domain adaptation method is applied. The segmentation model is trained with MaSTr-1325 and WaterDataset (ADE20K + RiverDataset) datasets and evaluated with Tampere-WaterSeg dataset.
  5) Multi-UDA: Multi source unsupervised domain adaptation: The model is trained with MaSTr-1325 and WaterDataset (ADE20K + RiverDataset) datasets and the acquired knowledge is adapted to the target domain (Tampere-WaterSeg) during training.
  
