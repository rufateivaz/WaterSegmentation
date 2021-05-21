## Master Thesis: Water Segmentation for Autonomous Maritime Vehicles

#### Keywords: 
Supervised Learning, Semantic Segmentation, Unsupervised Domain Adaptation, Adversarial Learning, PSPNet, UNet.


### Example Results.
<p align="center">
  <img width="800" height="1200" src="https://user-images.githubusercontent.com/25903137/117814527-b64e7180-b264-11eb-8209-3271850e701e.jpg">
</p>


### Idea: 
There is no any maritime vehicle that can autonomously operate on the water surface without human supervision. To provide secure navigation for the autonomous maritime vehicles, one important task might be being able to detect and recognize large variety of objects (e.g., fishes, maritime signs, different types of boats, etc) on the water surface. 

The main address of this thesis is **water segmentation** for autonomous maritime vehicles, which can be considered as a secure task. It allows discriminating water from all other non-water objects. Consequently, the boat can move forward in open water case through planned route and stop/slow down in all other cases to reduce the risk of dangerous accident.


### Definition
Providing dataset for a Semantic Segmentation task is expensive: high pixel-level annotation cost. To reduce the annotation/labeling cost, the common approach is to train the network with another but similar and already labeled dataset(s); i.e., training the network in another domain (source domain). However, in this case, the performance of the model in the target domain will more likely be poor because of the environmental distinctions of two domains (source and target domains). This is called **Domain Shift** problem. This problem can be tackled by **Domain Adaptation** techniques. There are many works that have applied different Domain Adaptation techniques to the road scene segmentation tasks, but there is no any work implementing such techniques for the water segmentation problem.

On the other hand, the our master thesis primarily focuses on "if an Unsupervised Domain Adaptation with Adversarial Learning method can be applied to our Water Segmentation task". This is the main contribution of the thesis.


### Pipeline
The operations, followed in the master thesis are listed as below:
1) We select publicly available Tampere-WaterSeg dataset as our use case; i.e., it is the **target domain**. This dataset includes the images of three distinct scenarios: **open** - the boat operates on the lake, **dock** - the boat operates on the lake but very close to the coastline, **channel** - the boat operates on a channel, where the environment is different than that of the lake.
2) We analyze which scenarios are difficult for our water segmentation task. For this, using the Supervised Learning approach, we train the segmentation models (UNet and PSPNet) with the Tampere-WaterSeg dataset and evaluate their performances accordingly. Based on the obtained results, we show that the **open** is the easiest and **dock** is the most challenging scenarios for segmenting water pixels from non-water ones. During our analyzes, we acquire a good segmentation score (0.9878) with the Supervised Learning method. However, this score is achieved with annotation cost. 
3) To research on "reducing the annotation cost", we assume that there are only unlabeled data in the target domain (Tampere-WaterSeg) for training the network. Then, we apply an established Unsupervised Domain Adaptation method to our water segmentation task. This method relies on an adversarial learning method that focuses on the output space rather than the feature space; note that there are more similarities between the source and target domains in the output space.
4) Firstly, we train our network with MaSTr-1325 dataset and adapt the obtained knowledge to the target domain (Tampere-WaterSeg) during training. We only provide very few **unlabeled** data from the target domain during training. We call this approach a single-source domain adaptation: **Single-UDA**. 
5) Secondly, we repeat step (4) but we use two distinct datasets in the source domain for training the network: MaSTr-1325 and WaterDataset. We call this approach a multi-source domain adaptation: **Multi-UDA**. 
6) Finally, we compare the results of three distinct approaches: Supervised Learning, Single-UDA, Multi-UDA. The Example Results (above) and the Comparison Table (below) sections show the differences among these approaches both numerically and visually. We see that the Supervised Learning method gives the best performance, but this is expected because the network is trained and evaluated in the same domain (target domain). However, this method is costly - annotation/labeling cost. On the other hand, **Multi-UDA** approach gives quiet remarkable performance with 0.9136 score, and this approach does not require any annotation cost.

Note that **NoDa** means "no domain adaptation" is performed. So, the network is trained in the source domain and evaluated in the target domain. 

### Comparison Table
<p align="center">
  <img src="https://user-images.githubusercontent.com/25903137/119140414-6af64900-ba44-11eb-831c-aa4d35c51337.png" width="600" height="200"/>
</p>
  
