# Water Segmentation for Autonomous Maritime Vehicles

<img src="https://user-images.githubusercontent.com/25903137/117814527-b64e7180-b264-11eb-8209-3271850e701e.jpg" width="600" height="800"/>

## The main steps of the master thesis are: 
  1) Tampere-WaterSeg dataset is chosen as the use case (or target domain).
  2) Using the Supervised Learning method, the established segmentation models (e.g., PSPNet/UNet) are trained with Tampere-WaterSeg dataset and evaluated accordingly.
  3) The human annotation cost is the fundamental problem for the semantic segmentation tasks. Therefore, the research is performed on "if an established Unsupervised Domain Adaptation (UDA) with adversarial learning method can be applied to the water segmentation task"; note that there is no need for any labeled/annotated data in the target domain for training the network. 
  4) Two types of UDA approaches are tested: 
  5) 
  6) With this method, it is assumed that 
  7) 
  8) it is assumed that there is no any data in the target domain for training the segmentation network. So, an Unsupervised Domain Adaptation with adversarial learning method is used. With this method, the models are trained with another but similar datasets, such as MaSTr1325 (single source domain adaptation), MaSTr1325 + WaterDataset (ADE20K + RiverDataset). During training, the learned knowledge is adapted to the target domain (Tampere-WaterSeg).  
  9) 

with Tampere-WaterSeg, WaterDataset 
