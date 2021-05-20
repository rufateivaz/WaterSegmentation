# Master Thesis - Water Segmentation for Autonomous Maritime Vehicles

## Motivation 
Providing datasets for semantic segmentation tasks is a fundamental issue due to manually pixel-level annotations that entail time consumption and high labor cost. So as to reduce the annotation cost for a given semantic segmentation task, another but similar dataset can be used to train the model to bring it to some level of knowledge. For example, consider a road scene segmentation problem for European roads and assume that there is no labeled data for training the network. In that case, a dataset that includes images taken in Asian roads can be used to train the network. However, it is well-known that the environmental information in both European and Asian roads are distinct; cars, road signs, weather conditions, etc. Therefore, if the network is trained with that dataset, it can give poor performance in European roads; the same holds true vice versa. This is a <b>domain shift</b>} problem that occurs when there is a gap in appearances between source (e.g., Asian roads) and target (e.g., European roads) domains.


<img src="https://user-images.githubusercontent.com/25903137/117814527-b64e7180-b264-11eb-8209-3271850e701e.jpg" width="600" height="800"/>


 ## Note: 
 1) Supervised Learning method is more accurate but very costly (annotation cost). Therefore, an unsupervised domain adaptation technique can be used to train the network with another but similar datasets (source domain) and the obtained knowledge can be adapted to the target domain during training.
 2) Tampere-WaterSeg is our use case (or target domain), while the source domain includes another but similar and already labeled datasets (e.g., WaterDataset, MaSTr-1325, etc).

## The explanation of the results given above: 
  1) Supervised: The segmentation model is trained and tested with Tampere-WaterSeg (costly; annotation cost).
  2) Single-NoDA: No domain adaptation method is applied. The segmentation model is trained with MaSTr-1325 dataset and evaluated with Tampere-WaterSeg dataset.
  3) Single-UDA: Single source unsupervised domain adaptation: The model is trained with MaSTr-1325 dataset (source domain) and the acquired knowledge is adapted to the target domain (Tampere-WaterSeg) during training.
  4) Multi-NoDA: No domain adaptation method is applied. The segmentation model is trained with MaSTr-1325 and WaterDataset (ADE20K + RiverDataset) datasets and evaluated with Tampere-WaterSeg dataset.
  5) Multi-UDA: Multi source unsupervised domain adaptation: The model is trained with MaSTr-1325 and WaterDataset (ADE20K + RiverDataset) datasets and the acquired knowledge is adapted to the target domain (Tampere-WaterSeg) during training.
  
