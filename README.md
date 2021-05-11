# Water Segmentation for Autonomous Maritime Vehicles

<img src="https://user-images.githubusercontent.com/25903137/117814527-b64e7180-b264-11eb-8209-3271850e701e.jpg" width="600" height="800"/>


 ## Note: 
 1) Supervised Learning method is more accurate but very costly (annotation cost). Therefore, an unsupervised domain adaptation technique can be used to train the network with another but similar datasets (source domain) and the obtained knowledge can be adapted to the target domain during training.
 2) Tampere-WaterSeg is our use case (or target domain), while the target domain includes other similar and already labeled datasets (e.g., WaterDataset, MaSTr-1325, etc).

## The explanation of the results given above: 
  1) Supervised: The segmentation model is trained and tested with Tampere-WaterSeg (costly; annotation cost).
  2) Single-NoDA: No domain adaptation method is applied. The segmentation model is trained with MaSTr-1325 dataset and evaluated with Tampere-WaterSeg dataset.
  3) Single-UDA: Single source unsupervised domain adaptation: The model is trained with MaSTr-1325 dataset (source domain) and the acquired knowledge is adapted to the target domain (Tampere-WaterSeg) during training.
  4) Multi-NoDA: No domain adaptation method is applied. The segmentation model is trained with MaSTr-1325 and WaterDataset (ADE20K + RiverDataset) datasets and evaluated with Tampere-WaterSeg dataset.
  5) Multi-UDA: Multi source unsupervised domain adaptation: The model is trained with MaSTr-1325 and WaterDataset (ADE20K + RiverDataset) datasets and the acquired knowledge is adapted to the target domain (Tampere-WaterSeg) during training.
  
