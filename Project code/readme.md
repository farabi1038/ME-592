
Before working on this you need to set up 3 repositories training process.Make sure that this 3 repositories standalone.
Here are the 3 repositories. 
1. DeepLabV3: https://github.com/leimao/DeepLab-V3
2. YolaAct: https://github.com/dbolya/yolact
3. Nvdia semantic semmentation: https://github.com/NVIDIA/semantic-segmentation
Once you are done with setting up all this 3 repositories you need to set up Boston Street calssifiers (https://github.com/ddehueck/BostonStreetCaster/tree/master/ml_models/segment_images/standard) which is using DeepLabV3. Once you are done with all these fours, then it is easy to go forward from here.
Almost all the pretrain models are here in the link for particular datasets(https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md)
The models on on CityScapes are freely available in the detctron2 repositories.
For Boston Dataset, you need to follow process of setting up training from https://github.com/ddehueck/BostonStreetCaster/tree/master/ml_models/.
As there are few datas here we used the pretrained version of Imagenets. You need to have https://github.com/qubvel/segmentation_models in you computer for processing further.
