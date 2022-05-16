
Before working on this you need to set up 3 repositories training process.Make sure that this 3 repositories standalone.
Here are the 3 repositories. 
1. DeepLabV3: https://github.com/leimao/DeepLab-V3
2. YolaAct: https://github.com/dbolya/yolact
3. Nvdia semantic semmentation: https://github.com/NVIDIA/semantic-segmentation



Once you are done with setting up all this 3 repositories you need to set up Boston Street calssifiers (https://github.com/ddehueck/BostonStreetCaster/tree/master/ml_models/segment_images/standard) which is using DeepLabV3. Once you are done with all these fours, then it is easy to go forward from here.



Almost all the pretrain models are here in the link for particular datasets(https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md)
The models on on CityScapes are freely available in the detctron2 repositories.


For Boston Dataset, you need to follow process of setting up training from https://github.com/ddehueck/BostonStreetCaster/tree/master/ml_models/.
As there are few datas here we used the pretrained version of Imagenets. You need to have https://github.com/qubvel/segmentation_models in you computer for processing further. For preprocessing the data to feed to model we followed the following 3 scripts :

1.https://github.com/ddehueck/BostonStreetCaster/blob/master/ml_models/segment_images/standard/deeplabv3/utils/preprocess_data.py (For preprocessing ade20 data)
Once you have all the models ready(pretrained version) follow the training script
And then once you have all outputs in a csv file for each model use the voting classifer from scikit  or this script(https://github.com/davidiommi/Ensemble-Segmentation/blob/main/predict_single_image.py)
