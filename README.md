
# DLMIAproject2025



<p align="center">
  <img src="https://github.com/mirthAI/CSA-Net/assets/26433669/f2f55c71-0361-478c-85e8-dedf3cc13659" alt="image">
  <br>
  <em>Figure 1: Visual representation of the CSA-Net architecture.</em>
</p>

For the Deep Learning for Medical Image Analysis course project at the University of Twente, we utilized the following PyTorch implementation:

[A Flexible 2.5D Medical Image Segmentation Approach with In-Slice and Cross-Slice Attention](https://doi.org/10.1016/j.compbiomed.2024.109173)

This original PyTorch code has been extended by us with k-fold cross-validation and data augmentation, and adapted for the ACDC dataset.


## Requirements
Python==3.9.16
```bash
pip install -r requirements.txt
```

## Usage

### 1. Download Google pre-trained ViT models
*[Get pretrained vision transformer model using this link](https://console.cloud.google.com/storage/browser/vit_models/imagenet21k?pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%255D%22))&prefix=&forceOnObjectsSortingFiltering=false) : R50-ViT-B_16
* Save your model into folder "model/vit_checkpoint/imagenet21k/".
```bash
../model/vit_checkpoint/imagenet21k/{MODEL_NAME}.npz
```

### 2. Prepare data
First download the ACDC dataset and partition it in training and testing dataset as :trainVol, trainMask, testVol, testMask. Put these folders under the data directory. Besides, put the information files under trainPatientInfo and testPatientinfo with names that should look like: patient101_info according to the patientID

1. The data needs to be preprocessed before it can be used by the network.
      * Run the preprocessing script, which would generate train_npz folder containing 2D images in folder "data/", data list files in folder "lists/" and train.csv for overview.
```
python preprocessing.py
```

The directory structure of the whole project is as follows:

```bash
.
├── CSANet
├── model
│   └── vit_checkpoint
│       └── imagenet21k
│           └── R50+ViT-B_16.npz
│           
└── data
    ├── trainVol
    ├── trainMask
    ├── trainPatientInfo
    ├── testVol
    ├── testMask
    ├── testPatientInfo
    └── train_npz     
```

### 3. Train/Test
* Please go to the folder "CSANet/" and it's ready for you to train the model. 

```
python train.py
```
* You can modify certain hyperparameters by passing arguments—for example, training with a different number of epochs, batch size, or learning rate can be done by using the following command:
```
python train.py --max_epochs 50 --batch_size 8 --base_lr 0.0005
```
* To test the network, you can run the following command. Make sure to use the same hyperparameters as those used during training:

```
python testing.py --max_epochs 50 --batch_size 8 --base_lr 0.0005
```
The segmented masks on the test set can be found in the Results/ folder

* To compute the dice scores, Hausdorff distances and volumes of the masks you need to run the following
```
python Metrics_organizers.py ground_truth/ prediction/
```
### 4. Classification
To perform classification a few steps need to be taken before you can run it. 
* Open the following file from the classification folder
```
main_classification.py
```
* Replace INSERT PATH TO GROUND TRUTH MAKS HERE with the path to the folder containing the ground truth masks of the training data. 
* Replace INSERT PATH TO PATIENT INFO OF TRAIN GROUP HERE with the path to the folder with the patient information of the train data
* Replace INSERT PATH TO SEGMENTATION MASKS YOU WANT TO CLASSIFY HERE with the path to the folder with the results of the segmentation masks of the test data   

* Replace INSERT PATH TO PATIENT INFO OF TEST GROUP HERE with the path to the folder with the patient information of the test data 

* If you want to train the model, set train_model to True. The default is not to train the model again but used the trained model saved in the files indicated in step 2 

* If you don’t want a ROC curve, set roc to False. In the default settings a roc curve of the result is made.  

* Run the main file with
  ```
  python main_classification.py
  ```   




## Reference
* [A Flexible 2.5D Medical Image Segmentation Approach with In-Slice and Cross-Slice Attention](https://doi.org/10.1016/j.compbiomed.2024.109173)
* [Google ViT](https://github.com/google-research/vision_transformer)
* [ViT-pytorch](https://github.com/jeonsworld/ViT-pytorch)
* [segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch)

This project incorporates concepts and implementations based on the following research papers and their corresponding code repositories:
   - "TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation": [Paper](https://arxiv.org/pdf/2102.04306) | [GitHub Repository](https://github.com/Beckschen/TransUNet)
   - "Non-local Neural Networks": [Paper](https://arxiv.org/abs/1711.07971) | [GitHub Repository](https://github.com/facebookresearch/video-nonlocal-net)
  



