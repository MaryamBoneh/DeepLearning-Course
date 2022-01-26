
# Estimating Age Pytorch - Resnet50 (Transfer Learning)

## Install
1. Clone Repository
```
git clone https://github.com/MaryamBoneh/DeepLearning-Course.git
```
2. CD EstimatingAge_Pytorch_TransferLearning
3. pip install requirements.txt

## Train
Run *train.py* for start training. you should select your device(cpu is default).

like following command: 

```
python train.py --device cuda
```

## Test

```
python test.py --device cuda --dataset dataset/test
```

## Inference

Give the input photo with the --input argument.
```
python inference.py --device cuda --weight age-estimating-resnet50-torch.pth --input test_images/image1.png
```

## Dataset

You can download utkface-new dataset from [here](https://www.kaggle.com/jangedoo/utkface-new).

