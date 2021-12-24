
# Persian mnist

<img width="1098" alt="Screen Shot 1400-10-03 at 15 42 15" src="https://user-images.githubusercontent.com/72157067/147352033-a77584f9-2333-4818-8713-372c36536093.png">



## Install
1. Clone Repository
```
git clone https://github.com/MaryamBoneh/DeepLearning-Course.git
```
2. CD Persian_MNIST_Pytorch
3. pip install requirements.txt

## Train
Run *train.py* for start training. you should select your device(cpu is default).

like following command: 

```
python train.py --device cuda
```

## Test

```
python test.py --device cuda
```

## Inference

Give the input photo with the --input argument.
```
python inference.py --device cuda --weight fashion-mnist.pth --input test_images/shirt.png
```

## Dataset

You can download persian mnist dataset from [here](https://drive.google.com/drive/folders/14aDOVDrczXi1uRDb8FMbisJPdiGxNF_2).

