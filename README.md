# SOLAR: Switchable Output Layer for Accuracy and Robustness in Once-for-All Training

This repository provides the official code used in the paper: **SOLAR: Switchable Output Layer for Accuracy and Robustness in Once-for-All Training**. 

## Requirements

Install the pre-requisites through following command:

```setup
pip install -r requirements.txt
```

## Training

To train a super-net for OATS or OATS-SOL, first set the "width_mult_list" in "slimmable_ops.py" file. Then run the pertinent command for the desired method:

```train
python train_OATS_SOL.py --method=OATS --dataset cifar10 --epochs 100
```
or
```train
python train_OATS_SOL.py --method=OATS_SOL --dataset cifar10 --epochs 100
```

Following datasets can be used in the OATS_SOL commands:
- cifar10
- svhn
- stl10

Similarly, to train a super-net for SNN or SNN-SOL, set the "width_mult_list" in "slimmable_ops.py" file. Then run the pertinent command for the desired method:

```train
python train_SNN_SOL.py --method=SNN --model ResNet_34 --epochs 100
```
or
```train
python train_SNN_SOL.py --method=SNN_SOL --model ResNet_34 --epochs 100
```

Following models can be used in the SNN_SOL based commands:
- ResNet_34
- WideResNet_16_8
- MobileNetV2

## Evaluation

To evaluate the trained model(s) use the following Jupyter Notebook Files for convenience:

- test_OATS_SOL (for OATS or OATS-SOL)
- test_SNN_SOL (for SNN or SNN-SOL)

