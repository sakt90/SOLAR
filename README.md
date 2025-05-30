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

Following datasets can be used in the OATS_SOL based commands:
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

- **test_OATS_SOL.ipynb** (for OATS or OATS-SOL)
- **test_SNN_SOL.ipynb** (for SNN or SNN-SOL)

## Fully Trained Models
The fully trained super-net backbones packed with 4, 8, 16, and 32 sub-nets using OATS, OATS-SOL, SNN, and SNN-SOL frameworks can be downloaded from the following link:
#### Models: https://drive.google.com/file/d/1RopQVt5hUl8Q0Xb2l2BtbXaXqB-uWSXV/view?usp=sharing


## References
- OAT: https://github.com/VITA-Group/Once-for-All-Adversarial-Training
- SNNs: https://github.com/JiahuiYu/slimmable_networks
```
@inproceedings{wang2020onceforall,
  title={Once-for-All Adversarial Training: In-Situ Tradeoff between Robustness and Accuracy for Free},
  author={Wang, Haotao and Chen, Tianlong and Gui, Shupeng and Hu, Ting-Kuei and Liu, Ji and Wang, Zhangyang},
  booktitle={NeurIPS},
  year={2020}
}
@article{yu2018slimmable,
  title={Slimmable neural networks},
  author={Yu, Jiahui and Yang, Linjie and Xu, Ning and Yang, Jianchao and Huang, Thomas},
  journal={arXiv preprint arXiv:1812.08928},
  year={2018}
}
```
