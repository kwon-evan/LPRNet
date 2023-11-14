# LPRNet
Spatial Transformer + License Plate Recognition Network \
A High Performance And Lightweight Korean License Plate Recognition.\
Pytorch-Lightning Implementation for [sirius-ai/LPRNet_Pytorch](https://github.com/sirius-ai/LPRNet_Pytorch)

## Performance
* inference time: about 5ms at each plate

# Dependencies
* lightning>=2.x
* numpy>=1.17.1
* torch>=1.10.0
* tqdm>=4.57.0
* PyYAML>=5.4
* opencv-python>=3.0
* imutils>=0.4.0
* rich>=10.2.

# Installation
```shell
git clone http://github.com/kwon-evan/LPRNet.git
cd LPRNet
python3 setup.py install
```

# References
1. [LPRNet: License Plate Recognition via Deep Neural Networks](https://arxiv.org/abs/1806.10447v1)
2. [sirius-ai/LPRNet_Pytorch](https://github.com/sirius-ai/LPRNet_Pytorch)
