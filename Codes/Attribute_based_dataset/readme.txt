####Code for Leveraging Seen and Unseen Semantic Relationships for Generative Zero-Shot Learning, Attribute based datasets 

####Setup Instructions

1. Python 3.6.8
2. Install PyTorch(1.3.0) and scikit-learn and other related packages
3. CUDA : 10.1.243

####How to reproduce the results:

1.Download data of CUB, AWA and SUN from www.mpi-inf.mpg.de/zsl-benchmark and add it to the ./data/AWA1 , ./data/CUB1 and ./data/SUN1 respectively

2.To reproduce the results run the scripts from ./Shell_Scripts 

####For the attribute based datasets, we mainly build our codebase on top of,  
@inproceedings {xianCVPR18,     
 title = {Feature Generating Networks for Zero-Shot Learning},  
 booktitle = {IEEE Computer Vision and Pattern Recognition (CVPR)},     
 year = {2018},     
 author = {Yongqin Xian and Tobias Lorenz and Bernt Schiele and Zeynep Akata} 
} 

