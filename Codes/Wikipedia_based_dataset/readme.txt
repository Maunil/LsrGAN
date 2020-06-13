####Code for Leveraging Seen and Unseen Semantic Relationships for Generative Zero-Shot Learning, Wikipedia based datasets 

####Setup Instructions

1. Python 3.6.8
2. Install PyTorch(1.3.0) and scikit-learn and other related packages

####How to reproduce the results:

1.Download data of CUB and NAB from https://drive.google.com/file/d/1YUcYHgv4HceHOzza8OGzMp092taKAAq1/view 
  and add it to the ./data/CUB2011 and ./data/NAbird accordingly

2. For the CUB dataset run,  
	python train_CUB.py --splitmode easy --epsilon 0.15 --correlation_penalty 0.15 --unseen_start 250 --mode_change 250
	python train_CUB.py --splitmode hard --epsilon 0.15 --correlation_penalty 0.15 --unseen_start 250 --mode_change 250

3. For the NAB dataset run,  
	python train_NAB.py --splitmode easy --epsilon 0.15 --correlation_penalty 0.15 --unseen_start 250 --mode_change 250 
	python train_NAB.py --splitmode hard --epsilon 0.15 --correlation_penalty 0.15 --unseen_start 250 --mode_change 250
	

#### For the wikipedia based dataset, we mainly build our codebase on top of,  
@inproceedings{zhu2018generative,
  title={A generative adversarial approach for zero-shot learning from noisy texts},
  author={Zhu, Yizhe and Elhoseiny, Mohamed and Liu, Bingchen and Peng, Xi and Elgammal, Ahmed},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={1004--1013},
  year={2018}
}
