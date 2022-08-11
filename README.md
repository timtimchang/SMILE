# SMILE: Sequence-to-Sequence Domain Adaption with Minimizing Latent Entropy for Text Image Recognition
Auther: Yen-Cheng Chang, [Yi-Chang Chen](https://github.com/GitYCC), Yu-Chuan Chang, Yi-Ren Yeh

Paper: https://arxiv.org/abs/2202.11949
- paper accepted by ICIP 2022 (IEEE International Conference on Image Processing).

Training recognition models with synthetic images have achieved remarkable results in text recognition. However, recognizing text from real-world images still faces challenges due to the domain shift between synthetic and real-world text images. One of the strategies to eliminate the domain difference without manual annotation is unsupervised domain adaptation (UDA). Due to the characteristic of sequential labeling tasks, most popular UDA methods cannot be directly applied to text recognition. To tackle this problem, we proposed a UDA method with minimizing latent entropy on sequence-to-sequence attention-based models with classbalanced self-paced learning. Our experiments show that our proposed framework achieves better recognition results than the existing methods on most UDA text recognition benchmarks.

## Overview
The proposed SMILE (**S**equence-to-sequence domain adaption with **M**inim**I**zing **L**atent **E**ntropy) is a UDA (unsupervised domain adaption) method with minimizing latent entropy on sequence-to-sequence attention-based models with class-balanced self-paced learning.

## installation
- building environment: ```cuda==11.0, python==3.7.10```

- install requierments:```pip3 install torch==1.2.0 pillow==6.2.1 torchvision==0.4.0 lmdb nltk natsort```
## training and evaluation
```
CUDA_VISIBLE_DEVICES=0 python train_smile_cbst.py --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn \
--src_train_data ./${src_train_data_path} \
--tar_train_data ./${tar_train_data_path} \
--tar_select_data ${benchmark} \
--valid_data ./${valid_data_path} \
--continue_model ./${continue_model_path} \
--batch_size 128 --lr 1 \
--init_portion 0.0 --add_portion 0.00005 --tar_lambda 1.0
```
## Citation
Please consider citing this work in your publications if it helps your research.
```
@article{chang2022smile,
      title={SMILE: Sequence-to-Sequence Domain Adaption with Minimizing Latent Entropy for Text Image Recognition}, 
      author={Yen-Cheng Chang and Yi-Chang Chen and Yu-Chuan Chang and Yi-Ren Yeh},
      year={2022},
      eprint={2202.11949},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
## Acknowledgement
The implementation was based on [deep-text-recognition-benchmark](https://github.com/clovaai/deep-text-recognition-benchmark) and [Seq2SeqAdapt](https://github.com/AprilYapingZhang/Seq2SeqAdapt)
