# Multi-GPU 환경에서 피라미드넷을 활용한 이미지 분류 분산 처리
Distributed Processing of Image Classification using Pyramid Net in Multi-GPU Environment

# 요약
딥 뉴럴 네트워크 모델은 고성능 그래픽 프로세서 유닛(GPU)을 통한 병렬처리로 느린 학습 속도라는 기존의 한계점을
개선하였다. 하지만 DNN 모델이 복잡할 경우 배치 사이즈가 제한되고 속도가 느리다는 문제는 여전히 존재한다. 본
논문에서는 이러한 문제를 해결하기위해 분산처리 환경에서 GPU를 사용하는 딥러닝 모델을 제안한다. 또한
Single-GPU와 Mulit-GPU 환경에서 학습 속도 차이를 비교 분석 한다. 이를 바탕으로 향후 분산 처리 환경에서 효율적
인 DNN을 설계하는데 도움이 되고자 한다.

# pytorch-multigpu
Multi GPU Training Code for Deep Learning with PyTorch. Train PyramidNet for CIFAR10 classification task. This code is for comparing several ways of multi-GPU training.

# Requirement
- Python 3
- PyTorch 1.0.0+
- TorchVision
- TensorboardX

# Usage
### single gpu
```
cd single_gpu
python train.py 
```

### DataParallel
```
cd data_parallel
python train.py --gpu_devices 0 1 2 3 --batch_size 768
```

### DistributedDataParallel
```
cd dist_parallel
python train.py --gpu_device 0 1 2 3 --batch_size 768
```

