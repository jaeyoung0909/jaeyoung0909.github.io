---
Layout: post
title: Distilling the Knowledge in a Neural Network
use_math: true
---

## 한 줄 요약

model compression 을 위한 데이타로 softmax 를 통과해 나온 확률 분포를 사용하고 speicalist model 을 이용한 새로운 방식의 ensemble technique 소개.

## Introduction

곤충이 환경에 따라 유충 상태로 있거나 성충이 되는 것 처럼, 머신러닝에서도 비슷한 과정이 필요하다. 연구와 개발은 다르기 때문이다. 연구에서는 경제성 등을 크게 고려할 필요는 없지만 실제로 개발하기 위해선 모델 크기, 실행 속도 등 고려해야할게 많아진다. 이를 해결하기 위한 좋은 방법이 큰 모델에서 작은 모델로 지식을 이전시키는 것이다. 

보통 모델을 학습할 때, 맞는 라벨에 대해 $log$ probability 를 증가하는 식으로 objective 를 정의한다. 이 과정에서 모델은 옳지 않은 라벨에 대한 확률도 나타낸다. 이 확률은 많은 정보를 준다. 가령 BMW 의 이미지가 주어졌을 때, garbage truck 에 대한 확률이 당근보다는 높은 식으로 말이다. 확률 라벨을 soft target 이라고 정의하고 기존의 라벨을 hard target 이라 정의하자. soft label 로 small model 을 학습 시키면, hard label 로 학습시켰을 때 보다 데이타 당 더 많은 정보를 학습할 수 있고 각 데이타가 주는 gradient 의 variance 도 줄어든다. 따라서 더 적은 데이타로 높은 학습률을 달성할 수 있다.

더 나아가서 softmax 함수의 temperature 을 올리는 방법을 제안한다. 이 뜻은, 확률 벡터가 충분히 커질 때 까지 softmax 를 살짝 변형한 $exp(z_{i}/T) \over \Sigma_{j}exp(z_{j}/T)$ 함수를 사용하겠다는 말이다. 이 함수를 이용하여 probability 가 $10^{-8}$ 이 되서 충분히 정보를 포함하지 못하는 라벨이 생기는 것을 막을 수 있다. 이 방법을 Distillation 이라고 부르겠다. 

## Distillation

$$
exp(z_{i}/T) \over \Sigma_{j}exp(z_{j}/T)
$$

여기서 T 는 temperature 이라 부르고 보통 softmax 에선 1 을 쓴다. 높은 T 값은 확률 분포를 더 soft 하게 만든다. 기본적인 Distillation 과정은 우선 큰 모델을 학습시키고 학습된 모델로부터 적당히 큰 T 값으로 transfer set 을 구한다. 이 transfer set 을 이용하여 distill 할 작은 모델을 큰 모델와 같은 T 값을 사용하여 학습시킨다. 학습이 끝난 후엔 distilled model 의 T 값을 1로 바꾼다.

Soft target 에 correct label 의 정보가 추가된다면 더 정확한 학습이 가능하다. 이를 위해 object function 으로 soft target 에 대한 cross entropy 와 correct label 에 대한 cross entropy 의 가중 평균을 사용한다. correct label 에 대한 cross entropy에 적은 가중을 두었을 때 결과가 잘 나왔다. 

## Experiment

MNIST 데이타 셋에서 실험해 본 결과, distilled 된 모델이 그렇지 않은 모델에 비해 훨씬 더 잘 나왔다. unit per layer 가 300 정도로 많으면 T 값은 8 이상이 적당했고, 30 정도 수준이라면 2.5 - 4 에서 잘 동작했다. bias 만 잘 맞춰준다면, 못 본 라벨에 대해서도 맞추었다. digit 3 에 대한 예를 한 번도 보여주지 않은 distilled model 도 3에 대해 높은 정확도를 가졌다. 대신 이 경우엔 bias 를 조정해주어야 한다.

Speech recognition 에서도 실험해 보았더니, Baseline 모델의 정확도가 58.9% 였고 이 모델 10개의 ensemble 이 61.1 % 의 정확도를 보여주었는데 distilled single model 이 60.8 % 의 정확도를 내었다. 

## Ensemble with specialist models 

cumbersome model 과 data 가 크지 않으면 위에서 언급한 방법으로 ensemble 을 해결 할 수 있지만 그렇지 않은 경우엔 병렬화해서 ensemble 시켜도 학습 시간이 길다. 이 문제는 애매한 label 만 분류하는 specialist model 을 global model 과 ensemble 하여 해결할 수 있다. 이렇게 했을 때 단점은 specialist model 의 overfitting 이 크다는 것인데, soft target 을 활용해서 보안할 수 있다. 먼저, general model 을 학습하고 그것의 weight 을 speical model 에 이식, data 의 반은 애매한 label, 나머지는 원래의 data 중 랜덤하게 뽑아서 training set 을 구성한 뒤 학습시킨다. 애매한 label 에 대한 cluster 는 generalist model 의 결과의 covariance matrix 의 column vector 에 K-means algorithm 을 적용해 구한다.

KD 할 때 soft target 을 사용하는 것은 regularization 효과도 크다. 논문의 실험에 따르면, hard target 을 사용한 distilled model 은 soft target 을 사용한 것과 training accuracy 에서 큰 차이를 보이진 않았으나 test accuracy 에서 soft target 이 월등히 좋았다. Soft target 을 사용한 것은 알아서 수렴했기 때문에 early stopping 을 사용할 필요도 없었다.