# Pytorch를 사용한 행동인식 딥러닝 모델 개발

**모델구조**

- Graph Convolution Network를통해 각 프레임마다 특징 추출 후 LSTM 방식으로 프레임간 관계 추출

<div align=center>
  
![img](https://user-images.githubusercontent.com/40749537/138597099-eaff5334-84b7-40e9-961a-b1aa83b38cb2.png)
</div>




**개발환경**

- 아나콘다
- tensorflow 1.13, pytorch v1.7.1 (CUDA 10.1)



**사전준비**

-  [train data](https://www.dropbox.com/sh/vt05irz7vqdgefw/AABHmWlJDcKIJrCizSgTri_3a?dl=0) 다운로드하기

  - train_data 다운로드해서 폴더내 다음과 같이 저장하기

  ```
  ├─data
  │  └─NTU-RGB-D
  │      ├─xsub
  │      └─xview
  ```

  

**Reference**

- ST-GCN :  https://github.com/woosangchul/st-gcn
- TS-LSTM : https://github.com/woosangchul/TS-LSTM



# 1. 학습

- 학습데이터로 사용되는  ntu-rgb-d 데이터셋은 xsub방식과 xview 방식으로 학습이 가능하다
  - xsub : 학습데이터와 test에 사용되는 행동class가 다른방식
  - xview : 학습데이터에 사용되는 카메라와 test에 사용되는 카메라 각도가 다른방식

- 명령어
  - batch_size의 경우 GPU의 메모리 용량에 따라 조절가능
  - 윈도우의 경우 --num-work옵션 반듯이 붙여줄것

```
python main.py --config config/st_gcn/nturgbd-cross-subject/train.yaml --device 0 --batch-size 1 --num-worker 0
```

