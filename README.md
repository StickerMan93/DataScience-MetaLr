# DataScience-MetaLr

## 1. Synthetic data generation
MetaLr_virtualDataGen.py를 통해 시뮬레이션 모델 (EnergyPlus idf)과 BCVTB를 이용한 co-simulation 수행.
co-simulation을 통해 sampling된 제어 signal을 입력하여 다양한 데이터 생성 및 저장.


## 2. Base model construction

2.1 MetaLr_model1_generation.py  
Deficient dataset (CDUs: 0 or 4)을 이용한 모델 생성 (model1).  
ANN 생성 및 저장 (saPred_bad.h5).  
Deficient dataset에 대한 test 결과 plotting.

2.2 MetaLr_model2_generation.py  
Synthetic dataset (EnergyPlus)을 이용한 모델 생성 (model2).  
ANN 생성 및 저장 (saPred_EP.h5).  
Synthetic dataset에 대한 test 결과 plotting.  

2.3 MetaLr_model3_target.py  
Full dataset (CDUs: 0, 1, 2, 3, 4)을 이용한 모델 생성 (model3).  
ANN 생성 및 저장 (saPred_Target.h5)  
Full dataset에 대한 test 결과 plotting.  


## 3. Transfer learning approaches

3.1 MetaLr_model4.py  
Model 1의 마지막 layer에 model 2의 weight copy.  
Full dataset 에 대한 test 결과 plotting.  

3.2 MetaLr_model5.py  
Model 2의 마지막 layer에 model 1의 weight copy.  
Full dataset 에 대한 test 결과 plotting.  

3.3 MetaLr_model6_finalTransfer.py  
Model 2의 마지막 layer를 deficient dataset을 이용해 fine-tuning.  
Full dataset 에 대한 test 결과 plotting.  


## 4. Causality plotting
MetaLr_causalityPlotting.py를 이용해 각 모델의 test 결과와 실제 데이터의 causality 분석.  
