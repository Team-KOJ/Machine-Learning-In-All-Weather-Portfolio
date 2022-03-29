from preprocessing import *
from backtesting import *

preprocessing = Clock_Predict()

# KOSIS 경기순환시계 crawling data => csv로 정리
# Input: data/KOSIS_data(folder)
# Output: data/clock_data.csv
preprocessing.crawlingdata_to_csv()

# KOSIS 경기순환시계를 RandomForest모델로 예측한다.
# Input: data/clock_data.csv
# Output: data/clock_data_predict.csv
preprocessing.clock_predict()

# clock_data_predict.csv의 # 초기 데이터는 예측값이 아닌 과거 데이터를 이용해서 manual하게 clock_data_predict_modify 생성

# 예측한 값을 토대로 KMeans Clustering모델로 국면을 구별하고 label을 달아준다.
# Input: data/clock_data_predict_modify.csv
# Output: data/clock_data_predict_label.csv
preprocessing.phase_clustering()

# clock_data_predict_label.csv를 가지고 최종적으로 manual하게 4개국면에 대해서 label을 달아서 clock_data_final 생성

becktesting = Strategy()
