import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


class customer_analysis():
    def customer_clustering(self):
        customer = pd.read_csv('data/bigdata.csv', low_memory=False)

        customer_use = customer[['DayTrading비중_202105', 'Swing비중_202105', 'Buy&Hold비중_202105',
                                 '주거래상품_202105', '선호시장_202105',
                                 '시총1천억미만선호도_202105', '시총1천억이상3천억미만선호도_202105',
                                 '시총3천억이상1조미만선호도_202105', '시총1조이상선호도_202105']]
        customer_use = customer_use.rename(columns={
                                           'DayTrading비중_202105': 'Day', 'Swing비중_202105': 'Swing', 'Buy&Hold비중_202105': 'Hold'})

        # 0, 0, 0인 데이터
        test_mask = (customer_use.Day == 0) & (
            customer_use.Swing == 0) & (customer_use.Hold == 0)
        test_data = customer_use.loc[test_mask, :]
        # print('0,0,0: \t'+str(len(test_data)))
        # 아닌 데이터
        mask = (customer_use.Day != 0) | (
            customer_use.Swing != 0) | (customer_use.Hold != 0)
        use_data = customer_use.loc[mask, :]
        # print('no 0,0,0: \t'+str(len(use_data)))
        # print('총 갯수: \t'+str(len(test_data)+len(use_data)))

        customer_use = use_data
        use_data = use_data[['Day']]

        km = KMeans(n_clusters=2, init='k-means++', n_init=10,
                    max_iter=300, tol=1e-04, random_state=0)

        y_km = km.fit_predict(use_data)
        customer_use["cluster_id"] = km.labels_
        # customer_use.to_csv("target_customer.csv", encoding="utf-8-sig")

        # customer_use = pd.read_csv("target_customer.csv", encoding="utf-8-sig")

        customer_use['타겟고객'] = ['O' if x == 1
                                else 'X' for x in customer_use['cluster_id']]

        highrisk = ['주식', '선물옵션', '파생결합상품', '신용공여']
        midrisk = ['해당없음', '펀드', '랩', '신탁', '기타', '상품현물']
        lowrisk = ['채권', '보험', '퇴직연금', 'CMA/RP', '예수금', '발행어음']
        customer_use['위험선호도'] = ['고위험' if x in highrisk
                                 else '중위험' if x in midrisk
                                 else '저위험'
                                 for x in customer_use['주거래상품_202105']]

        customer_use['시장선호도'] = customer_use['선호시장_202105']

        customer_use['시총선호도'] = ['X' if (w == 0 and x == 0 and y == 0 and z == 0)
                                 else '1조이상' if max(w, x, y, z) == w
                                 else '1조미만'
                                 for w, x, y, z in zip(customer_use['시총1조이상선호도_202105'],
                                                       customer_use['시총3천억이상1조미만선호도_202105'],
                                                       customer_use['시총1천억이상3천억미만선호도_202105'],
                                                       customer_use['시총1천억미만선호도_202105'])]

        customer_use['주식'] = ['코스피대형주' if (x == '코스피' and y == '1조이상')
                              else '코스피중형주' if (x == '코스피' and y == '1조미만')
                              else '코스피' if (x == '코스피' and y == 'X')
                              else '코스닥150' if x == '코스닥'
                              else '코스피/코스닥150'
                              for x, y in zip(customer_use['시장선호도'],
                                              customer_use['시총선호도'])]

        customer_use['장기채'] = ['중장기국채' if x == '고위험' else '중장기국채' if x ==
                               '중위험' else '국채3년' for x in customer_use['위험선호도']]

        customer_use['중기채'] = ['단기채권액티브' if x == '고위험' else '국채3년' if x ==
                               '중위험' else '단기통안채' for x in customer_use['위험선호도']]

        customer_use['원자재'] = [
            '금속선물/농산물선물/원유선물' for __ in customer_use['주거래상품_202105']]

        customer_use['금'] = ['금은선물' for __ in customer_use['주거래상품_202105']]

        customer_use.to_csv("./data/customer_summary.csv",
                            encoding="utf-8-sig")

    def customer_number(self, number):
        analysis_table = pd.read_csv(
            "data/customer_summary.csv", encoding="utf-8-sig")
        if number in analysis_table['Unnamed: 0']:
            index = list(analysis_table['Unnamed: 0']).index(number)
            if(analysis_table['타겟고객'][index] == 'O'):
                print(f'{number}번 고객님은 AWP 타겟 고객입니다.')
                data = analysis_table['위험선호도'][index]
                print(f'위험선호도: {data}')
                data = analysis_table['시장선호도'][index]
                print(f'시장선호도: {data}')
                data = analysis_table['시총선호도'][index]
                print(f'시총선호도: {data}')
                data = analysis_table['주식'][index]
                print(f'주식: {data}')
                data = analysis_table['장기채'][index]
                print(f'장기채: {data}')
                data = analysis_table['중기채'][index]
                print(f'중기채: {data}')
                data = analysis_table['원자재'][index]
                print(f'원자재: {data}')
                data = analysis_table['금'][index]
                print(f'금: {data}')
            else:
                print(f'{number}번 고객님은 AWP 타겟 고객이 아닙니다.')
        else:
            print(f'{number}번 고객님은 5월 거래데이터가 없어서 분석하지 못했습니다.')
        print('/////////////////////////////////////')
