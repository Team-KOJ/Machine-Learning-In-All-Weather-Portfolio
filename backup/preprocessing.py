import re
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans


class Clock_Predict():
    # KOSIS_Data안의 자료는 크롬개발자도구를 통해서 JavaScript사용하여 크롤링 하였습니다.
    def crawlingdata_to_csv(self):

        table = {
            'date': [],
            'icon01_x': [], 'icon01_y': [],
            'icon02_x': [], 'icon02_y': [],
            'icon03_x': [], 'icon03_y': [],
            'icon04_x': [], 'icon04_y': [],
            'icon05_x': [], 'icon05_y': [],
            'icon06_x': [], 'icon06_y': [],
            'icon07_x': [], 'icon07_y': [],
            'icon08_x': [], 'icon08_y': [],
            # 'icon09_x': [], 'icon09_y': [],
            # 'icon10_x': [], 'icon10_y': [],
            'icon11_x': [], 'icon11_y': [],
            # 'icon12_x': [], 'icon12_y': [],
            # 'icon13_x': [], 'icon13_y': [],
            # 'icon14_x': [], 'icon14_y': [],
            'icon15_x': [], 'icon15_y': []
            # 'icon16_x': [], 'icon16_y': [],
            # 'icon17_x': [], 'icon17_y': [],
            # 'icon18_x': [], 'icon18_y': [],
            # 'icon19_x': [], 'icon19_y': []
        }

        # KOSIS에서 크롤링하여 데이터를 가져왔을때 지표가 19개였지만 실제 사용하는 지표는 10개
        id_list = ['01', '02', '03', '04', '05', '06', '07', '08', '11', '15']

        # 각 지표들을 정규표현식을 사용하여 접근하여 긁어옵니다.
        for file in tqdm(os.listdir('./data/KOSIS_data')):
            with open('data/KOSIS_data/'+file, 'r', encoding='UTF-8') as f:
                data = f.read()

                p = re.compile('Graphbox[^`]*SVG')
                graphbox = p.findall(data)[0]

                p = re.compile('indicatorDate[^`]*span')
                date = p.findall(graphbox)[0][15:24]
                date = ''.join(date.split())
                # print(date)
                table['date'].append(date)

                p = re.compile('<g id="icon0[^>]*>')
                icon_list = p.findall(graphbox)

                for icon in icon_list:
                    id = icon[7:16].split('_')[1]
                    if id in id_list:
                        if 'none' in icon:
                            x, y = None, None
                        else:
                            x, y = map(float, icon[icon.index(
                                '(')+1:icon.index(')')].split())
                        # print(id, x, y)
                        table['icon'+id+'_x'].append(x)
                        table['icon'+id+'_y'].append(y)
                        # if x == None:
                        #     table['icon'+id+'_nx'].append(x)
                        #     table['icon'+id+'_ny'].append(y)
                        # else:
                        #     table['icon'+id+'_nx'].append((x-276.074)/41.4*0.2)
                        #     table['icon'+id+'_ny'].append(-(y-276.135)/36.8*1)

        # 지표들을 정규화한 값을 뽑기위한 과정
        wholex_list = []
        wholey_list = []
        for id in id_list:
            wholex_list.extend(table['icon'+id+'_x'])
            wholey_list.extend(table['icon'+id+'_y'])

        wholex_list = np.array(wholex_list)
        wholey_list = np.array(wholey_list)

        for id in id_list:
            originx_list = np.array(table['icon'+id+'_x'])
            originy_list = np.array(table['icon'+id+'_y'])
            normx_list = []
            normy_list = []
            for i in range(len(originx_list)):
                normx_list.append(
                    (originx_list[i] - wholex_list.mean()) / wholex_list.std())
                normy_list.append(
                    (originy_list[i] - wholey_list.mean()) / wholey_list.std())
            table['icon'+id+'_nx'] = normx_list
            table['icon'+id+'_ny'] = normy_list

        df = pd.DataFrame(table)
        # 최종적으로 data.csv는 날짜별 각 지표들의 값과 정규화한 값.
        df.to_csv('data/clock_data.csv')
        print("data/clock_data.csv 저장 완료")

    def clock_predict(self):
        table = pd.read_csv('data/clock_data.csv')
        # 10개의 지표에 대해서
        icon_list = ['01', '02', '03', '04',
                     '05', '06', '07', '08', '11', '15']

        # 4가지 케이스
        # 1. 6개월치를 학습시켜서 1개월 뒤를 예측
        # 2. 6개월치를 학습시켜서 3개월 뒤를 예측
        # 3. 12개월치를 학습시켜서 1개월 뒤를 예측
        # 4. 12개월치를 학습시켜서 3개월 뒤를 에측
        test_list = [
            #             [3,1],
            #             [3,2],
            #             [3,3]
                    [6, 1],
            #             [6,2],
                    [6, 3],
            #             [9,1],
            #             [9,2],
            #             [9,3],
                    [12, 1],
            #             [12,2],
                    [12, 3]
        ]

        # 각 지표의 x축,y축 예측
        axis_list = ['nx', 'ny']

        for i in tqdm(range(len(icon_list))):
            for t in range(len(test_list)):
                for a in range(len(axis_list)):

                    x_data = []
                    y_data = []
                    for j in range(len(table['icon'+icon_list[i]+'_'+axis_list[a]])-(test_list[t][0]+test_list[t][1]-1)):
                        x_data.append(table['icon'+icon_list[i] +
                                            '_'+axis_list[a]][j:j+test_list[t][0]])
                        y_data.append(table['icon'+icon_list[i]+'_'+axis_list[a]]
                                      [j+(test_list[t][0]+test_list[t][1]-1)])

                    x_train, x_test, y_train, y_test = train_test_split(
                        x_data, y_data)

                    model = RandomForestRegressor(n_estimators=10)
                    model.fit(x_train, y_train)
                    # print(
                    #     f'icon{icon_list[i]}_{axis_list[a]}_{test_list[t]}\ttrain_set score:\t{model.score(x_train, y_train)}')
                    # print(
                    #     f'icon{icon_list[i]}_{axis_list[a]}_{test_list[t]}\ttest_set score:\t\t{model.score(x_test, y_test)}')

                    y_predict = model.predict(x_data)
                    y_predict = np.append(
                        np.array([None]*(test_list[t][0]+test_list[t][1]-1)), y_predict)

                    table['icon'+icon_list[i]+'_'+axis_list[a] +
                          '_'+str(test_list[t])] = y_predict\

        table.to_csv('data/clock_data_predict.csv')
        print("clock_data_predict.csv 저장 완료")

    def phase_clustering(self):
        # 랜덤포레스트 모델로 예측한 국면 지표 예측값 입력
        economic = pd.read_csv('data/clock_data_predict_modify.csv')

        # [12, 1] 12개월 데이터로 한 달 후의 데이터 예측값
        economic_use = economic[['icon01_nx_[12, 1]', 'icon01_ny_[12, 1]', 'icon02_nx_[12, 1]', 'icon02_ny_[12, 1]',
                                 'icon03_nx_[12, 1]', 'icon03_ny_[12, 1]', 'icon04_nx_[12, 1]', 'icon04_ny_[12, 1]',
                                 'icon05_nx_[12, 1]', 'icon05_ny_[12, 1]', 'icon06_nx_[12, 1]', 'icon06_ny_[12, 1]',
                                 'icon07_nx_[12, 1]', 'icon07_ny_[12, 1]', 'icon08_nx_[12, 1]', 'icon08_ny_[12, 1]',
                                 'icon11_nx_[12, 1]', 'icon11_ny_[12, 1]', 'icon15_nx_[12, 1]', 'icon15_ny_[12, 1]']]

        # k-means clustering을 통해 각 경기 국면에 대한 라벨링
        clusters = 6
        kmeans = KMeans(n_clusters=clusters)
        kmeans.fit(economic_use)

        # 경기 국면 라벨링 데이터 엑셀로 출력
        economic_use['cluster_label'] = kmeans.labels_
        economic_use['date'] = economic['date']
        economic_use.to_csv('data/clock_data_predict_label.csv')
        print("data/clock_data_predict_label.csv 저장 완료")
