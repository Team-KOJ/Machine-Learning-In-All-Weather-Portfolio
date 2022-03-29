import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt


predic = pd.read_csv('data/clock_data_final.csv', low_memory=False)
indexing_predic = predic.set_index('date')  # 인덱스 날짜로 변경

bt = pd.read_csv('data/data_d.csv')
NaN_processing_bt = bt.fillna(0)  # nan 0으로 처리
indexing_bt = NaN_processing_bt.set_index('date')  # 인덱스 날짜로 변경


class Strategy:

    def __init__(self):
        while (1):
            try:
                print("\n전략 실행은 2003-03 부터 가능합니다. YYYY-MM 양식을 지켜주세요")
                self.S = input("전략실행 시작 연-월을 입력하세요. 월초부터 실행합니다. ex)2003-03 : ")
                print("\n전략 실행은 2021-05 까지 가능합니다. YYYY-MM 양식을 지켜주세요")
                pre_E = input("전략실행 종료 연-월을 입력하세요. 월말까지 실행합니다. ex)2020-12 : ")
                self.E = ''.join(indexing_predic.index[(
                    predic.index[predic.index[predic['date'] == pre_E]] + 1).tolist()])
            except:
                print('\n---------------------------------------------------')
                print(" 날짜입력 오류입니다. 1월인경우 01로 입력 바랍니다.")
                print('---------------------------------------------------\n')
                print("아무키나 입력하면 시작시간부터 다시 설정합니다.")
                self.Q = input("혹시 프로그램을 종료하고 싶다면 q를 입력해주세요.(대소문자 모두 가능) : ")
                print('\n\n\n')
                if self.Q == 'q' or self.Q == 'Q':
                    print('\n프로그램을 종료합니다. 감사합니다.')
                    exit()
                else:
                    continue

            self.predic12_1 = indexing_predic.loc[:, 'economic_cycle_12_1']
            self.predic12_1_set_SE = indexing_predic.loc[self.S: self.E]

            if self.predic12_1_set_SE.empty:
                print('\n---------------------------------------------------')
                print(" 날짜입력 오류입니다. 1월인경우 01로 입력 바랍니다.")
                print('---------------------------------------------------\n')
                print("아무키나 입력하면 시작시간부터 다시 설정합니다.")
                self.Q = input("혹시 프로그램을 종료하고 싶다면 q를 입력해주세요.(대소문자 모두 가능) : ")
                print('\n\n\n')
                if self.Q == 'q' or self.Q == 'Q':
                    print('\n프로그램을 종료합니다. 감사합니다.')
                    exit()
                else:
                    continue

            elif self.predic12_1_set_SE.index[0] == '2003-02':
                print('\n---------------------------------------------------')
                print(" 날짜입력 오류입니다. 2003-03 이후를 입력하시길 바랍니다.")
                print('---------------------------------------------------\n')
                print("아무키나 입력하면 시작시간부터 다시 설정합니다.")
                self.Q = input("혹시 프로그램을 종료하고 싶다면 q를 입력해주세요.(대소문자 모두 가능) : ")
                print('\n\n\n')
                if self.Q == 'q' or self.Q == 'Q':
                    print('\n프로그램을 종료합니다. 감사합니다.')
                    exit()
                else:
                    continue

            else:
                break

        self.Colected_Data_fix = pd.DataFrame()
        self.Colected_Data_fix = indexing_bt.loc['{}'.format(
            self.S):'{}'.format(self.E)]

    def Run(self):
        print('\n\n')
        self.Fixed_Allweather6_4 = pd.DataFrame(self.Backtesting_manualset(60, 20, 20, 0, 0, self.Colected_Data_fix),
                                                columns=['6_4'])
        self.Fixed_Allweather8_2 = pd.DataFrame(self.Backtesting_manualset(80, 10, 10, 0, 0, self.Colected_Data_fix),
                                                columns=['8_2'])
        self.Fixed_Allweather10_0 = pd.DataFrame(self.Backtesting_manualset(100, 0, 0, 0, 0, self.Colected_Data_fix),
                                                 columns=['10_0'])

        print('아래는 Reference인 All Weather Portfolio 입니다.')
        self.Fixed_Allweather_reference = pd.DataFrame(
            self.Backtesting_manualset(30, 40, 15, 7.5, 7.5, self.Colected_Data_fix), columns=['reference '])

        start = [100]
        array_for_STG_Averse = pd.DataFrame(start)
        array_for_STG_Neutral = pd.DataFrame(start)
        array_for_STG_Lover = pd.DataFrame(start)
        array_for_STG_No_Limit = pd.DataFrame(start)
        array_for_STG_Averse_var = []
        array_for_STG_Neutral_var = []
        array_for_STG_Lover_var = []
        array_for_STG_No_Limit_var = []

        print("Calculating our strategies...")

        for i in self.predic12_1_set_SE.index:

            if i == self.predic12_1_set_SE.index[-1]:
                break

            else:
                if self.predic12_1.loc[i] == '하강' or self.predic12_1.loc[i] == '급하강':

                    Colected_Data_Monthly = self.Select_Date_for_STG(i, ''.join(
                        self.predic12_1.index[(predic.index[predic.index[predic['date'] == i]] + 1).tolist()]))

                    array_for_STG_Neutral = pd.concat(
                        [array_for_STG_Neutral,
                         self.Backtesting_monthly(10, 50, 20, 10, 10, Colected_Data_Monthly,
                                                  array_for_STG_Neutral.iloc[-1], array_for_STG_Neutral_var)],
                        ignore_index=True)

                    Colected_Data_Monthly = self.Select_Date_for_STG(i, ''.join(
                        self.predic12_1.index[(predic.index[predic.index[predic['date'] == i]] + 1).tolist()]))
                    array_for_STG_Averse = pd.concat(
                        [array_for_STG_Averse,
                         self.Backtesting_monthly(10, 40, 30, 10, 10, Colected_Data_Monthly,
                                                  array_for_STG_Averse.iloc[-1], array_for_STG_Averse_var)],
                        ignore_index=True)

                    Colected_Data_Monthly = self.Select_Date_for_STG(i, ''.join(
                        self.predic12_1.index[(predic.index[predic.index[predic['date'] == i]] + 1).tolist()]))
                    array_for_STG_Lover = pd.concat(
                        [array_for_STG_Lover, self.Backtesting_monthly(10, 60, 10, 10, 10, Colected_Data_Monthly,
                                                                       array_for_STG_Lover.iloc[-1],
                                                                       array_for_STG_Lover_var)],
                        ignore_index=True)

                    Colected_Data_Monthly = self.Select_Date_for_STG(i, ''.join(
                        self.predic12_1.index[(predic.index[predic.index[predic['date'] == i]] + 1).tolist()]))
                    array_for_STG_No_Limit = pd.concat(
                        [array_for_STG_No_Limit, self.Backtesting_monthly(0, 35, 65, 0, 0, Colected_Data_Monthly,
                                                                          array_for_STG_No_Limit.iloc[-1],
                                                                          array_for_STG_No_Limit_var)],
                        ignore_index=True)

                elif self.predic12_1.loc[i] == '회복' or self.predic12_1.loc[i] == '급회복':

                    Colected_Data_Monthly = self.Select_Date_for_STG(i, ''.join(
                        self.predic12_1.index[(predic.index[predic.index[predic['date'] == i]] + 1).tolist()]))
                    array_for_STG_Neutral = pd.concat(
                        [array_for_STG_Neutral,
                         self.Backtesting_monthly(50, 10, 10, 12, 18, Colected_Data_Monthly,
                                                  array_for_STG_Neutral.iloc[-1], array_for_STG_Neutral_var)],
                        ignore_index=True)

                    Colected_Data_Monthly = self.Select_Date_for_STG(i, ''.join(
                        self.predic12_1.index[(predic.index[predic.index[predic['date'] == i]] + 1).tolist()]))
                    array_for_STG_Averse = pd.concat(
                        [array_for_STG_Averse,
                         self.Backtesting_monthly(40, 10, 10, 18, 22, Colected_Data_Monthly,
                                                  array_for_STG_Averse.iloc[-1], array_for_STG_Averse_var)],
                        ignore_index=True)

                    Colected_Data_Monthly = self.Select_Date_for_STG(i, ''.join(
                        self.predic12_1.index[(predic.index[predic.index[predic['date'] == i]] + 1).tolist()]))
                    array_for_STG_Lover = pd.concat(
                        [array_for_STG_Lover, self.Backtesting_monthly(60, 10, 10, 10, 10, Colected_Data_Monthly,
                                                                       array_for_STG_Lover.iloc[-1],
                                                                       array_for_STG_Lover_var)],
                        ignore_index=True)

                    Colected_Data_Monthly = self.Select_Date_for_STG(i, ''.join(
                        self.predic12_1.index[(predic.index[predic.index[predic['date'] == i]] + 1).tolist()]))
                    array_for_STG_No_Limit = pd.concat(
                        [array_for_STG_No_Limit, self.Backtesting_monthly(0, 25, 75, 0, 0, Colected_Data_Monthly,
                                                                          array_for_STG_No_Limit.iloc[-1],
                                                                          array_for_STG_No_Limit_var)],
                        ignore_index=True)

                elif self.predic12_1.loc[i] == '상승':

                    if (predic.index[predic.index[predic['date'] == i]] + 1).tolist()[0] < len(indexing_predic.index):
                        Colected_Data_Monthly = self.Select_Date_for_STG(i, ''.join(
                            self.predic12_1.index[(predic.index[predic.index[predic['date'] == i]] + 1).tolist()]))
                        array_for_STG_Neutral = pd.concat(
                            [array_for_STG_Neutral,
                             self.Backtesting_monthly(10, 20, 10, 20, 40, Colected_Data_Monthly,
                                                      array_for_STG_Neutral.iloc[-1],
                                                      array_for_STG_Neutral_var)],
                            ignore_index=True)

                        Colected_Data_Monthly = self.Select_Date_for_STG(i, ''.join(
                            self.predic12_1.index[(predic.index[predic.index[predic['date'] == i]] + 1).tolist()]))
                        array_for_STG_Averse = pd.concat(
                            [array_for_STG_Averse,
                             self.Backtesting_monthly(10, 30, 10, 16, 34, Colected_Data_Monthly,
                                                      array_for_STG_Averse.iloc[-1], array_for_STG_Averse_var)],
                            ignore_index=True)

                        Colected_Data_Monthly = self.Select_Date_for_STG(i, ''.join(
                            self.predic12_1.index[(predic.index[predic.index[predic['date'] == i]] + 1).tolist()]))
                        array_for_STG_Lover = pd.concat(
                            [array_for_STG_Lover,
                             self.Backtesting_monthly(10, 10, 10, 24, 46, Colected_Data_Monthly,
                                                      array_for_STG_Lover.iloc[-1], array_for_STG_Lover_var)],
                            ignore_index=True)

                        Colected_Data_Monthly = self.Select_Date_for_STG(i, ''.join(
                            self.predic12_1.index[(predic.index[predic.index[predic['date'] == i]] + 1).tolist()]))
                        array_for_STG_No_Limit = pd.concat(
                            [array_for_STG_No_Limit,
                             self.Backtesting_monthly(0, 100, 0, 0, 0, Colected_Data_Monthly,
                                                      array_for_STG_No_Limit.iloc[-1],
                                                      array_for_STG_No_Limit_var)],
                            ignore_index=True)

                    else:
                        array_for_STG_Neutral = array_for_STG_Neutral.drop(
                            index=0, axis=0)
                        array_for_STG_Averse = array_for_STG_Averse.drop(
                            index=0, axis=0)
                        array_for_STG_Lover = array_for_STG_Lover.drop(
                            index=0, axis=0)
                        array_for_STG_No_Limit = array_for_STG_No_Limit.drop(
                            index=0, axis=0)
                        continue

                elif self.predic12_1.loc[i] == '둔화':

                    Colected_Data_Monthly = self.Select_Date_for_STG(i, ''.join(
                        self.predic12_1.index[(predic.index[predic.index[predic['date'] == i]] + 1).tolist()]))
                    array_for_STG_Neutral = pd.concat(
                        [array_for_STG_Neutral,
                         self.Backtesting_monthly(10, 50, 20, 10, 10, Colected_Data_Monthly,
                                                  array_for_STG_Neutral.iloc[-1], array_for_STG_Neutral_var)],
                        ignore_index=True)

                    Colected_Data_Monthly = self.Select_Date_for_STG(i, ''.join(
                        self.predic12_1.index[(predic.index[predic.index[predic['date'] == i]] + 1).tolist()]))
                    array_for_STG_Averse = pd.concat(
                        [array_for_STG_Averse,
                         self.Backtesting_monthly(10, 40, 30, 10, 10, Colected_Data_Monthly,
                                                  array_for_STG_Averse.iloc[-1], array_for_STG_Averse_var)],
                        ignore_index=True)

                    Colected_Data_Monthly = self.Select_Date_for_STG(i, ''.join(
                        self.predic12_1.index[(predic.index[predic.index[predic['date'] == i]] + 1).tolist()]))
                    array_for_STG_Lover = pd.concat(
                        [array_for_STG_Lover, self.Backtesting_monthly(10, 60, 10, 10, 10, Colected_Data_Monthly,
                                                                       array_for_STG_Lover.iloc[-1],
                                                                       array_for_STG_Lover_var)],
                        ignore_index=True)

                    Colected_Data_Monthly = self.Select_Date_for_STG(i, ''.join(
                        self.predic12_1.index[(predic.index[predic.index[predic['date'] == i]] + 1).tolist()]))
                    array_for_STG_No_Limit = pd.concat(
                        [array_for_STG_No_Limit, self.Backtesting_monthly(0, 60, 40, 0, 0, Colected_Data_Monthly,
                                                                          array_for_STG_No_Limit.iloc[-1],
                                                                          array_for_STG_No_Limit_var)],
                        ignore_index=True)

                else:
                    print('Something.. Error')

        Averse_var = np.array(array_for_STG_Averse_var).var()
        Averse_standard_deviation = np.sqrt(Averse_var)
        array_for_STG_Averse = array_for_STG_Averse.drop([0])
        print("\n전략 Risk-Averse 10-40 의 결과")
        print('Return %: {}, Sharpe: {}, Volatility: {}\n'.format(round((array_for_STG_Averse.iloc[-1]) - 100, 4)[0],
                                                                  round(((array_for_STG_Averse.iloc[
                                                                      -1] - 100) / 100) / Averse_standard_deviation,
            4)[0], round(Averse_standard_deviation, 4)))

        Neutral_var = np.array(array_for_STG_Neutral_var).var()
        Neutral_standard_deviation = np.sqrt(Neutral_var)
        array_for_STG_Neutral = array_for_STG_Neutral.drop([0])
        print("전략 Risk-Neutral 10-50 의 결과 ")
        print('Return %: {}, Sharpe: {}, Volatility: {}\n'.format(round((array_for_STG_Neutral.iloc[-1]) - 100, 4)[0],
                                                                  round(((array_for_STG_Neutral.iloc[
                                                                      -1] - 100) / 100) / Neutral_standard_deviation,
            4)[0], round(Neutral_standard_deviation, 4)))

        Lover_var = np.array(array_for_STG_Lover_var).var()
        Lover_standard_deviation = np.sqrt(Lover_var)
        array_for_STG_Lover = array_for_STG_Lover.drop([0])
        print("전략 Risk-Lover 10-60 의 결과")
        print('Return %: {}, Sharpe: {}, Volatility: {}\n'.format(round((array_for_STG_Lover.iloc[-1]) - 100, 4)[0],
                                                                  round(((array_for_STG_Lover.iloc[
                                                                      -1] - 100) / 100) / Lover_standard_deviation,
            4)[0], round(Lover_standard_deviation, 4)))

        Nolimit_var = np.array(array_for_STG_No_Limit_var).var()
        Nolimit_standard_deviation = np.sqrt(Nolimit_var)
        array_for_STG_No_Limit = array_for_STG_No_Limit.drop([0])
        # print("전략 No-Limit 의 결과 ")
        # print('Return %: {}, Sharpe: {}, Volatility: {}\n'.format((round(array_for_STG_No_Limit.iloc[-1], 4)-100)[0], round(((array_for_STG_No_Limit.iloc[-1]-100)/100) / Nolimit_standard_deviation, 4)[0], round(Nolimit_standard_deviation, 4)))

        x = self.Fixed_Allweather6_4.index  # x축을 시간으로 지정
        y = self.Fixed_Allweather6_4.to_numpy()
        y1 = self.Fixed_Allweather8_2.to_numpy()
        y2 = self.Fixed_Allweather10_0.to_numpy()
        y10 = self.Fixed_Allweather_reference.to_numpy()

        y3 = array_for_STG_Averse.to_numpy() - 100
        y4 = array_for_STG_Neutral.to_numpy() - 100
        y5 = array_for_STG_Lover.to_numpy() - 100
        y6 = array_for_STG_No_Limit.to_numpy() - 100

        plt.figure(figsize=(30, 9))
        plt.plot(x, y, c='blue', alpha=0.2,
                 label='Stock 60%, Bond 40% Portfolio')
        plt.plot(x, y1, c='orange', alpha=0.2,
                 label='Stock 80%, Bond 20% Portfolio')
        plt.plot(x, y2, c='green', alpha=0.2, label='Stock 100% Portfolio')
        plt.plot(x, y10, c='black', alpha=0.6,
                 label='Reference All Weather Portfolio')

        plt.plot(x, y3, c='red', label='Our strategy Risk Averse')
        plt.plot(x, y4, c='purple', label='Our strategy Risk Neutral')
        plt.plot(x, y5, c='orchid', label='Our strategy Risk Lover')

        """
        plt.plot(x,y6, c='olive', label = 'No Limit')
        plt.plot(x,y11, c='red', label = 'k10 100')
        plt.plot(x,y12, c='purple', label = 'k5 100')
        plt.plot(x,y13, c='orchid', label = 'gold 100')
        plt.plot(x,y14, c='olive', label = 'commodity 100 ')
        """
        plt.legend(loc='upper left', frameon=False, fontsize=15)
        plt.axhline(y=0, color='black', linewidth=1, alpha=0.5, linestyle='--')

        # plt.xticks([0, 1480, 1600, 4950, 5065])
        # plt.xticks([0, 1367, 2733, 4100, 5465], labels = ['2003-03', '2007-09', '2012-04', '2016-11',  '2021-06'])

        if len(self.Fixed_Allweather6_4) < 31:
            plt.xticks([0, len(self.Fixed_Allweather6_4)],
                       labels=[self.S, self.E])
        elif 55 < len(self.Fixed_Allweather6_4) < 75:
            plt.xticks([0, len(self.Fixed_Allweather6_4)],
                       labels=[self.S, self.E])
        else:
            plt.xticks([0, int(len(self.Fixed_Allweather6_4) / 2), len(self.Fixed_Allweather6_4)],
                       labels=[self.S, ''.join(indexing_predic.index[int(len(self.predic12_1_set_SE) / 2) + 1]), self.E])
        plt.grid(True)
        plt.ylabel('Percent %')
        plt.show()

    def Backtesting_manualset(self, a, b, c, d, e, Data):

        kospi = a
        k10bond = b
        k5bond = c
        gold = d
        commodity = e

        self.list_for_percent = []
        list_for_var = []

        R = None

        for i in range(0, len(Data)):
            # [i] 값은 행을 의미함. (1+값)인 이유는 증감률이니깐
            gold = gold * (1 + Data['gold'][i])
            k10bond = k10bond * (1 + Data['k10bond'][i])
            k5bond = k5bond * (1 + Data['k5bond'][i])
            kospi = kospi * (1 + Data['kospi'][i])
            commodity = commodity * (1 + Data['commodity'][i])

            sum_for_var = gold + k10bond + k5bond + kospi + commodity
            self.list_for_percent.append(sum_for_var - 100)

            if R:  # 분산 구하기 위한 짓거리..
                r = (sum_for_var - R) / R * 100
                list_for_var.append(r)
            R = sum_for_var

        rate_of_return = (gold + k10bond + k5bond +
                          kospi + commodity - 100)  # 퍼센트임
        var_portfolio = np.array(list_for_var).var()  # 분산_ 모집단 전체 대상(varp)
        Volatility_standard_deviation = np.sqrt(var_portfolio)
        # expected_return / portfolio_volatility
        Sharpe_ratio = (rate_of_return / 100) / Volatility_standard_deviation

        # print(var_portfolio)
        # print(Volatility_standard_deviation)
        return_dataframe = pd.DataFrame(
            {'Return %': [round(rate_of_return, 4)], 'Sharpe': [round(Sharpe_ratio, 4)],
             'Volatility': [round(Volatility_standard_deviation, 4)]})

        print(
            "코스피지수 주식 {}%, 한국채 10년 {}%, 한국채 5년 {}%, 금 {}%, 원자재 {}% 포트폴리오 의 결과 ".format(
                a, b, c, d, e))

        print('Return %: {}, Sharpe: {}, Volatility: {}\n'.format(round(return_dataframe['Return %'][0], 4),
                                                                  round(
                                                                      return_dataframe['Sharpe'][0], 4),
                                                                  round(return_dataframe['Volatility'][0], 4)))

        return self.list_for_percent

    def Backtesting_monthly(self, a, b, c, d, e, Data, pre_sum, for_var):

        kospi = pre_sum * a / 100
        k10bond = pre_sum * b / 100
        k5bond = pre_sum * c / 100
        gold = pre_sum * d / 100
        commodity = pre_sum * e / 100

        list_for_mdd = []

        R = pre_sum
        for i in range(0, len(Data)):
            # [i] 값은 행을 의미함. (1+값)인 이유는 증감률이니깐
            gold = gold * (1 + Data['gold'][i])
            k10bond = k10bond * (1 + Data['k10bond'][i])
            k5bond = k5bond * (1 + Data['k5bond'][i])
            kospi = kospi * (1 + Data['kospi'][i])
            commodity = commodity * (1 + Data['commodity'][i])

            sum_for_mdd_var = gold + k10bond + k5bond + kospi + commodity
            list_for_mdd.append(sum_for_mdd_var)

            r = (sum_for_mdd_var - R) / R * 100
            for_var.append(r)

            R = sum_for_mdd_var

        self.df = pd.DataFrame(list_for_mdd)

        return self.df

    def Select_Date_for_STG(self, Start, End):  # 구간 데이터 모으는 함수

        self.total = indexing_bt.loc['{}'.format(Start):'{}'.format(End)]

        return self.total


class Get_weight:

    def __init__(self):

        while(1):
            print('이 작업은 2003년 3월부터 ')
            try:
                self.G = int(input(
                    "Backtesting 하고싶은 '국면'을 숫자로 입력합니다. \n회복 : 0 \n상승 : 1 \n둔화 : 2 \n하강 : 3 \n중에서 하나를 입력하세요. : "))
                if self.G == 0:
                    print('\n[회복 국면을 선택하셨습니다.]')
                    break
                elif self.G == 1:
                    print('\n[상승 국면을 선택하셨습니다.]')
                    break
                elif self.G == 2:
                    print('\n[둔화 국면을 선택하셨습니다.]')
                    break
                elif self.G == 3:
                    print('\n[하강 국면을 선택하셨습니다.]')
                    break
                else:
                    print('\n--------------------')
                    print("숫자를 다시 입력해주세요")
                    print('--------------------\n')
                    continue
            except:
                print('\n--------------------')
                print("숫자를 다시 입력해주세요")
                print('--------------------\n')
                continue

        while (1):
            try:

                self.H = int(input(
                    "\n이제 'Risk' 를 입력합니다. \nAverse : 0 \nNeutral : 1 \nLover : 2 \n중에서 하나를 입력하세요. : "))
                if self.H == 0:
                    print('\n[Risk Averse를 선택하셨습니다.]\n')
                    break
                elif self.H == 1:
                    print('\n[Risk Neutral을 선택하셨습니다.]\n')
                    break
                elif self.H == 2:
                    print('\n[Risk Lover를 선택하셨습니다.]\n')
                    break
                else:
                    print('\n--------------------')
                    print("숫자를 다시 입력해주세요")
                    print('--------------------\n')
                    continue
            except:
                print('\n--------------------')
                print("숫자를 다시 입력해주세요")
                print('--------------------\n')
                continue

    def Run(self):
        self.Colected_Data = pd.DataFrame()
        array_for_bt = pd.DataFrame()

        if self.G == 0:
            self.Colected_Data = self.Select_Date(
                '2003-08', '2004-02', self.Colected_Data)  # 회급회
            self.Colected_Data = self.Select_Date(
                '2005-06', '2006-02', self.Colected_Data)
            self.Colected_Data = self.Select_Date(
                '2006-09', '2006-12', self.Colected_Data)
            self.Colected_Data = self.Select_Date(
                '2009-02', '2009-09', self.Colected_Data)
            self.Colected_Data = self.Select_Date(
                '2009-09', '2010-06', self.Colected_Data)
            self.Colected_Data = self.Select_Date(
                '2013-08', '2014-02', self.Colected_Data)
            self.Colected_Data = self.Select_Date(
                '2016-12', '2017-07', self.Colected_Data)
            self.Colected_Data = self.Select_Date(
                '2020-07', '2020-11', self.Colected_Data)
            self.Colected_Data = self.Select_Date(
                '2020-11', '2021-03', self.Colected_Data)
        elif self.G == 1:
            self.Colected_Data = self.Select_Date(
                '2004-02', '2004-04', self.Colected_Data)  # 상승
            self.Colected_Data = self.Select_Date(
                '2006-12', '2008-04', self.Colected_Data)
            self.Colected_Data = self.Select_Date(
                '2010-06', '2011-09', self.Colected_Data)
            self.Colected_Data = self.Select_Date(
                '2017-07', '2018-06', self.Colected_Data)
            self.Colected_Data = self.Select_Date(
                '2021-03', '2021-07', self.Colected_Data)
        elif self.G == 2:
            self.Colected_Data = self.Select_Date(
                '2003-02', '2003-05', self.Colected_Data)  # 둔화
            self.Colected_Data = self.Select_Date(
                '2004-04', '2004-12', self.Colected_Data)
            self.Colected_Data = self.Select_Date(
                '2006-01', '2006-08', self.Colected_Data)
            self.Colected_Data = self.Select_Date(
                '2008-04', '2008-08', self.Colected_Data)
            self.Colected_Data = self.Select_Date(
                '2011-09', '2012-09', self.Colected_Data)
            self.Colected_Data = self.Select_Date(
                '2014-02', '2015-06', self.Colected_Data)
            self.Colected_Data = self.Select_Date(
                '2018-06', '2019-01', self.Colected_Data)
            self.Colected_Data = self.Select_Date(
                '2019-12', '2020-02', self.Colected_Data)
        elif self.G == 3:
            self.Colected_Data = self.Select_Date(
                '2003-04', '2003-08', self.Colected_Data)  # 하급강
            self.Colected_Data = self.Select_Date(
                '2004-12', '2005-06', self.Colected_Data)
            self.Colected_Data = self.Select_Date(
                '2008-08', '2009-02', self.Colected_Data)
            self.Colected_Data = self.Select_Date(
                '2012-09', '2013-08', self.Colected_Data)
            self.Colected_Data = self.Select_Date(
                '2015-06', '2016-11', self.Colected_Data)
            self.Colected_Data = self.Select_Date(
                '2019-01', '2019-11', self.Colected_Data)
            self.Colected_Data = self.Select_Date(
                '2020-02', '2020-06', self.Colected_Data)

        else:
            print("Error\n")
            self.__init__()

        if self.H == 0:

            array_for_bt = self.Backtesting_averse(self.Colected_Data)
            self.Analysis(array_for_bt)
            self.Scatter(array_for_bt)

        elif self.H == 1:
            array_for_bt = self.Backtesting_neutral(self.Colected_Data)
            self.Analysis(array_for_bt)
            self.Scatter(array_for_bt)

        elif self.H == 2:
            array_for_bt = self.Backtesting_lover(self.Colected_Data)
            self.Analysis(array_for_bt)
            self.Scatter(array_for_bt)
        else:
            print("Error\n")
            self.__init__()

    def Select_Date(self, Start, End, Dataframe):  # 구간 데이터 모으는 함수

        total = indexing_bt.loc['{}'.format(Start):'{}'.format(End)]

        self.Dataframe = pd.concat([Dataframe, total])  # 리스트형식.. tqtqtqtqtqtq
        return self.Dataframe

    def Backtesting_averse(self, Data):
        self.array_bt_all = pd.DataFrame()
        t = 0
        for a in range(0, 16):
            print("Backtesting Risk Averse {} % percent..".format(int(a * 6.7)))
            a += 5
            for b in range(0, 16):
                b += 5
                if a + b > 35:
                    break
                for c in range(0, 16):
                    c += 5
                    if a + b + c > 40:
                        break
                    for d in range(0, 16):
                        d += 5
                        if a + b + c + d > 45:
                            break

                        e = 50 - a - b - c - d
                        if e > 20:
                            break

                        random_weight = np.array([a, b, c, d, e]) * 2

                        kospi = random_weight[0]
                        k10bond = random_weight[1]
                        k5bond = random_weight[2]
                        gold = random_weight[3]
                        commodity = random_weight[4]

                        list_for_mdd = []
                        list_for_var = []

                        R = None

                        for i in range(0, len(Data)):
                            # [i] 값은 행을 의미함. (1+값)인 이유는 증감률이니깐
                            gold = gold * (1 + Data['gold'][i])
                            k10bond = k10bond * (1 + Data['k10bond'][i])
                            k5bond = k5bond * (1 + Data['k5bond'][i])
                            kospi = kospi * (1 + Data['kospi'][i])
                            commodity = commodity * (1 + Data['commodity'][i])

                            sum_for_mdd_var = gold + k10bond + k5bond + kospi + commodity
                            list_for_mdd.append(sum_for_mdd_var)

                            if R:  # 분산 구하기 위한 짓거리..
                                r = (sum_for_mdd_var - R) / R * 100
                                list_for_var.append(r)

                            R = sum_for_mdd_var

                        rate_of_return = (
                            gold + k10bond + k5bond + kospi + commodity - 100)  # 퍼센트임

                        Max = max(list_for_mdd)
                        Min = min(list_for_mdd)
                        Mdd = (Min - Max) / Max * 100

                        var_portfolio = np.array(
                            list_for_var).var()  # 분산_ 모집단 전체 대상 varp
                        Volatility_standard_deviation = np.sqrt(var_portfolio)
                        Sharpe_ratio = (
                            rate_of_return / 100) / Volatility_standard_deviation  # expected_return / portfolio_volatility
                        # print(var_portfolio)
                        # print(Volatility_standard_deviation)

                        infor_dataframe = pd.DataFrame(
                            {'Return %': [round(rate_of_return, 4)], 'MDD %': [round(Mdd, 4)],
                             'Sharpe': [round(Sharpe_ratio, 4)],
                             'Volatility': [round(Volatility_standard_deviation, 4)],
                             'kospi': [random_weight[0]], 'k10bond': [random_weight[1]],
                             'k5bond': [random_weight[2]], 'gold': [random_weight[3]], 'commodity': [random_weight[4]]},
                            index=[t])

                        self.array_bt_all = pd.concat(
                            [self.array_bt_all, infor_dataframe])

                        t += 1

        return self.array_bt_all

    def Backtesting_neutral(self, Data):
        self.array_bt_all = pd.DataFrame()
        t = 0
        for a in range(0, 21):  # 21
            print("Backtesting Risk Neutral {} % percent..".format(a*5))
            a += 5
            for b in range(0, 21):
                b += 5
                if a + b > 35:
                    break
                for c in range(0, 21):
                    c += 5
                    if a + b + c > 40:
                        break
                    for d in range(0, 21):
                        d += 5
                        if a + b + c + d > 45:
                            break
                        e = 50 - a - b - c - d

                        if e > 25:
                            break

                        random_weight = np.array([a, b, c, d, e]) * 2

                        kospi = random_weight[0]
                        k10bond = random_weight[1]
                        k5bond = random_weight[2]
                        gold = random_weight[3]
                        commodity = random_weight[4]

                        list_for_mdd = []
                        list_for_var = []

                        R = None

                        for i in range(0, len(Data)):
                            # [i] 값은 행을 의미함. (1+값)인 이유는 증감률이니깐
                            gold = gold * (1 + Data['gold'][i])
                            k10bond = k10bond * (1 + Data['k10bond'][i])
                            k5bond = k5bond * (1 + Data['k5bond'][i])
                            kospi = kospi * (1 + Data['kospi'][i])
                            commodity = commodity * (1 + Data['commodity'][i])

                            sum_for_mdd_var = gold + k10bond + k5bond + kospi + commodity
                            list_for_mdd.append(sum_for_mdd_var)

                            if R:  # 분산 구하기 위한 짓거리..
                                r = (sum_for_mdd_var - R) / R * 100
                                list_for_var.append(r)

                            R = sum_for_mdd_var

                        rate_of_return = (
                            gold + k10bond + k5bond + kospi + commodity - 100)  # 퍼센트임

                        Max = max(list_for_mdd)
                        Min = min(list_for_mdd)
                        Mdd = (Min - Max) / Max * 100

                        var_portfolio = np.array(
                            list_for_var).var()  # 분산_ 모집단 전체 대상 varp
                        Volatility_standard_deviation = np.sqrt(var_portfolio)
                        Sharpe_ratio = (
                            rate_of_return / 100) / Volatility_standard_deviation  # expected_return / portfolio_volatility
                        # print(var_portfolio)
                        # print(Volatility_standard_deviation)

                        infor_dataframe = pd.DataFrame(
                            {'Return %': [round(rate_of_return, 4)], 'MDD %': [round(Mdd, 4)],
                             'Sharpe': [round(Sharpe_ratio, 4)],
                             'Volatility': [round(Volatility_standard_deviation, 4)],
                             'kospi': [random_weight[0]], 'k10bond': [random_weight[1]],
                             'k5bond': [random_weight[2]], 'gold': [random_weight[3]], 'commodity': [random_weight[4]]},
                            index=[t])

                        self.array_bt_all = pd.concat(
                            [self.array_bt_all, infor_dataframe])

                        t += 1

        return self.array_bt_all

    def Backtesting_lover(self, Data):
        self.array_bt_all = pd.DataFrame()
        t = 0
        for a in range(0, 26):
            print("Backtesting Risk Lover {} % percent..".format(int(a * 4)))
            a += 5
            for b in range(0, 26):
                b += 5
                if a + b > 35:
                    break
                for c in range(0, 26):
                    c += 5
                    if a + b + c > 40:
                        break
                    for d in range(0, 26):
                        d += 5
                        if a + b + c + d > 45:
                            break

                        e = 50 - a - b - c - d
                        if e > 30:
                            break

                        random_weight = np.array([a, b, c, d, e]) * 2

                        kospi = random_weight[0]
                        k10bond = random_weight[1]
                        k5bond = random_weight[2]
                        gold = random_weight[3]
                        commodity = random_weight[4]

                        list_for_mdd = []
                        list_for_var = []

                        R = None

                        for i in range(0, len(Data)):
                            # [i] 값은 행을 의미함. (1+값)인 이유는 증감률이니깐
                            gold = gold * (1 + Data['gold'][i])
                            k10bond = k10bond * (1 + Data['k10bond'][i])
                            k5bond = k5bond * (1 + Data['k5bond'][i])
                            kospi = kospi * (1 + Data['kospi'][i])
                            commodity = commodity * (1 + Data['commodity'][i])

                            sum_for_mdd_var = gold + k10bond + k5bond + kospi + commodity
                            list_for_mdd.append(sum_for_mdd_var)

                            if R:  # 분산 구하기 위한 짓거리..
                                r = (sum_for_mdd_var - R) / R * 100
                                list_for_var.append(r)

                            R = sum_for_mdd_var

                        rate_of_return = (
                            gold + k10bond + k5bond + kospi + commodity - 100)  # 퍼센트임

                        Max = max(list_for_mdd)
                        Min = min(list_for_mdd)
                        Mdd = (Min - Max) / Max * 100

                        var_portfolio = np.array(
                            list_for_var).var()  # 분산_ 모집단 전체 대상 varp
                        Volatility_standard_deviation = np.sqrt(var_portfolio)
                        Sharpe_ratio = (
                            rate_of_return / 100) / Volatility_standard_deviation  # expected_return / portfolio_volatility
                        # print(var_portfolio)
                        # print(Volatility_standard_deviation)

                        infor_dataframe = pd.DataFrame(
                            {'Return %': [round(rate_of_return, 4)], 'MDD %': [round(Mdd, 4)],
                             'Sharpe': [round(Sharpe_ratio, 4)],
                             'Volatility': [round(Volatility_standard_deviation, 4)],
                             'kospi': [random_weight[0]], 'k10bond': [random_weight[1]],
                             'k5bond': [random_weight[2]], 'gold': [random_weight[3]], 'commodity': [random_weight[4]]},
                            index=[t])

                        self.array_bt_all = pd.concat(
                            [self.array_bt_all, infor_dataframe])

                        t += 1

        return self.array_bt_all

    def Analysis(self, Dataframe):

        print('----------Return Max------------')
        print(Dataframe.loc[Dataframe.idxmax()[0]])

        print('----------Sharpe Max------------')
        print(Dataframe.loc[Dataframe.idxmax()[2]])
        print('----------Volatility Min------------')
        print(Dataframe.loc[Dataframe.idxmin()[3]])
        print('----------MDD Min------------')
        print(Dataframe.loc[Dataframe.idxmax()[1]])
        """
        print('----------kospi Max------------')
        print(Dataframe.loc[Dataframe.idxmax()[4]])
        print('----------k10bond Max------------')
        print(Dataframe.loc[Dataframe.idxmax()[5]])
        print('----------k5bond Max------------')
        print(Dataframe.loc[Dataframe.idxmax()[6]])
        print('----------gold Max------------')
        print(Dataframe.loc[Dataframe.idxmax()[7]])
        print('----------commodity Max------------')
        print(Dataframe.loc[Dataframe.idxmax()[8]])
        """

        self.KNN(Dataframe)

    def KNN(self, Dataframe):

        M = 10  # 이웃 몇개 채취할거임?

        Sharpe = Dataframe.iloc[:, 2]  # 샤프
        Sharpe = Sharpe.values.reshape(len(Dataframe), 1)  # fit 시키기 위해 변형

        Sharpe_neighbor = NearestNeighbors(n_neighbors=M, radius=0.4)
        Sharpe_neighbor.fit(Sharpe)

        Return = Dataframe.iloc[:, 0]  # 샤프
        Return = Return.values.reshape(len(Dataframe), 1)  # fit 시키기 위해 변형

        Return_neighbor = NearestNeighbors(n_neighbors=M, radius=0.4)
        Return_neighbor.fit(Return)

        MDD = Dataframe.iloc[:, 1]  # 샤프
        MDD = MDD.values.reshape(len(Dataframe), 1)  # fit 시키기 위해 변형

        MDD_neighbor = NearestNeighbors(n_neighbors=M, radius=0.4)
        MDD_neighbor.fit(MDD)

        Volatility = Dataframe.iloc[:, 3]  # 샤프
        Volatility = Volatility.values.reshape(
            len(Dataframe), 1)  # fit 시키기 위해 변형

        Volatility_neighbor = NearestNeighbors(n_neighbors=M, radius=0.4)
        Volatility_neighbor.fit(Volatility)

        # print(Dataframe.loc[Dataframe.idxmax()[2]])
        # print(Dataframe.iloc[Dataframe.idxmax()[2], 2])

        nparray1 = Sharpe_neighbor.kneighbors(Dataframe.iloc[Dataframe.idxmax()[2], 2].reshape(-1, 1), M,
                                              return_distance=False)

        kospi = 0
        k10bond = 0
        k5bond = 0
        gold = 0
        commodity = 0

        for i in range(0, M):
            kospi += Dataframe.loc[nparray1[0][i], 'kospi']
            k10bond += Dataframe.loc[nparray1[0][i], 'k10bond']
            k5bond += Dataframe.loc[nparray1[0][i], 'k5bond']
            gold += Dataframe.loc[nparray1[0][i], 'gold']
            commodity += Dataframe.loc[nparray1[0][i], 'commodity']

        print("\n{}개의 포트폴리오 중에서".format(len(Dataframe)))
        print(
            "높은 Sharpe 지수를 가진 상위 {}개의 포트폴리오를 샘플링하여 평균낸 결과 \n코스피지수 주식 {}%, 한국채 10년 {}%, 한국채 5년 {}%, 금 {}%, 원자재 {}% 포트폴리오가 제안되었습니다. \n".format(
                round(M, 0), round(kospi / M, 1), round(k10bond / M,
                                                        1), round(k5bond / M, 1), round(gold / M, 1),
                round(commodity / M, 1)))

        nparray1 = Return_neighbor.kneighbors(Dataframe.iloc[Dataframe.idxmax()[0], 0].reshape(-1, 1), M,
                                              return_distance=False)
        kospi = 0
        k10bond = 0
        k5bond = 0
        gold = 0
        commodity = 0

        for i in range(0, M):
            kospi += Dataframe.loc[nparray1[0][i], 'kospi']
            k10bond += Dataframe.loc[nparray1[0][i], 'k10bond']
            k5bond += Dataframe.loc[nparray1[0][i], 'k5bond']
            gold += Dataframe.loc[nparray1[0][i], 'gold']
            commodity += Dataframe.loc[nparray1[0][i], 'commodity']

        print(
            "높은 Return 지수를 가진 상위 {}개의 포트폴리오를 샘플링하여 평균낸 결과 \n코스피지수 주식 {}%, 한국채 10년 {}%, 한국채 5년 {}%, 금 {}%, 원자재 {}% 포트폴리오가 제안되었습니다. \n".format(
                round(M, 0), round(kospi / M, 1), round(k10bond / M,
                                                        1), round(k5bond / M, 1), round(gold / M, 1),
                round(commodity / M, 1)))

        nparray1 = MDD_neighbor.kneighbors(Dataframe.iloc[Dataframe.idxmax()[1], 1].reshape(-1, 1), M,
                                           return_distance=False)
        kospi = 0
        k10bond = 0
        k5bond = 0
        gold = 0
        commodity = 0

        for i in range(0, M):
            kospi += Dataframe.loc[nparray1[0][i], 'kospi']
            k10bond += Dataframe.loc[nparray1[0][i], 'k10bond']
            k5bond += Dataframe.loc[nparray1[0][i], 'k5bond']
            gold += Dataframe.loc[nparray1[0][i], 'gold']
            commodity += Dataframe.loc[nparray1[0][i], 'commodity']
        print(
            "낮은 MDD 지수를 가진 상위 {}개의 포트폴리오를 샘플링하여 평균낸 결과 \n코스피지수 주식 {}%, 한국채 10년 {}%, 한국채 5년 {}%, 금 {}%, 원자재 {}% 포트폴리오가 제안되었습니다. \n".format(
                round(M, 0), round(kospi / M, 1), round(k10bond / M,
                                                        1), round(k5bond / M, 1), round(gold / M, 1),
                round(commodity / M, 1)))

        nparray1 = Volatility_neighbor.kneighbors(Dataframe.iloc[Dataframe.idxmin()[3], 3].reshape(-1, 1), M,
                                                  return_distance=False)
        kospi = 0
        k10bond = 0
        k5bond = 0
        gold = 0
        commodity = 0

        for i in range(0, M):
            kospi += Dataframe.loc[nparray1[0][i], 'kospi']
            k10bond += Dataframe.loc[nparray1[0][i], 'k10bond']
            k5bond += Dataframe.loc[nparray1[0][i], 'k5bond']
            gold += Dataframe.loc[nparray1[0][i], 'gold']
            commodity += Dataframe.loc[nparray1[0][i], 'commodity']

        print(
            "낮은 Volatility 지수를 가진 상위 {}개의 포트폴리오를 샘플링하여 평균낸 결과 \n코스피지수 주식 {}%, 한국채 10년 {}%, 한국채 5년 {}%, 금 {}%, 원자재 {}% 포트폴리오가 제안되었습니다. \n".format(
                round(M, 0), round(kospi / M, 1), round(k10bond / M,
                                                        1), round(k5bond / M, 1), round(gold / M, 1),
                round(commodity / M, 1)))

    def Scatter(self, Dataframe):

        # scatter
        plt.subplots(figsize=[10, 10])
        scatter1 = plt.scatter(
            Dataframe['Volatility'], Dataframe['Return %'], marker='o', s=10, alpha=0.5)  # 알파는 투명도

        # cursor1 = mplcursors.cursor(scatter1, hover=True)
        # cursor1.connect("add", lambda sel: sel.annotation.set_text('Portfolio \n Index : {}\n Return : {}%\n Volatility : {}\n Sharpe : {}\n  kospi : {}\n k10bond : {}\n k5bond : {}\n gold : {}\n commodity : {}'.format(sel.target.index, sel.target[1], sel.target[0], Dataframe.iloc[sel.target.index, 2], Dataframe.iloc[sel.target.index, 4], Dataframe.iloc[sel.target.index, 5], Dataframe.iloc[sel.target.index, 6], Dataframe.iloc[sel.target.index, 7], Dataframe.iloc[sel.target.index, 8])))
        # 되살리려면 mdd

        plt.scatter(Dataframe.iloc[Dataframe.idxmax()[2], 3], Dataframe.iloc[Dataframe.idxmax()[2], 0], marker='x',
                    c='red', s=15)
        plt.annotate('Sharpe Max', fontsize=10,
                     xy=(Dataframe.iloc[Dataframe.idxmax()[2], 3], Dataframe.iloc[Dataframe.idxmax()[2], 0]), xytext=(
                         Dataframe.iloc[Dataframe.idxmax()[2], 3] - 0.03, Dataframe.iloc[Dataframe.idxmax()[2], 0] + 0.15))

        plt.scatter(Dataframe.iloc[Dataframe.idxmin()[3], 3], Dataframe.iloc[Dataframe.idxmin()[3], 0], marker='x',
                    c='red', s=15)
        plt.annotate('Volatility Min', fontsize=10,
                     xy=(Dataframe.iloc[Dataframe.idxmin()[3], 3],
                         Dataframe.iloc[Dataframe.idxmin()[3], 0]),
                     xytext=(Dataframe.iloc[Dataframe.idxmin()[3], 3] - 0.03,
                             Dataframe.iloc[Dataframe.idxmin()[3], 0] + 0.15), )

        plt.scatter(Dataframe.iloc[Dataframe.idxmax()[0], 3], Dataframe.iloc[Dataframe.idxmax()[0], 0], marker='x',
                    c='red',
                    s=15)
        plt.annotate('Return Max', fontsize=10,
                     xy=(Dataframe.iloc[Dataframe.idxmax()[0], 3], Dataframe.iloc[Dataframe.idxmax()[0], 0]), xytext=(
                         Dataframe.iloc[Dataframe.idxmax()[0], 3] - 0.03, Dataframe.iloc[Dataframe.idxmax()[0], 0] + 0.15))

        plt.scatter(Dataframe.iloc[Dataframe.idxmax()[1], 3], Dataframe.iloc[Dataframe.idxmax()[1], 0], marker='x',
                    c='red',
                    s=15)

        plt.annotate('MDD Min', fontsize=10,
                     xy=(Dataframe.iloc[Dataframe.idxmax()[1], 3], Dataframe.iloc[Dataframe.idxmax()[1], 0]), xytext=(
                         Dataframe.iloc[Dataframe.idxmax()[1], 3] - 0.03, Dataframe.iloc[Dataframe.idxmax()[1], 0] + 0.15))

        # 람다식

        plt.xlabel('Volatility')
        plt.ylabel('Return %')
        plt.show()
