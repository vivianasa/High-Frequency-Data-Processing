#from decimal import Decimal
import pyodbc
import pymssql
import numpy as np
from pymssql import _mssql
from pymssql import _pymssql
import pandas as pd
from scipy import stats
from matplotlib import pyplot as plt
from scipy import stats
import matplotlib.ticker as ticker
import math
import  empyrical
import pyecharts.options as opts
from pyecharts.charts import Line
from pyecharts.faker import Faker
import imgkit


def VolSum(Product):
    # sql = f"select * from HF_rollover_day_ZZ where \
    #               symbol = \'{Product}\' and time between '2015-01-01' and '2020-07-20'and type = 'T+0' order by time  "
    sql = f"select * from HF_rollover_day_ZJ where \
                    symbol = \'{Product}\' and time between '2015-05-03' and '2020-07-30'and type = 'T+0' order by time  "
    cur.execute(sql)
    row = cur.fetchall()
    df = pd.DataFrame(np.array(row))
    df1=df[[3,1,5]]
    df1.set_index([3],inplace=True)  #index=time column=sym/close
    #df1.rename(index={'time'},columns={1:'symbol',5:'close'}, inplace=True)
    return df1   #df for fre=min time sym close


def data_resample(freq, df):
    return pd.DataFrame(df.resample(rule=freq).last().dropna())


def Dominant_future(df):
    df1= data_resample('D', df)

    new_price_info = pd.DataFrame(columns=['time', 'symbol', 'Current_price', 'Past_price', 'Gross_return'])
    #print(new_price_table)

    time_lst = df1.index.unique()  # 所有的时间段
    # max_accum_contract=df.iloc[index]
    # 循环时间序列
    for index in range(len(time_lst)):

        max_accum_contract = df1.iloc[index]
        max_accum_contract_past = df1.iloc[index-1]
        max_accum_contract[5]=pd.to_numeric(max_accum_contract[5],errors='coerce')
        max_accum_contract_past [5] = pd.to_numeric(max_accum_contract_past[5], errors='coerce')
        #log_R=np.log(max_accum_contract.iloc[1]/max_accum_contract_past.iloc[1])

       #l=max_accum_contract.iloc[1]/max_accum_contract_past.iloc[1]
        new_price_info = pd.concat([new_price_info,pd.DataFrame({'time': time_lst[index],
                                                       'symbol':max_accum_contract.iloc[0],
                                                       'Current_price':max_accum_contract.iloc[1],
                                                       'Past_price': max_accum_contract_past.iloc[1],
                                                       'Gross_return':(max_accum_contract[5]/max_accum_contract_past [5])-1},index=[0])])
                                                                         #np.log(max_accum_contract.iloc[1]/max_accum_contract_past.iloc[1])},index=[0]))



        #new_price_table = Dominant_future.pd.concat([new_price_table,new_price_info],axis=0)
        #print(f'The return of {time_lst[index]} has been calculated')
    new_price_info.set_index('time', inplace=True)
    return new_price_info


def  staRoll(df,Product):
    df = df.drop(df[df['Gross_return'] == ''].index)  #drop the null(first one)

    # # -------------------------------------------------------------
    # #             statistical    data
    # # -------------------------------------------------------------
    basicStatsRolling = pd.DataFrame(np.zeros((len(Product), 8)), columns=['Symbol','Maximum', 'Minimum', 'Mean', 'Std','median','skewness','kurtosis'])
    for i in range(0,len(Product)):
        df1=df.loc[df['symbol'] == Product[i]]
        print(df1)
        list = df1['Gross_return'].values.tolist() 
        print(list)
        basicStatsRolling.loc[i,'Symbol']=Product[i]
        basicStatsRolling.loc[i,'Maximum'] = np.max(list)
        basicStatsRolling.loc[i,'Minimum'] = np.min(list)
        basicStatsRolling.loc[i,'Mean'] = np.mean(list)
        basicStatsRolling.loc[i,'Std'] = np.std(list)
        basicStatsRolling.loc[i,'median'] = np.median(list)
        basicStatsRolling.loc[i,'skewness'] =  stats.skew(list)  
        basicStatsRolling.loc[i,'kurtosis'] = stats.kurtosis(list)


    print(basicStatsRolling)
    basicStatsRolling.to_csv("Day_stats_Gross_returnZJ.csv")

    return basicStatsRolling



if __name__ == '__main__':
    server = '10.7.6.92'
    user = 'temp'
    password = 'XJTLU12345shixi'
    database = 'HFData2'
    conn = pymssql.connect(server, user, password, database)
    cur = conn.cursor()
     #ZZ###############

    #DL  highVol & alltime
    #Product=['i', 'j', 'l', 'pp','cs','a','jd','jm','v', 'c', 'y', 'm', 'p']
    #ZJ
    Product=['TF','T','IF','IC','IH']
    #ZZ   highVol & alltime
    #Product=['FG', 'ZC','WH', 'MA','TA','SF','SM' , 'CF', 'SR','OI','RM']
    #SQ  all
    #Product=['ag','al','au','bu','cu','fu','hc','ni','pb','rb','ru','sn','ss','wr','zn']
    #SQ  highVol & alltime
    #Product=['ag','al','au','bu','cu','hc','ni','pb','rb','ru' ,'zn']
    W_return = pd.DataFrame()
    for a in Product:
            b=VolSum(a)
            W_return = pd.concat([W_return,Dominant_future(b)])
    print(W_return)
    W_return.to_csv("Gross_return_day_ZJ.csv")
    df=W_return
    staRoll(df,Product)
    print('done!')

#print(volSum.sort_values(by='Volume',ascending=False))
