import pyodbc
import pymssql
from pymssql import _mssql
from pymssql import _pymssql
import pandas as pd
import numpy as np
# from scipy import stats
from matplotlib import pyplot as plt
# dispkay all columns:
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', None)  


def connectSQL(symbol,fromdate,todate,cur):
# def connectSQL(symbol,cur):
    sql=f"select [symbol], [time], [close], [type] from HF_rollover_day_DL where symbol = \'{symbol}\' and time between '{fromdate}' and '{todate}' and type = 'T+0' order by time asc"
    # sql=f"select * from [HFData2].[dbo].[HF_business_day_DL] where date between '{fromdate}' and '{todate}'"
    index_list=['order','date']
    # index_list = ['symbol', 'time', 'close', 'type']
    cur.execute(sql)
    row=cur.fetchall()
    array=np.array(row)
    Data=pd.DataFrame(array)
    Data.rename(columns={col: col_name for col, col_name in enumerate(index_list)}, inplace=True)
    # Data.set_index('time', inplace=True) 
    Data.set_index('date', inplace=True)
    # print(Data)
    return Data

def data_resample(freq, df):
    return pd.DataFrame(df.resample(rule=freq).last().dropna())

def Dominant_future(df):
    df1= data_resample('B', df)
    new_price_info = pd.DataFrame(columns=['time', 'symbol', 'Current_price', 'Past_price', 'Log_return'])
    #print(new_price_table)
    print(df1)
    # print(df1.iloc[:,1])
    # a = df1.iloc[1,1]-df1.iloc[0,1]
    # print(a)
    time_lst = df1.index.unique()  # 所有的时间段
    # max_accum_contract=df.iloc[index]
    # 循环时间序列
    # print(time_lst)
    for index in range(len(time_lst)):
        max_accum_contract = df1.iloc[index+1]
        print("-------------------------")
        print("index is ",index)
        print(max_accum_contract)
        # print(max_accum_contract[1])
        # print("hh:",max_accum_contract[1])
        max_accum_contract_past = df1.iloc[index]
        # max_accum_contract[1]=pd.to_numeric(max_accum_contract[1],errors='coerce')
        # max_accum_contract_past[1] = pd.to_numeric(max_accum_contract_past[1], errors='coerce')
        #log_R=np.log(max_accum_contract.iloc[1]/max_accum_contract_past.iloc[1])

       #l=max_accum_contract.iloc[1]/max_accum_contract_past.iloc[1]
        new_price_info = pd.concat([new_price_info,pd.DataFrame({'time': time_lst[index],
                                                       'symbol':max_accum_contract.iloc[0],
                                                       'Current_price':max_accum_contract.iloc[1],
                                                       'Past_price': max_accum_contract_past.iloc[1],
                                                       'Log_return':np.log(max_accum_contract[1]/max_accum_contract_past[1])},index=[0])])
                                                                         #np.log(max_accum_contract.iloc[1]/max_accum_contract_past.iloc[1])},index=[0]))



        #new_price_table = Dominant_future.pd.concat([new_price_table,new_price_info],axis=0)
        #print(f'The return of {time_lst[index]} has been calculated')
    new_price_info.set_index('time', inplace=True)
    print(new_price_info)
    return new_price_info


if __name__ == '__main__':
    # connect sqlm
    server = "10.7.6.92"
    user = "temp"
    password = "XJTLU12345shixi"
    database = "HFData2"
    conn = pymssql.connect(server, user, password, database)
    cur = conn.cursor()

    # obtain original dataset and log return data
    # symbol = ['a','b','bb','c','cs','fb','i','j','jd','jm','l','m','pp','p','v','y']
    symbol = ['a', 'b']
    # symbol = 'pp'
    fromdate = '2018-01-01'
    todate = '2020-07-29'
    # dta = connectSQL(symbol, fromdate, todate, cur)
    # alter_dta = data_resample('W-Fri', dta)
    # print(alter_dta)
    # alter_dta.to_excel('BD_DL.xlsx')
    # print(len(alter_dta))
    all_log_R = pd.DataFrame()
    for x in symbol:
        Data = connectSQL(x, fromdate, todate, cur) # original dataset
        # print(Data)
        logr = Dominant_future(Data)
    #     # print(logr)
    #     all_log_R = all_log_R.append(logr)  # get log return dataset from all symbols
    # print(all_log_R)
    # print("length is: ",len(all_log_R))
    # sorted = all_log_R.sort_values(by=['Log_return'], ascending=False)  # sort all log returns from high to low
    # print(sorted)
    # all_log_R.to_excel('logReturn.xlsx')
