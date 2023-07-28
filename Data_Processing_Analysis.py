#from decimal import Decimal
import pyodbc
import pandas as pd
import numpy as np
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
from scipy.stats import norm

def runsqlcommand(sql, cur):
    '''
    :param sql: sql command
    :param cur: cursor
    :return: result generated from sql command
    '''
    cur.execute(sql)
    row = cur.fetchall()
    result = np.array(row)
    return result


def getBuyandSell(df):
    '''
    :param df: a dataframe contains [ log return of all selected contracts ] in formation period
    :return: two arrays, denote which to buy and sell
'''
    buy = [df.iloc[0, 0], df.iloc[1, 0]]  # 1st column
    sell = [df.iloc[-1, 0], df.iloc[-2, 0]]
    return buy, sell  # return symbol


def calAvgLogr(strtime2, strtime3, symbol, cur):
    '''
    :param time: time point, based on the length of holding period
    :param symbol: commodity which log return should be calculated
    :param cur: cursor needed to run command in sql server
    :return: a list of log return
    '''

    # strtime=str(time)

    sql2 = 'select time, Log_return from log_return_fri_fzx_ALL_DE where time>' \
           + '\'' + strtime2 + '\'' + ' and time<=' + '\'' + strtime3 + '\'' + ' and symbol=' + '\'' + symbol + '\'' \
           + ' order by time'
    logr = runsqlcommand(sql2, cur)[:, 1]

    return logr.astype(float)  # float(logr[0])

def data_resample(freq, df):
    return pd.DataFrame(df.resample(rule=freq).last().dropna())


def   sqlC_P(strtime1,strtime2,symbol):
    sql='IF EXISTS (SELECT ISNULL(sum(C_P), 0.0) FROM [log_return_profit_loss_2] WHERE time>='\
     + '\'' + strtime1 + '\'' + ' and time<=' + '\'' + strtime2+ '\'' +'and symbol  =' + '\'' + symbol + '\'' +') BEGIN SELECT ISNULL(sum(C_P), 0.0) FROM [log_return_profit_loss_2] WHERE time>='\
    + '\'' + strtime1 + '\'' + ' and time<=' + '\'' + strtime2+ '\'' +'and symbol  =' + '\'' + symbol + '\'' +'END ELSE BEGIN SELECT 0.0 END'


    # ime1 + '\'' + ' and time<=' + '\'' + strtsql = 'select  isnull(sum(C_P),0) FROM log_return__profit_loss  where time>=' \
    #     #       + '\'' + strtime2+ '\'' +'and symbol =' + '\'' + symbol + '\'' +'group by symbol'
    profit_loss = runsqlcommand(sql, cur)[0][0].astype(float)
    #print(profit_loss)

    return profit_loss


def benchmark(df):
    current_Past_bench = pd.DataFrame(np.zeros((len(df), 5)), columns=['time', 'PL_bench', 'Portfolio PL_bench', 'NAV_bench', 'return_bench'])
    for i in range(0,len(df)):  


        if(i==0):
            current_Past_bench.loc[0, 'Portfolio PL_bench'] = 1000000  #本金 two places
            current_Past_bench.loc[0, 'NAV_bench'] = 1
            current_Past_bench.loc[0, 'return_bench'] = ''
            strtime = str(df.index[0])
            current_Past_bench.loc[i,'PL_bench']=0
            current_Past_bench.loc[i,'time']=strtime[0:10]

        else:
            strtime1 = str(df.index[i])
            current_Past_bench.loc[i, 'PL_bench'] =   20*(df.iloc[i, 1] - df.iloc[i-1, 1])
            current_Past_bench.loc[i, 'return_bench'] = df.index[i]
            current_Past_bench.loc[i, 'time'] = strtime1[0:10]
            current_Past_bench.loc[i,'Portfolio PL_bench']=current_Past_bench.loc[i-1,'Portfolio PL_bench']+current_Past_bench.loc[i, 'PL_bench']   #累加
            current_Past_bench.loc[i,'NAV_bench']=current_Past_bench.loc[i,'Portfolio PL_bench']/1000000#df2.loc[i]['return']
            current_Past_bench.loc[i,'return_bench']=(current_Past_bench.loc[i,'NAV_bench'] -current_Past_bench.loc[i-1,'NAV_bench'] )/current_Past_bench.loc[i-1,'NAV_bench']



    print(current_Past_bench)   
    current_Past_bench = current_Past_bench.drop(current_Past_bench[current_Past_bench['time'] == 0].index)
    current_Past_bench.to_csv("RETURN_benchmark.csv")

    return current_Past_bench





def current_Past(df,cur):  #find C and P   profit and loss

    current_Past = pd.DataFrame(np.zeros((len(df), 5)), columns=['time','PL','Portfolio PL','NAV','return'])  


    for i in range(0,len(df)): 


        if(i==0):
            current_Past.loc[0, 'Portfolio PL'] = 1000000  
            current_Past.loc[0, 'NAV'] = 1
            current_Past.loc[0, 'return'] = ''
            strtime = df.iloc[i, 0]
            if df.iloc[i,5] !='No' :
                buy1=df.iloc[i,-4]
                buy2=df.iloc[i,-3]
                sell1=df.iloc[i,-2]
                sell2=df.iloc[i,-1]

                profit_loss_sum=20*sqlC_P(strtime, buy1)+20*sqlC_P(strtime, buy2)-20*sqlC_P(strtime, sell1)-20*sqlC_P(strtime, sell2)
                current_Past.loc[i,'PL']=profit_loss_sum
                current_Past.loc[i,'time']=strtime

            else:
                current_Past.loc[i,'time'] = strtime
                current_Past.loc[i,'PL'] = 0
        else:
            strtime1 = df.iloc[i-1, 0]   #holding start
            strtime2 = df.iloc[i, 0]      #holding end
            buy1 = df.iloc[i, -4]
            buy2 = df.iloc[i, -3]
            sell1 = df.iloc[i, -2]
            sell2 = df.iloc[i, -1]
            profit_loss_sum = 20*sqlC_P(strtime1,strtime2, buy1) + 20*sqlC_P(strtime1,strtime2, buy2) - 20*sqlC_P(strtime1,strtime2, sell1) -20* sqlC_P(strtime1,strtime2, sell2)
            current_Past.loc[i, 'PL'] = profit_loss_sum
            current_Past.loc[i, 'time'] = strtime2

            current_Past.loc[i,'Portfolio PL']=current_Past.loc[i-1,'Portfolio PL']+current_Past.loc[i, 'PL']  
            current_Past.loc[i,'NAV']=current_Past.loc[i,'Portfolio PL']/1000000#df2.loc[i]['return']
            current_Past.loc[i,'return']=(current_Past.loc[i,'NAV'] -current_Past.loc[i-1,'NAV'] )/current_Past.loc[i-1,'NAV']


    print(current_Past)   
    current_Past = current_Past.drop(current_Past[current_Past['time'] == 0].index)
    df4 = pd.merge(current_Past, df, on=('time'),how='right')
    df4.to_csv("RETURN_ALL.csv")

    return current_Past,df4

def concat_bench_mom(df_ben,df_mom):

    df  = pd.merge(df_ben, df_mom, on=('time'),how='right')
    df .to_csv("RETURN_ALL_bench.csv")
    df_T=df_mom[df_mom['momentum'] != 'No']  #only trade time
    df_T = pd.merge(df_ben, df_T, on=('time'),how='right')
    return df,df_T




def MaxDrawdown(return_list):
   
    i = np.argmax((np.maximum.accumulate(return_list) - return_list) / np.maximum.accumulate(return_list))  
    if i == 0:
        return 0
    j = np.argmax(return_list[:i]) 
    return (return_list[j] - return_list[i]) / (return_list[j])


def sharpe_ratio(return_series, N, rf):
   
    mean = return_series.mean() * N -rf
    sigma = return_series.std(ddof = 1) * np.sqrt(N)
    return   mean / sigma

def sortino_ratio(series, N,rf):
    
    mean = series.mean() * N -rf
    std_neg = series[series<0].std(ddof = 1)*np.sqrt(N)
    return mean/std_neg

def var_gaussian(r, level, modified):
    """
    Returns the Parametric Gauuian VaR of a Series or DataFrame
    If "modified" is True, then the modified VaR is returned,
    using the Cornish-Fisher modification
    Excess Kurtosis = Kurtosis – 3
    """
    # compute the Z score assuming it was Gaussian
    z = norm.ppf(level/100)
    if modified==True:
        # modify the Z score based on observed skewness and kurtosis
        s=r.skew(axis = 0)
        k=r.kurt()
        z = (z +
                (z**2 - 1)*s/6 +
                (z**3 -3*z)*(k-3)/24 -
                (2*z**3 - 5*z)*(s**2)/36
            )
    return  r.mean() - z*r.std(ddof = 1)/math.sqrt(52)

def  staRoll(df):
    df = df.drop(df[df['return'] == ''].index)  #drop the null(first one)
    list_w = df['return'].tolist()
    df1=df['return']*52              
    df = df.drop(df[df['NAV'] == ''].index)     
    list_y = df1.values.tolist()
    list1 = df['NAV'].values.tolist()

    # # -------------------------------------------------------------
    # #             statistical    data
    # # -------------------------------------------------------------
    basicStatsRolling = pd.DataFrame(np.zeros((1, 5)), columns=['Maximum', 'Minimum', 'Mean', 'Std','momentum'],
                                                       index=['MOM2/4'])
    Performance_measure = pd.DataFrame(np.zeros((1, 11)),
                                       columns=['Annualized return','t-value', 'Volatility','skewness', 'kurtosis','Max drawdown','Reward-to-risk ratio', 'Sharpe ratio','Sortino' ,
                                                '95% VaR', '99% VaR (Cornish–Fisher) ' ],
                                       index=['MOM4/2',])
    basicStatsRolling['Maximum'] = df1.max()
    basicStatsRolling['Minimum'] = df1.min()
    basicStatsRolling['Mean'] = df1.mean()   
    basicStatsRolling['Std'] = df1.std(ddof = 1)/math.sqrt(52)   
    basicStatsRolling['skewness'] =  df1.skew(axis = 0)  
    basicStatsRolling['kurtosis'] = df1.kurt()   
    rf=0.025
    A_r=df1.mean()
    Volatility=df1.std(ddof = 1)/math.sqrt(52)



    Max_drawdown=MaxDrawdown(list1)  
    SP_R= sharpe_ratio(df['return'], 52, 0)
    SP = sharpe_ratio(df['return'], 52, rf)
    SO=sortino_ratio(df['return'], 52, rf)
    VaR_cf=var_gaussian(df1, level=5, modified=True)

    t, p_ = stats.ttest_1samp(list_y, 0)
    # ------ p_ got from stats.ttest_1samp is the two-sided p-value, we should calculate one-side p-value-----
    if t > 0:
        p = p_ / 2
    else:
        p = 1 - p_ / 2

    if (p) >= 0.1:
        basicStatsRolling['momentum'] = 'No'
    elif (p) >= 0.05 and (p) < 0.1:
        basicStatsRolling['momentum'] = 'Yes*'
    else:
        basicStatsRolling['momentum'] = 'Yes**'

    Performance_measure['Annualized return'] = A_r   
    Performance_measure['t-value']=t
    Performance_measure['Volatility'] = Volatility     
    Performance_measure['skewness'] =  df1.skew(axis = 0)  
    Performance_measure['kurtosis'] = df1.kurt()   
    Performance_measure['Max drawdown'] = Max_drawdown
    Performance_measure['Reward-to-risk ratio'] = SP_R
    Performance_measure['Sharpe ratio'] = SP
    Performance_measure['Sortino'] = SO
    Performance_measure['95% VaR']=norm.ppf(0.95, A_r, Volatility)   
    Performance_measure['99% VaR (Cornish–Fisher) '] = VaR_cf

####################   result    ##############################
    print(list_y)
    print(basicStatsRolling)
    print(Performance_measure)
    basicStatsRolling.to_csv("basicStats_NIU.csv",mode='a+')
    Performance_measure.to_csv("Performance_YIQI.csv",mode='a+')
    return basicStatsRolling,Performance_measure




if __name__ == '__main__':
    ######################### sql connection ##########################

    server = "10.7.6.92"
    user = "temp"
    password = "XJTLU12345shixi"
    database = "HFData2"

conn = pyodbc.connect(DRIVER='{SQL Server}', SERVER=server, DATABASE=database, UID=user, PWD=password)

cur = conn.cursor()

sqlstart = 'use HFData2'
cur.execute(sqlstart)


######################### 2 week looking period ##########################
def weekReading():
    sql = f" SELECT all [time]\
  FROM [HFData2].[dbo].[BD_fzx_ALL]where time > \
   '2019-10-01' and time <= '2020-06-30' order by time "
    cur.execute(sql)
    row = cur.fetchall()
    df = pd.DataFrame(np.array(row))
    return df.iloc[:, 0]


return_week = []
timeperiod = [2]  # holding period base on the number of holding weeks
momentum = pd.DataFrame(np.zeros((2, 3)), columns=['winners', 'losers', 'momentum'])
buy_sell = pd.DataFrame(np.zeros((1, 4)), columns=['buy1', 'buy2', 'sell1','sell2'])
basicStats1 = pd.DataFrame(np.zeros((1, 4)), columns=['Maximum', 'Minimum', 'Mean', 'Std'],
                           index=['MOM12/4'])
hypothesis1 = pd.DataFrame(np.zeros((1, 6)), columns=['time','mean', 'std', 't-statistic', 'p-value', 'momentum'],
                           index=['MOM12/4'])           

df = weekReading()
hypothesis2 = pd.DataFrame()
for i in range(9, len(df) - 2, 2):  # F   1  H 5
   
    strtime = str(df.iloc[i - 2])[0:10]
    strtime1 = str(df.iloc[i])[0:10]   #holding start
    strtime2 = str(df.iloc[i])[0:10]
    strtime3 = str(df.iloc[i + 2])[0:10]
    sql1 = 'select t2.symbol,avg1 from((SELECT symbol, avg(Log_return) AS avg1  FROM   log_return_fri_fzx_ALL_C1 where time> ' \
               + '\'' + strtime + '\'' + ' and time<=' + '\'' + strtime1 + '\'' + ' GROUP BY symbol having sum(Log_return) !=0 )t1 inner JOIN (SELECT symbol FROM log_return_fri_fzx_ALL_C1  where time>' \
               + '\'' + strtime2 + '\'' + ' and time<=' + '\'' + strtime3 + '\'' + ' GROUP BY symbol  having sum(Log_return) !=0 ) t2 ON t1.symbol=t2.symbol) order by avg1  desc'
    df_avg7 = pd.DataFrame(runsqlcommand(sql1, cur), columns=['symbol', 'log_return'])

    if(hypothesis1.values[-1,-1]==0 or hypothesis1.values[-1,-1]=='Yes*'or  hypothesis1.values[-1,-1]=='Yes**'):
  hypothesis1.values[-1,-1]==0 
        buy, sell = getBuyandSell(df_avg7)  
        buy2 = calAvgLogr(strtime2, strtime3, buy[1], cur)
        sell1 = calAvgLogr(strtime2, strtime3, sell[0], cur)
        sell2 = calAvgLogr(strtime2, strtime3, sell[1], cur)

    else:  

        buy1 = calAvgLogr(strtime2, strtime3, buy_sell.iloc[-1, 0], cur)  #if  No then hold until Yes 所以symbol还是原来的symbol
        buy2 = calAvgLogr(strtime2, strtime3, buy_sell.iloc[-1, 1], cur)
        sell1 = calAvgLogr(strtime2, strtime3, buy_sell.iloc[-1, 2], cur)
        sell2 = calAvgLogr(strtime2, strtime3, buy_sell.iloc[-1, 3], cur)
        buy=[buy_sell.iloc[-1, 0],buy_sell.iloc[-1, 1]]
        sell=[buy_sell.iloc[-1, 2],buy_sell.iloc[-1, 3]]

    new = pd.DataFrame({"buy1": buy[0], "buy2": buy[1], "sell1": sell[0], "sell2": sell[1]}, index=["0"])
    buy_sell = buy_sell.append(new, ignore_index=True)
    momentum['winners'] = (buy1 + buy2) * 0.5
    momentum['losers'] = (sell1 + sell2) * 0.5
    momentum['momentum'] = (buy1 + buy2) * 0.5 - (sell1 + sell2) * 0.5  # equally weighted
    # momentum['momentum'] = -(buy1 + buy2) * 0.5 + (sell1 + sell2) * 0.5  
    print(momentum)
    print(momentum['momentum'].values)
    #
    # # -------------------------------------------------------------
    #              basic statisticsz
    # -------------------------------------------------------------
    # basicStats1 = pd.DataFrame(np.zeros((1, 4)), columns=['Maximum', 'Minimum', 'Mean', 'Std'],index=['MOM12/4'])
    maximum = np.max(momentum['momentum'])
    minimum = np.min(momentum['momentum'])
    mean = np.mean(momentum['momentum'])
    std = np.std(momentum['momentum'])

    basicStats1['Maximum'] = maximum
    basicStats1['Minimum'] = minimum
    basicStats1['Mean'] = mean
    basicStats1['Std'] = std
    print(basicStats1)


    # -------------------------------------------------------------
    #              test momentum
    # -------------------------------------------------------------
    df1 = pd.DataFrame(np.zeros((1,6)),columns=['time','mean','std','t-statistic','p-value','momentum'], index=["0"])  #一个临时df 储存一次循环里的一行
    array = momentum['momentum'].values
    mean = np.mean(momentum['momentum'])
    std = np.std(momentum['momentum'])

    t, p_ = stats.ttest_1samp(momentum['momentum'], 0)
    # ------ p_ got from stats.ttest_1samp is the two-sided p-value, we should calculate one-side p-value-----
    if t > 0:
        p = p_ / 2
    else:
        p = 1 - p_ / 2
    df1['time']=strtime3
    df1['mean'] = mean
    df1['std'] = std
    df1['t-statistic'] = t
    df1['p-value'] = p
    if (p) >= 0.1:
        df1['momentum'] = 'No'
    elif (p) >= 0.05 and (p) < 0.1:
        df1['momentum'] = 'Yes*'
    else:
        df1['momentum'] = 'Yes**'
    print(strtime1)
    hypothesis1 = hypothesis1.append(df1, ignore_index=True)  
    array = momentum['momentum'].values
    return_week.extend(array)  
    hypothesis2 = pd.concat([hypothesis1, buy_sell], axis=1)
    hypothesis2 = hypothesis2.drop(hypothesis2[hypothesis2['time'] == 0].index)

hypothesis2.to_csv("RETURN.csv")


########################################      benchamrk  ##############################
csv_data = pd.read_csv("CSI500_0313.csv", low_memory=False)  # remember to change the path!! under the code file
csv_df = pd.DataFrame(csv_data)
csv_df.set_index(csv_df.iloc[:, 0], inplace=True)
csv_df.index = pd.to_datetime(csv_df.index)  # turn index into timeindex for rasampling

CSI_df = data_resample('W-Fri', csv_df) 



df_ben=benchmark(CSI_df)
df,df_full=current_Past(hypothesis2,cur)
staRoll(df)
df1,df_T=concat_bench_mom(df_ben,df_full)

# ######################## cumulative profits ###########################
# # draw comparison plot to show difference between pure momentum strategy and the one with statistical arbitrage
# #


list1 = df1['time'].values.tolist()
list2 = df1['NAV'].values.tolist()
list3=df1['NAV_bench'].values.tolist()

print(list1)
print(list2)
print(list3)
hypothesis2.to_csv("NAV_1920all.csv")

(
    Line(
        init_opts=opts.InitOpts(width="800px", height="500px"))
    .add_xaxis(xaxis_data=list1

               )
        .add_yaxis(
        series_name="Momentum strategy",
        y_axis=list2,
        symbol="arrow",
        symbol_size=10,
        label_opts=opts.LabelOpts(is_show=False),
    )

    .add_yaxis(
        series_name="Benchmark",
        y_axis=list3,
        symbol='rect',
        symbol_size=10,
        label_opts=opts.LabelOpts(is_show=False),
    )

        .set_series_opts(
        markarea_opts=opts.MarkAreaOpts(

        )
    )
    .set_global_opts(
        legend_opts=opts.LegendOpts(
            align='left',
        ),
        xaxis_opts=opts.AxisOpts(
             splitline_opts=opts.SplitLineOpts(is_show=True)
        ),
        yaxis_opts=opts.AxisOpts(
            type_="value",

            axistick_opts=opts.AxisTickOpts(is_show=True),
            splitline_opts=opts.SplitLineOpts(is_show=True),
            min_=0.5
        ),
            title_opts=opts.TitleOpts(

        
        title='Net asset value',
       # subtitle='contrarian ',
                pos_top='bottom',
                pos_left='center',
            ),

        tooltip_opts=opts.TooltipOpts(is_show=False),
    )
    .render("C:/Users/Zixua/Desktop/LineALL4.html")
)
#
# (
#     Line(init_opts=opts.InitOpts(width="900px", height="500px"))
#     .add_xaxis(xaxis_data=list1
#
#                )
#         .add_yaxis(
#         series_name="All",
#         y_axis=list4,
#         symbol="arrow",
#         symbol_size=10,
#         label_opts=opts.LabelOpts(is_show=False),
#     )
#     .add_yaxis(
#         series_name="All excl. financial",
#         y_axis=list2 ,
#         symbol_size=10,
#         label_opts=opts.LabelOpts(is_show=False),
#     )
#     .add_yaxis(
#         series_name="Benchmark",
#         y_axis=list3,
#         symbol='rect',
#         symbol_size=10,
#         label_opts=opts.LabelOpts(is_show=False),
#     )
#
#         .set_series_opts(
#         markarea_opts=opts.MarkAreaOpts(
#             data=[
#                 #opts.MarkAreaItem(name="bull market", x=("2015-12-04", "2017-09-08")),
#                 #opts.MarkAreaItem(name="COVID-19 pandemic", x=("2019-12-27", "2020-06-19")),
#             ]
#         )
#     )
#     .set_global_opts(
#         xaxis_opts=opts.AxisOpts(
#              splitline_opts=opts.SplitLineOpts(is_show=True)
#         ),
#         yaxis_opts=opts.AxisOpts(
#             type_="value",
#
#             axistick_opts=opts.AxisTickOpts(is_show=True),
#             splitline_opts=opts.SplitLineOpts(is_show=True),
#             min_=0.5
#         ),
#             title_opts=opts.TitleOpts(
#
#         
#         title='NAV_All excl. financial ',
#         subtitle='momentum',
#         #pos_top=30,
#             ),
#
#         tooltip_opts=opts.TooltipOpts(is_show=False),
#     )
#     .render("C:/Users/Zixua/Desktop/LineALL.html")
# )
#
#
#
#
#
#
#
#
#
#
# list1 = df_T['time'].values.tolist()
# list2 = df_T['NAV'].values.tolist()
# list3=df_T['NAV_bench'].values.tolist()
#
#
# (
#     Line(init_opts=opts.InitOpts(width="900px", height="500px"))
#     .add_xaxis(xaxis_data=list1)
#     .add_yaxis(
#         series_name="MOM",
#         y_axis=list2 ,
#         symbol_size=10,
#         label_opts=opts.LabelOpts(is_show=False),
#     )
#     .add_yaxis(
#         series_name="Benchmark",
#         y_axis=list3,
#         symbol="arrow",
#         symbol_size=10,
#         label_opts=opts.LabelOpts(is_show=False),
#     )
#
#         .set_series_opts(
#         markarea_opts=opts.MarkAreaOpts(
#             data=[
#                 opts.MarkAreaItem(name="早高峰", x=("2015-12-04", "2017-09-08")),
#                 opts.MarkAreaItem(name="晚高峰", x=("2019-12-27", "2020-06-19")),
#             ]
#         )
#
#
#     )
#     .set_global_opts(
#         xaxis_opts=opts.AxisOpts(
#              splitline_opts=opts.SplitLineOpts(is_show=True)
#         ),
#         yaxis_opts=opts.AxisOpts(
#             type_="value",
#
#             axistick_opts=opts.AxisTickOpts(is_show=True),
#             splitline_opts=opts.SplitLineOpts(is_show=True),
#             min_=0.5
#         ),
#         visualmap_opts=opts.VisualMapOpts(
#                     is_piecewise=True,
#                     dimension=0,
#                     pieces=[
#                         {"lte": 6, "color": "green"},
#                         {"gt": 6, "lte": 8, "color": "red"},
#                         {"gt": 8, "lte": 14, "color": "green"},
#                         {"gt": 14, "lte": 17, "color": "red"},
#                         {"gt": 17, "color": "green"},
#                     ],
#                 ),
#         tooltip_opts=opts.TooltipOpts(is_show=False),
#     )
#     .render("C:/Users/Zixua/Desktop/LineALL_T.html")
# )
# print('done')
