# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 19:30:57 2025

@author: 86182
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import xgboost as xgb
from xgboost import XGBRegressor
from hyperopt import fmin, tpe, hp, Trials
from sklearn.model_selection import train_test_split
# from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics
from sklearn.metrics import r2_score
import os
import warnings

warnings.filterwarnings("ignore")

# 设置当前的工作路径
os.chdir(r'D:\NKU\2025（南开）\所做项目\周口市评估_20250210\春节评估')
print(os.getcwd())    

# 定义文件名变量
filename = 'ZK'

mydata = pd.read_excel(fr'{filename}_data.xlsx')
print(mydata.columns)

# ---------------------- data process --------------------------------

# 删除缺失值所在的行
mydata = mydata.dropna()

# 对类别变量cluster进行独热编码
encoder = OneHotEncoder(sparse=False)
cluster_encoded = encoder.fit_transform(mydata[['cluster']]) # 拟合并转换'cluster'列
cluster_encoded_df = pd.DataFrame(cluster_encoded, columns=encoder.get_feature_names_out(['cluster'])) # 将编码后的数组转换回DataFrame，并指定新的列名
mydata_encoded = pd.concat([mydata.reset_index(drop=True), cluster_encoded_df], axis=1) # 将原始数据集与编码后的数据集合并
mydata_encoded.drop(['cluster'], axis=1, inplace=True)  # 删除原始的'cluster'列

# 将'PM2.5'列移动到最后
cols = mydata_encoded.columns.tolist()
cols.remove('PM2.5')
cols.append('PM2.5')
mydata_encoded = mydata_encoded[cols]  # 根据新的列顺序重新排列数据框
print(mydata_encoded.columns)  # 验证是否成功将 'PM2.5' 移动到了最后一列

mydata_encoded['date'] = pd.to_datetime(mydata_encoded['date'])

# 设置起始和结束日期时间，包括具体的小时
start_datetime = '2024-02-09 19:00:00'  
end_datetime = '2024-02-10 19:59:59'   

date_mask = (mydata_encoded['date'] >= start_datetime) & (mydata_encoded['date'] <= end_datetime)  # 创建日期范围的掩码
pred_set = mydata_encoded.loc[date_mask]  # 在指定日期范围内的数据
model_train_set = mydata_encoded.loc[~date_mask]  # 不在指定日期范围内的数据

# 将date列设置为索引
model_train_set.set_index('date', inplace=True)
pred_set.set_index('date', inplace=True)

model_train_set.index = pd.to_datetime(model_train_set.index)
pred_set.index = pd.to_datetime(pred_set.index)

# ---------------------- model build @1 --------------------------------

X = model_train_set.iloc[:, 0:-1]
y = model_train_set.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 使用XGBoost进行回归
xg_reg = XGBRegressor(n_estimators=50, learning_rate=0.2, reg_alpha=0.1, reg_lambda=1, random_state=42)
xg_reg.fit(X_train, y_train)
print(xg_reg)
y_pred = xg_reg.predict(X_test)

print(f"R2 score:", r2_score(y_test, y_pred))
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

y_pred2 = xg_reg.predict(X)
y_pred2 = pd.Series(y_pred2, index=X.index)

# 设置 Series 的名称
y_pred2.name = 'PM2.5_pred'

results2 = pd.concat([y, y_pred2], axis=1, ignore_index=False) #列拼接series
results2.to_excel(excel_writer=r'results2.xlsx', sheet_name='sheet1', index=True, encoding='utf-8')   

# ----------------------------------------------------

mydata_encoded

mydata_encoded.set_index('date', inplace=True)
mydata_encoded.index = pd.to_datetime(mydata_encoded.index)

X_pred = mydata_encoded.iloc[:, 0:-1]
y_label = mydata_encoded.iloc[:, -1]

y_pred3 = xg_reg.predict(X_pred)
y_pred3 = pd.Series(y_pred3, index=mydata_encoded.index)
y_pred3.name = 'PM2.5_pred'

results3 = pd.concat([y_label, y_pred3], axis=1, ignore_index=False) #列拼接series
results3.to_excel(excel_writer=r'results4.xlsx', sheet_name='sheet1', index=True, encoding='utf-8')  

# --------------- plot -------------------------------------

from matplotlib import font_manager
import matplotlib.dates as mdates

my_font1 = font_manager.FontProperties(fname=("C:/Windows/Fonts/times.ttf"),size=45)
my_font4 = font_manager.FontProperties(fname=("C:/Windows/Fonts/times.ttf"),size=25)

date = results3.index

y_truth = results3['PM2.5']
y_sim = results3['PM2.5_pred']

# 同时设置数学字体
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = my_font1.get_name()
plt.rcParams['mathtext.it'] = my_font1.get_name()
plt.rcParams['mathtext.bf'] = my_font1.get_name()

min_date = pd.Timestamp('2024-2-9')
max_date = pd.Timestamp('2024-2-11')

# ----------

fig = plt.figure(figsize=(14,14),dpi=300,facecolor="w")
# plt.plot(x,y221,zorder=3,color="blue",linewidth=3,label='2022 MN')
plt.plot(date,y_truth,zorder=2,color="blue",linewidth=2,alpha=0.5,label='Observations')
# plt.plot(x,y231,zorder=3,color="red",linewidth=3,label='2023 MN')
plt.plot(date,y_sim,zorder=2,color="red",linewidth=2,alpha=0.8,label='Predictions')
# date_fmt = mdates.DateFormatter('%m-%d')  # 自定义日期格式
# 月名缩写-日
date_fmt = mdates.DateFormatter('%b-%d')
plt.gca().xaxis.set_major_formatter(date_fmt)
# 生成日期范围内的日期列表
# locator = mdates.MonthLocator(bymonth=None, interval=2)
# plt.gca().xaxis.set_major_locator(locator)

# 使用 DayLocator 来按日显示刻度
locator = mdates.DayLocator(interval=1)  # 每隔1天显示一个刻度
plt.gca().xaxis.set_major_locator(locator)

plt.xlim(min_date, max_date)
plt.gcf().autofmt_xdate(rotation=0)
plt.grid(alpha=0.7,linestyle=":",linewidth=1, zorder=1,axis="x")
plt.grid(alpha=0.7,linestyle=":",linewidth=1, zorder=1,axis="y")
plt.yticks(fontproperties=my_font1,rotation=0)
# 设置 x 轴的范围并向左右扩展一定距离
plt.xlim(min_date - pd.Timedelta(1.3, 'D'), max_date + pd.Timedelta(1.3, 'D'))
plt.xticks(fontproperties=my_font1,rotation=20)
plt.tick_params(axis='y', direction='out', length=8, pad=10)  
plt.tick_params(axis='x', direction='out', length=8, pad=10)
plt.ylim(25, 405)
plt.ylabel("PM$_{2.5}$ Concentrations (μg/m³)", fontproperties=my_font1, labelpad=15)
num_ticks = 6
plt.yticks(np.linspace(40, 390, num_ticks))
plt.legend(prop=my_font4,loc=0,handlelength=2, handleheight=1,ncol=1)

# 获取当前Figure的引用及设置Figure标题
# fig = plt.gcf()
# fig.autofmt_xdate()

# 对每个标签应用对齐设置
for label in plt.gca().get_xticklabels():
    label.set_ha('center')  # 水平对齐
#     label.set_va('center')  # 垂直对齐

# 设置边框的粗细
ax = plt.gca()  # 获取当前的Axes对象ax
ax.spines['top'].set_linewidth(2.2)    # 设置顶部边框的粗细
ax.spines['bottom'].set_linewidth(2.2) # 设置底部边框的粗细
ax.spines['left'].set_linewidth(2.2)   # 设置左侧边框的粗细
ax.spines['right'].set_linewidth(2.2)  # 设置右侧边框的粗细
# 设置x轴刻度的粗细
plt.tick_params(axis='x',length=13, width=2.1)  # 调整x轴刻度线的粗细为2
# 设置y轴刻度的粗细
plt.tick_params(axis='y',length=13, width=2.1)  # 调整y轴刻度线的粗细为2

# ----------













