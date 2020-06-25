#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import re
import jieba
from wordcloud import WordCloud
from imageio import imread
import warnings

sns.set(style="darkgrid")
mpl.rcParams["font.family"] = "SimHei"
mpl.rcParams["axes.unicode_minus"] = False
warnings.filterwarnings("ignore")

# ### 数据读取

# In[3]:
# path这里修改路径，就是你文件放的位置
#path = 'D:/python_workpace/Crawler/淘宝数据爬取/'
path='./'
# df = pd.read_csv(path+"笔记本电脑.csv", engine='python', encoding='utf-8-sig', header=None)
df = pd.read_csv(path + "口红.csv", engine='python', encoding='utf-8-sig', header=None)
df.columns = ["描述信息", "价格", "付款人数", "旗舰店", "发货地址"]

# ### 数据去重：我们认为“描述信息”和“价格”相同的记录，都是相同的记录。

# In[6]:


# 去重之前的记录数
#print("去重之前的记录数", df.shape)
# 记录去重
df.drop_duplicates(subset=["描述信息", "价格"], inplace=True)
# 去重之后的记录数
#print("去重之后的记录数", df.shape)

# In[7]:


# 删去所有含有缺失值的行
df = df.dropna()  # 默认是按行删除 即axis=0
print('*'*100)
print('-'*40+'输出最便宜的十件商品'+'-'*40)
print('\n')
df=pd.DataFrame(df)
print(df.sort_values(by='价格').head(10))