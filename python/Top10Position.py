
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
print("去重之前的记录数", df.shape)
# 记录去重
df.drop_duplicates(subset=["描述信息", "价格"], inplace=True)
# 去重之后的记录数
print("去重之后的记录数", df.shape)

# In[7]:


# 删去所有含有缺失值的行
df = df.dropna()  # 默认是按行删除 即axis=0

# ### 付款人数字段的处理

# In[8]:


df = df.reset_index(drop=True)  # 删去某列后index是不变的，得重新排序




df["付款人数"] = pd.DataFrame(df["付款人数"].astype(str))



for j in range(len(df["付款人数"])):
    test1 = df["付款人数"][j]
    #print(test1)
    if test1.find('万') != -1:
        df["付款人数"][j] = float(re.findall(r'\d+(?:\.\d+)?', test1)[0]) * 10000
    else:
        df["付款人数"][j] = float(re.findall(r'\d+(?:\.\d+)?', test1)[0])



# ### 发货地址的处理



df["发货地址"] = pd.DataFrame(df["发货地址"].astype(str))
for i in range(len(df["发货地址"])):
    pattern = r"[ ]"  # 定义分隔符   只要出现?或者$就将字符串进行分割
    kongge = 0
    x = df["发货地址"][i]
    for j in range(len((x))):
        if x[j] == ' ':
            kongge = 1
    if kongge == 1:
        result = re.split(pattern, x)
        df["发货地址"][i] = result[1]
print('*'*100)
print('-'*40+'输出出货最多的十个地址'+'-'*40)
print('\n')
print(df["发货地址"].value_counts().head(10))