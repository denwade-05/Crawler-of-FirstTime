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



tar_cpu = ['阿玛尼', '香奈儿', '迪奥', '魅可', '纪梵希', '圣罗兰', '古驰', '兰蔻', '卡姿兰', '完美日记', '曼秀雷敦', '欧莱雅', '珂莱欧', '资生堂', '屈臣氏', '爱马仕',
           'CL']
# tar_cpu = ['联想','惠普','酷睿','苹果','三星','华硕','索尼','宏碁','戴尔','海尔','长城','海尔','神舟','清华同方','方正','明基']
tar_cpu = np.array(tar_cpu)


def rename(x):
    index = [i in x for i in tar_cpu]
    if sum(index) > 0:
        return tar_cpu[index][0]
    else:
        return "牌子不详"


df["口红品牌"] = df["描述信息"].apply(rename)


x = df["口红品牌"].value_counts().reset_index()

x = df['描述信息'][0]
list(jieba.cut(x))


add_word = ['阿玛尼', '香奈儿', '迪奥', '魅可', '纪梵希', '圣罗兰', '古驰', '兰蔻', '卡姿兰', '完美日记', '曼秀雷敦', '欧莱雅', '珂莱欧', '资生堂', '屈臣氏',
            '爱马仕', 'CL']
for i in add_word:
    jieba.add_word(i)
df["切分后的描述信息"] = df["描述信息"].apply(lambda x: jieba.lcut(x))
df.head()

### 都去停用词
with open(path + "stoplist.txt", encoding="utf8") as f:
    stop = f.read()
stop = stop.split()
# stop = [" ","笔记本电脑"] + stop
stop = [" ", "口红"] + stop
stop[:10]



df["切分后的描述信息"] = df["切分后的描述信息"].apply(lambda x: [i for i in x if i not in stop])



# 计数出现词的个数
all_words = []
for i in df["切分后的描述信息"]:
    for j in i:
        all_words.extend(i)
word_count = pd.Series(all_words).value_counts()
print('*'*100)
print('-'*40+'出现最多的20个词'+'-'*40)
print('\n')
print(word_count[:20])

back_picture = imread(path + "aixin.jpg")


wc = WordCloud(font_path="G:\\6Tipdm\\wordcloud\\simhei.ttf",
               background_color="white",
               max_words=2000,
               mask=back_picture,
               max_font_size=200,
               random_state=42
               )
wc2 = wc.fit_words(word_count)

# 3、绘制词云图
plt.figure(figsize=(16, 8))
plt.imshow(wc2)
plt.axis("off")
plt.show()
wc.to_file("口红词云.png")






