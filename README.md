# Crawler-of-FirstTime
CSDN地址：https://blog.csdn.net/weixin_39459398/article/details/106959688
建议大家去看csdn，我写的详细一点。

首先这个算是自己第一个爬虫的项目，做这个是为了帮同学完成作用。
自己在github上或是csdn都找了好几份，因为自己之前也没接触过爬虫，
很多url都被封了也不知道改哪个地方。算是给同样入门的同学一个可以用的项目。

整个项目包括两个部分，第一部分就是淘宝爬虫，第二部分是对爬下来的数据进行数据分析和处理。ipynb文件夹里包含两份code，就和前面对应。大家用jupyter notebook 打开就好了。如果是要用pycharm或者别的打开的话我重新再发一份另外整理的code。

爬虫部分。自己也不太了解，这里就不细讲了。这份代码我是在  https://mp.weixin.qq.com/s/1n5QyFqsLcJ1h2PyK9RIgQ  这篇微信公众号的文章里找到的。里面讲的很清楚，也把webdriver配置也弄好了。公众号的爬虫代码可以运行，就是数据清洗的代码有问题，我就把一些删了自己重新写过，最后的效果是一样的。

数据分析部分。我主要对口红的进行分析，所以当时就没用公众号里打包好的 电脑.csv，自己重新爬了一份 口红.csv 。如果只想进行数据分析这部分的话，我里面已经包含了一个 口红.csv 的文件，大家可以直接用第二部分的代码来跑。  数据清洗的过程主要是对 付款人数 和 发货地点 进行预处理，后面分析的部分就是主要是找到付款人数最多的十家店，最便宜的十家店，发货最多的十个地点，还有通过jieba分词对商品的描述制作了一个词云。

