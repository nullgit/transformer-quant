根据 https://towardsdatascience.com/stock-predictions-with-state-of-the-art-transformer-and-time-embeddings-3a4485237de6 博文搭建一个简单的预测模型


## TODO

transformer历史，有关它以及之前的模型的历史，我记得这是有很多教程的
transformer搭建
transformer简单应用起来
看transformer变种论文


量化，选因子等过程是什么


## 问题

### 解决的问题是什么？输入输出是什么？

使用Transformer预测下一段时间的值？
decoder传的输入只是下一段时间的值还是滑窗的值？
是否只使用encoder？



预测的是涨幅？然后排序
使用Transformer选股？
有几篇论文都是量化选股
有一篇单步、多步预测μ和σ，然后采样
有一篇直接预测接下来的值，但是感觉这样效果不好，预测曲线相当于实际曲线右移（与今天股价最相关的是昨天的股价）

### 如何解决问题

Informer？

### 

期货的特殊点：
自带杠杆属性，交易方面会涉及平仓等在股票中没有的概念
T+0交易，可以做得更高频（但更高频的交易不一定好，要手续费等）

