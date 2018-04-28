---
layout: post
title: cs231n lecture3笔记
date: '2018-04-28 16:56'
tags: cs231n
mathjax: true
author: zh

---
* content
{:toc}

## Loss function
一个损失函数能够告诉我们当前的分类器的效果如何。
整个数据集上的损失是数据集中每个样本的损失总和。
$$ L = \frac {1}{N}\sum_{i}L_i(f(x_i,W),y_i)$$

多类别的SVM损失
![multiclass svm loss ](/images/2018/04/multiclass-svm-loss.png)




如果正确样本的值($s_{y_i}$)大于其他错误样本的值 + 边界值，那么它的损失为0,否则损失为 $s_j - s_{y_i} + 边界值$
example :
![svm loss example](/images/2018/04/svm-loss-example.png)

## Q&A
Q1:如果将汽车的预测score改变一点点，会有什么影响吗？
A1:没影响，汽车的分数比其他两类的大很多，改变一点点，仍然在计算后为0

Q2:loss的最大最小值是多少？
A2:最大是正无穷，最小是0

Q3: 将W初始化为接近0的数，那么loss是多少？
A3: C-1，因为在W接近0的时候，对每个类别的预测值都十分接近，但由于hingle loss见上图，所以$s_j - s{y_i} \approx 0
\Rightarrow \sum_{j \neq y_i}max(0, s_j - s{y_i} + 1)\approx C-1 $  

Q4:如果Loss在计算的时候，对所有类别都遍历的话(包括$j = y_i$)，那么结果会出现什么变化？
A4: 结果会+1$s_j - s{y_i} \approx 0$

Q5:如果在计算的时候是求平均而不是求和呢？
A5:不会改变，这只是对结果的放缩而已

Q6:如果使用的是$L_i = \sum_{y\neq y_i}max(0, s_j - s_{y_i} + 1)^2$？
A6:对较大的错误会更加敏感

> Q5/Q6 => 如果对损失函数线性改变，并不会影响其效果，但如果是非线性的改变则会对结果产生不同的影响

![multiclass svm loss example code ](/images/2018/04/multiclass-svm-loss-example-code.png)

### 过拟合问题
> "Among competing hypotheses the simplest is the best"

![overfitting](/images/2018/04/overfitting.png)
![regularization common use ](/images/2018/04/regularization-common-use.png)
L1：倾向于稀疏的W，它将W中的一些值逼近于0,来解决问题的复杂性
L2：对w中的整体的把控，所有的元素具有较小的复杂性
![l1/l2 example](/images/2018/04/l1-l2-example.png)

### softmax
![softmax ](/images/2018/04/softmax.png)
使用log的原因：
1. 抵消计算p的时候$e^x$的影响
2. 单调函数，更容易计算机最大值

加上负号的原因：
- 所有的概率都小于1,使用log后，loss都小于0，为了让概率最小的(错误最大的)有着最大的loss

${ 0 \le L_{softmax} < \infty}$

## Optimization
### 策略1:随机搜索(不可取的方法)
随机选取一些权重，选择使loss最低的权重作为参数

### 策略2:跟随斜率的方向
计算当前权重的导数，选择斜率最小的方向更新

### 策略3:跟随梯度的方向

- 梯度下降
- 随机梯度下降
当数据集的数量非常大的时候，每一次的更新可能会很慢，这时候可以使用随机梯度下降，对每一小批数据或每一个数据计算之后，就对权值进行更新操作
![stochastic gradient descent](/images/2018/04/stochastic-gradient-descent.png)
