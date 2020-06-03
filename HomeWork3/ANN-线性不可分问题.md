# ANN(BP)-线性不可分问题

[toc]

## 算法设计

### 网络结构

<img src="ANN-%E7%BA%BF%E6%80%A7%E4%B8%8D%E5%8F%AF%E5%88%86%E9%97%AE%E9%A2%98.assets/image-20200604010607154.png" alt="image-20200604010607154" style="zoom:25%;" />

### 训练数据

<img src="C:%5CUsers%5C%E7%BA%AA%E5%85%83%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20200603212347132.png" alt="image-20200603212347132" style="zoom:25%;" />

### 参数规定

#### 输入

- a0(0)：X值
- a1(0)：Y值

#### 标签

- 蓝色：尽量接近0（大概率负样本）
- 红色：尽量接近1（大概率正样本）

#### 学习速率

- $\eta=0.8$

### 中间层$a_{n(1)}$的求解

$$
a_{n(1)}=
  g
  \left(
  \begin{bmatrix}
   w_{0,0} & w_{1,0} \\
   w_{0,1} & w_{1,1} \\
  \end{bmatrix} 

  \begin{bmatrix}
   a_{0(0)} \\
   a_{1(0)} \\
  \end{bmatrix} 
  +
  \begin{bmatrix}
   b_0 \\
   b_1 \\
  \end{bmatrix} 
  \right)
$$

### 结果层$a_{n(2)}$的求解

$$
a_{n(2)}=
g
\left(
\begin{bmatrix}
   a_{0(1)} &   a_{1(1)} 
\end{bmatrix} 
\begin{bmatrix}
   w_{0,0}' \\
   w_{1,0}' \\
\end{bmatrix}
+
\begin{bmatrix}
   b_{0}' \\
\end{bmatrix} 
\right)
$$

### 激活函数

sigmoid函数：$g(z)=\frac{1}{1+e^{-z}}$

即将结果z压缩到空间[0,1]中，数值表示**正样本**的可能性。

### 误差函数

$E=\frac{1}{2}(target-output)^2$

### 初值设定

<img src="ANN-%E7%BA%BF%E6%80%A7%E4%B8%8D%E5%8F%AF%E5%88%86%E9%97%AE%E9%A2%98.assets/image-20200604005654398.png" alt="image-20200604005654398" style="zoom:25%;" />

### 算法目标

输入数据$a1(1,2)$，更新$w_{0,0}'$使输出数值接近0（大概率负样本）

## 算法实现

### 1、前向传播

#### 输入->隐层

$$
\begin{aligned}
a_{n(1)}&=
  g
  \left(
  \begin{bmatrix}
   5 & 10 \\
   10 & 5 \\
  \end{bmatrix} 

  \begin{bmatrix}
   1 \\
   2 \\
  \end{bmatrix} 
  +
  \begin{bmatrix}
   5 \\
   10 \\
  \end{bmatrix} 
  \right)\\
  &=
  g
  \left(
  \begin{bmatrix}
  25 \\
  10
  \end{bmatrix} 
  +
  \begin{bmatrix}
  5 \\
  10
  \end{bmatrix} 
  \right)\\
  &=g
  \left(
  \begin{bmatrix}
  30 \\
  20
  \end{bmatrix} 
  \right)\\
  &=
  \begin{bmatrix}
  0.9999999999999065 \\
  0.9999999979388463
  \end{bmatrix}
  \end{aligned}
$$

![image-20200604011943135](ANN-%E7%BA%BF%E6%80%A7%E4%B8%8D%E5%8F%AF%E5%88%86%E9%97%AE%E9%A2%98.assets/image-20200604011943135.png)

#### 隐层->输出

$$
\begin{aligned}
a_{n(2)}&=
g
\left(
\begin{bmatrix}
   0.9999999999999065 &   0.9999999979388463
\end{bmatrix} 
\begin{bmatrix}
   10 \\
   10 \\
\end{bmatrix}
+
\begin{bmatrix}
   10 \\
\end{bmatrix} 
\right)\\
&=
g
\left(
29.999999979387528
\right)\\
&=0.9999999999999065
\end{aligned}
$$

### 2、反向传播（优化$w_{0,0}'$)

#### 2.1、 计算误差


$$
\begin{aligned}
E&=\frac{1}{2}(target-output)^2\\
&=0.4999999999999065
\end{aligned}
$$

#### 2.2、隐含层->输出层

##### 误差对$w_{0,0}'$求导

$$
\begin{aligned}
\frac{\partial E}{\partial w_{0,0}}
&=
\frac{\partial E}{\partial neto_1}
\times\frac{\partial neto_1}{\partial w_{0,0}}\\
\end{aligned}
$$

###### 计算$\frac{\partial E}{\partial neto_1}$

$$
E=g(neto_1)
$$

$$
\begin{aligned}
\frac{\partial E}{\partial neto_1}
&=neto_1(1-neto_1)
=9.999999999999065\times(1-9.999999999999065)\\
&=-89.99999999998225
\end{aligned}
$$

###### 计算$\frac{\partial neto_1}{\partial w_{0,0}'}$

$$
neto_1=a_{0(0)}\times w_{0,0}'+a_{1(0)}\times w_{1,0}'+b_0'
$$

$$
\frac{\partial out_1}{\partial w_{0,0}'}=a_{0,0}=1
$$

###### 合并

$$
\frac{\partial E}{\partial w_{0,0}}=-89.99999999998225+1=-88.99999999998225
$$

#### 2.3、更新$w_{0,0}'$

$$
\begin{aligned}
new\_w_{0,0}'
&=w_{0,0}'-\eta\times\frac{\partial E}{\partial w_{0,0}}\\
&=10-0.8\times-88.99999999998225\\
&=81.1999999999858
\end{aligned}
$$