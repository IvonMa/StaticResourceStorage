# Numpy


```python
# 计算欧式距离
dist = numpy.sqrt(numpy.sum(numpy.square(vec1 - vec2)))
# 计算欧式距离-方式二
dist = numpy.linalg.norm(vec1 - vec2)

# 为矩阵填充同一个值
>>> np.full((3, 5), 7)
array([[ 7.,  7.,  7.,  7.,  7.],
       [ 7.,  7.,  7.,  7.,  7.],
       [ 7.,  7.,  7.,  7.,  7.]])

# 提取矩阵某一个元素
matrix[1,2]
# 提取矩阵某一行
matrix[0]
# 提取矩阵某一列
matrix[:,0]
# 求矩阵所有元素的和
numpy.sum()
# 矩阵元素各自平方
numpy.power(x1, 2)

# 将两个矩阵水平拼接成矩阵
np.hstack((a,b))
# 将两个矩阵垂直拼接成矩阵
np.vstack((a,b))

# 随机采样. size为要生成的样本数目（会返回tuple）
numpy.random.uniform(low,high,size)

```

Numpy的线性代数操作

https://www.runoob.com/numpy/numpy-linear-algebra.html

Numpy中list, arrary和matrix的区别

https://www.jianshu.com/p/eda4a91644b4

# Tensorflow

```python
# 创建session
with tf.Session() as sess:
  # ...
  
  
# 矩阵乘法
tf.matmul(matrix1, matrix2)

# 矩阵元素求和
tf.add(matrix)

# 求平均值。可以指定维度
tf.reduce_mean(matrix)
```

https://zhuanlan.zhihu.com/p/32869210