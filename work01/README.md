# 作业一

## 任务要求

本周课上介绍了用线性回归的方法预测房价。在这次作业中，请你使用Kaggle房价预测数据集实现一个预测模型。

如果你自学过多层感知机（MLP）可以直接根据.ipynb的代码引导完成这次作业。如果你还不会MLP，那么下周课上我们将做介绍，你可以先熟悉其他的代码处理部分。

提交要求：

1. 你完成的.ipynb文件（需要包含代码运行结果）

2. 你在Kaggle上面运行得分的截图

注意：一旦发现代码抄袭，本门课程得0分。

关于“”独热（one-hot）编码”的概念，可以上网搜索了解一下。

## 文件目录说明

- `Work01_kaggle_house_prise.ipynb`是所部署的最终代码，没有使用`nn.Module`，使用了`sklearn`的`KFold`和文档中建议的损失函数，不过在训练的时候实际使用的是（带根号的最后评测模型效果才使用）：
  $$
  \frac{1}{n}\sum_{i=1}^n\left(\log y_i -\log \hat{y}_i\right)^2
  $$

- `submission3.png`为最终提交结果截图

- `nndemo_test_linear_.ipynb`和`nndemo_test_mlp.ipynb`是自己用`nn.Module`看哪种模型优越的时候写的，最后选定用多层感知机，激活函数的使用`relu`，在截图中可以看到 用了`relu`的mlp比之前的用`sigmoid`的mlp分数一下子高起来了

- `Model.py`和`Trainer.py`是自己写的方便调试模型和训练可视化的两个文件，最后提交的结果没有使用这两个文件；

- `nonn_demo_test_mlp.ipynb`就是没有使用`nn.Module`的小demo，最后提交文件中的代码也是从这里迁移过去的

- `nonn_demo_test_optimizer.ipynb`是自己在重写优化器时的文件，但发现简单例子上`adam`并不见得会比`sgd`优，所以最后没有采用`adam`