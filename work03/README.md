# 作业三

## 任务要求

CIFAR-10 图像分类任务

本次作业和小作业1类似，请下载`Work03_CIFAR-10_Classification.ipynb`文件，根据提示信息，完成本次作业。

## 文件目录

```shell
    目录: D:\Desktop2\DL_Foundation\assignment\work03\code


Mode                 LastWriteTime         Length Name
----                 -------------         ------ ----
-a----         2023/4/12     19:52          11040 0_load_data.ipynb
-a----         2023/4/15     22:51          79140 1_LeNet.ipynb
-a----         2023/4/15     22:52          76572 1_LeNet_norm.ipynb
-a----         2023/4/11     10:45         280048 2_CIFAR10_AlexNet_colab.ipynb
-a----         2023/4/15     13:38          37907 3_dataProcessed_model_train.ipynb
-a----         2023/4/12     19:31          96777 3_data_preprocess.ipynb
-a----         2023/4/15     10:06          16615 3_data_proprocess.ipynb
-a----         2023/4/15     13:25           8095 3_testColab_trainParams.ipynb
-a----         2023/4/15     16:44         276451 4_VGG.ipynb
-a----         2023/4/15     16:42         398520 4_VGG_colab.ipynb
-a----         2023/4/15     17:34         514959 5_visualize_validate.ipynb
-a----         2023/4/15     16:15         463377 6_refer_tutorial_transfer_learning.ipynb
-a----         2023/4/17     17:01         214225 7_ResNet18_50_colab.ipynb
-a----         2023/4/10      9:57           3109 models.py
-a----         2023/4/15     17:25          24537 myCatPicture.jpg
-a----         2023/4/15     17:23         174041 myCatPicture.png
-a----         2023/4/16     11:37           4027 README.md
-a----         2023/4/10     19:43           4074 train.py
-a----          2023/4/9     21:40           8142 trainer.py
-a----         2023/4/15      9:58            736 utils.py
-a----         2023/4/16     11:17         734204 work3_董林康_1120212477.ipynb
```

## 文件说明

### main

- `models.py`是模型的定义文件，包括了LinearNet,LeNet,AlexNet
- `train.py`是训练模型的文件，包括了训练参数和快速调参的一些函数
- `trainer.py`是之前自己在第一次作业的时候写的用于kfold交叉验证的trainer，这次作业在除了VGG16之外的模型都用到了这
- `utils.py`是一些工具函数的定义文件，包括了可视化数据的函数
- `work3_董林康_1120212477.ipynb`是最终的作业文件，包括了所有的实验结果和分析

### others

- `0_load_data.ipynb`加载老师发布作业文件中所下载的数据，以及可视化数据的notebook；
- `1_LeNet.ipynb`和`1_LeNet_norm.ipynb`分别是LeNet网络的实现，其中`1_LeNet_norm.ipynb`是对数据进行了归一化处理（就是很简单地减每个像素每个通道除以255）；
- `2_CIFAR10_AlexNet_colab.ipynb`是在colab上运行的LinearNet,LeNet,AlexNet网络的实现，由于colab的GPU资源有限，所以只训练了10个epoch前三个模型，最后集中训练了AlexNet模型；
- `3_dataProcessed_model_train.ipynb`是对数据进行了预处理，然后单独训练AlexNet模型的notebook，但是最终发现效果并不好；
- `3_data_preprocess.ipynb`和`3_data_proprocess.ipynb`是对数据进行预处理的notebook，其中`3_data_preprocess.ipynb`是在本地运行的，`3_data_proprocess.ipynb`是在colab上运行的，这两个notebook是为了解决`3_dataProcessed_model_train.ipynb`中的transform的问题而写的，没有太大意义；
- `3_testColab_trainParams.ipynb`是在本地上测试从colab上所下载的训练参数的notebook，由于colab上的参数是挂载在cuda上的，我们加载的时候需要注意加一个`map_location='cpu'`参数；
- `4_VGG.ipynb`和`4_VGG_colab.ipynb`分别是VGG网络的实现，其中`4_VGG_colab.ipynb`是在colab上运行的，前者只在本地运行了一次（但超过了20min依然没有到达九十个batch，更不用说一个epoch了），后者则训练了四个epoch，最后达到83%的测试集准确率，但还是算力不够用了；
- `5_visualize_validate.ipynb`是对训练好VGG16模型进行可视化的notebook，上传了自己从网上找的一张猫的图片`myCatPicture.jpg`并预测；
- `6_refer_tutorial_transfer_learning.ipynb`是所参考的pytorch官方教程中的迁移学习的notebook
- `7_ResNet18_50_colab.ipynb`则是自己使用ResNet18从头训练、分别使用ResNet18和ResNet50进行迁移学习的notebook，最后达到90.09的测试集准确率，也是在本次作业中自己能达到的最好的准确率