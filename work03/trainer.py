import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import numpy as np

class Trainer:
    def __init__(self, model, X, y, num_epochs, batch_size, loss_fn, optimizer, n_splits=5, is_print=True, print_every=10):
        self.train_losses_per_fold = [] # 记录每个fold的训练损失
        self.test_losses_per_fold = [] # 记录每个fold的测试损失
        self.mean_train_losses = np.zeros(num_epochs) # 记录每个epoch的平均训练损失
        self.mean_test_losses = np.zeros(num_epochs) # 记录每个epoch的平均测试损失
        
        self.model = model

        self.X = X
        self.y = y

        self.num_epochs = num_epochs
        self.batch_size = batch_size

        self.loss_fn = loss_fn
        self.optimizer = optimizer

        self.n_splits = n_splits

        self.is_print = is_print
        self.print_every = print_every
        pass

    
    def reset_parameters(self, model):
        # 重置模型参数
        for layer in model.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
                print("reset_parameters")

    def reset_model(self):
        # 重置模型参数为最开始初始化的参数
        self.model.load_state_dict(torch.load("model.pth"))

    def train_kfold(self):
        # 保存模型
        torch.save(self.model.state_dict(), "model.pth")

        kf = KFold(n_splits=self.n_splits) # 创建KFold对象

        # 开始交叉验证循环
        for fold_index, (train_index, test_index) in enumerate(kf.split(self.X)):
            if self.is_print:
                print(f"Fold {fold_index}")

            # self.model.apply(self.reset_parameters) # 重置模型参数为最开始初始化的参数
            self.reset_model() # 重置模型参数为最开始初始化的参数
                
            X_train = self.X[train_index] # 训练集特征
            y_train = self.y[train_index] # 训练集标签
            X_test = self.X[test_index] # 测试集特征
            y_test = self.y[test_index] # 测试集标签

            train_loss_history = [] # 记录训练损失
            test_loss_history = [] # 记录测试损失
            
            train_dataset = TensorDataset(X_train, y_train) # 创建训练集数据集对象
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size) # 创建训练集数据加载器对象
            
            test_dataset = TensorDataset(X_test, y_test) # 创建测试集数据集对象
            test_loader = DataLoader(test_dataset, batch_size=self.batch_size) # 创建测试集数据加载器对象

            
            for epoch in range(self.num_epochs): # 训练num_epochs个周期

                train_loss = 0
                test_loss = 0

                self.model.eval() # 设置模型为评估模式
                
                with torch.no_grad(): 
                    
                    for batch_X, batch_y in test_loader: 
                        
                        output = self.model(batch_X).squeeze()
                        
                        loss = self.loss_fn(output, batch_y)

                        test_loss += loss.item() * batch_y.shape[0] # 记录测试损失
                
                
                self.model.train() # 设置模型为训练模式

                for batch_X, batch_y in train_loader: # 遍历训练集批次
                    
                    self.optimizer.zero_grad() # 清空梯度
                    
                    output = self.model(batch_X).squeeze() # 前向传播，得到输出
                    
                    loss = self.loss_fn(output, batch_y) # 计算损失
                    
                    loss.backward() # 反向传播，计算梯度
                    
                    self.optimizer.step() # 更新参数

                    train_loss += loss.item() * batch_y.shape[0] # 记录训练损失

                mean_train_loss = train_loss / X_train.shape[0] # 计算平均训练损失
                mean_test_loss = test_loss / X_test.shape[0]# 计算平均测试损失
                
                if self.is_print and (epoch+1) % self.print_every == 0: # 每print_every个周期打印一次训练信息(损失
                    print(f"Epoch | {epoch+1:2d}/{self.num_epochs:2d} | Train Loss: {mean_train_loss:.4f} | Test Loss: {mean_test_loss:.4f}")

                train_loss_history.append(mean_train_loss) # 记录训练损失
                test_loss_history.append(mean_test_loss) # 记录测试损失

            self.train_losses_per_fold.append(train_loss_history) # 记录每个fold的训练损失
            self.test_losses_per_fold.append(test_loss_history) # 记录每个fold的测试损失

        self.mean_train_losses = np.mean(np.array(self.train_losses_per_fold), axis=0)
        self.mean_test_losses = np.mean(np.array(self.test_losses_per_fold), axis=0)

    def train_normal(self):
        X_train = self.X # 训练集特征
        y_train = self.y # 训练集标签

        train_loss_history = [] # 记录训练损失
        
        train_dataset = TensorDataset(X_train, y_train) # 创建训练集数据集对象
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size) # 创建训练集数据加载器对象
                
        for epoch in range(self.num_epochs): # 训练num_epochs个周期

            train_loss = 0
            
            self.model.train() # 设置模型为训练模式

            for batch_X, batch_y in train_loader: # 遍历训练集批次
                
                self.optimizer.zero_grad() # 清空梯度
                
                output = self.model(batch_X).squeeze() # 前向传播，得到输出
                
                loss = self.loss_fn(output, batch_y) # 计算损失
                
                loss.backward() # 反向传播，计算梯度
                
                self.optimizer.step() # 更新参数

                train_loss += loss.item() * batch_y.shape[0] # 记录训练损失

            mean_train_loss = train_loss / X_train.shape[0] # 计算平均训练损失

            if self.is_print and (epoch+1) % self.print_every == 0: # 每print_every个周期打印一次训练信息(损失
                print(f"Epoch | {epoch+1:2d}/{self.num_epochs:2d} | Train Loss: {mean_train_loss:.4f} ")

            train_loss_history.append(mean_train_loss) # 记录训练损失

        self.train_losses_per_fold.append(train_loss_history) # 记录每个fold的训练损失
    
    def visualize_kfold(self,width=30, height=6):
        # 绘制训练损失和测试损失
        plt.figure(figsize=(width, height))
        for i in range(self.n_splits):
            plt.subplot(1, self.n_splits, i+1)
            plt.plot(self.train_losses_per_fold[i], label=f"Train Loss Fold {i+1}")
            plt.plot(self.test_losses_per_fold[i], label=f"Test Loss Fold {i+1}")
            plt.legend()
        plt.show()

    def plot_mean_loss(self, width=10, height=6):
        # 绘制平均训练损失和平均测试损失 (所有fold的平均)
        plt.figure(figsize=(width, height))
        plt.plot(self.mean_train_losses, label="Mean Train Loss")
        plt.plot(self.mean_test_losses, label="Mean Test Loss")
        plt.legend()
        plt.show()
    
    def visualize_normal(self,width=10, height=6):
        # 绘制训练损失
        plt.figure(figsize=(width, height))
        plt.plot(self.train_losses_per_fold[0], label=f"Train Loss")
        plt.legend()
        plt.show()

    def predict(self, test_X):
        self.model.eval()
        with torch.no_grad():
            output = self.model(test_X).squeeze() # .squeeze()将输出的形状从[batch_size, 1]转换为[batch_size]
            return output