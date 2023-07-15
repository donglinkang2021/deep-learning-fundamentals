# Path: train.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from models import LeNet, AlexNet
from tqdm import tqdm
import os
import numpy as np

class ArgumentConfig:
    def __init__(self,
                lr=0.001,
                resume=False,
                model='AlexNet',
                epochs=5,
                batch_size=16,
                n_splits=5,
                is_print=True,
                print_every=10
                ):
        self.lr = lr
        self.resume = resume
        self.model = model
        self.epochs = epochs
        self.batch_size = batch_size
        self.n_splits = n_splits
        self.is_print = is_print
        self.print_every = print_every

args = ArgumentConfig()

# 定义模型
if args.model == 'LeNet':
    model = LeNet()
elif args.model == 'AlexNet':
    model = AlexNet()
else:
    raise Exception("model name error")

# 定义损失函数
criterion = nn.CrossEntropyLoss()

# 定义优化器
optimizer = optim.Adam(model.parameters(), lr=args.lr)

# 定义学习率衰减
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

# 定义 tensorboard
writer = SummaryWriter(log_dir='runs/' + args.model)

# 定义是否使用 GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# 定义是否加载模型
if args.resume:
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    model.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

# 定义训练函数
def train(epoch):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        if batch_idx % args.print_every == 0 and args.is_print:
            print('Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAcc: {:.3f}%'.format(
                epoch, batch_idx * len(inputs), len(trainloader.dataset),
                       100. * batch_idx / len(trainloader), loss.item(),
                100. * correct / total))
    writer.add_scalar('Train/Loss', train_loss / (batch_idx + 1), epoch)
    writer.add_scalar('Train/Acc', 100. * correct / total, epoch)

# 定义测试函数
def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    writer.add_scalar('Test/Loss', test_loss / (batch_idx + 1), epoch)
    writer.add_scalar('Test/Acc', 100. * correct / total, epoch)
    return 100. * correct / total

# 定义主函数
def main():
    for epoch in range(args.epochs):
        train(epoch)
        acc = test(epoch)
        scheduler.step()
        # 保存模型
        state = {
            'net': model.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')

if __name__ == '__main__':
    main()
