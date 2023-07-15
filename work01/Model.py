import torch.nn as nn

class LinearNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearNet, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        x = self.fc(x)
        return x


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        # self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.sigmoid(x)
        x = self.fc2(x)
        return x
    
class MLP2(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP2, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        # self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.fc1(x)
        # x = self.sigmoid(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    

class MLP2_batchnorm(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP2_batchnorm, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.batchnorm = nn.BatchNorm1d(hidden_size)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
class MLP3(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP3, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        # self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)
    

class MLP3_batchnorm(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP3_batchnorm, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.batchnorm1 = nn.BatchNorm1d(hidden_size)
        self.batchnorm2 = nn.BatchNorm1d(hidden_size)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        return self.fc3(x)