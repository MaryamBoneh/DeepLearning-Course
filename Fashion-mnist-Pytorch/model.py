import torch

class mnist_classifire(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.L1=torch.nn.Linear(784,256)
        self.L2=torch.nn.Linear(256,64)
        self.L3=torch.nn.Linear(64,10)

    def forward(self,x):
        x = x.reshape((x.shape[0],784))

        z = self.L1(x)
        z = torch.relu(z)
        z = torch.dropout(z,0.2,train = True)

        z = self.L2(z)
        z = torch.relu(z)
        z = torch.dropout(z,0.2,train = True)
        z = self.L3(z)

        y = torch.softmax(z,dim = 1)
        return y