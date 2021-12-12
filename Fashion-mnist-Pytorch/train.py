import torch, torchvision

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

device=torch.device("cuda")
model=mnist_classifire()
model=model.to(device)
model.train(True)

batch=64
epoch=20
lr=0.001

data_transform=torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(),
     torchvision.transforms.Normalize((0),(1))]
)

dataset = torchvision.datasets.FashionMNIST("./dataset",train=True,download=True,transform=data_transform)

train_data = torch.utils.data.DataLoader(dataset,batch_size=batch,shuffle=True)

optimizer = torch.optim.Adam(model.parameters(),lr=lr)
loss_func = torch.nn.CrossEntropyLoss()

def cal_acc(y_hat,labels):
    _,y_hat_max=torch.max(y_hat,1)
    acc=torch.sum(y_hat_max==labels.data,dtype=torch.float64)/len(y_hat)
    return acc

#🔸🔸🔸🔸Train🔸🔸🔸🔸

for ep in range(epoch):
    train_loss = 0.0
    train_acc = 0.0

    for im,labels in train_data:
        im = im.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        #forwarding
        y_hat = model(im)

        #backwarding
        loss = loss_func(y_hat,labels)
        loss.backward()

        #update
        optimizer.step()

        train_loss += loss
        train_acc += cal_acc(y_hat,labels)

    total_loss  =  train_loss/len(train_data)
    total_acc  =  train_acc/len(train_data)

    print(f"epoch:{ep} , Loss:{total_loss} , accuracy: {total_acc}")


#🔸🔸🔸🔸Save weights🔸🔸🔸🔸
torch.save(model.state_dict(), "fashion-mnist.pth")