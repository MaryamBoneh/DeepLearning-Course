import torch, torchvision, argparse
import torch.nn as nn
import torch.nn.functional as F


my_parser = argparse.ArgumentParser()
my_parser.add_argument('--device', default="cpu")
args = my_parser.parse_args()

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, (3, 3), (1, 1), (1, 1))
        self.conv2 = nn.Conv2d(32, 32, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(32, 64, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(64, 128, (3, 3), (1, 1), (1, 1))

        self.fc1 = nn.Linear(128*8*8, 512)
        self.fc2 = nn.Linear(512, 10)


    def forward(self, x):
      x = F.relu(self.conv1(x))
      x = F.max_pool2d(x, kernel_size=(2, 2))
      x = F.relu(self.conv2(x))
      x = F.max_pool2d(x, kernel_size=(2, 2))
      x = F.relu(self.conv3(x))
      x = F.max_pool2d(x, kernel_size=(2, 2))
      x = F.relu(self.conv4(x))
      x = torch.flatten(x, start_dim=1)
      x = F.relu(self.fc1(x))
      x = torch.flatten(x, start_dim=1)
      x = torch.dropout(x, 0.2, train=True)
      x = self.fc2(x)
      x = torch.softmax(x, dim=1)

      return x

device=torch.device(args.device)
model=Model()
model=model.to(device)
model.train(True)

batch=32
epoch=20
lr=0.001

data_transform = torchvision.transforms.Compose([
                torchvision.transforms.RandomRotation(10),
                torchvision.transforms.Resize((70, 70)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                ])

dataset = torchvision.datasets.ImageFolder(root='/dataset', transform = data_transform)
dataset = torch.utils.data.DataLoader(dataset, batch_size = batch, shuffle = True)

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

    for im,labels in dataset:
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

    total_loss  =  train_loss/len(dataset)
    total_acc  =  train_acc/len(dataset)

    print(f"epoch:{ep} , Loss:{total_loss} , accuracy: {total_acc}")


#🔸🔸🔸🔸Save weights🔸🔸🔸🔸
torch.save(model.state_dict(), "persian-mnist.pth")