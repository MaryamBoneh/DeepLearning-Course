import torch, tqdm, argparse, cv2, os
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torchvision import transforms
from sklearn.model_selection import train_test_split
from model import Model


my_parser = argparse.ArgumentParser()
my_parser.add_argument('--device', default="cpu")
args = my_parser.parse_args()


batch = 64
epoch = 15
lr = 0.001
width = height = 224

images = []
ages = []
under4 = []
X = []
Y = []

for image_name in os.listdir('crop_part1')[0:9000]:
    part = image_name.split('_')
    ages.append(int(part[0]))

    image = cv2.imread(f'crop_part1/{image_name}')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    images.append(image)

images = pd.Series(images, name= 'Images')
ages = pd.Series(ages, name= 'Ages')
df = pd.concat([images, ages], axis= 1)

for i in range(len(df)):
    if df['Ages'].iloc[i] <= 4:
        under4.append(df.iloc[i])

under4 = pd.DataFrame(under4)
under4 = under4.sample(frac= 0.3)
up4 = df[df['Ages'] > 4]
df = pd.concat([under4, up4])
df = df[df['Ages'] < 90]

for i in range(len(df)):
    df['Images'].iloc[i] = cv2.resize(df['Images'].iloc[i], (width, height))

    X.append(df['Images'].iloc[i])
    Y.append(df['Ages'].iloc[i])

X = np.array(X)
Y = np.array(Y)

X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size = 0.2)

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, transform=None):
        self.data = X
        self.target = y
        self.transform = transform
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]

        if self.transform:
            x = self.transform(x)

        return x, y
    
    def __len__(self):
        return len(self.data)

data_transform = transforms.Compose([
                                     transforms.ToPILImage(),
                                     transforms.Resize((70, 70)),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                     ])


dataset = MyDataset(X_train, Y_train, data_transform)
dataset = torch.utils.data.DataLoader(dataset, batch_size = batch)

device = torch.device(args.device)
model = Model()
model = model.to(device)
model.train(True)

optimizer = torch.optim.Adam(model.parameters(), lr = lr)
loss_function = torch.nn.MSELoss()

#🔸🔸🔸🔸Train🔸🔸🔸🔸
for ep in range(epoch):
    train_loss = 0.0

    for im, labels in tqdm(dataset):
        im = im.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        #forwarding
        y_hat = model(im)

        #backwarding
        loss = loss_function(y_hat,labels.float())
        loss.backward()

        #update
        optimizer.step()

        train_loss += loss

    total_loss  =  train_loss/len(dataset)

    print(f"epoch:{ep} , Loss:{total_loss}")

#🔸🔸🔸🔸Save weights🔸🔸🔸🔸
torch.save(model.state_dict(), "age-estimating-resnet50-torch.pth")
