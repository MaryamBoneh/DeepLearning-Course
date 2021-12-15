import cv2, torch, argparse, torchvision, model
import numpy as np

my_parser = argparse.ArgumentParser()
my_parser.add_argument('--wieght')
my_parser.add_argument('--device')
args = my_parser.parse_args()

device = torch.device(args.device)

def cal_acc(y_hat,labels):
    _,y_hat_max=torch.max(y_hat,1)
    acc=torch.sum(y_hat_max==labels.data,dtype=torch.float64)/len(y_hat)
    return acc


data_transform=torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(),
     torchvision.transforms.Normalize((0),(1))]
)

dataset = torchvision.datasets.FashionMNIST("./dataset",train=False,download=True,transform=data_transform)
test_data = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

model = model.mnist_classifire()
model = model.to(device)
model.load_state_dict(torch.load(args.wieght))
model.eval()

test_acc=0.0

for im,labels in test_data:
    im = im.to(device)
    labels = labels.to(device)
    y_hat = model(im)
    test_acc += cal_acc(y_hat,labels)

test_acc  =  test_acc/len(test_data)
    
print(f"test acc: ", test_acc)