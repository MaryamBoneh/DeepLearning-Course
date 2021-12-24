import torch, argparse, torchvision, model

my_parser = argparse.ArgumentParser()
my_parser.add_argument('--device')
args = my_parser.parse_args()

device = torch.device(args.device)

def cal_acc(y_hat,labels):
    _,y_hat_max=torch.max(y_hat,1)
    acc=torch.sum(y_hat_max==labels.data,dtype=torch.float64)/len(y_hat)
    return acc

data_transform = torchvision.transforms.Compose([
                torchvision.transforms.RandomRotation(10),
                torchvision.transforms.Resize((70, 70)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                ])

test_data = torchvision.datasets.ImageFolder(root='/dataset', transform = data_transform)
test_data = torch.utils.data.DataLoader(test_data, batch_size = 32, shuffle = True)

model = model.Model()
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