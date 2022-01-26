import torch, argparse, torchvision, model
from torchvision import transforms

my_parser = argparse.ArgumentParser()
my_parser.add_argument('--device')
my_parser.add_argument('--dataset')
args = my_parser.parse_args()

batch = 64
epoch = 15
lr = 0.001
width = height = 224

device = torch.device(args.device)

data_transform = transforms.Compose([
                                     transforms.ToPILImage(),
                                     transforms.Resize((70, 70)),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                     ])

test_data = torchvision.datasets.ImageFolder(args.dataset, transform = data_transform)
test_data = torch.utils.data.DataLoader(test_data, batch_size = batch, shuffle = True)

model = model.Model()
model = model.to(device)
model.load_state_dict(torch.load(args.wieght))
model.eval()

cal_loss = torch.nn.MSELoss()

test_loss = 0.0

for im,labels in test_data:
    im = im.to(device)
    labels = labels.to(device)
    y_hat = model(im)
    test_loss += cal_loss(y_hat,labels)

test_loss  =  test_loss/len(test_data)
    
print(f"test loss: ", test_loss)