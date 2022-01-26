import cv2, torch, argparse, model
import numpy as np
from torchvision import transforms


my_parser = argparse.ArgumentParser()
my_parser.add_argument('--image')
my_parser.add_argument('--device')
my_parser.add_argument('--wieght')
args = my_parser.parse_args()

model.load_state_dict(torch.load('age-estimating-resnet50-torch.pth'))
model.eval()

device = torch.device(args.device)
model = model.Model()
model = model.to(device)

data_transform = transforms.Compose([
                                     transforms.ToPILImage(),
                                     transforms.Resize((70, 70)),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                     ])

model.load_state_dict(torch.load(args.wieght))
model.eval()

img = cv2.imread(args.image)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

tensor = data_transform(img).unsqueeze(0).to(device)

y_hat = model(tensor)
y_hat = y_hat.cpu().detach().numpy()
output = np.argmax(y_hat)
    
print(f"prediction: ", output)
