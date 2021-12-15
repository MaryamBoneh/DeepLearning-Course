import cv2, torch, argparse, torchvision, model
import numpy as np

my_parser = argparse.ArgumentParser()
my_parser.add_argument('--image')
my_parser.add_argument('--device')
my_parser.add_argument('--wieght')
args = my_parser.parse_args()

device = torch.device(args.device)

model = model.mnist_classifire()
model = model.to(device)

data_transform = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(),
     torchvision.transforms.Normalize((0),(1))]
)

img = cv2.imread(args.image)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.resize(img,(28,28))

model.load_state_dict(torch.load(args.wieght))
model.eval()

tensor = data_transform(img).unsqueeze(0).to(device)

y_hat = model(tensor)
y_hat = y_hat.cpu().detach().numpy()
output = np.argmax(y_hat)
    
print(f"prediction: ", output)