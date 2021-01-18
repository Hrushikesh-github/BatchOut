import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from batchout import BatchOut
import foolbox as fb
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

mnist_train = datasets.MNIST("./data", train=True, download=True, transform=transforms.ToTensor())
mnist_test = datasets.MNIST("./data", train=False, download=True, transform=transforms.ToTensor())

train_loader = DataLoader(mnist_train, batch_size = 512, shuffle=True)
test_loader = DataLoader(mnist_test, batch_size = 512, shuffle=False)

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)

batchout_model_cnn = nn.Sequential(nn.Conv2d(1, 32, 5, padding=1), BatchOut(0.05, 0.5), nn.ReLU(),
                          nn.MaxPool2d(2, stride=2),
                          nn.Conv2d(32, 32, 5, padding=1), BatchOut(0.1, 0.5), nn.ReLU(),
                          nn.MaxPool2d(2, stride=2), Flatten(),
                          nn.Dropout(p=0.5), nn.Linear(800, 256),
                          BatchOut(0.1, 0.5), nn.ReLU(), nn.Dropout(p=0.5), nn.Linear(256, 10)).to(device)

batchout_model_cnn.load_state_dict(torch.load('0.pt'))

batchout_model_cnn.eval()

X,y = next(iter(test_loader))
X = X.to(device)
y = y.to(device)

yp = batchout_model_cnn(X)
loss = nn.CrossEntropyLoss()(yp,y)

img = X[0].to('cpu')

plt.imshow(img.numpy().transpose(1, 2, 0))
#plt.show()
total_err = (yp.max(dim=1)[1] != y).sum().item() / 512

print(total_err)

fmodel = fb.PyTorchModel(batchout_model_cnn, bounds=(0, 255))

print(fb.utils.accuracy(fmodel, X, y))
attack = fb.attacks.FGSM()

for e in [1/255, 2/255, 3/255, 4/255, 5/255]:
    
    (raw, clipped, is_adv) = attack(fmodel, X, y, epsilons=e)

    total_err = (is_adv != True).sum().item() / 512

    print('For epsilon {} probability obtained is {}'.format(e, total_err))
