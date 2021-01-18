import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from batchout import BatchOut

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

mnist_train = datasets.MNIST("./data", train=True, download=True, transform=transforms.ToTensor())
mnist_test = datasets.MNIST("./data", train=False, download=True, transform=transforms.ToTensor())

train_loader = DataLoader(mnist_train, batch_size = 512, shuffle=True)
test_loader = DataLoader(mnist_test, batch_size = 512, shuffle=False)

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)    

model_cnn = nn.Sequential(nn.Conv2d(1, 32, 5, padding=1), nn.ReLU(),
                          nn.MaxPool2d(2, stride=2),
                          nn.Conv2d(32, 32, 5, padding=1), nn.ReLU(),
                          nn.MaxPool2d(2, stride=2), Flatten(), 
                          nn.Dropout(p=0.5), nn.Linear(800, 256), 
                          nn.ReLU(), nn.Dropout(p=0.5), nn.Linear(256, 10)).to(device)

batchout_model_cnn = nn.Sequential(nn.Conv2d(1, 32, 5, padding=1), BatchOut(0.05, 0.5), nn.ReLU(),
                          nn.MaxPool2d(2, stride=2),
                          nn.Conv2d(32, 32, 5, padding=1), BatchOut(0.1, 0.5), nn.ReLU(),
                          nn.MaxPool2d(2, stride=2), Flatten(),
                          nn.Dropout(p=0.5), nn.Linear(800, 256),
                          BatchOut(0.1, 0.5), nn.ReLU(), nn.Dropout(p=0.5), nn.Linear(256, 10)).to(device)

def epoch(loader, model, opt=None):
    """Standard training/evaluation epoch over the dataset"""
    total_loss, total_err = 0.,0.
    for X,y in loader:
        X,y = X.to(device), y.to(device)
        if not opt:
            model.eval()
            yp = model(X)
            loss = nn.CrossEntropyLoss()(yp,y)
        if opt:
            model.train()
            yp = model(X)
            loss = nn.CrossEntropyLoss()(yp, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

        total_err += (yp.max(dim=1)[1] != y).sum().item()
        total_loss += loss.item() * X.shape[0]
    return total_err / len(loader.dataset), total_loss / len(loader.dataset)

opt = optim.SGD(batchout_model_cnn.parameters(), lr=1e-2)

print('Training the model')

for t in range(10):
    train_err, train_loss = epoch(train_loader, batchout_model_cnn, opt)
    test_err, test_loss = epoch(test_loader, batchout_model_cnn)
    if t == 4:
        for param_group in opt.param_groups:
            param_group["lr"] = 1e-3
    print(*("{:.6f}".format(i) for i in (train_err, test_err)), sep="\t")

