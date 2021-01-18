import torch
import torch.nn as nn
from batchout import BatchOut

# A class to flatten the output of Conv2d
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)    

model_cnn = nn.Sequential(nn.Conv2d(1, 32, 5, padding=1), nn.ReLU(),
                          nn.MaxPool2d(2, stride=2),
                          nn.Conv2d(32, 32, 5, padding=1), nn.ReLU(),
                          nn.MaxPool2d(2, stride=2), Flatten(), 
                          nn.Dropout(p=0.5), nn.Linear(800, 256), 
                          nn.ReLU(), nn.Dropout(p=0.5), nn.Linear(256, 10))

batchout_model_cnn_all = nn.Sequential(nn.Conv2d(1, 32, 5, padding=1), BatchOut(0.05, 0.5), nn.ReLU(),
                          nn.MaxPool2d(2, stride=2),
                          nn.Conv2d(32, 32, 5, padding=1), BatchOut(0.1, 0.5), nn.ReLU(),
                          nn.MaxPool2d(2, stride=2), Flatten(),
                          nn.Dropout(p=0.5), nn.Linear(800, 256),
                          BatchOut(0.1, 0.5), nn.ReLU(), nn.Dropout(p=0.5), nn.Linear(256, 10))

batchout_model_cnn_c1 = nn.Sequential(nn.Conv2d(1, 32, 5, padding=1), BatchOut(0.05, 0.5), nn.ReLU(),
                          nn.MaxPool2d(2, stride=2),
                          nn.Conv2d(32, 32, 5, padding=1), nn.ReLU(),
                          nn.MaxPool2d(2, stride=2), Flatten(),
                          nn.Dropout(p=0.5), nn.Linear(800, 256),
                          nn.ReLU(), nn.Dropout(p=0.5), nn.Linear(256, 10))

batchout_model_cnn_c2 = nn.Sequential(nn.Conv2d(1, 32, 5, padding=1), nn.ReLU(),
                          nn.MaxPool2d(2, stride=2),
                          nn.Conv2d(32, 32, 5, padding=1), BatchOut(0.1, 0.5), nn.ReLU(),
                          nn.MaxPool2d(2, stride=2), Flatten(),
                          nn.Dropout(p=0.5), nn.Linear(800, 256),
                          nn.ReLU(), nn.Dropout(p=0.5), nn.Linear(256, 10))

batchout_model_cnn_c12 = nn.Sequential(nn.Conv2d(1, 32, 5, padding=1), BatchOut(0.05, 0.5), nn.ReLU(),
                          nn.MaxPool2d(2, stride=2),
                          nn.Conv2d(32, 32, 5, padding=1), BatchOut(0.1, 0.5), nn.ReLU(),
                          nn.MaxPool2d(2, stride=2), Flatten(),
                          nn.Dropout(p=0.5), nn.Linear(800, 256),
                          nn.ReLU(), nn.Dropout(p=0.5), nn.Linear(256, 10))

batchout_model_cnn_f1 = nn.Sequential(nn.Conv2d(1, 32, 5, padding=1), nn.ReLU(),
                          nn.MaxPool2d(2, stride=2),
                          nn.Conv2d(32, 32, 5, padding=1), nn.ReLU(),
                          nn.MaxPool2d(2, stride=2), Flatten(),
                          nn.Dropout(p=0.5), nn.Linear(800, 256),
                          BatchOut(0.1, 0.5), nn.ReLU(), nn.Dropout(p=0.5), nn.Linear(256, 10))

batchout_model_cnn_c2f1 = nn.Sequential(nn.Conv2d(1, 32, 5, padding=1), nn.ReLU(),
                          nn.MaxPool2d(2, stride=2),
                          nn.Conv2d(32, 32, 5, padding=1), BatchOut(0.1, 0.5), nn.ReLU(),
                          nn.MaxPool2d(2, stride=2), Flatten(),
                          nn.Dropout(p=0.5), nn.Linear(800, 256),
                          BatchOut(0.1, 0.5), nn.ReLU(), nn.Dropout(p=0.5), nn.Linear(256, 10))
