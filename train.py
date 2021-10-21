import os
import argparse
import yaml

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import CIFAR10
from model import ResNet
from utils.trainer import Trainer
from utils.transform import get_transform
import utils.scheduler as scheduler

parser = argparse.ArgumentParser(description='Plot ')
parser.add_argument('config_path', type=str)
args = parser.parse_args()

def get_layer_config(n):
    return [(16, 1, n), (32, 2, n), (64, 2, n)]

with open(args.config_path) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    
train_transform, test_transform = get_transform()

trainset = CIFAR10(train=True, transform=train_transform, **config['dataset'])
testset = CIFAR10(train=False, transform=test_transform, **config['dataset'])

trainloader = torch.utils.data.DataLoader(trainset, shuffle=True, drop_last=True, **config['dataloader'])
testloader = torch.utils.data.DataLoader(testset, shuffle=False, drop_last=False, **config['dataloader'])

device = 'cuda' if torch.cuda.is_available() else 'cpu'

trainer = Trainer(**config['trainer'])

for model_name, model_config in config['experiments'].items():    
    model = ResNet(layer_config=get_layer_config(model_config['n']), **config[model_config['model']])
    first_in_channels = config[model_config['model']]['first_in_channels']
    model.conv1 = nn.Sequential(nn.Conv2d(3, first_in_channels, kernel_size=3, stride=1, padding=1, bias=False),
                                        nn.BatchNorm2d(first_in_channels),
                                        nn.ReLU(inplace=True))
    model.to(device)
    
    if 'params' in config['criterion']:
        criterion = getattr(nn, config['criterion']['name'])(**config['criterion']['params'])
    else:
        criterion = getattr(nn, config['criterion']['name'])()

    optimizer = getattr(optim, config['optimizer']['name'])(model.parameters(), **config['optimizer']['params'])

    lr_scheduler = getattr(scheduler , config['scheduler']['name'])(optimizer, **config['scheduler']['params'])
    
    trainer.train(model_name, model, config['n_epochs'], trainloader, testloader, criterion, optimizer, lr_scheduler)