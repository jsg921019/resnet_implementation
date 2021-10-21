import os
import pandas as pd
import torch
from tqdm import tqdm
from .utils import seed_everything

class Trainer:
    
    def __init__(self, seed, log_dir, weight_dir):
        
        self.seed = seed
        self.log_dir = os.path.expanduser(log_dir)
        self.weight_dir = os.path.expanduser(weight_dir)
        
        for directory in [self.log_dir, self.weight_dir]:
            os.makedirs(directory, exist_ok=True)       

    def train(self, name, model, n_epochs, trainloader, testloader, criterion, optimizer, lr_scheduler):
        
        seed_everything(self.seed)
        
        log = {"train_error": [], "test_error": []}
        
        device = next(model.parameters()).device

        for epoch in tqdm(range(1, n_epochs+1)):

            # Training Phase

            running_incorrect = 0
            running_data = 0

            model.train()

            for inputs, labels in trainloader:

                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                lr_scheduler.step()

                with torch.no_grad():
                    pred = torch.argmax(outputs, dim=-1)
                    running_incorrect += torch.sum(pred != labels)
                    running_data += inputs.size(0)

            train_epoch_error = running_incorrect / running_data


            # Validation Phase

            running_incorrect = 0
            running_data = 0

            model.eval()

            for inputs, labels in testloader:

                inputs, labels = inputs.to(device), labels.to(device)

                with torch.no_grad():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    pred = torch.argmax(outputs, dim=-1)
                    running_incorrect += torch.sum(pred != labels)
                    running_data += inputs.size(0)

            test_epoch_error = running_incorrect / running_data

            # log results
            
            log["train_error"].append(train_epoch_error.item())
            log["test_error"].append(test_epoch_error.item())

        torch.save(model.state_dict(), os.path.join(self.weight_dir, f'{name}.pt'))
        log_df = pd.DataFrame(log, index=pd.Index(range(1, len(log["train_error"]) + 1), name='epoch'))
        log_df.to_csv(os.path.join(self.log_dir, f'{name}.csv'))

        print(f'Finished training {name}')
        print(f'  train_error : {100 * train_epoch_error:.1f}%')
        print(f'  test_error  : {100 * test_epoch_error:.1f}%\n')

        