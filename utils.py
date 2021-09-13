import random
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def show_batch_imgs(dataloader, show_img_num=24, nrow=4, figsize=(10, 10)):

    batch_tensor = next(iter(testloader))[0][:show_img_num]
    grid_img = make_grid(batch_tensor + 0.5, nrow=nrow)
    plt.figure(figsize=figsize)
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.show()