import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_transform():
    
    train_transform = A.Compose([A.PadIfNeeded(min_height=40, min_width=40),
                             A.RandomCrop(32, 32),
                             A.HorizontalFlip(),
                             ToTensorV2()])

    test_transform = ToTensorV2()
    
    return train_transform, test_transform