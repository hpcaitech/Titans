import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from colossalai.utils import get_dataloader
import torchvision.transforms as transforms


def build_imagenet(batch_size, crop=224, resize=256):
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(crop, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.)),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.IMAGENET),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    transform_test = transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(crop),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    data_root = os.environ['DATA']
    train_path = os.path.join(data_root, "train")
    test_path = os.path.join(data_root, "test")
    train_dataset = torchvision.datasets.ImageFolder(root=train_path, transform=transform_train)
    test_dataset = torchvision.datasets.ImageFolder(root=test_path, transform=transform_test)
    train_dataloader = get_dataloader(dataset=train_dataset, shuffle=True, batch_size=batch_size, pin_memory=True)
    test_dataloader = get_dataloader(dataset=test_dataset, batch_size=batch_size, pin_memory=True)
    return train_dataloader, test_dataloader
