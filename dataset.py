import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchvision import transforms

class KannadaMNISTDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, n):
        data = self.df.iloc[n]
        image = data[1:].values.reshape((28,28)).astype(np.uint8)
        label = data[0]
        if self.transform:
            image = self.transform(image)
        return (image, label)

def get_train_and_validation_dataset(batch_size, proportion_train=0.8, flatten=False):

    data_df = pd.read_csv("data/train.csv")

    idx_train = int(len(data_df)*proportion_train)
    train_df = data_df.iloc[:idx_train]
    val_df = data_df.iloc[idx_train:]


    data_mean = data_df.values[:,1:].mean()/255
    data_std = data_df.values[:,1:].std()/255

    train_transform = transforms.Compose(
                        [
                        transforms.ToPILImage(),
                        transforms.RandomRotation(10),
                        transforms.RandomAffine(
                                    degrees=20,
                                    translate=(0.12,0.12),
                                    scale=(0.95, 1.15),
                                    shear=0.2),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[data_mean], std=[data_std]),
                        ])

    normalise_transform = transforms.Compose(
                        [
                        transforms.ToPILImage(),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[data_mean], std=[data_std]),
                        ])

    train_dataset = KannadaMNISTDataset(train_df, transform=train_transform)
    trainloader = DataLoader(dataset=train_dataset,
                                    batch_size=batch_size, shuffle=True)

    val_dataset = KannadaMNISTDataset(val_df, transform=normalise_transform)
    valloader = DataLoader(dataset=val_dataset,
                                    batch_size=len(val_df),shuffle=False)

    test_df = pd.read_csv("data/test.csv")
    test_dataset = KannadaMNISTDataset(test_df, transform=normalise_transform)
    testloader = DataLoader(dataset=test_dataset,
                                    batch_size=len(test_df),shuffle=False)

    return trainloader, valloader, testloader
