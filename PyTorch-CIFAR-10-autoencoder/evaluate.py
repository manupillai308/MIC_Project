# Numpy
import numpy as np

# Torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from torch.autograd import Variable
import nibabel as nib
import torch.utils
from tqdm import tqdm
import h5py

# Torchvision
import torchvision
import torchvision.transforms as transforms

# Matplotlib
import matplotlib.pyplot as plt

# OS
import os
import argparse

# Set random seed for reproducibility
SEED = 87
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)


class BRATS(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.files = sorted(os.listdir(root))

        self.transforms = transforms
            
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, ix):
        ix = ix % len(self.files)
        f = self.files[ix]
        imgs = [nib.load(os.path.join(self.root, f"{f}/{f}-{m}.nii.gz")).get_fdata().astype(np.float32) for m in ["t2w", "t1c", "t1n", "t2f"]]
        out = torch.stack([torch.from_numpy(img) for img in imgs]).permute(3, 0, 1, 2)
        if self.transforms is not None:
            out = self.transforms(out)
            max_ = out.amax(dim=(2, 3), keepdims=True)
            if torch.any(max_ == 0):
                out = torch.zeros_like(out)
            else:
                out = out/max_
        return f, torch.maximum(out, torch.Tensor([0]))

def print_model(encoder, decoder):
    print("============== Encoder ==============")
    print(encoder)
    print("============== Decoder ==============")
    print(decoder)
    print("")


def create_model():
    autoencoder = Autoencoder()
    print_model(autoencoder.encoder, autoencoder.decoder)
    if torch.cuda.is_available():
        autoencoder = autoencoder.cuda()
        print("Model moved to GPU in order to speed up training.")
    return autoencoder



def imshow(img):
    npimg = img.cpu().numpy()
    plt.axis('off')
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 12, 4, stride=2, padding=1),            # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.Conv2d(12, 24, 4, stride=2, padding=1),           # [batch, 24, 8, 8]
            nn.ReLU(),
			nn.Conv2d(24, 48, 4, stride=2, padding=1),           # [batch, 48, 4, 4]
            nn.ReLU(),
# 			nn.Conv2d(48, 96, 4, stride=2, padding=1),           # [batch, 96, 2, 2]
#             nn.ReLU(),
        )
        self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(96, 48, 4, stride=2, padding=1),  # [batch, 48, 4, 4]
#             nn.ReLU(),
			nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            nn.ReLU(),
			nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1),  # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(12, 4, 4, stride=2, padding=1),   # [batch, 3, 32, 32]
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        # decoded = self.decoder(encoded)
        # return encoded, decoded
        return encoded

def main():
    dataset_root = "/media/ma293852/New Volume/BRATS/train"

    to_path = "/media/ma293852/New Volume/MIC/"
    os.makedirs(to_path, exist_ok=True)

    # Create model
    autoencoder = create_model()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Load data
    transform = transforms.Compose(
        [transforms.CenterCrop(170), transforms.Resize(32)])
    # trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
    #                                         download=True, transform=transform)
    trainset = BRATS(root=dataset_root, transforms=transform)
    autoencoder.load_state_dict(torch.load("./weights/autoencoder-99.pkl"))
    
    embeddings = h5py.File(os.path.join(to_path, 'embeddings.hdf5'), 'w')
    
    # classes = ('plane', 'car', 'bird', 'cat',
    #            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    with torch.no_grad():
        for i in tqdm(range(len(trainset))):
            f, slices = trainset[i]
            encoded = autoencoder(slices.to(device))
            dset = embeddings.create_dataset(f, (155, 48, 4, 4), dtype='f')
            dset[...] = encoded.cpu()
    embeddings.close()


if __name__ == '__main__':
    main()
    # f = h5py.File('/media/ma293852/New Volume/MIC/embeddings.hdf5', 'r')
    # print(list(f.keys()))
    # dset = f[list(f.keys())[0]]
    # print(dset.shape)
    # print(dset.dtype)



