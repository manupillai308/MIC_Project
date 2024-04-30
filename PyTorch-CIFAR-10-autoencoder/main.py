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
    def __init__(self, mode, root, transforms=None):
        self.root = root
        self.mode = mode
        self.files = sorted(os.listdir(root))
        if mode == 'train':
            self.files = self.files[:int(0.8*len(self.files))]
        else:
            self.files = self.files[int(0.8*len(self.files)):]

        self.transforms = transforms
        # self.files = pd.Series(self.files)
            
    def __len__(self):
        return len(self.files)*50
    
    def __getitem__(self, ix):
        ix = ix % len(self.files)
        f = self.files[ix]
        if self.mode == 'train':
            slice_ix = np.random.randint(50, 121)
        else:
            slice_ix = 100
        imgs = [nib.load(os.path.join(self.root, f"{f}/{f}-{m}.nii.gz")).get_fdata().astype(np.float32)[:, :, slice_ix] for m in ["t2w", "t1c", "t1n", "t2f"]]
        out = torch.stack([torch.from_numpy(img) for img in imgs])
        if self.transforms is not None:
            out = self.transforms(out)
            max_ = out.amax(dim=(1, 2), keepdims=True)
            if torch.any(max_ == 0):
                out = torch.zeros_like(out)
            else:
                out = out/max_
        return torch.maximum(out, torch.Tensor([0]))

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


def get_torch_vars(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return x

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
        decoded = self.decoder(encoded)
        return encoded, decoded

def main():
    parser = argparse.ArgumentParser(description="Train Autoencoder")
    parser.add_argument("--valid", action="store_true", default=False,
                        help="Perform validation only.")
    args = parser.parse_args()

    dataset_root = "/media/ma293852/New Volume/BRATS/train"

    # Create model
    autoencoder = create_model()

    # Load data
    transform = transforms.Compose(
        [transforms.CenterCrop(170), transforms.Resize(32)])
    # trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
    #                                         download=True, transform=transform)
    trainset = BRATS(mode='train', root=dataset_root, transforms=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=16,
                                              shuffle=True, num_workers=8)
    # testset = torchvision.datasets.CIFAR10(root='./data', train=False,
    #                                        download=True, transform=transform)
    testset = BRATS(mode='val', root=dataset_root, transforms=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=16,
                                             shuffle=False, num_workers=8)
    # classes = ('plane', 'car', 'bird', 'cat',
    #            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    if args.valid:
        print("Loading checkpoint...")
        autoencoder.load_state_dict(torch.load("./weights/autoencoder.pkl"))
        dataiter = iter(testloader)
        images = dataiter.next()
        # print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(16)))
        imshow(torchvision.utils.make_grid(images))

        images = Variable(images.cuda())

        decoded_imgs = autoencoder(images)[1]
        imshow(torchvision.utils.make_grid(decoded_imgs.data))

        exit(0)

    # Define an optimizer and criterion
    criterion = nn.BCELoss()
    optimizer = optim.Adam(autoencoder.parameters())

    for epoch in range(100):
        running_loss = 0.0
        for i, (inputs) in tqdm(enumerate(trainloader, 0), total=len(trainloader)):
            inputs = get_torch_vars(inputs)

            # ============ Forward ============
            encoded, outputs = autoencoder(inputs)
            loss = criterion(outputs, inputs)
            # ============ Backward ============
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # ============ Logging ============
            running_loss += loss.data
            if i % 200 == 199:
                fmtstr = '[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 200)
                print(fmtstr)
                running_loss = 0.0
                torchvision.utils.save_image(torchvision.utils.make_grid(outputs.detach()[:5].reshape(-1, 1, 32, 32), nrow=5, pad_value=0), f'outputs/{fmtstr}.jpg')

        print('Saving Model...')
        if not os.path.exists('./weights'):
            os.mkdir('./weights')
        torch.save(autoencoder.state_dict(), f"./weights/autoencoder-{epoch}.pkl")
    print('Finished Training')


if __name__ == '__main__':
    main()
