import argparse
import os
import numpy as np
from PIL import Image


import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm

from model import ConvVAE

""" This script is an example of Sigma VAE training in PyTorch. The code was adapted from:
https://github.com/pytorch/examples/blob/master/vae/main.py """


class NumpyDataset(torch.utils.data.Dataset):
    def __init__(self, np_array, transform=None):
        # self.array = (np_array * 255).astype(np.uint8)
        self.array = np_array
        # self.array = torch.from_numpy(np_array)
        # self.array = Image.fromarray(np.uint8(np_array)).covert('RGB')
        # __import__('ipdb').set_trace()
        self.transform = transform

    def __len__(self):
        return len(self.array)

    def __getitem__(self, idx):
        sample = self.array[idx]
        print('before', sample.max())
        test = np.uint8(sample * 25)
        print('test', test.max())
        sample = Image.fromarray(test, mode='RGB')
        print('mid', np.max(np.array(sample)))
        if self.transform:
            sample = self.transform(sample)
        print('after', np.max(np.array(sample)))
        # sample += 0.5
        return sample

## Arguments
parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
        help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
        help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
        help='enables CUDA training')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
        help='how many batches to wait before logging training status')
parser.add_argument('--model', type=str, default='mse', metavar='N',
        help='which model to use: mse_vae,  gaussian_vae, or sigma_vae or optimal_sigma_vae')
parser.add_argument('--log_dir', type=str, default='test', metavar='N', required=True)
args = parser.parse_args()

## Cuda
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")

## Dataset
kwargs = {'num_workers': 10, 'pin_memory': True} if args.cuda else {}
img_size = 28




transform = transforms.Compose([
    # transforms.ToPILImage('RGB'),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
])
train_dataset = datasets.SVHN('../../data', split='train', download=True, transform=transform)
test_dataset = datasets.SVHN('../../data', split='train', transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
print(train_dataset[0][0].max())
print(train_dataset[0][0].shape)

transform = transforms.Compose([
    # transforms.ToPILImage('RGB'),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
])
train_path = '/home/vitchyr/mnt/log/manual-upload/sets/OneObject-PickAndPlace-BigBall-RandomInit-2D-v1-ungrouped-train-28x28.npy'
test_path = '/home/vitchyr/mnt/log/manual-upload/sets/OneObject-PickAndPlace-BigBall-RandomInit-2D-v1-ungrouped-test-28x28.npy'
test_imgs = np.load(test_path)
# test_dataset = NumpyDataset(test_imgs, transform)
test_dataset = torch.from_numpy(test_imgs.astype(np.float32))
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

train_imgs = np.load(train_path)
# train_dataset = NumpyDataset(train_imgs, transform)
train_dataset = torch.from_numpy(train_imgs.astype(np.float32))
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

# __import__('ipdb').set_trace()
## Logging
os.makedirs('vae_logs/{}'.format(args.log_dir), exist_ok=True)
summary_writer = SummaryWriter(log_dir='vae_logs/' + args.log_dir, purge_step=0)

## Build Model
model = ConvVAE(device, 3, args, img_size=img_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        if len(data) == 2:
            data = data[0]
        data = data.to(device)
        optimizer.zero_grad()

        # Run VAE
        recon_batch, mu, logvar = model(data)
        # Compute loss
        rec, kl = model.loss_function(recon_batch, data, mu, logvar)

        total_loss = rec + kl
        total_loss.backward()
        train_loss += total_loss.item()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            log_sigma = model.log_sigma
            if isinstance(log_sigma, torch.Tensor):
                log_sigma = log_sigma.item()
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tMSE: {:.6f}\tKL: {:.6f}\tlog_sigma: {:f}'.format(
                epoch,
                batch_idx * len(data),
                len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                rec.item() / len(data),
                kl.item() / len(data),
                log_sigma,
            ))

            train_loss /=  len(train_loader.dataset)
    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss))
    summary_writer.add_scalar('train/elbo', train_loss, epoch)
    summary_writer.add_scalar('train/rec', rec.item() / len(data), epoch)
    summary_writer.add_scalar('train/kld', kl.item() / len(data), epoch)
    summary_writer.add_scalar('train/log_sigma', model.log_sigma, epoch)


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            if len(data) == 2:
                data = data[0]
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            # Pass the second value from posthoc VAE
            rec, kl = model.loss_function(recon_batch, data, mu, logvar)
            test_loss += rec + kl
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n], recon_batch.view(args.batch_size, -1, 28, 28)[:n]])
                save_image(comparison.cpu(), 'vae_logs/{}/reconstruction_{}.png'.format(args.log_dir, str(epoch)), nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    summary_writer.add_scalar('test/elbo', test_loss, epoch)


if __name__ == "__main__":
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test(epoch)
        with torch.no_grad():
            sample = model.sample(64).cpu()
            save_image(sample.view(64, -1, 28, 28),
                    'vae_logs/{}/sample_{}.png'.format(args.log_dir, str(epoch)))
            summary_writer.file_writer.flush()

    torch.save(model.state_dict(), 'vae_logs/{}/checkpoint_{}.pt'.format(args.log_dir, str(epoch)))
