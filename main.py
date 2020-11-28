import torch 
from torch import nn, optim
from torch.nn import functional as F
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp

import os
import argparse
import time

from sys import exit as e


class ConvNet(nn.Module):
  def __init__(self,):
    super(ConvNet, self).__init__()
    self.conv1 = nn.Conv2d(1, 32, 3, 1)
    self.conv2 = nn.Conv2d(32, 64, 3, 1)
    self.dropout1 = nn.Dropout2d(0.25)
    self.dropout2 = nn.Dropout2d(0.5)
    self.fc1 = nn.Linear(9216, 128)
    self.fc2 = nn.Linear(128, 10)
    self.batchnorm = nn.BatchNorm1d(128)

  def forward(self, x):
    x = self.conv1(x)
    x = F.relu(x)
    x = self.conv2(x)
    x = F.relu(x)
    x = F.max_pool2d(x, 2)
    x = self.dropout1(x)
    x = torch.flatten(x, 1)
    x = self.fc1(x)
    x = self.batchnorm(x)
    x = F.relu(x)
    x = self.dropout2(x)
    x = self.fc2(x)
    output = F.log_softmax(x, dim=1)
    return output

def setup(rank, world_size):
  os.environ["MASTER_ADDR"] = "localhost"
  os.environ["MASTER_PORT"] = "12355"
  dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
  dist.destroy_process_group()

def train(opt, model, rank, train_loader, optimizer, epoch):
  model.train()
  for b, (img, label) in enumerate(train_loader):
    img, label = img.to(rank), label.to(rank)
    optimizer.zero_grad()
    out = model(img)
    loss = F.nll_loss(out, label)
    loss.backward()
    optimizer.step()
    if b % 10 == 0:
      dist.all_reduce(loss, op=dist.ReduceOp.SUM)
      if rank == 0:
        print(f"epoch: {epoch} [{b}/{len(train_loader)} \
          ({100. * b/len(train_loader)}%)]\t Loss:{loss.item()}")

def test(opt, model, rank, test_loader, optimizer, epoch):
  model.eval()
  test_loss=0
  correct = 0
  test_loss_tensor = 0
  correct_tensor = 0
  datalen = 0
  with torch.no_grad():
    for img, label in test_loader:
      img, label = img.to(rank), label.to(rank)
      out = model(img)
      test_loss += F.nll_loss(out, label, reduction='sum').item()
      test_loss_tensor += F.nll_loss(out, label, reduction='sum')
      pred = out.argmax(dim=1, keepdim=True)
      correct += pred.eq(label.view_as(pred)).sum().item() 
      correct_tensor += pred.eq(label.view_as(pred)).sum()
      datalen += len(img)

  test_loss /= datalen
  test_loss /= len(test_loader.dataset)
  dist.all_reduce(test_loss_tensor, op=dist.ReduceOp.SUM)
  dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)

  if rank==0:
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(\
      test_loss_tensor, correct_tensor, len(test_loader.dataset),100. * correct_tensor / len(test_loader.dataset)))


def proc(rank, world_size, opt, use_cuda):
  # rank = 'cpu' if not torch.cuda.is_available() else rank
  print(f"running DDP example on rank {rank}")
  setup(rank, world_size)

  torch.manual_seed(0)
  device = torch.device('cuda' if use_cuda else 'cpu')
  kwargs = {'num_workers':1, 'pin_memory': True} if use_cuda else {}
  
  train_dataset = datasets.MNIST('./data/', train=True, download=True, transform=transforms.Compose(\
    [transforms.ToTensor(), transforms.Normalize((0.1307), (0.3081))]))
  train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
  train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=False, \
    num_workers=0, pin_memory=True, sampler=train_sampler)

  test_dataset = datasets.MNIST('./data/', train=False, download=True, transform=transforms.Compose(\
    [transforms.ToTensor(), transforms.Normalize((0.1307), (0.3081))]))
  test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank)
  test_loader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, sampler=test_sampler)

  model = ConvNet().to(rank)
  if torch.cuda.device_count() > 1:
    # print(f"There are {torch.cuda.device_count()} GPUs available")
    # model =nn.DataParallel(model, device_ids=[0,1,2,3])
    model = DistributedDataParallel(model, device_ids=[rank])

  optimizer = optim.Adadelta(model.parameters(), lr=opt.lr)
  scheduler = StepLR(optimizer, step_size=1, gamma=opt.gamma)

  for epoch in range(opt.epochs):
    t0 = time.time()
    train(opt, model, rank, train_loader, optimizer, epoch)
    test(opt, model, rank, train_loader, optimizer, epoch)
    scheduler.step()
    print(f"time to complete this epoch: {time.time() - t0} seconds")
  torch.save(model.state_dict(), "mnist.pt")
  cleanup()


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--batch_size', type=int, default=128, help="batch size of train and test")
  parser.add_argument('--lr', type=float, default=0.001, help="LR of optimizer")
  parser.add_argument('--epochs', type=int, default=20, help="#of epochs")
  parser.add_argument('--no-cuda', default=False, help="disables CUDA if True")
  parser.add_argument('--gamma', default=0.7, type=float, help="gamma value for lr update")
  parser.add_argument('--gpus', default=4, type=int, help="# of GPUs")
  opt = parser.parse_args()

  use_cuda = not opt.no_cuda and torch.cuda.is_available()
  world_size = opt.gpus if torch.cuda.is_available() else 2
  # proc(0, world_size, opt, use_cuda)
  mp.spawn(proc, args=(world_size, opt, use_cuda), nprocs=world_size, join=True)

  


if __name__ == '__main__':
  main()
