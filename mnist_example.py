import os
import numpy as np
import torch
from torchvision import datasets, transforms
from torch import nn, optim
from magnet_loss_iclr2016 import MagnetSampler
from magnet_loss_iclr2016 import MagnetLoss
from magnet_loss_iclr2016 import AverageMeter
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(self, emb_dim):
        super(LeNet, self).__init__()
        self.emb_dim = emb_dim
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.emb = nn.Linear(64 * 7 * 7, self.emb_dim)
        self.logmax_layer = nn.LogSoftmax(dim=1)
        self.layer1 = None
        self.layer2 = None
        self.features = None
        self.embeddings = None
        self.norm_embeddings = None

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        self.layer1 = x
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        self.layer2 = x
        x = x.view(-1, self.num_flat_features(x))
        self.features = x
        x = self.emb(x)
        # embeddings = x
        # x = self.logmax_layer(x)

        return x, self.features

    def num_flat_features(self, x):
        """
        Calculate the total tensor x feature amount
        """

        size = x.size()[1:]  # All dimensions except batch dimension
        num_features = 1
        for s in size:
            num_features *= s

        return num_features

    def name(self):
        return "lenet-magnet"


class MyMagnetSampler(MagnetSampler):
    def __init__(self, dataset, model, k, m, d):
        super().__init__(dataset, model, k, m, d)

    def get_labels(self):
        all_labels = []
        trainloader = torch.utils.data.DataLoader(self.dataset, batch_size=self.m * self.d, shuffle=False)
        for _, target in trainloader:
            all_labels.extend(target.cpu().numpy())
        return np.array(all_labels)

    def get_reps(self):
        all_reps = []
        trainloader = torch.utils.data.DataLoader(self.dataset, batch_size=self.m * self.d, shuffle=False)
        for img, _ in trainloader:
            img = img.cuda()
            _, train_features = self.model(img)
            # Apparently no need to use data between function calls, because it's deprecated
            all_reps.extend(train_features.detach().cpu().numpy())
        return np.array(all_reps)


def train_magnet():
    # Below the lenet is chosen, because of following reasons
    # For every forward pass, it gives back the final prediction as well as features
    # Used to train magnet loss. For your model, if you want to train it with magnet loss, you have to make a similar model
    # where you return the prediction as well as the features, same as Lenet, implemntation above
    model = LeNet(10).cuda()

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    dir_path = os.path.dirname(os.path.realpath(__file__))
    trainset = datasets.MNIST(f"{dir_path}/datasets", download=True, train=True, transform=transform)
    # Reducing the size, so that it can work on TSNE
    trainset, _ = torch.utils.data.random_split(trainset, [2000, 58000])
    # valset = datasets.MNIST(f"{dir_path}/datasets", download=True, train=False, transform=transform)
    k = 3
    m = 4
    d = 8
    alpha = 1.0
    my_magnet_sampler = MyMagnetSampler(trainset, model, k, m, d)
    criterion = MagnetLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    e = 1
    model.train()
    while 1:
        epoch_loss = AverageMeter()
        my_magnet_sampler.update_clusters()
        batch_class_inds = [ids for ids in my_magnet_sampler]
        for each_batch in batch_class_inds:
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=m * d, sampler=iter(each_batch))
            # The training needs to be done for n times, before updating the reps, so that it's fast
            for _ in range(3):
                for images, labels in trainloader:
                    img = images.cuda()
                    optimizer.zero_grad()
                    output, _ = model(img)
                    batch_loss, batch_example_losses = criterion(output, labels.numpy(), m, d, alpha)
                    batch_loss.backward()
                    optimizer.step()
                    my_magnet_sampler.update_losses(each_batch, batch_example_losses)
                    epoch_loss.update(batch_loss.detach().cpu().numpy())
        my_magnet_sampler.save_tsne_to_image("tsne_image.png")
        e += 1
        if e % 1 == 0:
            print(f"Batch Loss {epoch_loss.avg}")


if __name__ == "__main__":
    train_magnet()
