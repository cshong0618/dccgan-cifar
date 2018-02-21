import sys
import argparse
import time

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch.autograd import Variable

import numpy as np

import model
from sampler import ChunkSampler
from utils import generate_batch_images

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epoch", help="Number of epochs to run", type=int, default=100)
    parser.add_argument("--sample_output", help="Output of in-training samples", default="./training")
    parser.add_argument("--sample_nums", help="Number of times to produce in-training samples", type=int, default=100)
    parser.add_argument("--batch_size", help="Batch size", type=int, default=32)

    args = parser.parse_args(sys.argv[1:])
    return args

def one_hot(labels, output_size=10):
    try:
        ret = labels.numpy()
    except:
        ret = labels
    ret = (np.arange(output_size) == ret[:,None]).astype(np.float)
    return ret

def main():
    args = parse_args()

    epochs = args.epoch
    sample_output = args.sample_output
    sample_nums = args.sample_nums
    batch_size = args.batch_size

    sample_interval = epochs // sample_nums

    learning_rate_d = 1e-3
    learning_rate_g = 1e-3

    # CIFAR dataset
    train_dataset = datasets.CIFAR10("./data", train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = datasets.CIFAR10("./data", train=False, transform=transforms.ToTensor(), download=True)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # Build model
    _d = model.D(3, 11)
    _d.cuda()

    _g = model.CCNN()
    _g.cuda()

    # Loss and optimizer
    criterion_d = nn.CrossEntropyLoss().cuda()
    optimizer_d = torch.optim.Adam(_d.parameters(), lr=learning_rate_d)

    criterion_g = nn.CrossEntropyLoss().cuda()
    optimizer_g = torch.optim.Adam(_g.parameters(), lr=learning_rate_g)

    total_batches = len(train_dataset) // batch_size
    _i = 1
    for epoch in range(epochs):
        for i, (images, labels) in enumerate(train_loader):
            print("Training batch: %d / %d" % (i, total_batches), end="\r")
            sys.stdout.flush()

            # Prepare images and labels
            images = Variable(images.cuda())
            labels_onehot = torch.from_numpy(one_hot(labels)).float().cuda()
            labels_g = Variable(labels_onehot)
            
            labels = Variable(labels.cuda())

            # Train D network
            optimizer_d.zero_grad()
            outputs_d = _d(images)
            real_loss = criterion_d(outputs_d, labels)

            noise = Variable(torch.cuda.FloatTensor(batch_size, 3, 32, 32).normal_())
            fake_labels = np.zeros(batch_size) + 10
            fake_labels_d = Variable(torch.from_numpy(fake_labels).long().cuda())
            
            fake_images = _g(labels_g, noise)
            fake_outputs= _d(fake_images)
            fake_loss = criterion_d(fake_outputs, fake_labels_d)

            loss_d = real_loss + fake_loss
            loss_d.backward()
            optimizer_d.step()

            # Train G network
            noise = Variable(torch.cuda.FloatTensor(batch_size, 3, 32, 32).normal_())
            
            fake_labels = np.random.randint(0, 10, batch_size)      
            labels_fake_onehot = torch.from_numpy(one_hot(fake_labels)).float().cuda()
            labels_fake_onehot = Variable(labels_fake_onehot)

            fake_labels = torch.from_numpy(fake_labels)
            fake_labels = Variable(fake_labels.cuda())

            images_g = _g(labels_fake_onehot, noise)
            optimizer_g.zero_grad()
            truth = _d(images_g)

            loss_g = criterion_g(truth, fake_labels)
            loss_g.backward()
            optimizer_g.step()

        # Log and outputs
        _d.eval()
        correct_d = 0
        total_d = 0

        for images, labels in test_loader:
            images = Variable(images).cuda()
            outputs = _d(images)

            _, predicted = torch.max(outputs.data, 1)
            total_d += labels.size(0)
            correct_d += (predicted.cpu() == labels).sum()
        
        print("Epoch [%d/%d], Iter [%d/%d] D loss:%.5f D accuracy: %.2f%% G loss:%.5f" % (
            epoch + 1,
            epochs,
            i + 1,
            len(train_dataset) // batch_size,
            loss_d.data[0],
            (100 * correct_d / total_d),
            loss_g.data[0]
        ))
                

        if _i == sample_interval:
            _i = 1
            print("Generating images: ", end="\r")
            generate_batch_images(_g, 5, start=0, end=9, prefix="training-epoch-%d" % (epoch + 1), figure_path=sample_output)
            sys.stdout.flush()
            print("Generated images for epoch %d" % (epoch + 1))
        else:
            _i += 1

    return 0

if __name__ == "__main__":
    main()