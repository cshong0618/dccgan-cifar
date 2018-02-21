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

    # Params
    epochs = args.epoch
    sample_output = args.sample_output
    sample_nums = args.sample_nums
    batch_size = args.batch_size

    sample_interval = epochs // sample_nums

    # Hyperparams
    learning_rate_d = 1e-3
    learning_rate_g = 1e-3

    # Class names 
    class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    # CIFAR dataset
    #transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))]) # https://github.com/kuangliu/pytorch-cifar/issues/19
    #transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = datasets.CIFAR10("./data", train=True, transform=transform, download=True)
    test_dataset = datasets.CIFAR10("./data", train=False, transform=transform, download=True)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # Build model
    _d = model.D(3, 11)
    _d.cuda()

    _g = model.CCNN(input_class=11)
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
            current_batch_size = len(images)
            print("Training batch: %d / %d" % (i, total_batches), end="\r")
            sys.stdout.flush()

            # Prepare images and labels
            images = Variable(images.cuda())

            # Prepare one hot labels for G
            labels_onehot = torch.from_numpy(one_hot(labels, 11)).float().cuda()
            labels_g = Variable(labels_onehot)

            # Put labels to GPU
            labels = Variable(labels.cuda())
            #print(labels)

            # Train D network
            optimizer_d.zero_grad()
            outputs_d = _d(images)
            real_loss = criterion_d(outputs_d, labels)
            real_loss.backward()            
            #print(outputs_d)

            # Generate fake labels
            noise = Variable(torch.cuda.FloatTensor(current_batch_size, 3, 32, 32).normal_())
            fake_labels = np.zeros(current_batch_size) + 10
            
            #print(fake_labels)
            #input()
            
            fake_labels_d = Variable(torch.from_numpy(fake_labels).long().cuda())

            # Generate fake images and classify
            fake_labels_g = np.random.randint(0, 10, current_batch_size)      
            labels_fake_onehot = torch.from_numpy(one_hot(fake_labels_g, 11)).float().cuda()
            labels_fake_onehot = Variable(labels_fake_onehot)
            fake_images = _g(labels_fake_onehot, noise)
            fake_outputs= _d(fake_images)

            # Calculate loss
            fake_loss = criterion_d(fake_outputs, fake_labels_d)
            fake_loss.backward()

            loss_d = real_loss + fake_loss
            #loss_d = real_loss
            optimizer_d.step()

            # Train G network
            optimizer_g.zero_grad()
            noise = Variable(torch.cuda.FloatTensor(current_batch_size, 3, 32, 32).normal_())
            
            fake_labels = np.random.randint(0, 10, current_batch_size)      
            labels_fake_onehot = torch.from_numpy(one_hot(fake_labels, 11)).float().cuda()
            labels_fake_onehot = Variable(labels_fake_onehot)

            fake_labels = torch.from_numpy(fake_labels)
            fake_labels = Variable(fake_labels.cuda())

            images_g = _g(labels_fake_onehot, noise)
            truth = _d(images_g)

            loss_g = criterion_g(truth, fake_labels)
            loss_g.backward()
            optimizer_g.step()

            if i % 100 == 99:
                print("Epoch [%d/%d], Iter [%d/%d] D loss:%.5f G loss:%.5f" % (
                    epoch + 1,
                    epochs,
                    i + 1,
                    len(train_dataset) // batch_size,
                    loss_d.data[0],
                    loss_g.data[0]
                ))

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
            generate_batch_images(_g, 5, start=0, end=9, prefix="training-epoch-%d" % (epoch + 1), figure_path=sample_output, labels=class_names)
            sys.stdout.flush()
            print("Generated images for epoch %d" % (epoch + 1))
        else:
            _i += 1

    return 0

if __name__ == "__main__":
    main()