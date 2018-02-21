import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch.autograd import Variable

import pathlib
import os

import matplotlib.pyplot as plt
import numpy as np

import PIL

def generate_batch_images(_g, batch_size, start=0, end=9, prefix="", suffix="", figure_path="./samples", height=32, width=32, depth=3, labels=None, normalized=False):
    if start >= end:
        raise ArithmeticError("start is higher than end [%d > %d]" % (start, end))

    _g.cuda()
    pathlib.Path(figure_path).mkdir(parents=True, exist_ok=True)    

    for n in range(start, end + 1):
        noise = Variable(torch.cuda.FloatTensor(batch_size, depth, height, width).normal_())

        label = np.full((batch_size, 1), n)
        label_one_hot = (np.arange(11) == label[:,None]).astype(np.float)
        label_one_hot = torch.from_numpy(label_one_hot)
        label_one_hot = Variable(label_one_hot.cuda())

        im_outputs = _g(label_one_hot.float(), noise)

        if labels is not None:
            name = labels[n]
        else:
            name = str(n)

        for i, img in enumerate(im_outputs):
            trans = transforms.ToPILImage()
            data = img.data.cpu()

            if normalized:
                data = data / 2 + 0.5

            im = trans(data)
            im.save(os.path.join(figure_path, "%s-%s-%d-%s.png" % (prefix, name, i, suffix)))