import os
import numpy as np
import copy

import torch
from torch.optim import Adam

from mnist import Net
from PIL import Image


MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


def save_image(img, path):
    img = Image.fromarray(img[0].astype(np.uint8))
    img.save(path)


def img_to_tensor(img):
    img_float = np.float32(img)
    img_float[0] /= 255
    img_float[0] -= MNIST_MEAN
    img_float[0] /= MNIST_STD
    img_tensor = torch.from_numpy(img_float).float()
    img_tensor.unsqueeze_(0)  # Make a batch
    img_tensor.requires_grad_()
    return img_tensor


def tensor_to_img(img_variable):
    img = copy.copy(img_variable.data.numpy()[0])
    # Inverted restoration of std / mean
    img[0] *= MNIST_STD
    img[0] += MNIST_MEAN
    img[img > 1] = 1
    img[img < 0] = 0
    img = np.round(img * 255)
    return img


def find_best_input(model, target_class, num_loops=50, lr=0.5):
    img = np.uint8(np.zeros((1, 28, 28)))
    for i in range(num_loops):
        tensor = img_to_tensor(img)
        optimizer = Adam([tensor], lr=lr)
        output = model(tensor)
        # The model output uses log_softmax, so the output tends towards negative numbers where the smallest one is the best prediction.
        # We want to maximize the target class, so we multiply it by -10, so the goal is to make it as small as possible and ideally 0
        # Furthermore, we add the other negative non-targets, so the goal is to make them more negative -> more unlikely
        loss = -10 * output[0, target_class] + torch.sum(output)
        print('Target Class: {0}, Iteration: {1}, Loss: {2:.2f}'.format(str(target_class), str(i), loss.data.numpy()))
        model.zero_grad()
        loss.backward()
        optimizer.step()
        img = tensor_to_img(tensor)
        img_path = './generated/img_{}_iteration_{}.jpg'.format(str(target_class), str(i))
        save_image(img, img_path)


if __name__ == '__main__':
    model = Net()
    model.load_state_dict(torch.load('./mnist_cnn.pt'))
    model.eval()  # Don't change model weights
    if not os.path.exists('./generated'):
        os.makedirs('./generated')
    for target_class in range(10):
        find_best_input(model, target_class, num_loops=50)
