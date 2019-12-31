# PyTorch input visualization of a neural network

This repository explores how to visualize what a neural network learned after having trained it.
The basic idea is that we keep the network weights fixed while we run backpropagation on the input image to change the input image to excite our target output the most.

We thus obtain images which are "idealized" versions of our target classes.

For MNIST this is the result of these "idealized" input images that the network likes most for the numbers 0 - 9:

![png](results.png)

If you want to learn more: check out my blog entry explaining this [visualization technique for deep neural networks](https://www.paepper.com/blog/posts/do-you-know-which-inputs-your-neural-network-likes-most/)
