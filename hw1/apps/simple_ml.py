import struct
import gzip
import numpy as np

import sys
sys.path.append('python/')
import needle as ndl


def parse_mnist(image_filename, label_filename):
    """ Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0.

            y (numpy.ndarray[dypte=np.int8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.int8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR SOLUTION
    with gzip.open(image_filename, 'rb') as f:
      # read first 16 bytes and interprets them as C struct values
      # '>' means big endian
      # 'I' character indicates that the next 4 bytes should be interpreted as an unsigned integer (4 bytes each).
      magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
      image_size = rows * cols
      # create ndarray from read buffer, convert into dtype=float32
      data = np.frombuffer(f.read(), dtype=np.uint8).astype(np.float32)
      # normalization uint8 ranges in [0,255], so divide values by 255.0
      data = data / 255.0
      data = data.reshape(num_images, image_size)

    with gzip.open(label_filename, 'rb') as f:
      magic, num_items = struct.unpack('>II', f.read(8))
      labels = np.frombuffer(f.read(), dtype=np.uint8)

    return data, labels
    ### END YOUR SOLUTION


def softmax_loss(Z, y_one_hot):
    """ Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (ndl.Tensor[np.float32]): 2D Tensor of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (ndl.Tensor[np.int8]): 2D Tensor of shape (batch_size, num_classes)
            containing a 1 at the index of the true label of each example and
            zeros elsewhere.

    Returns:
        Average softmax loss over the sample. (ndl.Tensor[np.float32])
    """
    ### BEGIN YOUR SOLUTION
    e_z_sum = ndl.ops.exp(Z).sum(axes=(1,))
    z_i_log_sum = ndl.ops.log(e_z_sum).sum()
    # z_y = Z[range(Z.shape[0]), y_one_hot]  (batch_size, )
    z_y = (Z * y_one_hot).sum()
    loss = z_i_log_sum - z_y
    return loss / Z.shape[0]
    ### END YOUR SOLUTION


def nn_epoch(X, y, W1, W2, lr = 0.1, batch=100):
    """ Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W2
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (ndl.Tensor[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (ndl.Tensor[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD mini-batch

    Returns:
        Tuple: (W1, W2)
            W1: ndl.Tensor[np.float32]
            W2: ndl.Tensor[np.float32]
    """

    ### BEGIN YOUR SOLUTION
    num_examples = X.shape[0]
    num_classes = W2.shape[1]

    for i in range(0,num_examples, batch):
      batch_size = min(batch, num_examples-i)
      X_batch = X[i:i+batch_size]
      y_batch = y[i:i+batch_size]

      X_batch = ndl.Tensor(X_batch)
      y_one_hot = np.zeros((batch_size, num_classes))
      y_one_hot[np.arange(batch_size),y_batch] = 1
      y_one_hot = ndl.Tensor(y_one_hot)

      # forward pass
      logits = ndl.ops.relu(X_batch @ W1) @ W2

      # compute loss
      loss = softmax_loss(logits, y_one_hot)

      # backpropagation
      loss.backward()      

      # update W1 and W2 (compute in Numpy, avoid creating new ndl.Tensor)
      W1_new = W1.numpy() - lr * W1.grad.numpy()
      W2_new = W2.numpy() - lr * W2.grad.numpy()

      # convert to Tensor
      W1 = ndl.Tensor(W1_new)
      W2 = ndl.Tensor(W2_new)
      
    return (W1,W2)
    ### END YOUR SOLUTION


### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT

def loss_err(h,y):
    """ Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h,y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)
