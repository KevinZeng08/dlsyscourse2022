import sys
sys.path.append('../python')
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)

def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    linear1 = nn.Linear(dim, hidden_dim)
    norm1 = norm(dim=hidden_dim)
    relu1 = nn.ReLU()
    dropout = nn.Dropout(drop_prob)
    linear2 = nn.Linear(hidden_dim, dim)
    norm2 = norm(dim=dim)
    fn = nn.Sequential(linear1, norm1, relu1, dropout, linear2, norm2)
    residual = nn.Residual(fn)

    relu2 = nn.ReLU()
    block = nn.Sequential(residual, relu2) 
    return block
    ### END YOUR SOLUTION


def MLPResNet(dim, hidden_dim=100, num_blocks=3, num_classes=10, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    linear1 = nn.Linear(in_features=dim, out_features=hidden_dim)
    relu1 = nn.ReLU()
    residuals = []
    for _ in range(num_blocks):
        residual = ResidualBlock(dim=hidden_dim, hidden_dim=hidden_dim // 2, norm=norm, drop_prob=drop_prob)
        residuals.append(residual)
    linear2 = nn.Linear(in_features=hidden_dim, out_features=num_classes)

    block = nn.Sequential(linear1, relu1, *residuals, linear2)
    return block
    ### END YOUR SOLUTION



def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    if opt:
        model.train()
    else:
        model.eval()
    
    loss_fn = nn.SoftmaxLoss()
    losses = []
    num_examples = 0
    correct = 0
    error_rate = 0.0
    for batch in dataloader:
        x,y = batch[0], batch[1] # x is samples, y is true labels
        x = x.reshape((x.shape[0], -1))

        y_hat = model(x)
        loss = loss_fn(y_hat, y)
        losses.append(loss.cached_data)
        
        # 求正确分类的个数：取y_hat中最大值的索引，与y比较，相等则为正确分类
        label_hat = np.argmax(y_hat.cached_data, axis=1)
        correct += np.sum(label_hat == y.cached_data)
        num_examples += y.shape[0]

        if opt:
            opt.reset_grad()
            loss.backward()
            opt.step()
    
    error_rate = 1 - correct / num_examples
    avg_loss = np.mean(losses)
    return error_rate, avg_loss
    ### END YOUR SOLUTION



def train_mnist(batch_size=100, epochs=10, optimizer=ndl.optim.Adam,
                lr=0.001, weight_decay=0.001, hidden_dim=100, data_dir="data"):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    # load train and test data
    train_image_filename = os.path.join(data_dir, "train-images-idx3-ubyte.gz")
    train_label_filename = os.path.join(data_dir, "train-labels-idx1-ubyte.gz")
    test_image_filename = os.path.join(data_dir, "t10k-images-idx3-ubyte.gz")
    test_label_filename = os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz")

    train_dataset = ndl.data.MNISTDataset(train_image_filename, train_label_filename)
    test_dataset = ndl.data.MNISTDataset(test_image_filename, test_label_filename)

    train_dataloader = ndl.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = ndl.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # create MLPResNet
    model = MLPResNet(dim=784, hidden_dim=hidden_dim)

    # create optimizer
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)

    # training epoches
    for i in range(epochs):
        print("Epoch %d" % i)
        train_error, train_loss = epoch(dataloader=train_dataloader, model=model, opt=opt)
        test_error, test_loss = epoch(dataloader=test_dataloader, model=model)

    return train_error, train_loss, test_error, test_loss
    ### END YOUR SOLUTION



if __name__ == "__main__":
    train_mnist(data_dir="../data")
