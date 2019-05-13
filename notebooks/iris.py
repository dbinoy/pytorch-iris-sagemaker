import argparse
import json
import logging
import os
import sagemaker_containers
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

class IrisDataset(torch.utils.data.Dataset):
    def __init__(self, x, y,iscuda=False):
        self.X = np.array(x)
        self.y = np.array(y)
        self.cuda = iscuda
    
    def __getitem__(self, index):
        x_val = self.X[index]
        x_val = torch.from_numpy(x_val)
        y_val = torch.from_numpy(np.array([self.y[index]]))
        if self.cuda:
            x_val = x_val.cuda()
            y_val = y_val.cuda()
        return x_val, y_val

    def __len__(self):
        return len(self.X)

    def close(self):
        self.archive.close()
        
class IrisClassifier(nn.Module):

    def __init__(self):
        super(IrisClassifier, self).__init__()
        self.fc1 = nn.Linear(4, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 3)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, X):
        X = F.relu(self.fc1(X))
        X = self.fc2(X)
        X = self.fc3(X)
        X = self.softmax(X)
        return X


def _get_train_data_loader(batch_size, training_dir, **kwargs):
    logger.info("Get train data loader") 
    X_train = np.load(training_dir + '/iris_train_data.npy')
    y_train = np.load(training_dir + '/iris_train_target.npy')
    return torch.utils.data.DataLoader(dataset = IrisDataset(X_train, y_train), batch_size=batch_size, shuffle=True, **kwargs)


def _get_test_data_loader(batch_size, training_dir, **kwargs):
    logger.info("Get test data loader")
    X_test = np.load(training_dir + '/iris_test_data.npy')
    y_test = np.load(training_dir + '/iris_test_target.npy')

    return torch.utils.data.DataLoader(IrisDataset(X_test, y_test), batch_size=batch_size, shuffle=True, **kwargs)


def train(args):
    kwargs = {}
    train_loader = _get_train_data_loader(args.batch_size, args.data_dir, **kwargs)
    test_loader = _get_test_data_loader(args.test_batch_size, args.data_dir, **kwargs)

    logger.debug("Processes {}/{} ({:.0f}%) of train data".format(
        len(train_loader.sampler), len(train_loader.dataset),
        100. * len(train_loader.sampler) / len(train_loader.dataset)
    ))

    logger.debug("Processes {}/{} ({:.0f}%) of test data".format(
        len(test_loader.sampler), len(test_loader.dataset),
        100. * len(test_loader.sampler) / len(test_loader.dataset)
    ))

    model = IrisClassifier()
    model = torch.nn.DataParallel(model)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader, 1):
            data = Variable(data.float())
            target = Variable(target.reshape(target.shape[0]))            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                logger.info('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.sampler),
                    100. * batch_idx / len(train_loader), loss.item()))
        test(model, criterion, test_loader)
    save_model(model, optimizer, args.model_dir)


def test(model, criterion, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = Variable(data.float())
            target = Variable(target.reshape(target.shape[0]))              
            output = model(data)
            test_loss = test_loss + criterion(output, target)  # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct = correct + pred.eq(target.data.view_as(pred)).sum()

    test_loss = test_loss / len(test_loader.dataset)
    logger.info('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def model_fn(model_dir):
    model = torch.nn.DataParallel(IrisClassifier())
    with open(os.path.join(model_dir, 'model.pth'), 'rb') as f:
        model.load_state_dict(torch.load(f))
    return model


def save_model(model, optimizer, model_dir):
    logger.info("Saving the model.")
    path = os.path.join(model_dir, 'model.pth')
    torch.save(model.state_dict(), os.path.join(model_dir, 'model.pth'))
    torch.save(optimizer.state_dict(), os.path.join(model_dir, 'optimizer.pth'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--batch-size', type=int, default=20, metavar='N',
                        help='input batch size for training (default: 20)')
    parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                        help='input batch size for testing (default: 100)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    # Container environment
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])

    train(parser.parse_args())
