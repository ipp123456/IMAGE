from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import time
import os
from efficientnet.model import EfficientNet
import pickle

# some parameters
use_gpu = torch.cuda.is_available()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_dir = 'data_split'
batch_size = 32
lr = 0.01
momentum = 0.9
num_epochs = 10
input_size = 224
class_num = 20
net_name = 'efficientnet-b4'


def loaddata(data_dir, batch_size, set_name, shuffle):
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in [set_name]}
    # num_workers=0 if CPU else =1
    dataset_loaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                      batch_size=batch_size,
                                                      shuffle=shuffle) for x in [set_name]}
    data_set_sizes = len(image_datasets[set_name])
    return dataset_loaders, data_set_sizes


def train_model(model_ft, criterion, optimizer, lr_scheduler, num_epochs=50):
    metrics_data = {'epoch': [], 'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    train_loss = []
    since = time.time()
    best_model_wts = model_ft.state_dict()
    best_acc = 0.0
    model_ft.train(True)
    for epoch in range(num_epochs):
        dset_loaders, dset_sizes = loaddata(data_dir=data_dir, batch_size=batch_size, set_name='train', shuffle=True)

        optimizer = lr_scheduler(optimizer, epoch)

        running_loss = 0.0
        running_corrects = 0
        count = 0

        for data in dset_loaders['train']:
            inputs, labels = data
            labels = torch.squeeze(labels.type(torch.LongTensor))
            if use_gpu:
                inputs, labels = Variable(inputs.to(device)), Variable(labels.to(device))
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            outputs = model_ft(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs.data, 1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            count += 1

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / dset_sizes
        epoch_acc = running_corrects.double() / dset_sizes

        with torch.no_grad():
            val_loss, val_acc = test_model(model_ft, criterion, 'val')
        print('-' * 10)
        print('Epoch: {},train_loss: {:.4f}, train_acc: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}'.format(epoch,
                                                                      epoch_loss, epoch_acc, val_loss, val_acc))
        metrics_data['epoch'].append(epoch)
        metrics_data['train_loss'].append(epoch_loss)
        metrics_data['train_acc'].append(epoch_acc.item())
        metrics_data['val_loss'].append(val_loss)
        metrics_data['val_acc'].append(val_acc.item())

        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = model_ft.state_dict()
        if epoch_acc > 0.999:
            break

    # save best model
    save_dir = 'model'
    model_ft.load_state_dict(best_model_wts)
    model_out_path = save_dir + "/" + net_name + '.pth'
    torch.save(model_ft, model_out_path)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    return metrics_data, best_model_wts


def test_model(model, criterion, option):
    if option == 'val':
        model.eval()
    running_loss = 0.0
    running_corrects = 0
    cont = 0
    outPre = []
    outLabel = []
    dset_loaders, dset_sizes = loaddata(data_dir=data_dir, batch_size=16, set_name=option, shuffle=False)
    for data in dset_loaders[option]:
        inputs, labels = data
        labels = torch.squeeze(labels.type(torch.LongTensor))
        inputs, labels = Variable(inputs.to(device)), Variable(labels.to(device))
        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)
        loss = criterion(outputs, labels)
        if cont == 0:
            outPre = outputs.data.cpu()
            outLabel = labels.data.cpu()
        else:
            outPre = torch.cat((outPre, outputs.data.cpu()), 0)
            outLabel = torch.cat((outLabel, labels.data.cpu()), 0)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        cont += 1

    return running_loss / dset_sizes, running_corrects.double() / dset_sizes


def exp_lr_scheduler(optimizer, epoch, init_lr=0.01, lr_decay_epoch=10):
    """Decay learning rate by a f#            model_out_path ="./model/W_epoch_{}.pth".format(epoch)
#            torch.save(model_W, model_out_path) actor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.8**(epoch // lr_decay_epoch))
    # print('LR is set to {}'.format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer


# train
pth_map = {
    'efficientnet-b0': 'efficientnet-b0-355c32eb.pth',
    'efficientnet-b1': 'efficientnet-b1-f1951068.pth',
    'efficientnet-b2': 'efficientnet-b2-8bb594d6.pth',
    'efficientnet-b3': 'efficientnet-b3-5fb5a3c3.pth',
    'efficientnet-b4': 'efficientnet-b4-6ed6700e.pth',
    'efficientnet-b5': 'efficientnet-b5-b6417697.pth',
    'efficientnet-b6': 'efficientnet-b6-c76e70fd.pth',
    'efficientnet-b7': 'efficientnet-b7-dcc49843.pth',
}
# 离线加载预训练，需要事先下载好
model_ft = EfficientNet.from_name(net_name)
net_weight = 'eff_weights/' + pth_map[net_name]
state_dict = torch.load(net_weight)
model_ft.load_state_dict(state_dict)

# 修改全连接层
num_ftrs = model_ft._fc.in_features
model_ft._fc = nn.Linear(num_ftrs, class_num)

criterion = nn.CrossEntropyLoss()
if use_gpu:
    model_ft = model_ft.to(device)
    criterion = criterion.to(device)
# 定义优化器
optimizer = optim.SGD((model_ft.parameters()), lr=lr,
                      momentum=momentum, weight_decay=0.0004)
#训练模型
metrics_data, best_model_wts = train_model(model_ft, criterion, optimizer, exp_lr_scheduler, num_epochs=num_epochs)

# test
model_ft.load_state_dict(best_model_wts)
criterion = nn.CrossEntropyLoss().to(device)
test_loss, test_acc = test_model(model_ft, criterion, 'test')

print('-' * 10)
print('Test Accuracy:', test_acc.item())

with open('record.pkl', 'wb') as wt:
    pickle.dump(metrics_data, wt)
