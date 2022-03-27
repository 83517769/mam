# -*- coding: utf-8 -*-
"""
Created by hwh


"""
import pandas as pd
import seaborn as sns
import cv2
import matplotlib.pyplot as plt
import argparse
import os
import time
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import misc
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
#import mat4py
from pipeline1 import *
#import model
import alexnet
import torchvision.transforms as transforms
#from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
#import inbreast
import numpy as np
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import matplotlib.image as pltt
import skimage.io
from sklearn.metrics import precision_recall_curve


def changetheway(feature_map,data,p,label):
    pdata = torch.squeeze(data)
    R = pdata[0, :, :]
    G = pdata[1, :, :]
    B = pdata[2, :, :]
    print(R.shape)
    print(G.shape)
    print(B.shape)
    Gray = R * 0.299 + G * 0.587 + B * 0.114
    output = F.interpolate(feature_map, size=[227, 227] , mode="bilinear")
    Newdata = output + R
    Newdata = torch.squeeze(Newdata)
    Newdata = Newdata.cpu().detach().numpy()
    #print((Newdata.shape))
    #pfeature = torch.squeeze(feature_map)
    outputs = torch.squeeze(output)
    coutput = outputs.cpu().detach().numpy()
    #print(outputs.shape)
    Gray = torch.squeeze(Gray).cpu().detach().numpy()
    feature_map =torch.squeeze(feature_map)
    feature_map1 =feature_map.cpu().detach().numpy()
    #pdata = torch.squeeze(pdata)
    #pdata = pdata.cpu().detach().numpy()  #喂入模型的数据 转换回cpu
    #print('转换后的三维：')
    #print(pdata.shape)
    #outputs = outputs + b
    #print(outputs.shape)
    #gray_image_orginal = cv2.cvtColor(Pdata, cv2.COLOR_BGR2GRAY) #这句话还是有问题接着改
    #gray_image = outputs + gray_image_orginal
    #print(outputs.shape)
    outputs = torch.squeeze(outputs)
    outputs = outputs.cpu().detach().numpy()
    #print(outputs.shape)
    feature_map = torch.squeeze(feature_map).cpu().detach().numpy()
    # cmap = sns.cubehelix_palette(n_colors=3,start=1, gamma=0.6, reverse=True)
    # fm = sns.heatmap(Newdata, vmin=0, vmax=1.0, cmap= 'viridis', square=True)
    fm1= fm = sns.heatmap(feature_map, vmin=0, vmax=1.0, cmap= 'viridis', square=True)
    plt.show()
    pltt.imsave("G:/mam/feature_map/photo_{%d}_{%d}.jpg" % (p, label), feature_map)  #保存6*6的特征图
    # pltt.imsave("G:/mam/small feature_map/photo test _bilinear2{%d}_{%d}.jpg" % (p, label), Newdata)
    # skimage.io.imsave("G:/mam/orginal/photo test _bilinear2{%d}_{%d}.jpg" % (p, label), Gray) #保存训练样本
    # img_BGR = cv2.imread("G:/mam/orginal/photo_{%d}_{%d}.jpg" % (p, label))
    # img_GRAY = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2GRAY)


def compute_AUCs(gt_np, pred_np):
    AUROCs = []
    AUROCs.append(roc_auc_score(gt_np, pred_np))
    return AUROCs

parser = argparse.ArgumentParser(description='PyTorch CIFAR-X Example')
parser.add_argument('--wd', type=float, default=0.00001, help='weight decay')
parser.add_argument('--wd_mil', type=float, default=0.000005, help='weight decay')
parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
parser.add_argument('--epoch', type=int, default=1, help='epoch to start training from')
parser.add_argument('--n_epochs', type=int, default=2, help='number of epochs of training')
parser.add_argument('--lr', type=float, default=0.00005, help='learning rate (default: 1e-3)')
parser.add_argument('--seed', type=int, default=117, help='random seed (default: 1)')
parser.add_argument('--sample_interval', type=int, default=100, help='interval between sampling of images from generators')
parser.add_argument('--test_interval', type=int, default=2,  help='how many epochs to wait before another test')
parser.add_argument('--logdir', default='log', help='folder to save to the log')
parser.add_argument('--decreasing_lr', default='3', help='decreasing strategy')
args = parser.parse_args()
args.logdir = os.path.join(os.path.dirname(__file__), args.logdir)
misc.logger.init(args.logdir, 'log_5-15_testagain')
print = misc.logger.info
os.makedirs('models5-15', exist_ok=True)
# logger
misc.ensure_dir(args.logdir)
print("=================FLAGS==================")
for k, v in args.__dict__.items():
    print('{}: {}'.format(k, v))
print("========================================")
#valid_img = mat4py.loadmat('../crops_new/valid_img.mat')['valid_img']
#train_img = mat4py.loadmat('../crops_new/train_img1.mat')['train_img1']
#train_data,train_label,valid_data,valid_label = inbreast.loaddataenhance(0,5)
data_all = np.load('data3.npz')
train_data = data_all['arr_0']
train_label = data_all['arr_1']
valid_data = data_all['arr_2']
valid_label = data_all['arr_3']

args.cuda = torch.cuda.is_available()
torch.manual_seed(args.seed)
print(args.cuda)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
transform = transforms.Compose([
                    transforms.RandomResizedCrop(227,scale=(0.9,1),ratio=(0.9,10/9.0)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(45),
                    transforms.ToTensor(),
#                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])
transform1 = transforms.Compose([
#                    transforms.RandomCrop(224),
                transforms.ToTensor(),
#                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
train_loader = DataLoader(ImageDataset(train_data,train_label,transform=transform),batch_size=args.batch_size, shuffle=True, num_workers=0)
valid_loader = DataLoader(ImageDataset_test(valid_data,valid_label,transform=transform1),batch_size=1, shuffle=False, num_workers=0)
if args.epoch != 0:
    # Load pretrained models
    model = alexnet.alexnet(pretrained=False)
#    model.load_state_dict(torch.load('save_models_152/model_%d.pth'%(args.epoch-1)))
    model.load_state_dict(torch.load('models5-15/model_14.pth'))
else:
    model = alexnet.alexnet(pretrained=True, model_root=args.logdir)
criterion = torch.nn.BCELoss()#二分类常用的损失函数（交叉熵）
if args.cuda:
    model.cuda()
    criterion = criterion.cuda()

optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
decreasing_lr = list(map(int, args.decreasing_lr.split(',')))
print('decreasing_lr: ' + str(decreasing_lr))
#begin = time.time()
# ready to go
print(model)
for epoch_ in range(args.epoch, args.n_epochs):
    print('Epoch {}/{}'.format(epoch_, args.n_epochs))
    print('-' * 10)
    model.train()
    train_correct = 0
    all_loss = 0
    p=0
    test_correct = 0
    train_list = []
    if epoch_ in decreasing_lr :
        optimizer.param_groups[0]['lr'] *= 0.1  #衰减学习率，疑问的是每次衰减学习率的的幅度会不会太大了？0.1
    for i, sample in enumerate(train_loader): #遍历训练集迭代器
        data = Variable(sample['img'].type(torch.FloatTensor).cuda())
        labels = Variable(sample['labels'].type(torch.FloatTensor).unsqueeze(-1).cuda())#unsqueeze？
        #print(sample)
        #p=p+1
        #print(labels.shape)
        #print(data.shape)
#        labelsc = Variable(sample['labelc'].type(torch.LongTensor).cuda())
        #print(i)
        optimizer.zero_grad()#优化器的初始化
        outputs,feature_map = model(data)#经过Alexnet处理过后得到特征图和对应的label
        #changetheway(feature_map, data, p, labels)
        #cv2.imwrite("./orginal/photo_{}.jpg".format(p), data)
        #cv2.imwrite("./feature_map/photo_{}.jpg".format(p), feature_map)
        #print(feature_map.shape)
        #print(outputs.shape)
#        loss1=criterion(outputs, labels)
#        loss2=criterion(outputsc, labelsc)
#        loss = loss1+0.1*loss2
        loss=criterion(outputs, labels)+torch.sum(feature_map)*args.wd_mil/args.batch_size
        #loss = criterion(outputs, labels)
        all_loss += loss.item()
        if i % args.sample_interval == 0:
            print('Train Epoch: {} [{}/{}] Loss: {:.6f}'.format(
                    epoch_,i*args.batch_size,len(train_loader.dataset),loss.item()))
        preds=(outputs.data > 0.5).type(torch.FloatTensor).cuda()   #outputs这个输出量里除了data还有其他量吗？
        train_correct += torch.sum(preds == labels.data).type(torch.FloatTensor)
        loss.backward()
        optimizer.step()
        '''
        上面的这段没太理解 args.sample_interval 这个参数没太理解用来干嘛的 
        '''
#        softmax_out = F.log_softmax(outputs, dim=1)
#        pred = softmax_out.data.max(1, keepdim=True)[1]
#        train_correct += pred.eq(labels.data.view_as(pred)).cpu().sum()
    train_loss = all_loss / (len(train_loader))
    train_acc = train_correct.type(torch.FloatTensor) / (len(train_loader.dataset))
    train_list.append(train_loss)

    print('Train Loss: {:.6f} Acc: {:.4f}'.format(train_loss, train_acc))
#        elapse_time = time.time() - t_begin
#        speed_epoch = elapse_time / (epoch + 1)
#        speed_batch = speed_epoch / len(train_loader)
#        eta = speed_epoch * args.epochs - elapse_time
#        print("Elapsed {:.2f}s, {:.2f} s/epoch, {:.2f} s/batch, ets {:.2f}s".format(
#            elapse_time, speed_epoch, speed_batch, eta))
    if (epoch_+1)%1 == 0:
        model.eval()
        p = 0
        TP = 0.0
        TN = 0.0
        FN = 0.0
        FP = 0.0
        all_loss = 0
        gt = torch.FloatTensor()
        gt = gt.cuda()
        pred = torch.FloatTensor()
        pred = pred.cuda()
        test_list = []
    with torch.no_grad():
        for i,sample in enumerate(valid_loader):
            data = Variable(sample['img'].type(torch.FloatTensor).cuda())
            labels = Variable(sample['labels'].type(torch.FloatTensor).unsqueeze(-1).cuda())
            outputs,feature_map = model(data)
    #        loss1 = criterion(outputs, labels)
    #        loss2 = criterion(outputsc, labelsc)
    #        loss = loss1+0.1*loss2
            loss=criterion(outputs, labels)+torch.sum(feature_map)*args.wd_mil/args.batch_size
            #loss = criterion(outputs, labels)
            all_loss += loss.item()
            preds=(outputs.data > 0.5).type(torch.FloatTensor).cuda()
            test_correct += torch.sum(preds == labels.data).type(torch.FloatTensor)
            if i % args.test_interval == 0:
                print('Test Epoch: {} [{}/{}] Loss: {:.6f}'.format(
                    epoch_, i, len(valid_loader.dataset), loss.item()))
            TP += torch.sum((preds == 1) & (labels.data == 1)).type(torch.FloatTensor)
            # TN    predict 和 label 同时为0
            TN += torch.sum((preds == 0) & (labels.data == 0)).type(torch.FloatTensor)
            # FN    predict 0 label 1
            FN += torch.sum((preds == 0) & (labels.data == 1)).type(torch.FloatTensor)
            # FP    predict 1 label 0
            FP += torch.sum((preds == 1) & (labels.data == 0)).type(torch.FloatTensor)
            gt = torch.cat((gt, labels.data), 0)
            pred = torch.cat((pred, outputs.data), 0)
            changetheway(feature_map, data, i, labels)
        gt_npy = gt.cpu().numpy()
        pred_npy = pred.cpu().numpy()
        AUROCs = compute_AUCs(gt_npy, pred_npy)
        AUROC_avg = np.array(AUROCs).mean()
        test_loss = all_loss / (len(valid_loader))
        test_list.append(test_loss)
        test_acc = test_correct.type(torch.FloatTensor) / (len(valid_loader.dataset))
        print('Test Loss: {:.6f} Acc: {:.4f} AUC:{:.4f}\n'.format(test_loss, test_acc,AUROC_avg))
        #torch.save(model.state_dict(), 'models5-15/model_%d.pth' % epoch_)
        # p = TP / (TP + FP)
        # r = TP / (TP + FN)  # lingmingdu
        # te = FP / (FP + TN)
        # F1 = 2 * r * p / (r + p)
        # print('precision:{:.4f} recall(sen):{:.4f} te:{:.4f}\n'.format(p, r, te))
        # F1 = F1.detach().cpu().numpy()
        # print('F1 Score :{}'.format(F1))
        precision, recall, thresholds = precision_recall_curve(gt_npy, pred_npy)
        #plt.plot(recall, precision)
        #plt.show()


