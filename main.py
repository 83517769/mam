from PyQt5 import QtGui
from PyQt5.QtWidgets import *
from Show import Ui_MainWindow
import sys
from PyQt5 import QtCore
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
#import mat4py
import alexnet
import torchvision.transforms as transforms
#from torchvision.utils import save_image
import matplotlib.pyplot as plt
import matplotlib.image as pltt
#import inbreast
import numpy as np
from sklearn.metrics import roc_auc_score
from PIL import Image
import os
import seaborn as sns
import matplotlib.pyplot as plt
imglist = []
i = 0
"""
Created by ATRer.hwh
"""
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

transform1 = transforms.Compose([
#                    transforms.RandomCrop(224),
                transforms.ToTensor(),
#                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])


path1=os.path.abspath('.')   #表示当前所处的文件夹的绝对路径
#print(path1)
path2=os.path.abspath('..')  #表示当前所处的文件夹上一级文件夹的绝对路径
#print(path2)
os.makedirs('modelsresult', exist_ok=True)
modelpath ="models4-13-3\model_10.pth"
model_path = os.path.join(path1, modelpath)
print(model_path)
def wait():
    data_all = np.load('2.npz')
    print(data_all['arr_0'])
    test_data = data_all['arr_0']
    print(test_data)
    model = alexnet.alexnet(pretrained=False)
    model.load_state_dict(torch.load(model_path))
    model.cuda()
    print(model)
    Img = Image.fromarray(np.uint8(test_data)).convert("RGB")
    print(Img)
    Img = transform1(Img)
    data = Variable(Img.type(torch.FloatTensor).cuda()).unsqueeze(0) # 使用pytorch中的unsqueeze(0)函数,在第0维的位置增加一个维度.
    print(data.shape)
    print('模型加载成功')
    outputs, feature_map = model(data)
    preds = (outputs.data > 0.5).type(torch.FloatTensor).cuda()
    preds = preds.cpu().detach().numpy()
    #print(preds)
    #changetheway(feature_map, data)
    return preds, feature_map, data

imgsave = "modelsresult/model data result.jpg"
img_save = os.path.join(path1, imgsave)
print(img_save)
def changetheway(feature_map,img):
    pdata = torch.squeeze(img)
    R = pdata[0, :, :]
    G = pdata[1, :, :]
    B = pdata[2, :, :]
    print(R.shape)
    print(G.shape)
    print(B.shape)
    Gray = R * 0.299 + G * 0.587 + B * 0.114
    output = F.interpolate(feature_map, size=[227, 227] , mode="bilinear",align_corners= True)
    Newdata = output + Gray
    Newdata = torch.squeeze(Newdata)
    Newdata = Newdata.cpu().detach().numpy()
    print((Newdata.shape))
    pltt.imsave(img_save, Newdata)



class MyMain(QMainWindow,Ui_MainWindow):
    def __init__(self):
        super(MyMain, self).__init__()
        self.setupUi(self)
        self.pushButton.clicked.connect(self.on_op1_img_clicked)
        self.pushButton_2.clicked.connect(self.click_success)
        self.pushButton_3.clicked.connect(self.close)
        self.label_6 = QLabel(self)
        self.label_6.setGeometry(QtCore.QRect(850, 480, 350, 277))
        self.label_6.setStyleSheet("QLabel{background:white;}"
                                 "QLabel{color:rgb(300,300,300,120);font-size:300px;font-weight:bold;font-family:宋体;}"
                                 )
        self.label_6.setScaledContents(True)
        self.label_7 = QLabel(self)
        self.label_7.setGeometry(QtCore.QRect(170, 480, 441, 401))
        self.label_7.setStyleSheet("QLabel{background:white;}"
                                   "QLabel{color:rgb(300,300,300,120);font-size:20px;font-weight:bold;font-family:宋体;}"
                                   )
        self.label_7.setScaledContents(True)
        self.label_7.setText("等候样本图")
        self.label_7.setScaledContents(True)


    def on_op1_img_clicked(self):
        imgName, imgType = QFileDialog.getOpenFileName(None, "打开图片", "", "*.jpg;;*.png;;All Files(*)")
        print(type(imgName))
        jpg = QtGui.QPixmap(imgName)
        self.label_6.setPixmap(jpg)
        img = Image.open(imgName)
        img = np.array(img)
        np.savez('2.npz', img)

    def click_success(self):
         preds, feature_map,data = wait()
         print('测试成功正在上传数据')
         pred = int(preds)
         if pred!=1:
             str = '模型预测结果为：阴性'
             self.textEdit.setText(str)
         else:
            str = '模型预测结果为：阳性'
            self.textEdit.setText(str)
         print('结果显示完成')
         changetheway(feature_map,data)
         img_path = img_save
         image = QtGui.QPixmap(img_path)
         self.label_7.setPixmap(image)
         # imgName, imgType = QFileDialog.getOpenFileName(None, "打开图片", "G:\\ShowQt", "*.jpg;;*.png;;All Files(*)")
         # jpg = QtGui.QPixmap(imgName).scaled(self.label.width(), self.label.height())
         # self.label_7.setPixmap(jpg)




if __name__ =="__main__":
    app = QApplication(sys.argv)
    main = MyMain()
    main.show()#显示窗口
    sys.exit(app.exec_())

