# coding=utf8
# 你配不上自己的野心，也就辜负了先前的苦难
# 整段注释Control+/


#debug1:class_head = self.class_head.expand(batch_size, 1, -1)
#debug2:weight_sum = weight_sum.transpose(1, 2).contiguous().view(weight_sum.size()[0], -1).
# contiguous() 方法：在调用 view 之前，先调用 .contiguous() 方法，这将返回一个新的连续张量，然后您可以在这个新张量上使用 view 方法


#1.Embedding
#Patch编码
import math
import os

import numpy as np
import torch.nn
import torchvision.datasets
from matplotlib import pyplot as plt
from torch import nn
from torch.optim import optimizer
from torch.utils.data import DataLoader
from torchvision import transforms


class PatchEmbedding(nn.Module):
    def __init__(self,img_size,patch_size,input_channel,embedding_size):
        super(PatchEmbedding, self).__init__()
        img_size=(img_size,img_size)
        patch_size=(patch_size,patch_size)
        num_patch=(img_size[0]//patch_size[0])*(img_size[1]//patch_size[1])
        self.img_size=img_size
        self.patch_size=patch_size
        self.patch_num=num_patch        #将图像大小、小块大小和小块数量保存为类的属性。
        self.patch=nn.Conv2d(in_channels=input_channel,out_channels=self.patch_num,kernel_size=self.patch_size,
                             stride=self.patch_size,padding=0)
        self.patch_embedding=nn.Linear(in_features=self.patch_num,out_features=embedding_size)

    def forward(self,x):
        x=self.patch(x)         # x:(batch_size,channel=patch_num,width=14,height=14)
        #在CNN中卷积或者池化之后需要连接全连接层，所以需要把多维度的tensor展平成一维
        #view可以调整tensor的形状
        #flatten可以展平张量的一个或多个维度
        #reshape类似view
        #permute可以改变tensor的顺序
        seq_len=x.size(2)*x.size(3)
        x=x.view(x.size(0),x.size(1),seq_len) #x:b,num_patch,seq_len(H*W)
        #     size(0):batch_size         size(1):patch_num--->embedding_size
        x=x.permute(0,2,1)      #x:b,seq_len,num_patch
        x=self.patch_embedding(x)
        return x


#[class]Embedding
class ClassHead(nn.Module):
    def __init__(self,embedding_size):
        super(ClassHead, self).__init__()
    #nn.Parameter的作用：
    # 将一个不可训练的tensor转换成可以训练的类型parameter，并将这个parameter绑定到这个module里面。
    # 即在定义网络时这个tensor就是一个可以训练的参数了。使用这个函数的目的也是想让某些变量在学习的过程中不断的修改其值以达到最优化。
        self.class_head=torch.nn.Parameter(data=torch.rand(1,1,embedding_size),requires_grad=True)

    def forward(self,x):
        batch_size=x.size(0)
        class_head=self.class_head.expand(batch_size,1,-1)    # # (batch_size, 1, embedding_size)
        x=torch.cat((class_head,x),dim=1)       #  (batch_size, N+1, embedding_size)
        return x


#PositionEmbedding
class PositionEncoding(nn.Module):      #直接sum在原来的x上
    def __init__(self,embedding_size,num_patch):
        super(PositionEncoding, self).__init__()
        self.position_embedding=nn.Parameter(torch.rand(1,num_patch+1,embedding_size))

    def forward(self,x):
        x=self.position_embedding+x     # (batch_size, N+1, embedding_size)
        return x

class Embedding(nn.Module):
    def __init__(self,img_size,patch_size,input_channel,embedding_size):
        super(Embedding, self).__init__()
        self.patch_embedding=PatchEmbedding(img_size=img_size,patch_size=patch_size,input_channel=input_channel,
                                            embedding_size=embedding_size)
        self.class_head=ClassHead(embedding_size=embedding_size)
        self.positional_encoding=PositionEncoding(embedding_size=embedding_size,num_patch=(img_size//patch_size)**2)

    def forward(self,x):
        x=self.patch_embedding(x)
        x=self.class_head(x)
        x=self.positional_encoding(x)
        return x


#2.Transformer Encoder
class MultiHeadAttention(nn.Module):
    def __init__(self,embedding_size,q_size,k_size,v_size,num_head):
        super(MultiHeadAttention, self).__init__()
        self.embedding_size=embedding_size
        self.q_size=q_size
        self.k_size=k_size
        self.v_size=v_size
        self.num_heads=num_head

        self.w_q=nn.Linear(in_features=embedding_size,out_features=num_head*q_size)
        self.w_k=nn.Linear(in_features=embedding_size,out_features=num_head*k_size)
        self.w_v=nn.Linear(in_features=embedding_size,out_features=num_head*v_size)

        self.softmax=nn.Softmax(dim=-1) #用于计算注意力权重
    def forward(self,q,k,v):
        q_transformed=self.w_q(q)   #queries: (batch_size, sequence_length, total_embedding_size)
                                             # (batch_size, sequence_length, num_heads * size_per_head)
        k_transformed=self.w_k(k)    #key:(batch_size, sequence_length, num_heads * size_per_head)
        v_transformed=self.w_v(v)   #value:(batch_size, sequence_length, num_heads * value_size)
        q_multi_head=q_transformed.view(q_transformed.size()[0],q_transformed.size()[1],self.num_heads,self.q_size).transpose(1,2)
        # (batch_size,num_heads, sequence_length,  size_per_head)
        k_multi_head=k_transformed.view(k_transformed.size()[0],k_transformed.size()[1],self.num_heads,self.k_size).transpose(1,2).transpose(2,3)
        # (batch_size,num_heads, size_per_head,sequence_length) 进行矩阵乘法
        attention_score=torch.matmul(input=q_multi_head,other=k_multi_head)/math.sqrt(self.q_size)  #或者k_size都可以
        attention_weight=self.softmax(attention_score)
        v_multi_head=v_transformed.view(v_transformed.size()[0],v_transformed.size()[1],self.num_heads,self.v_size).transpose(1,2)
        # (batch_size,num_heads, sequence_length,  size_per_head)
        weight_sum=torch.matmul(input=attention_weight,other=v_multi_head)
        weight_sum=weight_sum.transpose(1,2)
        # (batch_size, sequence_length, num_heads * value_size)
        return weight_sum.reshape(weight_sum.size()[0],weight_sum.size()[1],-1)

class EncoderLayer(nn.Module):
    def __init__(self,q_size,k_size,v_size,hidden_dimension,num_head,embedding_size):
        super(EncoderLayer, self).__init__()
        self.multi_head_attention=MultiHeadAttention(embedding_size=embedding_size,q_size=q_size,k_size=k_size,
                                                     v_size=v_size,num_head=num_head)
        self.linear=nn.Linear(in_features=v_size*num_head,out_features=embedding_size)
        self.add_norm1=nn.LayerNorm(normalized_shape=embedding_size)
        self.feedforward=nn.Sequential(
            nn.Linear(in_features=embedding_size,out_features=hidden_dimension),
            nn.Linear(in_features=hidden_dimension,out_features=embedding_size),
            nn.ReLU(),
            nn.Dropout(p=0.5)

        )
        self.add_norm2=nn.LayerNorm(normalized_shape=embedding_size)

    def forward(self,x):
        _x=x
        x=self.multi_head_attention(x,x,x)
        x=self.linear(x)
        x=self.add_norm1(x+_x)
        _x=x
        x=self.feedforward(x)
        x=self.add_norm2(x+_x)
        return x

class Encoder(nn.Module):
    def __init__(self,embedding_size,q_size,k_size,v_size,num_head,num_layer,hidden_dimension):
        super(Encoder, self).__init__()
        self.layers=nn.ModuleList(modules=[EncoderLayer(q_size=q_size,k_size=k_size,v_size=v_size,hidden_dimension=hidden_dimension,
                                                        num_head=num_head,embedding_size=embedding_size)
                                                        for _ in range(num_layer)])
        #nn.ModuleList表示同时管理多个模块，最后需要for _ in range()

    def forward(self,x):
        for layer in self.layers:
            x=layer(x)
        return x


#3.ViT
class VisionTransformer(nn.Module):
    def __init__(self,img_size,patch_size,input_channel,embedding_size,q_size,k_size,v_size,hidden_dimension,
                 num_head,num_layer,class_number):
        super(VisionTransformer, self).__init__()
        self.embedding=Embedding(img_size=img_size,patch_size=patch_size,input_channel=input_channel,embedding_size=embedding_size)
        self.encoder=Encoder(embedding_size=embedding_size,q_size=q_size,k_size=k_size,v_size=v_size,num_head=num_head,
                             num_layer=num_layer,hidden_dimension=hidden_dimension)
        self.class_linear=nn.Linear(in_features=embedding_size,out_features=class_number)
        self.softmax=nn.Softmax(dim=1)

    def forward(self,x):
        x=self.embedding(x)
        x=self.encoder(x)
        x=self.class_linear(x[:,0,:])       # (batch_size, sequence_length, embedding_size)
        # 0 表示选择序列中的第0个元素，这通常是一个特殊的类别补丁（class token）
        output=self.softmax(x)
        return output


#4.Train/Test@CIFAR10
data_transform={
    "Train":transforms.Compose([transforms.Resize(64),
                                transforms.RandomCrop(size=64,padding=4),
                                transforms.RandomHorizontalFlip(p=0.5),
    # RandomHorizontalFlip():
    # 这个变换在训练图像识别模型时非常有用，因为它能够模拟训练数据中可能出现的水平翻转情况，从而增加模型的泛化能力。
    # 例如，在训练一个识别不同方向物体的模型时，通过随机水平翻转图像，模型可以学会识别无论物体朝向如何都具有相同特征的物体
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])]),
    "Val":transforms.Compose([transforms.Resize(64),
                              transforms.ToTensor(),
                              transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])])

}
train_data=torchvision.datasets.CIFAR10(root='./root',train=True,transform=data_transform['Train'],download=True)
val_data=torchvision.datasets.CIFAR10(root='./root',train=False,transform=data_transform['Val'],download=True)
device='cuda' if torch.cuda.is_available() else 'cpu'
print("使用的device是：",device)
vit=VisionTransformer(img_size=64,patch_size=8,input_channel=3,embedding_size=64,q_size=64,k_size=64,v_size=64,
                      hidden_dimension=128,num_head=2,num_layer=3,class_number=10).to(device)

#储存模型参数
model_save_path='./checkpoints/model.pth'
checkpoints_dir=os.path.dirname(model_save_path)
if not os.path.exists(checkpoints_dir):
    os.makedirs(checkpoints_dir,exist_ok=True)

if os.path.exists(model_save_path):
    try:
        vit.load_state_dict(torch.load(model_save_path))
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
else:
    print(f"No model found at {model_save_path}.")

#定义loss和optim和其他hyperparameters
optim=torch.optim.Adam(params=vit.parameters(),lr=1e-5)
criterion=nn.CrossEntropyLoss()
epochs=300
batch_size=256
iter_num=0
all_train_cost=[]
all_train_acc=[]
test_cost=[]
test_acc=[]



train_loader=DataLoader(dataset=train_data,batch_size=batch_size,shuffle=True,num_workers=0)
val_loader=DataLoader(dataset=val_data,batch_size=batch_size,shuffle=False,num_workers=0)

for epoch in range(epochs):
    vit.train()
    train_loss=0
    total=0
    correct=0.0
    for imgs,labels in train_loader:
        optim.zero_grad()
        output=vit(imgs.to(device))
        loss=criterion(output,labels.to(device))
        loss.backward()
        optim.step()

        train_loss=train_loss+loss.item()
        predicted=torch.max(output.data,1)[1]
        total=total+labels.size(0)
        correct = correct + (predicted == labels.to(device)).sum().item()
        if iter_num%100==0:
            torch.save(vit.state_dict(),'./checkpoints/model.pth')
            print("-----训练次数：{}，Loss值为：{}------".format(iter_num,train_loss/len(train_loader)))
        iter_num=iter_num+1

        all_train_cost.append(train_loss/total)
        all_train_acc.append(100.*correct/total)

    #测试
    vit.eval()
    val_loss=0
    correct=0.0
    total=0

    with torch.no_grad():
        for imgs,labels in val_loader:
            if torch.cuda.is_available():
                imgs, labels = imgs.to(device), labels.to(device)
            output=vit(imgs.to(device))
            loss=criterion(output,labels.to(device))
            val_loss=val_loss+loss.item()
            _,predicted=output.max(1)
            total=total+labels.size(0)
            correct=correct+(predicted==labels).sum().item()

        test_cost.append(val_loss/len(val_loader))
        test_acc.append(100.*correct/total)

    print(f'----Epoch:{epoch+1}/{epochs}----'
          f'-----TrainLoss:{train_loss:.4f},TrainAcc:{all_train_acc[-1]:.2f}%----'
          f'----ValLoss:{val_loss:.4f},ValAcc:{test_acc[-1]:.2f}%------')

#5.Predict



#6.Visualization
def draw_loss_acc(loss,acc,mode='train'):

    loss_acc_dir = 'loss_acc'
    if not os.path.exists(loss_acc_dir):
        os.makedirs(loss_acc_dir)

    iters=len(loss)     #计算传入的 loss 列表的长度，这将决定图表的x轴的刻度数量。

    fig, ax1 = plt.subplots()
    ax1.plot(range(1,iters+1),loss,'b-',label='Loss')
    ax1.set_xlabel('Epoch' if mode!='train' else 'Train_Step')
    ax1.set_ylabel('Loss',color='b')
    ax2=ax1.twinx()
    acc_percent=[a/100 for a in acc]
    ax2.plot(range(1,iters+1),acc_percent,'r-',label='ACC')
    ax2.set_ylabel('Accuracy',color='r')
    ax1.set_title(f'{mode}_Loss&Accuracy')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.plot(range(1,iters+1),np.ones(iters),'r--',label='1')
    plt.savefig(os.path.join('loss_acc','loss_acc_{}.png'.format(mode)))
    plt.show()
    plt.clf()

draw_loss_acc(all_train_cost,all_train_acc,mode='train')
draw_loss_acc(test_cost,test_acc,mode='test')

