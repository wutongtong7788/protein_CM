import torch
import torch.nn as nn
import h5py
import torch.utils.data.dataset as Dataset
from torch.utils.data import DataLoader
# import esm
import pickle
import numpy as np
import os
import torch.nn.functional as F
from test import calculateEvaluationStats,get_evaluation_result
from resnet_7_27 import ResNet,block
import GAT_7_24 as GAT
from my_work import tri_model
cuda=torch.cuda.is_available()
name_date='7_27'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
ss=400
i_epoch=20
batch_size=1
num_workers=0
#####################################################data
def get_seq(path):
    # path='examples\example.fasta'
    f = open(path, 'r')
    lines = f.readlines()
    index = 0
    name=[]
    seq=[]
    data=[]
    for i in lines:
        index = index + 1
        if i[0] == '>':
            a = i.split('\n')
            a = a[0]
            j = lines[index]
            b = j.split('\n')
            b = b[0]
            # name.append(a[1:])
            # seq.append(b)
            da = (a[1:], b)
            dat = [da]
            data.append(dat)
    return data
def one_hot(seq):
    RNN_seq = seq
    BASES = 'ARNDCQEGHILKMFPSTWYV'
    bases = np.array([base for base in BASES])
    feat = np.concatenate(
        [[(bases == base.upper()).astype(int)] if str(base).upper() in BASES else np.array([[-1] * len(BASES)]) for base
         in RNN_seq])
    return feat
def get_fasta_2d(seq):
    one_hot_feat = one_hot(seq)
    temp = one_hot_feat[None, :, :]
    temp = np.tile(temp, (temp.shape[1], 1, 1))
    feature = np.concatenate([temp, np.transpose(temp, [1, 0, 2])], 2)
    return feature
class Dataset(Dataset.Dataset):
#####################序列和representation
    def __init__(self,path,path_repre,path_label,path_contact,path_dismap,flag,transform=None):
        self.prot_list = get_seq(path)
        self.path_repre=path_repre
        self.path_label = path_label
        self.path_contact = path_contact
        self.path_dismap= path_dismap
        self.flag=flag
    def __getitem__(self,index):             ####index的这个特征
        da=self.prot_list[index][0]           ####da包括名字和seq
        name=da[0]
        seq=da[1]
        seq_l=len(seq)
        ###########hot
        hot=get_fasta_2d(seq)
        ##########repre
        repre= self.path_repre[name][:, :]
        ###########得到attention数据
        if self.flag == 1:
            with open(os.path.join('./train_3', name + '.pkl'), 'rb') as fo:  # 读取pkl文件数据
                attention = pickle.load(fo, encoding='bytes')  # 1,m,m,660
                attention = attention.squeeze()
            # print(1)


        elif self.flag == 0:

            with open(os.path.join('./test_3', name + '.pkl'), 'rb') as fo:  # 读取pkl文件数据

                attention = pickle.load(fo, encoding='bytes')  # 1,m,m,660

                attention = attention.squeeze()

        elif self.flag == 3:

            with open(os.path.join('./test_28_3', name + '.pkl'), 'rb') as fo:  # 读取pkl文件数据

                attention = pickle.load(fo, encoding='bytes')  # 1,m,m,660

                attention = attention.squeeze()
        ##########得到label，单体的contact
        label = self.path_label[name][:, :]
        label= torch.tensor(label)
        contact = self.path_contact[name][:, :]
        dismap = self.path_dismap[name][:, :]####
        # aaa=dismap[33,9,:]
        # bbb=dismap[45,78,:]
        # ccc=dismap[92,7,:]
        # aaa,bbb,ccc
        # contact = torch.tensor(contact)
        l = hot.shape[0]
        hot = torch.tensor(hot)
        dismap= torch.FloatTensor(dismap)
        if l < ss:
            padding = ss - l
            pad = nn.ZeroPad2d(padding=(0, 0, 0, padding, 0, padding))
            pad_2 = nn.ZeroPad2d(padding=(0, padding, 0, padding))
            # label = pad_2(label)
            dismap= pad(dismap)
            attention = pad(attention)
            hot = pad(hot)
            new_label = np.zeros((l, l))
            ss_l=l
        else:
            label = label[:ss, :ss]
            dismap = dismap[:ss, :ss,:]
            attention = attention[:ss, :ss, :]
            hot = hot[:ss, :ss, :]
            new_label = np.zeros((ss,ss))
            ss_l=ss
        for i in range(ss_l):
            for j in range(ss_l):
                if label[i][j] < 8 and label[i][j] > 0:
                    new_label[i][j] = 1
                else:
                    new_label[i][j] = 0
        G=GAT.default_loader(contact,l)
        # print(G)
        repre = torch.tensor(repre)
        new_label = torch.tensor(new_label)

        return repre,name,seq,new_label,attention,hot,G,dismap

    def __len__(self):
        return len(self.prot_list)
#####################################################3data
def _collate_fn(samples):

    repre,name,seq,label,attention,hot,G,dismap = map(list, zip(*samples))
    return repre,name,seq,label,attention,hot,G,dismap
# def _collate_fn(batch):
#     repre,name,seq,label,attention,hot,G= batch
#     G= dgl.batch(G)
#     # labels = torch.tensor(labels, dtype=torch.long)
#     return repre,name,seq,label,attention,hot,G
############################################################
# class FullyConnectedEmbed(nn.Module):
#     """
#     Protein Projection Module. Takes embedding from language model and outputs low-dimensional interaction aware projection.
#
#     :param nin: Size of language model output
#     :type nin: int
#     :param nout: Dimension of projection
#     :type nout: int
#     :param dropout: Proportion of weights to drop out [default: 0.5]
#     :type dropout: float
#     :param activation: Activation for linear projection model
#     :type activation: torch.nn.Module
#     """
#     def __init__(self, nin=1280, nout=64, dropout=0.5, activation=nn.ReLU()):
#         super(FullyConnectedEmbed, self).__init__()
#         self.nin = nin
#         self.nout = nout
#         self.dropout_p = dropout
#
#         self.transform = nn.Linear(nin, nout)
#         self.drop = nn.Dropout(p=self.dropout_p)
#         self.activation = activation
#
#     def forward(self, x):
#         """
#         :param x: Input language model embedding :math:`(b \\times N \\times d_0)`
#         :type x: torch.Tensor
#         :return: Low dimensional projection of embedding
#         :rtype: torch.Tensor
#         """
#         t = self.transform(x)
#         t = self.activation(t)
#         t = self.drop(t)
#         return t
#####################################################model
# ###################处理repre,把m*64变成m*m*128,两个蛋白各个残基的差异和乘积和conv，处理完了repre
class FullyConnected1(nn.Module):
    """
    Performs part 1 of Contact Prediction Module. Takes embeddings from Projection module and produces broadcast tensor.

    Input embeddings of dimension :math:`d` are combined into a :math:`2d` length MLP input :math:`z_{cat}`, where :math:`z_{cat} = [z_0 \\ominus z_1 | z_0 \\odot z_1]`

    :param embed_dim: Output dimension of `dscript.models.embedding <#module-dscript.models.embedding>`_ model :math:`d` [default: 100]
    :type embed_dim: int
    :param hidden_dim: Hidden dimension :math:`h` [default: 50]
    :type hidden_dim: int
    :param activation: Activation function for broadcast tensor [default: torch.nn.ReLU()]
    :type activation: torch.nn.Module
    """

    def __init__(self, embed_dim=64, hidden_dim=64, activation=nn.ReLU()):
        super(FullyConnected1, self).__init__()

        self.D = embed_dim
        self.H = hidden_dim
        self.conv1 = nn.Conv2d(2 * self.D, self.H, 1)
        # self.batchnorm1 = nn.BatchNorm2d(self.H)
        self.batchnorm1= nn.InstanceNorm2d(self.H)
        self.activation1= activation

    def forward(self, z0):
        """
        :param z0: Projection module embedding :math:`(b \\times N \\times d)`
        :type z0: torch.Tensor
        :param z1: Projection module embedding :math:`(b \\times M \\times d)`
        :type z1: torch.Tensor
        :return: Predicted broadcast tensor :math:`(b \\times N \\times M \\times h)`
        :rtype: torch.Tensor
        """

        # z0 is (b,N,d), z1 is (b,M,d)
        z0 = z0.transpose(1, 2)
        # z1 = z1.transpose(1, 2)
        # z0 is (b,d,N), z1 is (b,d,M)

        z_dif = torch.abs(z0.unsqueeze(3) - z0.unsqueeze(2))
        z_mul = z0.unsqueeze(3) * z0.unsqueeze(2)
        z_cat = torch.cat([z_dif, z_mul], 1)

        b = self.conv1(z_cat)
        b = self.batchnorm1(b)
        b = self.activation1(b)


        return b
# class FullyConnected2(nn.Module):
#     """
#     Performs part 1 of Contact Prediction Module. Takes embeddings from Projection module and produces broadcast tensor.
#
#     Input embeddings of dimension :math:`d` are combined into a :math:`2d` length MLP input :math:`z_{cat}`, where :math:`z_{cat} = [z_0 \\ominus z_1 | z_0 \\odot z_1]`
#
#     :param embed_dim: Output dimension of `dscript.models.embedding <#module-dscript.models.embedding>`_ model :math:`d` [default: 100]
#     :type embed_dim: int
#     :param hidden_dim: Hidden dimension :math:`h` [default: 50]
#     :type hidden_dim: int
#     :param activation: Activation function for broadcast tensor [default: torch.nn.ReLU()]
#     :type activation: torch.nn.Module
#     """
#
#     def __init__(self, embed_dim=64, hidden_dim=64, activation=nn.ReLU()):
#         super(FullyConnected2, self).__init__()
#
#         self.D = embed_dim
#         self.H = hidden_dim
#         self.conv2 = nn.Conv2d(2 * self.D, self.H, 1)
#         # self.batchnorm2 = nn.BatchNorm2d(self.H)
#         self.batchnorm2 = nn.InstanceNorm2d(self.H)
#         self.activation2 = activation
#
#     def forward(self, z0):
#         """
#         :param z0: Projection module embedding :math:`(b \\times N \\times d)`
#         :type z0: torch.Tensor
#         :param z1: Projection module embedding :math:`(b \\times M \\times d)`
#         :type z1: torch.Tensor
#         :return: Predicted broadcast tensor :math:`(b \\times N \\times M \\times h)`
#         :rtype: torch.Tensor
#         """
#
#         # z0 is (b,N,d), z1 is (b,M,d)
#         z0 = z0.transpose(1, 2)
#         # z1 = z1.transpose(1, 2)
#         # z0 is (b,d,N), z1 is (b,d,M)
#
#         z_dif = torch.abs(z0.unsqueeze(3) - z0.unsqueeze(2))
#         z_mul = z0.unsqueeze(3) * z0.unsqueeze(2)
#         z_cat = torch.cat([z_dif, z_mul], 1)
#
#         b = self.conv2(z_cat)
#         b = self.batchnorm2(b)
#         b = self.activation2(b)
#         # print(b.cpu().detach().numpy().mean())
#         # print(b.cpu().detach().numpy().var())
#         # print(self.batchnorm2.running_mean)
#         # print(self.batchnorm2.running_var)
#
#
#         return b
######################得到attention map
class model_model(nn.Module):
    def __init__(self,in_features=784,bias=True):
        super(model_model, self).__init__()
        # self.P=FullyConnectedEmbed()
        self.F1=FullyConnected1()
        # self.F2 = FullyConnected2()
        self.conv1 = nn.Conv2d(in_features,64, kernel_size=1, stride=1, padding=0, bias=False)
        # self.ins = nn.BatchNorm2d(self.filter)
        self.ins = nn.InstanceNorm2d(64)
        self.elu = nn.ELU()
        self.tri_model1=tri_model(64,32)         #输入是128，隐藏层是64
        self.tri_model2 = tri_model(64, 32)
        self.M=ResNet(block, [16], in_features, ss*ss).to(device)
        self.GGAT=GAT.Net().to(device)
        self.last_layer = nn.Conv2d(64,1, kernel_size=1, stride=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()
        # self.regression = nn.Linear(in_features, 1, bias)
        # self.activation = nn.Sigmoid()
    def forward(self, repre,attention_map,hot,G,dismap):
        out_2 = self.GGAT(G, repre)
        # out_2=out_2.unsqueeze(0)
        # repre=self.P(repre)
        # # out_2=self.P(out_2)
        # feature_repre=self.F1(repre)
        out_2_repre = self.F1(out_2)
        # feature_repre=feature_repre.permute(0, 2, 3, 1)             #feature_repre是1，64，208，208，
        out_2_repre = out_2_repre.permute(0, 2, 3, 1)
        l = repre.shape[1]
        if l < ss:
            padding = ss - l
            pad = nn.ZeroPad2d(padding=(0, 0, 0, padding, 0, padding))
            # feature_repre= pad(feature_repre)
            out_2_repre = pad(out_2_repre)
        else:
            # feature_repre = feature_repre[:, :ss, :ss, :]
            out_2_repre= out_2_repre[:, :ss, :ss, :]
        dismap=dismap.unsqueeze(0)
        # dismap = dismap.unsqueeze( 3)
        feature=torch.cat((out_2_repre,attention_map,hot,dismap),axis=3)  #attention_map是1，208，208，660
        feature = feature.permute(0, 3, 1, 2)
        feature = self.conv1(feature)
        feature = self.ins(feature)
        feature = self.elu(feature)
        feature = feature.permute(0, 2, 3, 1)
        feature=self.tri_model1(feature)
        feature = self.tri_model2(feature)
        feature = feature.permute(0, 3, 1, 2)
        out=self.M(feature)              #输入得是764，m,m
        ############处理图出来后的repre
        # out_2_repre = out_2_repre.permute(0, 3, 1, 2)
        # out=torch.cat((out_1,out_2_repre),1)
        out = self.last_layer(out)
        out = self.sigmoid(out)
        # feature=self.regression(feature)
        # feature=feature.squeeze(3)
        # out=self.activation(feature)
        return out
#########################
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, size_average=False):
        super(FocalLoss, self).__init__()
        self.alpha = torch.tensor(alpha)
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, pred, target):
        # 如果模型最后没有 nn.Sigmoid()，那么这里就需要对预测结果计算一次 Sigmoid 操作
        # pred = nn.Sigmoid()(pred)

        # 展开 pred 和 target,此时 pred.size = target.size = (BatchSize,1)
        pred = pred.contiguous().view(-1, 1)
        target = target.view(-1, 1)

        # 此处将预测样本为正负的概率都计算出来，此时 pred.size = (BatchSize,2)
        pred = torch.cat((1 - pred, pred), dim=1)

        # 根据 target 生成 mask，即根据 ground truth 选择所需概率
        # 用大白话讲就是：
        # 当标签为 1 时，我们就将模型预测该样本为正类的概率代入公式中进行计算
        # 当标签为 0 时，我们就将模型预测该样本为负类的概率代入公式中进行计算
        class_mask = torch.zeros(pred.shape[0], pred.shape[1])
        class_mask = class_mask.to(device)
        # 这里的 scatter_ 操作不常用，其函数原型为:
        # scatter_(dim,index,src)->Tensor
        # Writes all values from the tensor src into self at the indices specified in the index tensor.

        class_mask.scatter_(1, target.view(-1, 1).long(), 1.)


        # 利用 mask 将所需概率值挑选出来
        probs = (pred * class_mask).sum(dim=1).view(-1, 1)
        probs = probs.clamp(min=0.0001, max=1.0)

        # 计算概率的 log 值
        log_p = probs.log()

        # 根据论文中所述，对 alpha　进行设置（该参数用于调整正负样本数量不均衡带来的问题）
        alpha = torch.ones(pred.shape[0], pred.shape[1])
        alpha[:, 0] = alpha[:, 0] * (1 - self.alpha)
        alpha[:, 1] = alpha[:, 1] * self.alpha
        alpha = alpha.to(device)
        alpha = (alpha * class_mask).sum(dim=1).view(-1, 1)

    # 根据 Focal Loss 的公式计算 Loss
        batch_loss = -alpha * (torch.pow((1-probs), self.gamma)) * log_p

    # Loss Function的常规操作，mean 与 sum 的区别不大，相当于学习率设置不一样而已
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
            # loss=loss/(ss*ss)

        return loss
######################################################训练
def main():
    # path_train = './examples/train.fasta'
    # path_train_repre = './examples/train_hdf5_file.h5'
    # path_train_label = './examples/train_label_hdf5_file.h5'
    # path_train_contact = './examples/train_contact_hdf5_file.h5'
    # path_train_dismap = './examples/train_dismap_hdf5_file.h5'
    path_test = './examples/test_28.fasta'
    path_test_repre = './examples/test_28_hdf5_file.h5'
    path_test_label = './examples/test_28_label_hdf5_file.h5'
    path_test_contact = './examples/test_28_contact_hdf5_file.h5'
    path_test_dismap = './examples/test_28_dismap_20_hdf5_file.h5'
    # train_h5fi_repre = h5py.File(path_train_repre, "r")
    test_h5fi_repre = h5py.File(path_test_repre, "r")
    # repre = h5fi_repre[name][:, :]
    # train_h5fi_label = h5py.File(path_train_label, "r")
    test_h5fi_label = h5py.File(path_test_label, "r")
    # train_h5fi_contact = h5py.File(path_train_contact, "r")
    test_h5fi_contact = h5py.File(path_test_contact, "r")
    # train_h5fi_dismap = h5py.File(path_train_dismap, "r")
    test_h5fi_dismap = h5py.File(path_test_dismap, "r")

    # label = h5fi_label[name][:, :]
    # new_label = np.zeros((seq_l, seq_l))
    # train = Dataset(path_train,train_h5fi_repre,train_h5fi_label,train_h5fi_contact,train_h5fi_dismap,flag=1)
    test = Dataset(path_test, test_h5fi_repre,test_h5fi_label,test_h5fi_contact,test_h5fi_dismap,flag=3)
    # trainloader = DataLoader(dataset=train, batch_size=batch_size, num_workers=num_workers,collate_fn=_collate_fn)       ##linux下可以设为4
    testloader = DataLoader(dataset=test, batch_size=batch_size, num_workers=num_workers,collate_fn=_collate_fn)
    my_model = model_model().to(device)
    # model_path='checkpoints7_24/19.pt'
    model_path =r'E:\contact\7_27\10.pt'
    my_model.load_state_dict((torch.load(model_path, map_location='cpu')))
    # FL = FocalLoss()
    # learning_rate = 1e-3  # 太大了！！
    # optimizer = torch.optim.Adam(my_model.parameters(), lr=learning_rate)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=2, min_lr=1e-6)
    # print("初始化的学习率：", optimizer.defaults['lr'])
    # epoch = i_epoch
    # 4. train
    val_acc_list = []
    # out_dir = "checkpoints" + name_date + "/"
    # if not os.path.exists(out_dir):
    #     os.makedirs(out_dir)
    # for epoch in range(0, epoch):
    #     my_model.train()
    #     sum_loss = 0.0
    #     val_sum_loss = 0.0
    #     for batch_idx,(repre,name,seq,label,attention,hot,G,dismap) in enumerate(trainloader):          #repre是1，m,1280,label是1，m,m
    #         da=name+seq
    #         da=[da]
    #         length = len(trainloader)
    #         optimizer.zero_grad()
    #         repre, attention,hot,G,dismap=repre[0].to(device), attention[0].to(device),hot[0].to(device),G[0].to(device),dismap[0].to(device)
    #         repre, attention, hot=repre.unsqueeze(0), attention.unsqueeze(0),hot.unsqueeze(0)
    #
    #         # G=G[0]
    #         outputs=my_model(repre,attention,hot,G,dismap)            #repre是1,208,1280，attention是1，208，208，660
    #         outputs=outputs.squeeze(0)
    #         ######################改，7_9
    #         l = len(seq[0])
    #         if  l<400:
    #             outputs = outputs[:, 0:l, 0:l]
    #         del repre,attention,hot,G
    #         torch.cuda.empty_cache()
    #         label=label[0].to(device)
    #         label=label.unsqueeze(0)
    #
    #         bce_loss  = FL(outputs, label).to(device)
    #         # bce_loss = F_loss(outputs.float(),label.float()).to(device)
    #         del label
    #         torch.cuda.empty_cache()
    #         bce_loss.requires_grad_(True).to(device)
    #         bce_loss.backward()
    #         optimizer.step()
    #         print("第%d个epoch的学习率：%f" % (epoch, optimizer.param_groups[0]['lr']))
    #         sum_loss += bce_loss.item()
    #         # predicted = torch.where(outputs > 0.5, 1, 0)
    #         print('[epoch:%d, iter:%d] Loss: %.06f' % (
    #             epoch + 1, (batch_idx + 1 + epoch * length), sum_loss / (batch_idx + 1)))
    #     torch.save(my_model.state_dict(), out_dir + str(epoch) + ".pt")
    relax_0 = []
#############测试
    relax_0 = []
    my_model.eval()
    with torch.no_grad():

        ####################################################################################################验证集，有之前写的基础真好！！！1
        for batch_idx, (repre, name, seq, label, attention, hot, G, dismap) in enumerate(
                testloader):  # repre是1，m,1280,label是1，m,m
            da = name + seq
            da = [da]
            repre, attention, hot, G, dismap = repre[0].to(device), attention[0].to(device), hot[0].to(device), G[0].to(
                device), dismap[0].to(device)
            repre, attention, hot = repre.unsqueeze(0), attention.unsqueeze(0), hot.unsqueeze(0)
            outputs = my_model(repre, attention, hot, G, dismap)  # repre是1,208,1280，attention是1，208，208，660
            outputs = outputs.squeeze(0)
            ######################改，7_9
            l = len(seq[0])
            if l < 400:
                outputs = outputs[:, 0:l, 0:l]
            del repre, attention, hot, G
            label = label[0].to(device)
            torch.cuda.empty_cache()

            # path=name_date+'_test'+'/'
            # path='5_test/'+'/'
            # pre_path='7_9_2_300_pre_new_18'+'/'
            # pre_path=r'E:\contact\re'+'/'
            pre_path = r'E:\contact\7_27\7_27_10_pre' + '/'
            label_path = r'E:\contact\7_27\7_27_10_label' + '/'
            # label_path='7_9_2_300_label_new_18'+'/'
            # label_path=r'E:\contact\la'+'/'
            name = name[0]
            outputs = outputs.squeeze()
            # label = label[0].squeeze()
            outputs = outputs.cpu().detach().numpy()
            # label = label.cpu().detach().numpy()
            np.savetxt(pre_path + name + '.cmap', outputs)
            np.savetxt(label_path + name + '.cmap', label, fmt='%d')

            relax_0.append(calculateEvaluationStats(outputs, label, label.shape[0], name))
            # val_loss_epoch = val_sum_loss / 28
            # if len(relax_0) == 28:
            #     print('len(relax_0)==28!')
        relax_data_0, T10, L30 = get_evaluation_result(relax_0, 0, 28)
        T10 = float(T10)
        L30 = float(L30)
        print('[epoch:%d] val_loss: %.06f| T10: %.03f' % (1, 2, T10))
        print('[epoch:%d] val_loss: %.06f| L30: %.03f' % (1, 2, L30))


if __name__ == "__main__":
    main()

