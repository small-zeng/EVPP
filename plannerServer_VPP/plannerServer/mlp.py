import torch
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 空间三维画图
import numpy as np
import time

from torch._C import Size


device = torch.device('cuda:0'if torch.cuda.is_available() else 'cpu')

class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1=torch.nn.Linear(3,64)
        self.layer2_1=torch.nn.Linear(64,64)
        self.layer2_2=torch.nn.Linear(64,64)
        self.layer2_3=torch.nn.Linear(64,64)
        self.layer2_4=torch.nn.Linear(64,64)
        self.layer2_5=torch.nn.Linear(64,64)
        self.layer2_6=torch.nn.Linear(64,64)
        self.layer2_7=torch.nn.Linear(64,64)
        self.layer2_8=torch.nn.Linear(64,64)
        self.layer3=torch.nn.Linear(64,1)

    def forward(self,x):
        x=self.layer1(x)
        x=torch.nn.functional.relu(x)

        x=self.layer2_1(x)
        x=torch.nn.functional.relu(x)

        x=self.layer2_3(x)
        x=torch.nn.functional.relu(x)

        x=self.layer2_4(x)
        x=torch.nn.functional.relu(x)
        
        x=self.layer2_5(x)
        x=torch.nn.functional.relu(x)

        x=self.layer2_6(x)
        x=torch.nn.functional.relu(x)

        x=self.layer2_7(x)
        x=torch.nn.functional.relu(x)

        x=self.layer2_8(x)
        x=torch.nn.functional.relu(x)

        x=self.layer3(x)

        return x



def  PlotCurve(test_num, test_loss, x_train,y_train,z_train,g_train_input,g_train,x,y,z,g_test_input,g):  

    fig = plt.figure(figsize=(12, 9))
    

    plt.subplot(221)
    plt.plot([i + 1 for i in range(50,EPOCH)], mlp_loss[50:], label='MLP_train')
    plt.plot(test_num[5:], test_loss[5:], label='MLP_eval')
    plt.title('loss')
    plt.legend()

    
    ax3d = fig.add_subplot(222,projection="3d")  # 创建三维坐标
    sc = ax3d.scatter(x_train,y_train,z_train, marker='*', c=g_train_input, cmap='rainbow')
    plt.colorbar(sc)
    ax3d.set_xlabel('x', fontsize=10)
    ax3d.set_ylabel('y', fontsize=10)
    ax3d.set_zlabel('z', fontsize=10)
    ax3d.set_title('3D GMM distribution (train input)', fontsize=10)



    ax3d = fig.add_subplot(223,projection="3d")  # 创建三维坐标
    sc = ax3d.scatter(x_train,y_train,z_train, marker='*', c=g_train, cmap='rainbow')
    plt.colorbar(sc)
    ax3d.set_xlabel('x', fontsize=10)
    ax3d.set_ylabel('y', fontsize=10)
    ax3d.set_zlabel('z', fontsize=10)
    ax3d.set_title('3D GMM distribution (train MLP)', fontsize=10)

    ax3d = fig.add_subplot(224,projection="3d")  # 创建三维坐标
    sc = ax3d.scatter(x,y,z, marker='*', c=g, cmap='rainbow')
    plt.colorbar(sc)
    ax3d.set_xlabel('x', fontsize=10)
    ax3d.set_ylabel('y', fontsize=10)
    ax3d.set_zlabel('z', fontsize=10)
    ax3d.set_title('3D GMM distribution (train MLP)', fontsize=10)
    
    fig = plt.figure(figsize=(12, 5))
    
    ax3d = fig.add_subplot(121,projection="3d")  # 创建三维坐标
    g_train_error = np.absolute(g_train_input-g_train)
    sc = ax3d.scatter(x_train,y_train,z_train, marker='*', c=g_train_error, cmap='rainbow')
    plt.colorbar(sc)
    ax3d.set_xlabel('x', fontsize=10)
    ax3d.set_ylabel('y', fontsize=10)
    ax3d.set_zlabel('z', fontsize=10)
    ax3d.set_title('3D GMM distribution (train MLP)', fontsize=10)

    ax3d = fig.add_subplot(122,projection="3d")  # 创建三维坐标
    g_test_error = np.absolute(g_test_input-g)
    sc = ax3d.scatter(x,y,z, marker='*', c=g_test_error, cmap='rainbow')
    plt.colorbar(sc)
    ax3d.set_xlabel('x', fontsize=10)
    ax3d.set_ylabel('y', fontsize=10)
    ax3d.set_zlabel('z', fontsize=10)
    ax3d.set_title('3D GMM distribution (train MLP)', fontsize=10)

    print('平均误差：',np.mean(g_train_error),np.mean(g_test_error))



    plt.tight_layout()
    plt.show()


def gauss_3D(mean, x, C= np.array([[1,0,0],[0,1,0],[0,0,1]])):
    u = x-mean
    norm = np.linalg.norm(C, ord= 2)
    y = 1/(np.power(2*np.pi,1.5)*np.power(norm,0.5))*np.exp(-0.5* u@np.linalg.inv(C)@u.transpose())
    return y


def gmm_pdf(x):
    alpha = [0.5, 0.5]
    z = []
    for i in range(len(x)):
        gauss_1 = gauss_3D(np.array([0, 0,0]),x[i],C = np.array([[4,0,0],[0,4,0],[0,0,4]]))
        gauss_2 = gauss_3D(np.array([2, 2,2]),x[i],C = np.array([[4,0,0],[0,4,0],[0,0,4]]))
        # z.append(gauss_1)
        z.append(alpha[0] * gauss_1 + alpha[1] * gauss_2)
    return np.array(z)


# N = 10000
# x = np.random.uniform(-2,4,N).reshape(N,1)
# y = np.random.uniform(-2,4,N).reshape(N,1)
# z = np.random.uniform(-2,4,N).reshape(N,1)
# XX = np.concatenate((x,y,z),axis=1)
# g = gmm_pdf(XX).reshape(N,1)
# print(min(g),max(g))
# data = np.concatenate((x,y,z,g),axis=1)
# data_show= data[data[:,3]>10e-3][0:1000]
# print(data_show.shape)


# fig = plt.figure(figsize=(12, 8))
# ax3d = plt.gca(projection="3d")  # 创建三维坐标

# sc = ax3d.scatter(data_show[:,0],data_show[:,1],data_show[:,2], marker='*', c=data_show[:,3], cmap='rainbow')
# plt.colorbar(sc)
# ax3d.set_xlabel('x', fontsize=14)
# ax3d.set_ylabel('y', fontsize=14)
# ax3d.set_zlabel('z', fontsize=14)
# ax3d.set_title('3D GMM distribution', fontsize=14)
# plt.show()

#常量都取出来，以便改动
EPOCH=500
BATCH_SIZE = 100
MLP_LR=0.001

if __name__ == '__main__':

    mlp=MLP().to(device)
    t0 = time.time()


    # ###train相关
    N_train = 100
    BATCH_SIZE = N_train

    x_train = np.random.uniform(-2,4,N_train).reshape(N_train,1)
    y_train = np.random.uniform(-2,4,N_train).reshape(N_train,1)
    z_train = np.random.uniform(-2,4,N_train).reshape(N_train,1)
    XX_train = np.concatenate((x_train,y_train,z_train),axis=1)
    g_train_input = gmm_pdf(XX_train).reshape(N_train,1)


    t_start = time.time()
    input_x=torch.tensor(XX_train).to(torch.float32).to(device)
    labels=torch.tensor(g_train_input).to(torch.float32).to(device)
    print(time.time() - t_start,input_x.device )
    N = len(labels)
    BATCH_NUM = int((N/BATCH_SIZE))
    BATCH_index = [[i*BATCH_SIZE,min((i+1)*BATCH_SIZE,N)] for i in range(BATCH_NUM)]
    print(N,BATCH_NUM)

    print(min(g_train_input),max(g_train_input))

    # plt.figure(figsize=(12, 8))
    # sc = plt.scatter(x_train,y_train, marker='*', c=z_train, cmap='rainbow')
    # plt.colorbar(sc)
    # plt.show()

    

    ###test相关
    N_test = 10000
    test_num = []
    for i in range(EPOCH):
        if i % 10 ==0:
            test_num.append(i)
    test_loss = []
    x = np.random.uniform(-2,4,N_test).reshape(N_test,1)
    y = np.random.uniform(-2,4,N_test).reshape(N_test,1)
    z = np.random.uniform(-2,4,N_test).reshape(N_test,1)
    XX = np.concatenate((x,y,z),axis=1)
    g_test_input = gmm_pdf(XX).reshape(N_test,1)
    test_labels = torch.tensor(g_test_input).to(torch.float32).to(device).reshape(len(XX),1)
    XX_test = torch.tensor(XX).to(torch.float32).to(device)
    
    t1 = time.time()
    #训练mlp
    mlp_optimizer=torch.optim.Adam(mlp.parameters(), lr=MLP_LR)
    mlp_loss=[]
    for epoch in range(EPOCH):
        loss_train = 0.0
        print("epoch --->",epoch)
        for i in range(BATCH_NUM):
            mlp_optimizer.zero_grad()
            preds=mlp(input_x[BATCH_index[i][0]:BATCH_index[i][1]])
            loss=torch.nn.functional.mse_loss(preds,labels[BATCH_index[i][0]:BATCH_index[i][1]])
            loss.backward()
            mlp_optimizer.step()
            loss_train += loss.item()
            # print(type(loss),type(loss_train))

        mlp_loss.append(loss_train/BATCH_NUM)

        #测试mlp
        if epoch % 10 == 0:
            mlp_eval = mlp.eval()
            mlp_z = mlp_eval(XX_test)
            # print(mlp_z.shape,test_labels.shape)
            loss = torch.nn.functional.mse_loss(mlp_z,test_labels).cpu().detach().numpy()
            test_loss.append(loss.item() )
            g = mlp_z.cpu().detach().numpy().reshape(N_test,1)
            mlp.train()
    print(test_loss)
    mlp_eval = mlp.eval()
    mlp_z_train = mlp_eval(input_x)
    g_train = mlp_z_train.cpu().detach().numpy().reshape(N_train,1)
    
    # print(test_loss)
    print("用时：",time.time()-t0,time.time()-t1)
    print(max(z_train),min(z_train))
    PlotCurve(test_num, test_loss, x_train,y_train,z_train,g_train_input,g_train,x,y,z,g_test_input,g)

