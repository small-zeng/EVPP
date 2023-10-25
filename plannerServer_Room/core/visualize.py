import torch
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 空间三维画图
import numpy as np
import time
from core.Vector3d import *
import os
import sys
import math
from scipy.stats import norm



class Visualize():


     ## 画训练图调试
    def PlotTrain(self, test_num,mlp_loss,test_loss,XX_train,g_train_input,g_train,XX_test,g_test_input,g_test):
        # print(len(mlp_loss))
        # print(EPOCH)
        fig = plt.figure(figsize=(12, 12))

        plt.subplot(221)
        plt.plot([i + 1 for i in range(50,300)], mlp_loss[50:], label='MLP_train')
        plt.plot(test_num[5:], test_loss[5:], label='MLP_eval')
        plt.title('loss')
        plt.legend()

        ax3d = fig.add_subplot(222,projection="3d")  # 创建三维坐标
        ax3d.view_init(elev=-13., azim=88.)
        sc = ax3d.scatter(XX_train[:,0],XX_train[:,1],XX_train[:,2], marker='*', c=g_train_input, cmap='rainbow')
        plt.colorbar(sc)
        ax3d.invert_xaxis()
        ax3d.set_xlabel('x', fontsize=10)
        ax3d.set_ylabel('y', fontsize=10)
        ax3d.set_zlabel('z', fontsize=10)
        ax3d.set_title('3D NeRF gain distribution (train input)', fontsize=10)

        ax3d = fig.add_subplot(223,projection="3d")  # 创建三维坐标
        ax3d.view_init(elev=-13., azim=88.)
        sc = ax3d.scatter(XX_train[:,0],XX_train[:,1],XX_train[:,2], marker='*', c=g_train, cmap='rainbow')
        plt.colorbar(sc)
        ax3d.invert_xaxis()
        ax3d.set_xlabel('x', fontsize=10)
        ax3d.set_ylabel('y', fontsize=10)
        ax3d.set_zlabel('z', fontsize=10)
        ax3d.set_title('3D NeRF gain distribution (train MLP)', fontsize=10)

        ax3d = fig.add_subplot(224,projection="3d")  # 创建三维坐标
        ax3d.view_init(elev=-13., azim=88.)
        sc = ax3d.scatter(XX_test[:,0],XX_test[:,1],XX_test[:,2], marker='*', c=g_test, cmap='rainbow')
        plt.colorbar(sc)
        ax3d.invert_xaxis()
        ax3d.set_xlabel('x', fontsize=10)
        ax3d.set_ylabel('y', fontsize=10)
        ax3d.set_zlabel('z', fontsize=10)
        ax3d.set_title('3D NeRF gain distribution (test MLP)', fontsize=10)

        fig2 = plt.figure(figsize=(12, 12))
        ax3d = fig2.add_subplot(221,projection="3d")  # 创建三维坐标
        ax3d.view_init(elev=-13., azim=88.)
        g_train_error = np.abs(g_train_input-g_train)
        print("g_train_error = ",np.mean(g_train_error))
        sc = ax3d.scatter(XX_train[:,0],XX_train[:,1],XX_train[:,2], marker='*', c=g_train_error, cmap='rainbow')
        plt.colorbar(sc)
        ax3d.invert_xaxis()
        ax3d.set_xlabel('x', fontsize=10)
        ax3d.set_ylabel('y', fontsize=10)
        ax3d.set_zlabel('z', fontsize=10)
        ax3d.set_title('3D NeRF gain distribution (train error)', fontsize=10)

        ax3d = fig2.add_subplot(222,projection="3d")  # 创建三维坐标
        ax3d.view_init(elev=-13., azim=88.)
        g_test_error = np.abs(g_test_input-g_test)
        print("g_test_error = ",np.mean(g_test_error))
        sc = ax3d.scatter(XX_test[:,0],XX_test[:,1],XX_test[:,2], marker='*', c=g_test_error, cmap='rainbow')
        plt.colorbar(sc)
        ax3d.invert_xaxis()
        ax3d.set_xlabel('x', fontsize=10)
        ax3d.set_ylabel('y', fontsize=10)
        ax3d.set_zlabel('z', fontsize=10)
        ax3d.set_title('3D NeRF gain distribution (test error)', fontsize=10)

        plt.show()

    ## 画NBV调试
    def Plot_NBV(self, testdata,path):
        print(path.shape)

        fig = plt.figure(figsize=(12, 12))
        
        ax3d = fig.add_subplot(221,projection="3d")  # 创建三维坐标
        ax3d.view_init(elev=-13., azim=88.)
        sc = ax3d.scatter(testdata[:,0],testdata[:,1],testdata[:,2], marker='*', c=testdata[:,3], cmap='rainbow', alpha = .5)
        plt.colorbar(sc)
        ax3d.invert_xaxis()
        ax3d.set_xlabel('x', fontsize=10)
        ax3d.set_ylabel('y', fontsize=10)
        ax3d.set_zlabel('z', fontsize=10)
        ax3d.set_title('3D NeRF NBV', fontsize=10)
        for i in range(len(path)-1):
            ax3d.quiver(path[i,0], path[i,1], path[i,2], path[i+1,0]-path[i,0],  path[i+1,1]-path[i,1],  path[i+1,2]-path[i,2],
             color = 'black', alpha = .8, lw = 2)

        
        # 画出信息增益变化
        ax = fig.add_subplot(222)  # 创建三维坐标
        ax.set_xlabel('iter num', fontsize=10)
        ax.set_ylabel('igain', fontsize=10)
        ax.set_title('3D NeRF NBV path gain', fontsize=10)
        iter = [0]
        for i in range(1,len(path)):
            iter.append((i-1)*10+1)
        ax.plot(np.array(iter),path[:,3])

        plt.show()


    def PlotField(self,data_all,f_gain,f_att,f_rep,f_add): 
       
        fig = plt.figure(figsize=(12, 12))
        
        ax3d = fig.add_subplot(221,projection="3d")  # 创建三维坐标
        ax3d.view_init(elev=-13., azim=88.)
        ax3d.invert_xaxis()
        sc = ax3d.scatter(data_all[:,0],data_all[:,1],data_all[:,2], marker='*', c=data_all[:,3], cmap='rainbow', alpha = .5)
        plt.colorbar(sc)
        for i in range(100):
            ax3d.quiver(data_all[i,0],data_all[i,1],data_all[i,2],  f_gain[i,0],f_gain[i,1],f_gain[i,2],color = 'black',
            length = 0.2,  alpha = .8, lw = 2)

        ax3d.set_xlabel('x', fontsize=10)
        ax3d.set_ylabel('y', fontsize=10)
        ax3d.set_zlabel('z', fontsize=10)
        ax3d.set_title('Gain Field', fontsize=10)

        ax3d = fig.add_subplot(222,projection="3d")  # 创建三维坐标
        ax3d.view_init(elev=-13., azim=88.)
        ax3d.invert_xaxis()
        sc = ax3d.scatter(data_all[:,0],data_all[:,1],data_all[:,2], marker='*', c=data_all[:,3], cmap='rainbow', alpha = .5)
        plt.colorbar(sc)
        for i in range(100):
            ax3d.quiver(data_all[i,0],data_all[i,1],data_all[i,2],  f_att[i,0],f_att[i,1],f_att[i,2],color = 'black',
            length = 0.2,  alpha = .8, lw = 2)

        ax3d.set_xlabel('x', fontsize=10)
        ax3d.set_ylabel('y', fontsize=10)
        ax3d.set_zlabel('z', fontsize=10)
        ax3d.set_title('Attractive Field', fontsize=10)

        ax3d = fig.add_subplot(223,projection="3d")  # 创建三维坐标
        ax3d.view_init(elev=-13., azim=88.)
        ax3d.invert_xaxis()
        sc = ax3d.scatter(data_all[:,0],data_all[:,1],data_all[:,2], marker='*', c=data_all[:,3], cmap='rainbow', alpha = .5)
        plt.colorbar(sc)
        for i in range(100):
            ax3d.quiver(data_all[i,0],data_all[i,1],data_all[i,2],  f_rep[i,0],f_rep[i,1],f_rep[i,2],color = 'black',
            length = 0.2,  alpha = .8, lw = 2)

        ax3d.set_xlabel('x', fontsize=10)
        ax3d.set_ylabel('y', fontsize=10)
        ax3d.set_zlabel('z', fontsize=10)
        ax3d.set_title('Repulsion Field', fontsize=10)


        ax3d = fig.add_subplot(224,projection="3d")  # 创建三维坐标
        ax3d.view_init(elev=-13., azim=88.)
        ax3d.invert_xaxis()
        sc = ax3d.scatter(data_all[:,0],data_all[:,1],data_all[:,2], marker='*', c=data_all[:,3], cmap='rainbow', alpha = .5)
        plt.colorbar(sc)
        for i in range(100):
            ax3d.quiver(data_all[i,0],data_all[i,1],data_all[i,2],  f_add[i,0],f_add[i,1],f_add[i,2],color = 'black',
            length = 0.2,  alpha = .8, lw = 2)

        ax3d.set_xlabel('x', fontsize=10)
        ax3d.set_ylabel('y', fontsize=10)
        ax3d.set_zlabel('z', fontsize=10)
        ax3d.set_title('All Field', fontsize=10)


        plt.show()


    ##显示采样点
    def draw_sample(self,data):
        fig = plt.figure(figsize=(12, 12))

        ax3d = fig.add_subplot(221,projection="3d")  # 创建三维坐标
        ax3d.view_init(elev=-13., azim=88.)
        # sc = ax3d.scatter(data[:,0],data[:,1],data[:,2], marker='*', color= "b")
        sc = ax3d.scatter(data[:,0],data[:,1],data[:,2], marker='*', c=data[:,5], cmap='rainbow')
        plt.colorbar(sc)
        ax3d.invert_xaxis()
        ax3d.set_xlabel('x', fontsize=10)
        ax3d.set_ylabel('y', fontsize=10)
        ax3d.set_zlabel('z', fontsize=10)
        ax3d.set_title('3D NeRF uncer distribution (train input)', fontsize=10)

        ax3d = fig.add_subplot(222,projection="3d")  # 创建三维坐标
        ax3d.view_init(elev=-13., azim=88.)
        sc = ax3d.scatter(data[:,0],data[:,1],data[:,2], marker='*', c=data[:,6], cmap='rainbow')
        plt.colorbar(sc)
        ax3d.invert_xaxis()
        ax3d.set_xlabel('x', fontsize=10)
        ax3d.set_ylabel('y', fontsize=10)
        ax3d.set_zlabel('z', fontsize=10)
        ax3d.set_title('3D NeRF avg_distance distribution (train input)', fontsize=10)

        # ax3d = fig.add_subplot(223,projection="3d")  # 创建三维坐标
        # ax3d.view_init(elev=-13., azim=88.)
        # sc = ax3d.scatter(data[:,0],data[:,1],data[:,2], marker='*', c=data[:,7], cmap='rainbow')
        # plt.colorbar(sc)
        # ax3d.invert_xaxis()
        # ax3d.set_xlabel('x', fontsize=10)
        # ax3d.set_ylabel('y', fontsize=10)
        # ax3d.set_zlabel('z', fontsize=10)
        # ax3d.set_title('3D NeRF gain distribution (train input)', fontsize=10)
        
        ## 保存
        plt.savefig(fname="planner_result/result_imgs/sample_train_1000.png",figsize=[12,12])

        plt.show()

            ## 画APF路径调试
    def Plot_APF(self, testdata,path):
        fig = plt.figure(figsize=(12, 12))
        
        ax3d = fig.add_subplot(111,projection="3d")  # 创建三维坐标
        ax3d.view_init(elev=-13., azim=88.)
        sc = ax3d.scatter(testdata[:,0],testdata[:,1],testdata[:,2], marker='*', c=testdata[:,3], cmap='rainbow', alpha = .5)
        plt.colorbar(sc)
        ax3d.invert_xaxis()
        ax3d.set_xlabel('x', fontsize=10)
        ax3d.set_ylabel('y', fontsize=10)
        ax3d.set_zlabel('z', fontsize=10)
        ax3d.set_title('3D NeRF APF)', fontsize=10)
        for i in range(len(path)-1):
            ax3d.quiver(path[i,0], path[i,1], path[i,2], path[i+1,0]-path[i,0],  path[i+1,1]-path[i,1],  path[i+1,2]-path[i,2],
             color = 'black', alpha = .8, lw = 2)

        plt.show()

    ## 画RRT路径调试
    def Plot_RRT(self, testdata,path):
        fig = plt.figure(figsize=(12, 12))
        
        ax3d = fig.add_subplot(111,projection="3d")  # 创建三维坐标
        ax3d.view_init(elev=-13., azim=88.)
        sc = ax3d.scatter(testdata[:,0],testdata[:,1],testdata[:,2], marker='*', c=testdata[:,5], cmap='rainbow', alpha = .5)
        plt.colorbar(sc)
        ax3d.invert_xaxis()
        ax3d.set_xlabel('x', fontsize=10)
        ax3d.set_ylabel('y', fontsize=10)
        ax3d.set_zlabel('z', fontsize=10)
        ax3d.set_title('3D NeRF RRT', fontsize=10)
        for i in range(len(path)-1):
            ax3d.quiver(path[i,0], path[i,1], path[i,2], path[i+1,0]-path[i,0],  path[i+1,1]-path[i,1],  path[i+1,2]-path[i,2],
             color = 'black', alpha = .8, lw = 2)

        plt.show()