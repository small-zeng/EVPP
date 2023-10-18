"""
This is rrt star code for 3D
@author: yue qi
"""
from matplotlib.cm import register_cmap
import numpy as np
from numpy.matlib import repmat
from collections import defaultdict
import time
import matplotlib.pyplot as plt

import os
import sys
from core.visualize import Visualize
from scipy.stats import norm
from core.utils3D import getDist, steer, isCollide,  path
from core.tsdf_gpu import *
from core.interface2 import get_uncertainty


viz = Visualize()

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

mlp=MLP().to(device)

class RRT():
    def __init__(self, x_start,object_center,r0,pdf_mean,pdf_std):
         
        self.x_start = x_start
        self.object_center = object_center
        self.r0 = r0
        self.sample_N = 100
        self.ymin = 0.1
        self.nbv_stepsize = 1.0

        self.Parent = {}
        self.V = []
        self.U = []
        self.Path = []
        # self.E = edgeset()
        self.i = 0
        self.maxiter = 10000
        self.stepsize = 0.05
        # self.Path = []
        self.done = False
        self.x0 = tuple(self.x_start)
        self.xt = tuple(np.array([-1.86,0.30,-2.24]))
        self.Point = []
        self.Point.append([np.array(self.x0),0,0,1]) # 点的位置,到起点的路径长度,路径增益和，点的数目

        
        self.ind = 0
        # self.fig = plt.figure(figsize=(10, 8))
        
        self.ymin = 0.3
        self.ymax = 4.3
        
        # empty space 索引
        self.emptyspace_index_gpu = np.array([])
        self.emptyspace_index = np.array([])
        self.emptyspace_index_sphere = np.array([])
        self.State_vol = np.array([])

        ## tsdf参数入口
        self.tsdf = TSDF(self.ymin,self.ymax,self.object_center)
        
        ## 距离参数
        self.pdf_mean = pdf_mean
        self.pdf_std = pdf_std

        ## 模型参数
        self.mlp = None

        ## 查询点数、时间
        self.query_num = 0
        self.query_time = 0



     ## train 
    def train_only(self,mlp, data,N=100):
        
        EPOCH=300
        BATCH_SIZE = 100
        MLP_LR=0.001
        t0 = time.time()
        

        ## train相关
        N_train = N
        x_train = data[0:N_train,0].reshape(N_train,1)
        y_train = data[0:N_train,1].reshape(N_train,1)
        z_train = data[0:N_train,2].reshape(N_train,1)
        xyz_train = np.concatenate((x_train,y_train,z_train),axis=1)
        g_train_input = data[0:N_train,5].reshape(N_train,1)
        input_x=torch.tensor(xyz_train).to(torch.float32).to(device)
        labels=torch.tensor(g_train_input).to(torch.float32).to(device)
        mlp_loss=[]

        print("用时:",time.time()-t0)
        

        #训练mlp
        mlp_optimizer=torch.optim.Adam(mlp.parameters(), lr=MLP_LR)
        mlp_loss=[]
        for epoch in range(EPOCH):
            # print("epoch --->",epoch)
            mlp_optimizer.zero_grad()
            preds=mlp(input_x)
            loss=torch.nn.functional.mse_loss(preds,labels)
            loss.backward()
            mlp_optimizer.step()
            mlp_loss.append(loss.item())

        return mlp
  
    def get_gain(self,location):
        location = np.array(location).astype(np.double).tolist()
        return self.mlp(torch.tensor(location).to(device)).detach().cpu().numpy()[0]

  # ---------------------- leaf node extending algorithms
    def nearest(self, x, isset=False):
        P = np.array(self.Point)
        V = np.array(self.V)
        if self.i == 0:
            return self.V[0],0
        xr = repmat(x, len(V), 1)
        dists = np.linalg.norm(xr - V, axis=1) + P[:,1]
        heuristic = dists - 0.5*dists*P[:,2]/P[:,3]
        return (tuple(self.V[np.argmin(heuristic)]), np.argmin(heuristic))

     ## 在上一个NBV一定半径圆内采样位置
    def sample_location(self,node, N =1000,R = 4.0):
        n = 0
        locations = []
        print(self.emptyspace_index_sphere.shape)
        
        emptyspace_index_gpu = torch.nonzero(self.State_vol==1)
        emptyspace_index = emptyspace_index_gpu.cpu().numpy()
        start_index = torch.tensor(((node[0:3]-self.tsdf.tsdf_vol._vol_origin)/self.tsdf.tsdf_vol._voxel_size)).to(device)
        start_index = start_index.repeat(emptyspace_index.shape[0],1)
        vox2start_dist = torch.norm(emptyspace_index_gpu.to(device)-start_index,p =2,dim=1)*voxel_res
        emptyspace_index_sphere = emptyspace_index_gpu[vox2start_dist<R].cpu().numpy()
        print(emptyspace_index_sphere.shape)
        
        num = emptyspace_index_sphere.shape[0]

        while n < N:
    
            index = emptyspace_index_sphere[np.random.randint(0,num,size =1)[0]]
            coord = self.tsdf.tsdf_vol._vol_bnds[:,0]+ index* self.tsdf.tsdf_vol._voxel_size 
        
            if self.tsdf.is_valid(coord) :
                locations.append(coord.tolist())
                # print("index = ", index)
                # print(n,[coord,node],self.tsdf.get_state_cpu(coord))
                n = n + 1
        
        locations_array = np.array(locations)

        return locations_array

    
     ## 输入有效位置采样方向
    def sample_direction(self,locations):
        ## 计算中每个点的tsdf uncertainty,每个点有5*3=15个方向 (100,15)
        views = []
        depths = []
        sample_Num = locations.shape[0]
        for i in range(sample_Num):
             x,y,z = locations[i,0],locations[i,1],locations[i,2]
             dx,dy,dz =  self.object_center[0]-x,self.object_center[1]-y,self.object_center[2]-z
             #u_center = np.arctan2(-dy,np.sqrt(dx**2+dz**2))
             u_center = np.arctan2(-dy,0.6)
             v_center = np.arctan2(dx,dz)
             for u in np.linspace(u_center-30/180*np.pi,u_center+30/180*np.pi,5):
                 for v in np.linspace((0)/180*np.pi,(360)/180*np.pi,12):
                     if u < -np.pi/2.0:
                         u = -np.pi/2.0
                     if u > np.pi/2.0:
                         u = np.pi/2.0

                     views.append([x,y,z,u,v])
        views = np.array(views)
        print(views,views.shape)

        ## 通过tsdf筛选3个视角
        uncer,depth, = self.tsdf.get_uncertainty_tsdf(views[:,0:3].tolist(),views[:,3].tolist(),views[:,4].tolist())
        depth = np.array(depth)
        print("uncer = ",uncer)

        data_buff = np.concatenate((views, np.array(uncer).reshape(locations.shape[0]*5*12,1)),axis=1)
        select_views = []
        select_depth = []
        for i in range(sample_Num):
            dir_buff = data_buff[i*5*12:(i+1)*5*12]
            depth_buff = depth[i*5*12:(i+1)*5*12]
            index = (np.argsort(dir_buff[:,5]))[::-1]
            for j in range(3):
                select_views.append(dir_buff[index[j]].tolist())
                select_depth.append(depth_buff[index[j]])
        select_views = np.array(select_views)
        select_depth = np.array(select_depth)

        print(select_depth,select_depth.shape)

        ## 计算这3个方向的nerf uncertainty
        uncer,_,_ = get_uncertainty(select_views[:,0:3].tolist(), select_views[:,3].tolist(), select_views[:,4].tolist())

        depth_clip = np.ones(select_depth.shape)
        depth_clip[select_depth<2.0] = 0
        # depth_clip[select_depth<1.0]= np.exp(-2*np.abs(select_depth[select_depth<1.5] - 3.0))
        # depth_clip[select_depth>4.5] = np.exp(-2*np.abs(select_depth[select_depth>4.5] - 3.0))

        uncer =  np.array(uncer)
        uncer = np.array(uncer).reshape((sample_Num*3,1))
        data_buff = np.concatenate((select_views[:,0:5], np.array(uncer).reshape(sample_Num*3,1)),axis=1)
        data = self.get_maxdirdata(data_buff)
        # print("data = ",data)
        
        ## uncer 归一化
        data[:,5] = data[:,5]/np.max(data[:,5])
        # print("data_norm = ",data,data.shape)

        return data


    def wireup(self, x, y):
        # self.E.add_edge([s, y])  # add edge
        self.Parent[x] = y


    def path_plan(self):
        
        t0 = time.time()
        ## 采样数据
        locations = self.sample_location(self.x_start,N = 100,R = self.r0)  
        # send_SamplePoints(locations.tolist())
        data_select = self.sample_direction(locations)
        print("采样用时：", time.time()-t0)

        ## 提取 cost space 
        sample_space, data_all = copy.deepcopy(data_select),0
        # viz.Plot_SMC(sample_space,np.array([]))

        ## 计算NBV
        data_sort = sample_space[sample_space[:,5].argsort()] #按照第5列对行排序
        nbv = data_sort[len(data_sort)-1,:]
        self.xt = tuple(nbv[0:3])
        print("nbv ：",nbv)
        
        ## 训练NBV
        t1 = time.time()
        mlp_trained = self.train_only(mlp,data_sort[0:100],N=100).to(device)
        self.mlp = mlp_trained
        self.mlp.eval()
        print("训练用时:", time.time()-t1)
       
        ## 开始RRT规划
        # nbv = np.array([-2.29999995,  2.91000009, -3.0000001,   0.73818687,  0.93232092,  1.   ])
        # nbv = np.array([-0.5,         1.7,         0.3,         1.21833705,  4.56958931,  1.        ])
        # self.xt = tuple(nbv[0:3])
        self.V.append(self.x0)
        data = self.get_gain(np.array(self.x0).reshape((1,3)))
        print(data,data.shape)
        self.U.append(data[0])
        tp = time.time()
        while self.ind < self.maxiter:
            if self.ind%300==0:
                print(self.ind)
            xrand = self.sample(sample_space)
            xnearest,near_index = self.nearest(xrand)
            xnew, dist = steer(self, xnearest, xrand)
            collide, dist = isCollide(self, xnearest, xnew, dist=dist)
            if not collide and dist != 0 and xnew[1]>self.ymin and (self.xt not in self.V):
                self.V.append(xnew)  # add point
                tq = time.time()
                data = self.get_gain(np.array(xnew).reshape((1,3)))
                self.query_num += 1
                self.query_time += (time.time()- tq)
                dis = getDist(xnew, xnearest)
                pl = self.Point[near_index][1] + dis  # xj到起点路径长度
                p_gain = self.Point[near_index][2] + data[0] # xj到起点路径总增益
                p_num = self.Point[near_index][3] + 1 # xj到起点路径总点数
                self.Point.append([np.array(xnew),pl,p_gain,p_num])

                self.U.append(data[0])
                self.wireup(xnew, xnearest)
                if self.ind == self.maxiter-1:
                    self.done = False
                    break
                elif getDist(xnew, self.xt) <= 0.2:
                    self.wireup(self.xt, xnew)
                    print("查询点数、时间",self.query_num,self.query_time)
                    print("规划时间",time.time()-tp)
                    self.Path, D = path(self)
                    print('Total distance = ' + str(D))
                    self.done = True
                    break
                self.i += 1
            self.ind += 1
            # if the goal is really reached
        print(self.Path)
        # viz.Plot_RRT(data_select,np.array(self.Path))
        
        print(np.array(self.Path).shape)
        return nbv, np.array(self.Path), data_all,self.done
    
    ## 采样点
    def sample(self, sample_space):
        goal_rate = 0.1
        resampling_rate = 0.5
        p = np.random.uniform(0,1)
        if p < goal_rate:
             x_new = self.xt
        elif p < goal_rate + resampling_rate:
            index = np.random.randint(0,len(sample_space))
            x_new = sample_space[index,0:3]
            x_new = tuple(x_new.tolist()) 

        else:
            x_new = self.sample_location(self.x_start,N = 1,R = self.r0)[0]  
            x_new = tuple(x_new.tolist()) 

        return x_new


      
    ## 路径上选择视角
    def get_path_view(self,nbv_final,plan_path,step=2.0):
        path_view = []
        path_length = 0.0
        d0 =1.0
        n = 0
        for i in range(len(plan_path)-1):
            delta_dis = np.linalg.norm(plan_path[i+1,0:3]-plan_path[i,0:3],ord=2)
            path_length += delta_dis
            if path_length >= (n+1)*step:
                pos = plan_path[i,0:3] + (1-(path_length-(n+1)*step)/delta_dis)*(plan_path[i+1,0:3]-plan_path[i,0:3])
                print("get view:",n,pos,np.linalg.norm(pos-np.array(nbv_final[0:3]),ord=2))
                if np.linalg.norm(pos-np.array(nbv_final[0:3]),ord=2)>d0:
                    pos_final = self.sample_direction(np.array(pos).reshape(1,3))[0].tolist()
                    path_view.append(pos_final)
                n += 1
        path_view.append(nbv_final)
        return path_view

   
    ## 得到最佳方向数据
    def get_maxdirdata(self, data):
        maxdirdata = []
        data_num = int(data.shape[0]/(1*3))
        for i in range(data_num):
            dir_buff = data[i*1*3:(i+1)*1*3]
            maxdirdata.append(list(dir_buff[np.argmax(dir_buff[:,5])]))
        return np.array(maxdirdata)


  

    ## 计算路径长度
    def get_pathlength(self,path):
        path_length = 0.0
        for i in range(len(path)-1):
            path_length += np.linalg.norm(path[i+1,0:3]-path[i,0:3],ord=2)
        return path_length


    def savepath(self,points):
        filename = "core/results/path_rrt_"+str(version)+".txt"
        with open(filename,'a',encoding='utf-8') as f:
            f.writelines("规划路径"+ '\n')
            for i in range(len(points)):
                data= ""
                for j in range(len(points[i])):
                   data += (str(points[i,j])+",") 
                f.writelines(data + '\n')
    
    
    def saveview(self,views):
        filename = "core/results/views_rrt_"+str(version)+".txt"
        with open(filename,'a',encoding='utf-8') as f:

            f.writelines("规划视角"+ '\n')
            for i in range(len(views)):
                data= ""
                for j in range(len(views[i])):
                   data += (str(views[i][j])+",") 
                f.writelines(data + '\n')

    def savetime(self,time):
        filename = "core/results/time_rrt_"+str(version)+".txt"
        with open(filename,'a',encoding='utf-8') as f:
            f.writelines("规划时间"+ '\n')
            f.writelines(str(time) + '\n')


 

if __name__ == '__main__':
    x_start = np.array([0, 1.0,-3])  # Starting node
    object_center = [0.0,1.0,0.0] 
    r0 = 3.0; r_near=3.0;r_far= 4.0

    rrt = RRT(x_start,object_center,r0,r_near,r_far)
    starttime = time.time()
    
    ## 建立tsdf栅格地图，便于采样和规划
    rrt.State_vol = rrt.tsdf.tsdf_test()
    rrt.emptyspace_index_gpu = torch.nonzero(rrt.State_vol==1)
    rrt.emptyspace_index = rrt.emptyspace_index_gpu.cpu().numpy()
    start_index = torch.tensor(((rrt.x_start[0:3]-rrt.tsdf.tsdf_vol._vol_origin)/rrt.tsdf.tsdf_vol._voxel_size)).to(device)
    start_index = start_index.repeat(rrt.emptyspace_index.shape[0],1)
    vox2start_dist = torch.norm(rrt.emptyspace_index_gpu.to(device)-start_index,p =2,dim=1)*voxel_res
    rrt.emptyspace_index_sphere = rrt.emptyspace_index_gpu[vox2start_dist<rrt.r0].cpu().numpy()
    
    ## rrt 规划
    rrt.path_plan()
    print('time used = ' + str(time.time() - starttime))
