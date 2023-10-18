from tracemalloc import start
from numpy.lib.nanfunctions import _remove_nan_1d
import torch
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 空间三维画图
import numpy as np
import time
import core.Vector3d as Vector3d
import os
import sys
import math
from scipy.stats import norm
import copy
from core.visualize import Visualize



from core.interface2 import *
from core.tsdf_gpu import *
from core.localPlanner import Weighted_A_star, MinheapPQ


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


class Viewpath_planner:
    def __init__(self, x_start,object_center=0.,r0=0.,pdf_mean = 3, pdf_std = 1.0):

        self.x_start = x_start
        self.object_center = object_center
        self.r0 = r0
        self.ymin = 0.3
        self.ymax = 4.5
        self.y_center = (self.ymin + self.ymax)/2.0
        self.y_dis = self.ymax - self.ymin
        self.sample_N = 100

        ###定义势力场参数
        self.k_att = 2.0
        self.k_rep = 0.01
        self.rr = 2.0
        self.step_size, self.max_iters, self.goal_threashold = 0.01, 300, 1.0  # 步长0.5寻路1000次用时4.37s, 步长0.1寻路1000次用时21s
        self.step_size_ =2
        self.lam = 0.3


        # empty space 索引
        self.emptyspace_index_gpu = np.array([])
        self.emptyspace_index = np.array([])
        self.emptyspace_index_sphere = np.array([])
        self.State_vol = np.array([])

        ## tsdf参数入口
        self.tsdf = TSDF(self.ymin,self.ymax,self.object_center)
    
        ## 模型参数
        self.mlp = None
        
        ## 距离参数
        self.pdf_mean = pdf_mean
        self.pdf_std = pdf_std
        

    
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
            dist = np.linalg.norm(coord-self.x_start[0:3])
        
            if  dist<self.r0 and self.tsdf.is_valid(coord) :
                locations.append(coord.tolist())
                # print("index = ", index)
                # print(n,[coord,node],self.tsdf.get_state(torch.tensor(coord).to(device)))
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
             u_center = np.arctan2(-dy,np.sqrt(dx**2+dz**2))
             v_center = np.arctan2(dx,dz)
             for u in np.linspace(u_center-15/180*np.pi,u_center+15/180*np.pi,3):
                 for v in np.linspace(v_center-30/180*np.pi,v_center+30/180*np.pi,5):
                     if u < -np.pi/2.0:
                         u = -np.pi/2.0
                     if u > np.pi/2.0:
                         u = np.pi/2.0

                     views.append([x,y,z,u,v])
        views = np.array(views)
        # print(views,views.shape)

        ## 通过tsdf筛选3个视角
        uncer,depth, = self.tsdf.get_uncertainty_tsdf(views[:,0:3].tolist(),views[:,3].tolist(),views[:,4].tolist())
        depth = np.array(depth)
        print("uncer = ",uncer)
        # print(depth,np.min(depth),np.max(depth))
        # if sample_Num == 1:
        #     print("depth = ", depth)

        # print("uncer shape = ",uncer.shape,"depth shape = ",depth.shape)
        data_buff = np.concatenate((views, np.array(uncer).reshape(locations.shape[0]*15,1)),axis=1)
        select_views = []
        select_depth = []
        for i in range(sample_Num):
            dir_buff = data_buff[i*1*15:(i+1)*1*15]
            depth_buff = depth[i*1*15:(i+1)*1*15]
            index = (np.argsort(dir_buff[:,5]))[::-1]
            for j in range(15):
                select_views.append(dir_buff[index[j]].tolist())
                select_depth.append(depth_buff[index[j]])
        select_views = np.array(select_views)
        select_depth = np.array(select_depth)
        # print(select_views,select_views.shape)
        print(select_depth,select_depth.shape)

        ## 计算这3个方向的nerf uncertainty
        uncer,_,_ = get_uncertainty(select_views[:,0:3].tolist(), select_views[:,3].tolist(), select_views[:,4].tolist())
        # depth_clip = np.array(select_depth).copy()
        # np.exp(-np.abs(select_depth[select_depth<2.5] - 3.5))
        depth_clip = np.ones(select_depth.shape)
        depth_clip[select_depth<1.5]= np.exp(-2*np.abs(select_depth[select_depth<1.5] - 3.0))
        depth_clip[select_depth>4.5] = np.exp(-2*np.abs(select_depth[select_depth>4.5] - 3.0))
        # print(depth_clip,np.min(depth_clip),np.max(depth_clip))
        # print(np.array(uncer).shape,select_depth.shape,depth_clip.shape)

        # uncer = np.array(uncer)*norm(self.pdf_mean,self.pdf_std).pdf(select_depth)*depth_clip
        uncer =  np.array(uncer)*depth_clip
        uncer = np.array(uncer).reshape((sample_Num*15,1))
        data_buff = np.concatenate((select_views[:,0:5], np.array(uncer).reshape(sample_Num*15,1)),axis=1)
        data = self.get_maxdirdata(data_buff)
        # print("data = ",data)
        
        ## uncer 归一化
        data[:,5] = data[:,5]/np.max(data[:,5])
        # print("data_norm = ",data,data.shape)

        return data

    
    ## 得到最佳方向数据
    def get_maxdirdata(self, data):
        maxdirdata = []
        data_num = int(data.shape[0]/(1*15))
        for i in range(data_num):
            dir_buff = data[i*1*15:(i+1)*1*15]
            maxdirdata.append(list(dir_buff[np.argmax(dir_buff[:,5])]))
        return np.array(maxdirdata)
    
    ## 一定范围内采样，使用拟合的MLP模型求最大值
    def get_mlp_maxdata(self,node,mlp,NUM=1000,r = 4.0):
        ###采样1000个点,选择初始点
        N_test = NUM
        init_test_data = self.sample_location(node,N=N_test,R = r)
        x = init_test_data[:,0].reshape(N_test,1)
        y = init_test_data[:,1].reshape(N_test,1) 
        z = init_test_data[:,2].reshape(N_test,1)
        xyz = np.concatenate((x,y,z),axis=1)
        test_input=torch.tensor(xyz).to(torch.float32).to(device)
        mlp.eval()
        g = mlp(test_input).detach()
        data_all = torch.cat((test_input,g),axis=1).cpu().numpy()
        print(data_all.shape)
        data_select = data_all
        data_sort = data_select[data_select[:,3].argsort()] #按照第3列对行排序
        N_num = N_test
        N_MAX = 1                                                                                                                                                                                                                                   
        x_init,y_init,z_init = float(np.mean(data_sort[N_num-N_MAX:,0])),float(np.mean(data_sort[N_num-N_MAX:,1])),float(np.mean(data_sort[N_num-N_MAX:,2]))
        init_node = np.array([x_init,y_init,z_init])  
        init_node_gain = mlp(torch.tensor(init_node).to(torch.float32).to(device)).detach().cpu().numpy()[0]
        
        return data_all,init_node,init_node_gain

    ## 梯度下降计算NBV
    def get_optimal_NBV(self,node,mlp,NUM=1000):
        t0 = time.time()
        print("NBV起始点:", node)
        data_all,init_node,init_node_gain = self.get_mlp_maxdata(node,mlp,NUM=1000,r = self.r0)
        print("NBV优化初始点 :",init_node,init_node_gain)
        nbv = init_node

        # path = [[init_node[0],init_node[1],init_node[2], init_node_gain]]
        # x = torch.tensor([float(init_node[0]),float(init_node[1]),float(init_node[2])]).to(torch.float32).to(device)
        # x.requires_grad = True
        # optimizer = torch.optim.Adam([x], lr=0.001)  # 使用Adam编译器，对自变量使用，学习速率：lr=1e-3
        # sample_center = torch.tensor(np.array(self.x_start[0:3])).to(device)
        # R = torch.tensor(self.r0).to(device)
        # y_center = torch.tensor(self.y_center).to(device)
        # y_dis = torch.tensor(self.y_dis).to(device)
        # sigma = 10
        # for step in range(200):  # 迭代200次
        #     sigma = sigma*1.1
            
        #     vec = x[0:3]-sample_center
        #     hs = (torch.maximum(torch.tensor(0).to(device),vec[0]**2 + vec[1]**2 + vec[2]**2-R**2))**2
        #     ho = (torch.maximum(torch.tensor(0).to(device), (x[1]-y_center)**2-(y_dis/2.0)**2))**2
        #     hg = (self.tsdf.get_state(x[0:3])-1.0)**2

        #     pred = -mlp(x[0:3]) +  sigma*hs + sigma*ho + sigma*hg # 预测的值是pred，要使pred达到最小
        #     optimizer.zero_grad()  # 首先将所有梯度清零
        #     pred.backward()  # 对pred反向传播，求得x，y每次的梯度
        #     optimizer.step()  # 这是代表更新一步，在这里面完成了自动利用梯度更新权值。这里是更新x,y

        #     if step % 10 == 0:
        #         gain_pred = mlp(x[0:3]).cpu().detach().numpy().tolist()[0]
        #         point = x.cpu().tolist()
        #         print('step {}: x = {}, f(x) = {}'.format(step, point, pred.item()),gain_pred)
        #         y = point[0:3]
        #         y.append(gain_pred)
        #         path.append(y)
        #     if step % 10 == 0:  
        #         for param_group in optimizer.param_groups:
        #             param_group['lr'] *= 0.96

        # gain_pred = mlp(x[0:3]).cpu().detach().numpy().tolist()[0]
        # point = x.cpu().tolist()
        # y = point[0:3]
        # y.append(gain_pred)
        # path.append(y)
        # nbv = np.array(point[0:3])
        # path = np.array(path)
        # print(path.shape)
        # print("计算NBV用时:", time.time()-t0)
        # print("NBV:", nbv, self.get_pathlength(path))
        # nbv_state = self.tsdf.get_state_cpu(x[0:3].detach().cpu().numpy())
        # print("NBV state: ",nbv_state)
        # data_all_resample = None
        # if not (nbv_state == 1.0):
        #     data_all_resample,nbv,nbv_gain_resample = self.get_mlp_maxdata(init_node,mlp,NUM=1000,r = 1.0)
        #     print("重采样的NBV:", nbv,nbv_gain_resample)
        
        
      

        # # ###人工势场法规划路径
        # # 计算视角最佳方向，仅考虑航向角
        # nbv_final = self.sample_direction(np.array(nbv).reshape(1,3))[0]
        # print("最终NBV:", nbv_final)
        
        # t0 = time.time() 
        # plan_path = self.path_plan(mlp, Vector3d.Vector3d(self.x_start[0:3]),Vector3d.Vector3d(nbv_final[0:3]))
         
        # ####  TODO ########
        # local_path = self.local_path_plan(plan_path)
         
         
         
        # ######## TODO #########   
            
            
        # path_view = self.get_path_view(nbv_final,plan_path,step=2.0)
        # print("选择视角:",path_view)
        # print("人工势场法用时:", time.time()-t0)
        
        # viz.Plot_APF(data_all,plan_path,local_path)

        
       
        return nbv 
    
    ## 计算路径长度
    def get_pathlength(self,path):
        path_length = 0.0
        for i in range(len(path)-1):
            path_length += np.linalg.norm(path[i+1,0:3]-path[i,0:3],ord=2)
        return path_length

 
    ## 人工势力场全局路径规划
    def path_plan(self,mlp,start,goal):
        """
        path plan
        :return:
        """

        self.is_plot = True 
        self.is_path_plan_success = False
        self.delta_t = 0.01
        plan_path = []
        self.iters = 0
        
        pos = [start.deltaX,start.deltaY,start.deltaZ]
        x = torch.tensor(pos, requires_grad=True).to(torch.float32)
        
        print("路径规划")
        x_init,y_init,z_init =float(start.deltaX),float(start.deltaY),float(start.deltaZ)
        plan_path = [[x_init,y_init,z_init]]
        x = torch.tensor([float(x_init),float(y_init),float(z_init)]).to(torch.float32).to(device)
        x.requires_grad = True
        optimizer = torch.optim.SGD([x], lr=0.05)  # 使用Adam编译器，对自变量使用，学习速率：lr=1e-3
        sample_center = torch.tensor(np.array(self.x_start[0:3])).to(device)
        R = torch.tensor(self.r0).to(device)
        y_center = torch.tensor(self.y_center).to(device)
        y_dis = torch.tensor(self.y_dis).to(device)
        sigma = 10
  
        for step in range(300):  # 迭代200次
            sigma = 1.1*sigma
            f_value_att = self.Attractive(x[0:3], torch.tensor([goal.deltaX,goal.deltaY,goal.deltaZ]).to(device))
            f_value = f_value_att 

            vec = x[0:3]-sample_center
            hs = (torch.maximum(torch.tensor(0).to(device),vec[0]**2 + vec[1]**2 + vec[2]**2-R**2))**2
            ho = (torch.maximum(torch.tensor(0).to(device), (x[1]-y_center)**2-(y_dis/2.0)**2))**2
            hg = (self.tsdf.get_state(x[0:3])-1.0)**2

            pred = -mlp(x[0:3]) +  self.lam* f_value + sigma*hs  + sigma*ho + 0*hg # 预测的值是pred，要使pred达到最小
            # pred = f_value
            optimizer.zero_grad()  # 首先将所有梯度清零
            pred.backward()  # 对pred反向传播，求得x，y每次的梯度
            optimizer.step()  # 这是代表更新一步，在这里面完成了自动利用梯度更新权值。这里是更新x,y
            
            ## 已经到达终点，终止规划
            if (Vector3d.Vector3d(x.tolist()[0:3])-goal).length <0.3:
                break
            if step % 10 == 0:
                # gain_pred = mlp(torch.tensor([x.tolist()[0:3]])).cpu().detach().numpy()[0,0]
                print('step {}: x = {}, f(x) = {}'.format(step, x.tolist(), pred.item()))
                y = x.tolist()[0:3]
                # y.append(gain_pred)
                # print(y)
            if step % 5 == 0:
                plan_path.append(y)
            if step % 10 == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.95

        plan_path.append( x.tolist()[0:3])
        plan_path.append([goal.deltaX,goal.deltaY,goal.deltaZ])

        plan_path = np.array(plan_path)
        print(plan_path.shape)
        print("路径为:")
        print(plan_path)
        print([start.deltaX,start.deltaY,start.deltaZ],"---------------->",[goal.deltaX,goal.deltaY,goal.deltaZ])
        print("全局路径长度:",  self.get_pathlength(plan_path))
        
        return np.array(plan_path)
    
     ## A_star + 启发函数局部路径规划
    def local_path_plan(self,plan_path):
        ## 路径检测分段
        path_state = []
        sec_indexs = []
        sec_pts = []
        j = 0
        path_state.append(1.0)
        print(0,"state: ",1.0)
        for i in range(1,plan_path.shape[0]-1,1):
            pt_state,_ = self.tsdf.get_state_cpu(plan_path[i])
            print(i,"state: ",pt_state)
            path_state.append(pt_state)
        path_state.append(1.0)
        print(i+1,"state: ",1.0)
        for i in range(len(path_state)-1):
            if path_state[i]==1 and not(path_state[i+1] == 1):
                sec_indexs.append([])
                sec_pts.append([])
                print("start = ",i)
                sec_indexs[j].append(i)
                sec_pts[j].append(plan_path[i].tolist())
            if not(path_state[i] == 1) and path_state[i+1]==1:
                print("end = ",i+1)
                sec_indexs[j].append(i+1)
                sec_pts[j].append(plan_path[i+1].tolist())
                j = j + 1
        
        print("sec_indexs = ",sec_indexs)
        print("sec_pts = ",sec_pts)
        sec_indexs = np.array(sec_indexs)
        sec_pts = np.array(sec_pts)
        
        ## 路径上的所有点都是有效的
        if sec_pts.shape[0]== 0:
            local_path = copy.deepcopy(plan_path)
            return local_path
        
        
        ## 启发函数测试
        # print("g = ",self.get_gain([0.0,1.0,-3.0]))
        ## 局部规划
        local_path = []
        sec_path_list = []
        for i in range(sec_pts.shape[0]):
            # local_path =  plan_path
            print("start ------> goal",sec_pts[i,0],"-------------------->",sec_pts[i,1])
            ###  TODO ##
            Astar_Planner = Weighted_A_star(sec_pts[i,0], sec_pts[i,1], self)
            sec_path = Astar_Planner.run()
            sec_path.insert(0, sec_pts[i,0].tolist())
            sec_path.append(sec_pts[i,1].tolist())
            sec_path_list.append(sec_path)
            # print(sec_path)
        
        ## 拼接路径
        j = 0
        k = 0
        n = 0
        for i in range(len(path_state)-1):
            if i == sec_indexs[j,0] :
                local_path = local_path + sec_path_list[k]
                n = sec_indexs[j,1]
                if j < sec_indexs.shape[0]-1:
                    j = j + 1
                    k = k + 1
            elif i>= n:
                local_path.append(plan_path[i].tolist()) 
        # print("local_path = ",local_path)
    
    
        return np.array(local_path)
    
    def get_gain(self,location):
        location = np.array(location).astype(np.double).tolist()
        return self.mlp(torch.tensor(location).to(device)).detach().cpu().numpy()[0]

    def get_gain_discrete(self,location):
        location = np.array(location).astype(np.double).tolist()
        data = self.sample_direction(np.array([location]))[0]
        return data[5]
        
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
        

    ## 引力场
    def attractive(self,current_pos,goal):
        """
        引力计算
        :return: 引力
        """
        d0 = 1.0
        dis_goal =np.sqrt((current_pos.deltaX- goal.deltaX)**2+(current_pos.deltaY- goal.deltaY)**2+(current_pos.deltaZ-  goal.deltaZ)**2)
        if dis_goal < d0:
           att = (goal -current_pos) * self.k_att  # 方向由机器人指向目标点
        else:
           att =  (goal -current_pos) * (d0*self.k_att/dis_goal)

        return att

    ## 引力
    def Attractive(self,current_pos,goal):
        """
        引力计算
        :return: 引力
        """
        d0 = 1.0
        dis_goal =torch.norm(current_pos-goal,p=2)
        if dis_goal < d0:
           Att =0.5 * self.k_att * (dis_goal**2)  # 方向由机器人指向目标点
        else:
           Att = (d0*dis_goal -0.5*(d0**2))*self.k_att

        return Att



    
    ## 训练MLP(调试)
    def train(self,mlp, data,N=100):
        
        EPOCH=300
        BATCH_SIZE = 100
        MLP_LR=0.001
        t0 = time.time()
        

        ## train相关
        N_train = N
        x_train = data[0:N_train,0].reshape(N_train,1)
        y_train = data[0:N_train,1].reshape(N_train,1)
        z_train = data[0:N_train,2].reshape(N_train,1)
        XX_train = np.concatenate((x_train,y_train,z_train),axis=1)
        g_train_input = data[0:N_train,5].reshape(N_train,1)
        input_x=torch.tensor(XX_train).to(torch.float32).to(device)
        labels=torch.tensor(g_train_input).to(torch.float32).to(device)
        mlp_loss=[]

        ## test相关
        N_test = 900
        test_num = []
        for i in range(EPOCH):
            if i % 10 ==0:
                test_num.append(i)
        x_test = data[1000-N_test:,0].reshape(N_test,1)
        y_test = data[1000-N_test:,1].reshape(N_test,1)
        z_test = data[1000-N_test:,2].reshape(N_test,1)
        XX_test = np.concatenate((x_test,y_test,z_test),axis=1)
        g_test_input = data[1000-N_test:,5].reshape(N_test,1)
        test_input=torch.tensor(XX_test).to(torch.float32).to(device)
        test_labels=torch.tensor(g_test_input).to(torch.float32).to(device)
        test_loss=[]

        print("用时:",time.time()-t0)
        

        #训练mlp
        mlp_optimizer=torch.optim.Adam(mlp.parameters(), lr=MLP_LR)
        mlp_loss=[]
        for epoch in range(EPOCH):
            print("epoch --->",epoch)
            mlp_optimizer.zero_grad()
            preds=mlp(input_x)
            loss=torch.nn.functional.mse_loss(preds,labels)
            loss.backward()
            mlp_optimizer.step()
            mlp_loss.append(loss.item())

            #测试mlp
            if epoch % 10 == 0:
                mlp_eval = mlp.eval()
                mlp_z = mlp_eval(test_input)
                # print(mlp_z.shape,test_labels.shape)
                loss = torch.nn.functional.mse_loss(mlp_z,test_labels).cpu().detach().numpy()
                test_loss.append(loss.item() )
                g_test = mlp_z.cpu().detach().numpy().reshape(N_test,1)
                mlp.train()
       
        print(test_loss)
        mlp_eval = mlp.eval()
        mlp_z_train = mlp_eval(input_x)
        g_train = mlp_z_train.cpu().detach().numpy().reshape(N_train,1)
        mlp_z = mlp_eval(test_input)
        g_test = mlp_z.cpu().detach().numpy().reshape(N_test,1)
        print("用时:",time.time()-t0)

        viz.PlotTrain(test_num,mlp_loss,test_loss,XX_train,g_train_input,g_train,XX_test,g_test_input,g_test)
        return mlp.cpu()
    
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

       
    
