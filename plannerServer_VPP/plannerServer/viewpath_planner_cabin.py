import torch
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 空间三维画图
import numpy as np
import time
import Vector3d
import os
import sys
import math
from scipy.stats import norm
from visualize import Visualize

sys.path.append("/home/zengjing/zj/Projects/nerfplanning/") 
print(sys.path)
from  testscript import get_uncertainty,get_ray_depth,get_surface_points


device = torch.device('cuda:7'if torch.cuda.is_available() else 'cpu')

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
    def __init__(self, x_start):

        self.x_start = x_start

        ###定义势力场参数
        self.k_att = 1.0
        self.k_rep = 10
        self.rr = 1.0
        self.step_size, self.max_iters, self.goal_threashold = 0.01, 300, 1.0  # 步长0.5寻路1000次用时4.37s, 步长0.1寻路1000次用时21s
        self.step_size_ =2
        self.lam = 0.5

    ## 在上一个NBV一定半径圆内采样并筛选，计算权重,得到分布
    def get_sample(self,node, N =1000, R = 2):
        n = 0 
        data_select = []
        data_select_array = np.array(data_select)
        center = list(node[0:3])

        for i in range(N):
            theta = np.random.uniform(0,np.pi)
            pha = np.random.uniform(0,2*np.pi)
            dir_vec = [np.sin(theta)*np.cos(pha),np.sin(theta)*np.sin(pha),np.cos(theta)]
            dis = get_ray_depth([center],[dir_vec]).cpu().detach().numpy()
            # print(dis)

            r = min(dis[0],R)* np.power(np.random.uniform(0,1),1/3.0)
            x = r * dir_vec[0] + center[0]
            y = r * dir_vec[1] + center[1]
            z = r * dir_vec[2] + center[2]
            angle = np.arctan2(-x,-z)
            
            if y>0:
                uncer,avg_distance,ratio = get_uncertainty([x,y,-z],0,angle)
                uncer = uncer.cpu().detach().numpy()
                avg_distance = avg_distance.cpu().detach().numpy()
                gain = uncer
                # if uncer>0.004:
                #     print([x,y,z], angle,uncer,avg_distance,ratio)
                data_select.append([x,y,z,angle,uncer])

        data_select_array = np.array(data_select)

        return data_select_array

    ## 在上一个NBV一定半径圆内采样并筛选，方向为看向目标中心，计算权重,得到分布
    def get_sample_object_center(self,node, N =1000, R = 2):
        object_center = [0,1,0]
        n = 0 
        i = 0
        data_select = []
        data_select_array = np.array(data_select)
        center = list(node[0:3])

        while n < N:
            theta = np.random.uniform(0,np.pi)
            pha = np.random.uniform(0,2*np.pi)
            dir_vec = [np.sin(theta)*np.cos(pha),np.sin(theta)*np.sin(pha),np.cos(theta)]
            dis = get_ray_depth([center],[dir_vec]).cpu().detach().numpy()
            # print(dis)
            
            ##看向空白区域，给很大距离
            if dis[0] == 0:
                dis[0] = 100
            r = min(dis[0],R)* np.power(np.random.uniform(0,1),1/3.0)
            x = r * dir_vec[0] + center[0]
            y = r * dir_vec[1] + center[1]
            z = r * dir_vec[2] + center[2]
            dx,dy,dz =  object_center[0]-x,object_center[1]-y,object_center[2]-z
            # if np.abs(dz) < 1:
            #     continue
            if y > 0:
            
                u = np.arctan2(-dy,np.sqrt(dx**2+dz**2))
                v = np.arctan2(dx,dz)
                
                uncer,avg_distance,ratio = get_uncertainty([x,y,z],u,v)
                uncer = uncer.cpu().detach().numpy()
                avg_distance = avg_distance.cpu().detach().numpy()
                gain = uncer
                # if uncer>0.004:
                #     print([x,y,z], u,v,uncer,avg_distance,ratio)
                ##数据有效
                # if not math.isnan(avg_distance):
                data_select.append([x,y,z,u,v,uncer,avg_distance])
                n = n + 1
                print(n,[x,y,z], u,v,avg_distance)
                i = i + 1
                # print(i)
        data_select_array = np.array(data_select)

        return data_select_array
    
    ## 以物体中心一定半径上球面上采样
    def get_sample_sphere(self,node, N =1000, R = 2):
        object_center = [0,1,0]
        n = 0 
        i = 0
        data_select = []
        data_select_array = np.array(data_select)
        center = list(object_center)

        while n < N:
            theta = np.random.uniform(0,np.pi)
            pha = np.random.uniform(0,2*np.pi)
            dir_vec = [np.sin(theta)*np.cos(pha),np.sin(theta)*np.sin(pha),np.cos(theta)]

            r = R
            x = r * dir_vec[0] + center[0]
            y = r * dir_vec[1] + center[1]
            z = r * dir_vec[2] + center[2]
            dx,dy,dz =  object_center[0]-x,object_center[1]-y,object_center[2]-z
            # if np.abs(dz) < 1:
            #     continue
            if y > 0:
            
                u = np.arctan2(-dy,np.sqrt(dx**2+dz**2))
                v = np.arctan2(dx,dz)
                
                uncer,avg_distance,ratio = get_uncertainty([x,y,z],u,v)
                uncer = uncer.cpu().detach().numpy()
                avg_distance = avg_distance.cpu().detach().numpy()
                gain = uncer
                # if uncer>0.004:
                #     print([x,y,z], u,v,uncer,avg_distance,ratio)
                ##数据有效
                # if not math.isnan(avg_distance):
                data_select.append([x,y,z,u,v,uncer,avg_distance])
                n = n + 1
                print(n,[x,y,z], u,v,avg_distance)
                i = i + 1
                # print(i)
        data_select_array = np.array(data_select)

        return data_select_array

    ## 在上一个NBV一定半径圆内采样并筛选，方向为看向目标中心
    def get_sample_sphere_center(self,node, N =100, R = 1):
        object_center = [0,1,0]
        dir_range = 15/180.0*np.pi
        dir_resolution = 5/180.0*np.pi
        dir_num = int(2*dir_range/dir_resolution) + 1
        
        n = 0
        i = 0
        center = list(node[0:3]) 
        data_select = []
        data_select_array = np.array(data_select)

        while n < N:
            theta = np.random.uniform(0,np.pi)
            pha = np.random.uniform(0,2*np.pi)
            dir_vec = [np.sin(theta)*np.cos(pha),np.sin(theta)*np.sin(pha),np.cos(theta)]

            dis = get_ray_depth([center],[dir_vec]).cpu().detach().numpy()
            if dis[0] == 0.0:
                dis[0]=10.0
            # print(dis)

            r = min(dis[0],R)* np.power(np.random.uniform(0,1),1/3.0)

            # r = np.power(np.random.uniform(0,1),1.0/3)
            x = r * dir_vec[1] + node[0]
            y = r * dir_vec[2] + node[1]
            z = r * dir_vec[0] + node[2]
            dx,dy,dz =  object_center[0]-x,object_center[1]-y,object_center[2]-z
 
            dis_object2sample =  np.sqrt(dx**2+dy**2+dz**2)     
            # print(x,y,z,dis_node2sample)                                                                                              
            if y > 0.3 and 3.0 <= dis_object2sample <= 4.0:
            
                u = np.arctan2(-dy,np.sqrt(dx**2+dz**2))
                v_center = np.arctan2(dx,dz)
                dir_buff = []
                for v in np.linspace(v_center-dir_range,  v_center+dir_range, dir_num):
                
                    uncer,avg_distance,ratio = get_uncertainty([x,y,z],u,v)
                    uncer = uncer.cpu().detach().numpy()
                    avg_distance = avg_distance.cpu().detach().numpy()
                    dir_buff.append([x,y,z,u,v,uncer,avg_distance])
                dir_buff = np.array(dir_buff)
                max_dir_data = list(dir_buff[np.argmax(dir_buff[:,5])])

                data_select.append(max_dir_data)
                print(n,max_dir_data,[dis[0],r])
                n = n + 1

        data_select_array = np.array(data_select)

        return data_select_array

        ## 在上一个NBV一定半径圆内采样并筛选，方向为看向目标中心，计算权重,得到分布
    def get_testdata_sphere_center(self,node, N =1000, R = 1):
        object_center = [0,1,0]
        n = 0 
        i = 0
        data_select = []
        data_select_array = np.array(data_select)

        while n < N:
            theta = np.random.uniform(0,np.pi)
            pha = np.random.uniform(0,2*np.pi)
            dir_vec = [np.sin(theta)*np.cos(pha),np.sin(theta)*np.sin(pha),np.cos(theta)]

            r = R * np.power(np.random.uniform(0,1),1/3.0)
            x = r * dir_vec[1] + node[0]
            y = r * dir_vec[2] + node[1]
            z = r * dir_vec[0] + node[2]
            dx,dy,dz =  object_center[0]-x,object_center[1]-y,object_center[2]-z
 
            dis_object2sample =  np.sqrt(dx**2+dy**2+dz**2)                                                                                                   
            if y > 0.3 and 3.0 <= dis_object2sample <= 4.0:    
                data_select.append([x,y,z])
                # print([x,y,z,dis_object2sample,r,node])
                n = n + 1

        data_select_array = np.array(data_select)

        return data_select_array

    ## 在上一个NBV一定半径圆内采样并筛选，方向为看向目标中心，计算权重,得到分布
    def get_sample_line(self,node, N =100, R = 2):
        object_center = [0,1,0]
        n = 0 
        i = 0
        data_select = []
        data_select_array = np.array(data_select)
        center = list(object_center)

        while n < N:
            theta = np.random.uniform(0,np.pi)
            pha = np.random.uniform(0,2*np.pi)
            dir_vec = [np.sin(theta)*np.cos(pha),np.sin(theta)*np.sin(pha),np.cos(theta)]
            
            for r in np.linspace(0,R,20):
                x = r * dir_vec[1] + center[0]
                y = r * dir_vec[2] + center[1]
                z = -r * dir_vec[0] + center[2]
                dx,dy,dz =  object_center[0]-x,object_center[1]-y,object_center[2]-z
                # if np.abs(dz) < 1:
                #     continue
                if y > 0:
                
                    u = np.arctan2(-dy,np.sqrt(dx**2+dz**2))
                    v = np.arctan2(dx,dz)
                    
                    uncer,avg_distance,ratio = get_uncertainty([x,y,z],u,v)
                    uncer = uncer.cpu().detach().numpy()
                    avg_distance = avg_distance.cpu().detach().numpy()
                    gain = uncer
                    # if uncer>0.004:
                    #     print([x,y,z], u,v,uncer,avg_distance,ratio)
                    ##数据有效
                    # if not math.isnan(avg_distance):
                    data_select.append([x,y,z,u,v,uncer,avg_distance])
                    
                    print(n,[x,y,z], u,v,avg_distance)
            n = n + 1
                    
        data_select_array = np.array(data_select)
        print(data_select_array.shape)

        return data_select_array


    ## 梯度下降计算NBV
    def get_optimal_NBV(self,node,mlp,NUM=100):
        
        NUM=1000
        t0 = time.time()
        ###采样1000个点,选择初始点
        N_test = NUM
        init_test_data = self.get_testdata_sphere_center(node,N=N_test,R=1)
        # init_test_data = np.loadtxt("planner_result/result_data/resultdata_1000_all.txt")
        x = init_test_data[:,0].reshape(N_test,1)
        y = init_test_data[:,1].reshape(N_test,1)
        z = init_test_data[:,2].reshape(N_test,1)
        XX = np.concatenate((x,y,z),axis=1)
        test_input=torch.tensor(XX).to(torch.float32)
        mlp.eval()
        mlp_g = mlp(test_input)
        g = mlp_g.cpu().detach().numpy().reshape(N_test,1)
        data_all = np.concatenate((x,y,z,g),axis=1)
        print(data_all.shape)
        data_select = data_all[0:100]
        data_sort = data_select[data_select[:,3].argsort()] #按照第3列对行排序
        N_num = 100
        # print(N_num)
        # N_select = 1000
        # data_select = data_sort[N_num-N_select:N_num,:]
        # print(data_select.shape)
        N_MAX = 5                                                                                                                                                                                                                                   
        x_init,y_init,z_init = float(np.mean(data_sort[N_num-N_MAX:,0])),float(np.mean(data_sort[N_num-N_MAX:,1])),float(np.mean(data_sort[N_num-N_MAX:,2]))
        print("初始点 ：",x_init,y_init,z_init)
        
        x_init,y_init,z_init = 0.0, 1.0, -3.0

        lamda_hs = 10.0;lamda_ho = 10.0;lamda_ho = 10.0;u = 10
        path = [[x_init,y_init,z_init, mlp(torch.tensor([x_init,y_init,z_init]).reshape(1,3)).cpu().detach().numpy()[0,0]]]
        x = torch.tensor([float(x_init),float(y_init),float(z_init),float(lamda_hs),float(lamda_ho),float(u)], requires_grad=True).to(torch.float32)
        optimizer = torch.optim.Adam([x], lr=0.01)  # 使用Adam编译器，对自变量使用，学习速率：lr=1e-3
        object_center = torch.tensor(np.array([0,1,0]))
        sample_center = torch.tensor(np.array([0,1,-3]))
        for step in range(200):  # 迭代200次
    
            hs = torch.abs(torch.norm(x[0:3]-sample_center)-1.0)
            ho = torch.abs(torch.abs(torch.norm(x[0:3]-object_center)-3.5)-0.5)
            g = torch.abs(-x[1]+0.3)
            pred = -mlp(x[0:3]) +  x[3]*hs + x[4]*ho + x[5]*g # 预测的值是pred，要使pred达到最小

            optimizer.zero_grad()  # 首先将所有梯度清零
            pred.backward()  # 对pred反向传播，求得x，y每次的梯度
            optimizer.step()  # 这是代表更新一步，在这里面完成了自动利用梯度更新权值。这里是更新x,y

            if step % 10 == 0:
                gain_pred = mlp(torch.tensor([x.tolist()[0:3]])).cpu().detach().numpy()[0,0]
                print('step {}: x = {}, f(x) = {}'.format(step, x.tolist(), pred.item()),gain_pred)
                y = x.tolist()[0:3]
                y.append(gain_pred)
                # print(y)
                path.append(y)
            if step % 10 == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.95

        gain_pred = mlp(torch.tensor([x.tolist()[0:3]])).cpu().detach().numpy()[0,0]
        y = x.tolist()[0:3]
        y.append(gain_pred)
        print(y)
        path.append(y)
        nbv = x.tolist()[0:3]
        path = np.array(path)
        print(path.shape)
        print("计算NBV用时：", time.time()-t0)
        print("NBV：", nbv, self.get_pathlength(path))
        nbv = [0.693,0.30,-2.834]
        goal = Vector3d.Vector3d(nbv)
        # viz.Plot_NBV(data_all,path)
        
        
        # ## 人工势场法路径规划
        # # print(self.get_surface_points([16.0,13.0]))
        # t0 = time.time()
        # plan_path = self.path_plan(mlp, Vector3d.Vector3d([0.,1.,-3.]),goal)
        # # print(plan_path)
        # print("人工势场法用时：", time.time()-t0)
        # print("路径长度：", self.get_pathlength(plan_path))
        # viz.Plot_NBV(data_all,path)
        # # self.Plot_APF(data_all,plan_path)

        
        ##计算增益场
        num = 500
        f_gain = []
        f_force = []
        f_add =  []
        input_pos = data_all[0:num,0:3].astype(np.float32)
        # input =  torch.tensor(input_pos).to(torch.float32).reshape(100,2)
        x_pos = torch.tensor(input_pos[0], requires_grad=True).to(torch.float32)
        # optimizer = torch.optim.Adam([x], lr=0.01)  # 使用Adam编译器，对自变量使用，学习速率：lr=1e-3\
        x_pos = torch.tensor([0.,1.,-3.], requires_grad=True).to(torch.float32)
        y = mlp(x_pos)
        y.backward()  # 对pred反向传播，求得x，y每次的梯度
        gain_grad1 = x_pos.grad.cpu().detach().numpy()
        gain_grad = gain_grad1/np.linalg.norm(gain_grad1)
        print(gain_grad)

        for i in range(input_pos.shape[0]):
            x_pos = torch.tensor(input_pos[i], requires_grad=True).to(torch.float32)
            y = mlp(x_pos)
            # optimizer.zero_grad()  # 首先将所有梯度清零
            y.backward()  # 对pred反向传播，求得x，y每次的梯度
            gain_grad1 = x_pos.grad.cpu().detach().numpy()
            gain_grad = gain_grad1/np.linalg.norm(gain_grad1)
            x_pos_ = input_pos[i]
            current_pos = Vector3d.Vector3d(x_pos_)
            # f_vec = self.attractive(current_pos,goal) + self.repulsion(current_pos,goal)
            f_vec = self.attractive(current_pos,goal)
            f_vec_rep = self.repulsion(current_pos,goal)
            # print(np.array(f_vec.direction))
            f_force.append(f_vec.direction)
            f_gain.append(gain_grad)
            
            f_all = gain_grad1 + self.lam * np.array(f_vec.direction)
            f_all = f_all/np.linalg.norm(f_all)
            f_add.append(f_all.tolist())
            if i % 10 == 0:
                print(i,list(x_pos_),list(gain_grad),[f_vec.direction])
        # # print(f_gain)

        viz.PlotField(data_all[0:num],np.array(f_gain),np.array(f_force),np.array(f_add))
        
       
        return x.tolist() 
    
    ## 计算路径长度
    def get_pathlength(self,path):
        path_length = 0.0
        for i in range(len(path)-1):
            path_length += np.linalg.norm(path[i+1,0:3]-path[i,0:3],ord=2)
        return path_length

    ## 得到当前点的附近目标表面坐标
    def get_surface_points(self, pose,R = 5): 
        surface_points = []
        N = R
        for angle in range(0,360,20):
            for n in range(1,N+1):
                x = pose[0] + n/N *R*np.cos(np.radians(angle))
                y = pose[1] + n/N *R*np.sin(np.radians(angle))
                x_index,y_index = int(np.around(x-0.5)),int(np.around(y-0.5))
                if self.grid_map[x_index,y_index]==1:
                    surface_points.append([x,y])

        return np.array(surface_points)
 
    ## 人工势力场路径规划
    def path_plan(self,mlp,start,goal):
        """
        path plan
        :return:
        """

        self.is_plot = True 
        self.is_path_plan_success = False
        self.delta_t = 0.01
        plan_path = []
        current_pos = start
        self.iters = 0
        goal_final = goal
        
        points_list = []
        num = 50
        Ridius = 1
        lamda_hs = 10.0;lamda_ho = 10.0;lamda_ho = 10.0;u = 10
        pos = [start.deltaX,start.deltaY,start.deltaZ,float(lamda_hs),float(lamda_ho),float(u)]
        x = torch.tensor(pos, requires_grad=True).to(torch.float32)
        object_center = torch.tensor(np.array([0,1,0]))
        sample_center = torch.tensor(np.array([0,1,-3]))
        
        print([start.deltaX,start.deltaY,start.deltaZ],"---------------->",[goal.deltaX,goal.deltaY,goal.deltaZ])

        x_init,y_init,z_init =start.deltaX,start.deltaY,start.deltaZ
        lamda_hs = 10.0;lamda_ho = 10.0;lamda_ho = 10.0;u = 10
        plan_path = [[x_init,y_init,z_init, mlp(torch.tensor([x_init,y_init,z_init]).reshape(1,3)).cpu().detach().numpy()[0,0]]]
        x = torch.tensor([float(x_init),float(y_init),float(z_init),float(lamda_hs),float(lamda_ho),float(u)], requires_grad=True).to(torch.float32)
        optimizer = torch.optim.Adam([x], lr=0.01)  # 使用Adam编译器，对自变量使用，学习速率：lr=1e-3
        object_center = torch.tensor(np.array([0,1,0]))
        sample_center = torch.tensor(np.array([0,1,-3]))
        for step in range(200):  # 迭代200次
            f_value = self.Attractive(x[0:3], torch.tensor([goal.deltaX,goal.deltaY,goal.deltaZ])) 
            # current_pos = Vector3d.Vector3d(x.tolist()[0:3])
            # f_vec = self.attractive(current_pos, goal)
            hs = torch.abs(torch.norm(x[0:3]-sample_center)-1.0)
            ho = torch.abs(torch.abs(torch.norm(x[0:3]-object_center)-3.5)-0.5)
            g = torch.abs(-x[1]+0.3)
            pred = -mlp(x[0:3]) + 0.5*f_value +  x[3]*hs + x[4]*ho + x[5]*g# 预测的值是pred，要使pred达到最小

            optimizer.zero_grad()  # 首先将所有梯度清零
            pred.backward()  # 对pred反向传播，求得x，y每次的梯度
            optimizer.step()  # 这是代表更新一步，在这里面完成了自动利用梯度更新权值。这里是更新x,y

            if step % 10 == 0:
                gain_pred = mlp(torch.tensor([x.tolist()[0:3]])).cpu().detach().numpy()[0,0]
                print('step {}: x = {}, f(x) = {}'.format(step, x.tolist(), pred.item()),gain_pred)
                y = x.tolist()[0:3]
                y.append(gain_pred)
                # print(y)
                plan_path.append(y)
            if step % 10 == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.95

        gain_pred = mlp(torch.tensor([x.tolist()[0:3]])).cpu().detach().numpy()[0,0]
        y = x.tolist()[0:3]
        y.append(gain_pred)
        print(y)
        plan_path.append(y)
        nbv = x.tolist()[0:3]
        plan_path = np.array(plan_path)
        print(plan_path.shape)
        print("NBV：", nbv, self.get_pathlength(plan_path))
    
           



        # while  self.iters < self.max_iters and (current_pos - goal_final).length > self.goal_threashold:
        #     # f_vec = self.attractive(current_pos, goal) 
        #     x = torch.tensor([float(current_pos.deltaX),float(current_pos.deltaY),float(current_pos.deltaZ),float(lamda),float(u)], requires_grad=True).to(torch.float32)
        #     # y = mlp(x)
        #     optimizer.zero_grad()  # 首先将所有梯度清零
        #     # y.backward()  # 对pred反向传播，求得x，y每次的梯度
        #     # print([f_vec.deltaX,f_vec.deltaY,f_vec.deltaZ],x.grad)
        #     # gain_grad = x.grad.cpu().detach().numpy()
        #     # f_vec = Vector3d.Vector3d(gain_grad) * self.lam
        #     h = torch.abs(torch.norm(x[0:3]-point)-2)
        #     g = torch.abs(-x[1]+0.3)
        #     pred = -mlp(x[0:3]) +  x[3]*h + x[4]*g # 预测的值是pred，要使pred达到最小
        #     pred.backward()
        #     gain_grad = x.grad
        #     x = x+ self.step_size*gain_grad
        #     # print(x)
            
        #     # current_pos += Vector3d.Vector3d([f_vec.direction[0], f_vec.direction[1], f_vec.direction[2]]) * self.step_size
        #     self.iters += 1
        #     # plan_path.append([current_pos.deltaX, current_pos.deltaY, current_pos.deltaZ])
        #     # print([current_pos.deltaX, current_pos.deltaY, current_pos.deltaZ])
        #     # if len(points_list)==num:
        #     #     dis = math.hypot(points_list[num-1][0]- points_list[0][0],points_list[num-1][1]- points_list[0][1])
        #     #     dis_goal = math.hypot(points_list[num-1][0]- goal.deltaX,points_list[num-1][1]-  goal.deltaY)
        #     #     if dis < Ridius and dis_goal >  self.goal_threashold: 
        #     #        print("陷入局部最优")
        #     #        self.k_att = 3.0
        #     #        goal = current_pos + Vector3d.Vector3d((goal-current_pos).direction) * 3.0
        #     #        points_list = []
        #     #     #    print(goal)
        #     #     #    return plan_path
        #     #     else:
        #     #        self.k_att = 1.0
        #     #        goal = goal_final
        #     #     if len(points_list)>0:
        #     #         points_list.pop(0)
        #     # points_list.append([current_pos.deltaX, current_pos.deltaY])
        
        # print(x)
        # print(current_pos.deltaX, current_pos.deltaY, current_pos.deltaZ)
        # plan_path.append([goal_final.deltaX, goal_final.deltaY, goal_final.deltaZ])
        # if (current_pos - goal_final).length <= self.goal_threashold:
        #     self.is_path_plan_success = True
 
        # if self.is_path_plan_success:
        #     print('path plan success')
        # else:
        #     print('path plan failed')


        return np.array(plan_path)
    
    
    ## 引力场
    def attractive(self,current_pos,goal):
        """
        引力计算
        :return: 引力
        """
        d0 = 0.2
        dis_goal =np.sqrt((current_pos.deltaX- goal.deltaX)**2+(current_pos.deltaY- goal.deltaY)**2+(current_pos.deltaZ-  goal.deltaZ)**2)
        if dis_goal < d0:
           att = (goal -current_pos) * self.k_att  # 方向由机器人指向目标点
        else:
           att =  (goal -current_pos) * (d0*self.k_att/dis_goal)


        return att

    def Attractive(self,current_pos,goal):
        """
        引力计算
        :return: 引力
        """
        d0 = 0.2
        dis_goal =torch.norm(current_pos-goal,p=2)
        if dis_goal < d0:
           Att =0.5 * self.k_att * (dis_goal**2)  # 方向由机器人指向目标点
        else:
           Att = (d0*dis_goal -0.5*(d0**2))*self.k_att


        return Att


    ## 斥力场
    def repulsion(self,current_pos,goal):
        """
        斥力计算, 改进斥力函数, 解决不可达问题
        :return: 斥力大小
        """
        rep = Vector3d.Vector3d([0, 0, 0])  # 所有障碍物总斥力
        ## TODO
        obstacles = get_surface_points([current_pos.deltaX,current_pos.deltaY,current_pos.deltaZ],2,0.5).cpu().detach().numpy()
        # print(obstacles)
        for obstacle in obstacles:
            obstacle = Vector3d.Vector3d(obstacle)
            print(obstacle)
            # obstacle = Vector3d(0, 0, 0)
            obs_to_rob = current_pos - obstacle
            rob_to_goal = goal - current_pos
            if (obs_to_rob.length > self.rr):  # 超出障碍物斥力影响范围
                pass
            else:
                print(obs_to_rob.length,rob_to_goal.length)
                rep_1 = Vector3d.Vector3d([obs_to_rob.direction[0], obs_to_rob.direction[1], obs_to_rob.direction[2]]) * self.k_rep * (
                    1.0 / obs_to_rob.length - 1.0 / self.rr) / (obs_to_rob.length ** 2) * (rob_to_goal.length ** 2)
                rep_2 = Vector3d.Vector3d([rob_to_goal.direction[0], rob_to_goal.direction[1], rob_to_goal.direction[2]]) * self.k_rep * (
                    (1.0 / obs_to_rob.length - 1.0 / self.rr) ** 2) * rob_to_goal.length
                rep +=(rep_1+rep_2)
        return rep
    
    ## 训练MLP
    def train(self,mlp, data,N=100):
    
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

        print("用时：",time.time()-t0)
        

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
        print("用时：",time.time()-t0)

        self.PlotTrain(test_num,mlp_loss,test_loss,XX_train,g_train_input,g_train,XX_test,g_test_input,g_test)
        return mlp.cpu()
    

## 测试
def test():
    ## 前三个为坐标，后两个为方向
    x_start = np.array([0.0, 1.0,-3.0,0,0])  # Starting node
    x_goal = np.array([1.0, 1.0,1.0,30,120])  # Goal node

    vpp = Viewpath_planner(x_start)
    
    ## 采集视角数据
    t0 = time.time()
    # data = vpp.get_sample_sphere(x_start,N =1000,R=3)
    # data= vpp.get_sample_sphere_center(x_start,N=100,R=1)
    # np.savetxt("planner_result/result_data/resultdata_1000_all.txt", data,fmt='%f')

    # 显示
    # viz.draw_sample(data)
    
    # ## 加载数据
    # data = np.loadtxt("planner_result/result_data/resultdata_1000_all.txt")
    # print(data.shape)
    # # ## 去除无效数据
    # data_select_list = list(data)
    # # ## ucer 归一化处理，并计算信息增益
    # uncer_min,uncer_max = np.min(np.array(data_select_list)[:,5]),np.max(np.array(data_select_list)[:,5])
    # for j in range(len(data_select_list)):
    #     data_select_list[j][5] = (data_select_list[j][5]-uncer_min)/(uncer_max-uncer_min)
        
    # data_select = np.array(data_select_list) 
    # print(data_select.shape)
    # print("用时：", time.time()-t0)

    # # 显示
    # viz.draw_sample(data_select)

 
    
    # ## train mlp
    # mlp=MLP().to(device)
    # t0 = time.time()
    # vpp.train(mlp,data_select,N=100)
    # ## model存储
    # torch.save(mlp.state_dict(), 'planner_result/model/params_mlp.pkl')

    # ## 加载模型,计算NBV
    mlp = MLP()
    mlp.load_state_dict(torch.load('planner_result/model/params_mlp.pkl'))
    vpp.get_optimal_NBV(x_start,mlp,NUM=100)



    # t0 = time.time()
    # # print(get_ray_depth([[0,1.5,-2]],[[0,0,1]]))
    # uncer,avg_distance,ratio = get_uncertainty([0,1,-2],0,-90)
    # print([uncer,avg_distance,ratio])
    # print("用时：", time.time()-t0)


    # t0 = time.time()
    # uncer,avg_distance,ratio = get_uncertainty([0,1,-2],0,0)
    # print("用时：", time.time()-t0)
    # print([uncer,avg_distance,ratio])
    # # print(get_surface_points([0,1,-2],4,0.1))
    # print(get_ray_depth([[0,1,-2]],[[1,0,0]]))
    # # print(get_ray_depth([[0,1,-2]],[[0,-1,0]]))


if __name__ == '__main__':
    #常量都取出来，以便改动
    EPOCH=300
    BATCH_SIZE = 100
    MLP_LR=0.001

    viz = Visualize()



    ## 测试
    test()


    print("finished")