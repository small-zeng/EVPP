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

sys.path.append("/home/zengjing/zj/Projects/nerfplanning/") 
print(sys.path)
from  testscript import get_uncertainty,get_ray_depth,get_surface_points


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


class Viewpath_planner:
    def __init__(self, x_start):

        self.x_start = x_start

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
    
    ## 在上一个NBV一定半径圆内采样并筛选，方向为看向目标中心，计算权重,得到分布
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

    ## 在上一个NBV一定半径圆内采样并筛选，方向为看向目标中心，计算权重,得到分布
    def get_sample_sphere_center(self,node, N =1000, R = 2):
        object_center = [0,1,0]
        n = 0 
        i = 0
        data_select = []
        data_select_array = np.array(data_select)

        while n < N:
            theta = np.random.uniform(0,np.pi)
            pha = np.random.uniform(0,2*np.pi)
            dir_vec = [np.sin(theta)*np.cos(pha),np.sin(theta)*np.sin(pha),np.cos(theta)]

            r = R
            x = r * dir_vec[1] + object_center[0]
            y = r * dir_vec[2] + object_center[1]
            z = -r * dir_vec[0] + object_center[2]
            dx,dy,dz =  object_center[0]-x,object_center[1]-y,object_center[2]-z
            # if np.abs(dz) < 1:
            #     continue  
            dis_node2sample =  np.sqrt((node[0]-x)**2+(node[1]-y)**2+(node[2]-z)**2)                                                                                                   
            if y > 0.3 and dis_node2sample<1.0:
            
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

        ## 在上一个NBV一定半径圆内采样并筛选，方向为看向目标中心，计算权重,得到分布
    def get_testdata_sphere_center(self,node, N =1000, R = 2):
        object_center = [0,1,0]
        n = 0 
        i = 0
        data_select = []
        data_select_array = np.array(data_select)

        while n < N:
            theta = np.random.uniform(0,np.pi)
            pha = np.random.uniform(0,2*np.pi)
            dir_vec = [np.sin(theta)*np.cos(pha),np.sin(theta)*np.sin(pha),np.cos(theta)]

            r = R
            x = r * dir_vec[1] + object_center[0]
            y = r * dir_vec[2] + object_center[1]
            z = -r * dir_vec[0] + object_center[2]
            dx,dy,dz =  object_center[0]-x,object_center[1]-y,object_center[2]-z
            # if np.abs(dz) < 1:
            #     continue  
            dis_node2sample =  np.sqrt((node[0]-x)**2+(node[1]-y)**2+(node[2]-z)**2)                                                                                                   
            if y > 0.3 and dis_node2sample<1.0:    
                data_select.append([x,y,z])
                # print([x,y,z,dis_node2sample])
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
        init_test_data = self.get_testdata_sphere_center(node,N=N_test,R=2)
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
        data_sort = data_all[data_all[:,3].argsort()] #按照第3列对行排序
        # N_num = len(data_sort)
        # print(N_num)
        # N_select = 1000
        # data_select = data_sort[N_num-N_select:N_num,:]
        # print(data_select.shape)
        N_MAX = 1                                                                                                                                                                                                                                   
        x_init,y_init,z_init = np.mean(data_sort[N_test-N_MAX:,0]),np.mean(data_sort[N_test-N_MAX:,1]),np.mean(data_sort[N_test-N_MAX:,2])
        print("初始点 ：",x_init,y_init,z_init)
        
        # x_init,y_init,z_init = 0, 1, -2
        lamda = 10.0;u = 10
        path = [[x_init,y_init,z_init]]
        x = torch.tensor([float(x_init),float(y_init),float(z_init),float(lamda),float(u)], requires_grad=True).to(torch.float32)
        optimizer = torch.optim.Adam([x], lr=0.01)  # 使用Adam编译器，对自变量使用，学习速率：lr=1e-3
        point = torch.tensor(np.array([0,1,0]))
        for step in range(300):  # 迭代20000次
            
    
            h = torch.abs(torch.norm(x[0:3]-point)-2)

            g = torch.abs(-x[1]+0.3)
            pred = -mlp(x[0:3]) +  x[3]*h + x[4]*g # 预测的值是pred，要使pred达到最小

            optimizer.zero_grad()  # 首先将所有梯度清零
            pred.backward()  # 对pred反向传播，求得x，y每次的梯度
            optimizer.step()  # 这是代表更新一步，在这里面完成了自动利用梯度更新权值。这里是更新x,y

            if step % 10 == 0:
                print('step {}: x = {}, f(x) = {}'.format(step, x.tolist(), pred.item()))
                path.append( x.tolist()[0:3])
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.95
        path.append(x.tolist()[0:3])
        nbv = x.tolist()[0:3]
        path = np.array(path)
        print(path.shape)
        print("计算NBV用时：", time.time()-t0)
        print("NBV：", nbv)
        # self.Plot_NBV(data_all,path)
        
        
        ## 人工势场法路径规划
        # print(self.get_surface_points([16.0,13.0]))
        goal = Vector3d.Vector3d(nbv)
        plan_path = self.path_plan(mlp, Vector3d.Vector3d([0,1,-2]),goal)
        print(plan_path)
        self.Plot_APF(data_all,plan_path)
        
        # ##计算增益场
        # num = 500
        # f_gain = []
        # f_force = []
        # f_add =  []
        # input_pos = test_data[0:num,0:2].astype(float32)
        # # input =  torch.tensor(input_pos).to(torch.float32).reshape(100,2)
        # x_pos = torch.tensor(input_pos[0], requires_grad=True).to(torch.float32)
        # # optimizer = torch.optim.Adam([x], lr=0.01)  # 使用Adam编译器，对自变量使用，学习速率：lr=1e-3
        # for i in ra nge(input_pos.shape[0]):
        #     x_pos = torch.tensor(input_pos[i], requires_grad=True).to(torch.float32)
        #     y = mlp(x_pos)
        #     # optimizer.zero_grad()  # 首先将所有梯度清零
        #     y.backward()  # 对pred反向传播，求得x，y每次的梯度
        #     gain_grad1 = x_pos.grad.cpu().detach().numpy()
        #     gain_grad = gain_grad1/np.linalg.norm(gain_grad1)
        #     x_pos_ = input_pos[i]
        #     current_pos = Vector2d.Vector2d(x_pos_)
        #     f_vec = self.attractive(current_pos,goal) + self.repulsion(current_pos,goal)
        #     f_force.append(np.concatenate((x_pos_,np.array(f_vec.direction)),axis=0).tolist())
        #     f_gain.append(np.concatenate((x_pos_,gain_grad),axis=0).tolist())
        #     f_all = gain_grad1 + self.lam * np.array(f_vec.direction)
        #     f_all = f_all/np.linalg.norm(f_all)
        #     f_add.append(np.concatenate((x_pos_,f_all),axis=0).tolist())
        # # print(f_gain)

        # PlotCurve_resampling(test_result,path,plan_path,np.array(f_gain),np.array(f_force),np.array(f_add))
        
       
        return x.tolist()

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
        ###定义势力场参数
        self.k_att = 1.0
        self.k_rep = 10
        self.rr = 10.0
        self.step_size, self.max_iters, self.goal_threashold = 0.01, 300, 1.0  # 步长0.5寻路1000次用时4.37s, 步长0.1寻路1000次用时21s
        self.step_size_ =2
        self.is_plot = True 
        self.is_path_plan_success = False
        self.delta_t = 0.01
        self.lam = 100
        plan_path = []
        current_pos = start
        self.iters = 0
        goal_final = goal
        
        points_list = []
        num = 50
        Ridius = 1
        lamda = 10.0;u = 10
        pos = [float(current_pos.deltaX),float(current_pos.deltaY),float(current_pos.deltaZ),float(lamda),float(u)]
        x = torch.tensor(pos, requires_grad=True).to(torch.float32)
        point = torch.tensor(np.array([0,1,0]))
        
        print([start.deltaX,start.deltaY,start.deltaZ],"---------------->",[goal.deltaX,goal.deltaY,goal.deltaZ])
        lr = 0.01
        for step in range(300):  # 迭代20000次

            f_value = self.Attractive(x[0:3], torch.tensor([goal.deltaX,goal.deltaY,goal.deltaZ])) 
            f_vec = self.attractive(current_pos, goal)
            h = torch.abs(torch.norm(x[0:3]-point)-2)
            g = torch.abs(-x[1]+0.3)
            # print(-mlp(x[0:3]),f_value)
            pred = -mlp(x[0:3]) + 0.1*f_value +  x[3]*h + x[4]*g # 预测的值是pred，要使pred达到最小

            pred.backward()  # 对pred反向传播，求得x，y每次的梯度
            
            x.grad.tolist()
            a = x - x.grad * lr
            x = torch.tensor(a, requires_grad=True).to(torch.float32)
            current_pos = Vector3d.Vector3d(x.tolist()[0:3])
            
          
            if step % 10 == 0:
                print("grad = ", f_vec.direction, x.grad)
                print('step {}: x = {}, f(x) = {}'.format(step, x.tolist(), pred.item()))
                plan_path.append( x.tolist()[0:3])
                lr *=0.9

            plan_path.append([current_pos.deltaX, current_pos.deltaY, current_pos.deltaZ])
           



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
        
        print(x)
        print(current_pos.deltaX, current_pos.deltaY, current_pos.deltaZ)
        plan_path.append([goal_final.deltaX, goal_final.deltaY, goal_final.deltaZ])
        if (current_pos - goal_final).length <= self.goal_threashold:
            self.is_path_plan_success = True
 
        if self.is_path_plan_success:
            print('path plan success')
        else:
            print('path plan failed')


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
        obstacles = self.get_surface_points([current_pos.deltaX,current_pos.deltaY,current_pos.deltaZ])
        for obstacle in obstacles:
            obstacle = Vector3d.Vector3d(obstacle)
            # obstacle = Vector3d(0, 0, 0)
            obs_to_rob = current_pos - obstacle
            rob_to_goal = goal - current_pos
            if (obs_to_rob.length > self.rr):  # 超出障碍物斥力影响范围
                pass
            else:
                rep_1 = Vector3d.Vector3d([obs_to_rob.direction[0], obs_to_rob.direction[1], obs_to_rob.direction[2]]) * self.k_rep * (
                        1.0 / obs_to_rob.length - 1.0 / self.rr) / (obs_to_rob.length ** 2) * (rob_to_goal.length ** 2)
                rep_2 = Vector3d.Vector3d([rob_to_goal.direction[0], rob_to_goal.direction[1], rob_to_goal.direction[2]]) * self.k_rep * ((1.0 / obs_to_rob.length - 1.0 / self.rr) ** 2) * rob_to_goal.length
                rep +=(rep_1+rep_2)
        return rep
    
    ## 训练MLP
    def train(self,model, data,N=100):
        
    
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
    
    ## 画训练图调试
    def PlotTrain(self, test_num,mlp_loss,test_loss,XX_train,g_train_input,g_train,XX_test,g_test_input,g_test):
        # print(len(mlp_loss))
        # print(EPOCH)
        fig = plt.figure(figsize=(12, 12))

        plt.subplot(221)
        plt.plot([i + 1 for i in range(50,EPOCH)], mlp_loss[50:], label='MLP_train')
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
        fig = plt.figure(figsize=(12, 12))
        
        ax3d = fig.add_subplot(111,projection="3d")  # 创建三维坐标
        ax3d.view_init(elev=-13., azim=88.)
        sc = ax3d.scatter(testdata[:,0],testdata[:,1],testdata[:,2], marker='*', c=testdata[:,3], cmap='rainbow', alpha = .5)
        plt.colorbar(sc)
        ax3d.invert_xaxis()
        ax3d.set_xlabel('x', fontsize=10)
        ax3d.set_ylabel('y', fontsize=10)
        ax3d.set_zlabel('z', fontsize=10)
        ax3d.set_title('3D NeRF NBV)', fontsize=10)
        for i in range(len(path)-1):
            ax3d.quiver(path[i,0], path[i,1], path[i,2], path[i+1,0]-path[i,0],  path[i+1,1]-path[i,1],  path[i+1,2]-path[i,2],
             color = 'black', alpha = .8, lw = 2)

        plt.show()

    ## 画路径调试
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

    
    ## 画路径鬼规划调试
    def PlotCurve(test_result,path,plan_path,f_gain,f_force,f_add):  

        plt.figure(figsize=(12, 9))
        
        ax = plt.subplot(221)
        sc = plt.scatter(test_result[:,0],test_result[:,1], marker='*', c=test_result[:,2], cmap='rainbow')
        plt.colorbar(sc)
        for i in range(len(path)-1):
            ax.arrow(path[i,0], path[i,1], path[i+1,0]-path[i,0],  path[i+1,1]-path[i,1], head_width=0.05, head_length=0.02,linewidth=1,color ='black')
        
        #画出规划的路径
        plan_path = np.array(plan_path)
        plt.plot(plan_path[:,0],plan_path[:,1], marker='o', c='g', markersize=2)
        plt.pause(0.01)

        plt.title('Artificial Potential Field')
        
        ax = plt.subplot(222)
        sc = plt.scatter(test_result[:,0],test_result[:,1], marker='*', c=test_result[:,2], cmap='rainbow')
        plt.colorbar(sc)
        # plt.scatter(f_gain[:,0],f_gain[:,1], marker='*', c='black')
        print(f_gain[0])
        for i in range(f_gain.shape[0]):
            ax.arrow(f_add[i,0], f_add[i,1],f_add[i,2],  f_add[i,3], head_width=0.5, head_length=0.2,linewidth=1,color ='black')
        
        plt.pause(0.01)

        plt.title('All Field')
        
        ax = plt.subplot(223)
        sc = plt.scatter(test_result[:,0],test_result[:,1], marker='*', c=test_result[:,2], cmap='rainbow')
        plt.colorbar(sc)
        # plt.scatter(f_gain[:,0],f_gain[:,1], marker='*', c='black')
        print(f_gain[0])
        for i in range(f_gain.shape[0]):
            ax.arrow(f_gain[i,0], f_gain[i,1],f_gain[i,2],  f_gain[i,3], head_width=0.5, head_length=0.2,linewidth=1,color ='black')
        
        plt.pause(0.01)

        plt.title('Gain Field')

        ax = plt.subplot(224)
        sc = plt.scatter(test_result[:,0],test_result[:,1], marker='*', c=test_result[:,2], cmap='rainbow')
        plt.colorbar(sc)
        # plt.scatter(f_gain[:,0],f_gain[:,1], marker='*', c='black')
        print(f_gain[0])
        for i in range(f_gain.shape[0]):
            ax.arrow(f_force[i,0], f_force[i,1],f_force[i,2],  f_force[i,3], head_width=0.5, head_length=0.2,linewidth=1,color ='black')
        
        plt.pause(0.01)

        plt.title('Potential Field')



        plt.tight_layout() 
        # plt.show()


##显示采样点
def draw_sample(data):
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


    plt.show()


 
## 测试
def test():
    ## 前三个为坐标，后两个为方向
    x_start = np.array([0.0, 1.0,-2.0,30,120])  # Starting node
    x_start1 = np.array([1.0, 1.0,0.0,30,120])
    x_start2 = np.array([0.0, 1.0,1.0,30,120])
    x_start3 = np.array([-1.0, 1.0,0.0,30,120])
    x_goal = np.array([1.0, 1.0,1.0,30,120])  # Goal node

    vpp = Viewpath_planner(x_start)
    
    # t0 = time.time()
    # data = vpp.get_sample_object_center(x_start,N =100,R=2)
    # data1 = vpp.get_sample_object_center(x_start1,N =100,R=2)
    # data2 = vpp.get_sample_object_center(x_start2,N =100,R=2)
    # data3 = vpp.get_sample_object_center(x_start3,N =100,R=2)
    # data = np.concatenate((data,data1,data2,data3),axis=0)
    
    # np.savetxt("data_400_all.txt", data,fmt='%f')
    # data = np.loadtxt("/home/zengjing/zj/Projects/nerfplanning/data_1000_all.txt")
    
    # data = vpp.get_sample_sphere_center(x_start,N =1000,R=2)
    # np.savetxt("data_400_all.txt", data,fmt='%f')


    data = np.loadtxt("/home/zengjing/zj/Projects/nerfplanning/data_1000_all.txt")
    print(data.shape)
    # ## 去除无效数据
    data_select_list = list(data)
    # for i in range(1000):
    #     # if np.abs(data[i,2])>1 and data[i,6]<5:
    #     #     data_select_list.append(list(data[i]))
    #     if not math.isnan(data[i,6]) and np.abs(data[i,2])>0.5:
    #         data_select_list.append(list(data[i]))
    # ## ucer 归一化处理，并计算信息增益
    uncer_min,uncer_max = np.min(np.array(data_select_list)[:,5]),np.max(np.array(data_select_list)[:,5])
    for j in range(len(data_select_list)):
        data_select_list[j][5] = (data_select_list[j][5]-uncer_min)/(uncer_max-uncer_min)
        # view_IG_dist_prob = norm(2.75,1).pdf(data_select_list[j][6])/norm(2.75,1).pdf(2.75)
        # view_IG = 0.7 * data_select_list[j][5] +  0.3 * view_IG_dist_prob
        # data_select_list[j].append(view_IG)
        # print(data_select_list[j],view_IG_dist_prob)
    data_select = np.array(data_select_list) 
    # print(data_select.shape)
    # print("用时：", time.time()-t0)

    ## 显示
    # draw_sample(data_select)

    # ## train mlp
    # vpp.train(mlp,data_select,N=100)
    # ## model存储
    # torch.save(mlp.state_dict(), 'params_mlp.pkl')

    ## 加载模型
    mlp = MLP()
    mlp.load_state_dict(torch.load('params_mlp.pkl'))
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

    mlp=MLP().to(device)
    t0 = time.time()


    ## 测试
    test()



   


    print("finished")