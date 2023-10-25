
import threading
import requests
import numpy as np
import sys
import json
import copy
import time
import os
import torch

from core.interface2 import *
from core.vpp import *
# from core.rrt3D_mlp_dvcv import *

global first_Flag
first_Flag = [True] 
stop_Flag = [False]
view_num = 0
view_maxnum = 35


## IdahoStateCapitol
# x_start = np.array([2.0,1.0,-3.5,0,0])  # Starting node
# object_center = [0.0,1.0,0.0] 
# r0 = 4.0
# pos = [[2.0,1.0,-4.4],[2.5,1.0,-4.1],[3.0,1.0,-4.0],[3.5,1.0,-3.0],[5.0,1.0,0.0]]  
# pdf_mean = 3.0; pdf_std = 0.6

# ## cabin
x_start = np.array([0.0,1.0,-3.0,0,0])  # Starting node
object_center = [0.0,1.0,0.0] 
r0 = 3.0;step_size = 2.0
pos = [[0.0,1.0,-2.9],[0.5,1.0,-2.8],[1.0,1.0,-2.7],[1.5,1.0,-2.6],[2.0,1.0,-2.3]]
pdf_mean =3.5; pdf_std = 1.0


# # ## drum模型
# x_start = np.array([0.0,1.0,-3.0,0,0])  # Starting node
# object_center = [0.0,1.0,0.0] 
# r0 = 3.0;step_size = 2.0
# pos = [[0.0,1.0,-2.9],[0.5,1.0,-2.8],[1.0,1.0,-2.7],[1.5,1.0,-2.6],[2.0,1.0,-2.3]]
# pdf_mean =3.5; pdf_std = 1.0

    
mlp=MLP().to(device)
#### 路径规划主函数
def path_plan():
    #################################### code here #############################
    ## our
    our_planner()
    ## rrt
    # rrt_planner()

    



## 单个物体
def our_planner():
    global x_start,mlp,r0,view_num,pos,step_size
    print(x_start)
    
    if view_num < view_maxnum:
        if first_Flag[0]:

            ### 给定起始位姿
            first_Flag[0]=False
       
            for i in range(len(pos)):
                dx,dy,dz =  object_center[0]-pos[i][0],object_center[1]-pos[i][1],object_center[2]-pos[i][2]
                u = np.arctan2(-dy,np.sqrt(dx**2+dz**2))
                v = np.arctan2(dx,dz)
                send_NBV(pos[i],u,v)
                print("send NBV")
                time.sleep(1)
            
            if os.path.exists("core/results/path_rrt_"+str(version)+".txt"):
                os.remove("core/results/path_rrt_"+str(version)+".txt")
            if os.path.exists("core/results/views_rrt_"+str(version)+".txt"):
                os.remove("core/results/views_rrt_"+str(version)+".txt")  
            if os.path.exists("core/results/time_rrt_"+str(version)+".txt"):
                os.remove("core/results/time_rrt_"+str(version)+".txt")  
                

        else:
       
            vpp = Viewpath_planner(x_start,object_center,r0,pdf_mean,pdf_std)
            ## 建立tsdf栅格地图，便于采样和规划
            vpp.State_vol = vpp.tsdf.tsdf_test()
            vpp.emptyspace_index_gpu = torch.nonzero(vpp.State_vol==1)
            vpp.emptyspace_index = vpp.emptyspace_index_gpu.cpu().numpy()
            start_index = torch.tensor(((vpp.x_start[0:3]-vpp.tsdf.tsdf_vol._vol_origin)/vpp.tsdf.tsdf_vol._voxel_size)).to(device)
            start_index = start_index.repeat(vpp.emptyspace_index.shape[0],1)
            vox2start_dist = torch.norm(vpp.emptyspace_index_gpu.to(device)-start_index,p =2,dim=1)*voxel_res
            vpp.emptyspace_index_sphere = vpp.emptyspace_index_gpu[vox2start_dist<vpp.r0].cpu().numpy()

            ## 采集数据
            t0 = time.time()
            locations = vpp.sample_location(x_start,N = 100,R = vpp.r0)
            data = vpp.sample_direction(locations)
            # print(data)
            print("采样用时:", time.time()-t0)

            ## 训练MLP
            t1 = time.time()
            model = copy.deepcopy(mlp)
            mlp_trained = vpp.train_only(model,data[0:100],N=100).to(device)
            vpp.mlp = mlp_trained
            vpp.mlp.eval()
            print("训练用时:", time.time()-t1)


            ## 离散采样方法计算NBV
            data_sort = data[data[:,5].argsort()] #按照第5列对行排序
            nbv_final = data_sort[len(data_sort)-1,:]
            print("最终NBV:", nbv_final)
            
            ## A star 规划路径
            t3 = time.time()
            # nbv_final = np.array([-2.29999995,  2.91000009, -2.9000001,   0.73818687,  0.93232092,  1.   ])
            Astar_Planner = Weighted_A_star(x_start[0:3], nbv_final[0:3], vpp)
            local_path = Astar_Planner.run()
            local_path = np.array(local_path)
            print("规划查询点数、时间 = ", Astar_Planner.query_num,Astar_Planner.query_time)
            print("A*用时:", time.time()-t3)
            print("local_path = ",local_path)
            print("最终路径长度: ",vpp.get_pathlength(local_path))
            path_view = vpp.get_path_view(nbv_final,local_path,step=step_size)

            
            # 存储
            time_use = time.time()-t0
            print("规划用时：", time_use)
            vpp.savepath(local_path)
            vpp.saveview(path_view)
            vpp.savetime(time_use)

            ## 发送NBV
            for i in range(len(path_view)):
                send_NBV(path_view[i][0:3],path_view[i][3],path_view[i][4])
                view_num += 1
                print("规划视角数目：",view_num)
                time.sleep(1)

            x_start = nbv_final


## rrt
def rrt_planner():
    global x_start,r0,view_num,pos
    print(x_start)
    if view_num < view_maxnum:
        if first_Flag[0]:
            ### 给定起始位姿
            first_Flag[0]=False
            for i in range(len(pos)):
                dx,dy,dz =  object_center[0]-pos[i][0],object_center[1]-pos[i][1],object_center[2]-pos[i][2]
                u = np.arctan2(-dy,np.sqrt(dx**2+dz**2))
                v = np.arctan2(dx,dz)
                send_NBV(pos[i],u,v)
                print("send NBV")
                time.sleep(1)
            
            if os.path.exists("core/results/path_rrt_"+str(version)+".txt"):
                os.remove("core/results/path_rrt_"+str(version)+".txt")
            if os.path.exists("core/results/views_rrt_"+str(version)+".txt"):
                os.remove("core/results/views_rrt_"+str(version)+".txt")  
            if os.path.exists("core/results/time_rrt_"+str(version)+".txt"):
                os.remove("core/results/time_rrt_"+str(version)+".txt")  
                
        else:
            t0 = time.time()
            rrt = RRT(x_start[0:3],object_center,r0,pdf_mean,pdf_std)

            rrt.State_vol = rrt.tsdf.tsdf_test()
            rrt.emptyspace_index_gpu = torch.nonzero(rrt.State_vol==1)
            rrt.emptyspace_index = rrt.emptyspace_index_gpu.cpu().numpy()
            start_index = torch.tensor(((rrt.x_start[0:3]-rrt.tsdf.tsdf_vol._vol_origin)/rrt.tsdf.tsdf_vol._voxel_size)).to(device)
            start_index = start_index.repeat(rrt.emptyspace_index.shape[0],1)
            vox2start_dist = torch.norm(rrt.emptyspace_index_gpu.to(device)-start_index,p =2,dim=1)*voxel_res
            rrt.emptyspace_index_sphere = rrt.emptyspace_index_gpu[vox2start_dist<rrt.r0].cpu().numpy()


            nbv,plan_path,data_all,is_sucess = rrt.path_plan()
            if is_sucess and getDist(nbv,x_start) > 0:
                # 计算视角最佳方向，仅考虑航向角
                nbv_final = nbv[0:5]
                print("最终NBV：", nbv_final)
                path_view = rrt.get_path_view(nbv_final,plan_path,step=2.0)
                print("规划视角数量：",len(path_view))
                np.savetxt("core/results/planned_view_num.txt",np.array([len(path_view),0]))
                ## 发送NBV
                for i in range(len(path_view)):
                    view_num += 1
                    if view_num < view_maxnum:
                        send_NBV(path_view[i][0:3],path_view[i][3],path_view[i][4])   
                        time.sleep(1)
                    else:
                        print("规划结束")
                rrt.savepath(plan_path)
                rrt.saveview(path_view)
                t_use = time.time()-t0
                rrt.savetime(t_use)

                print('time used = ' + str(t_use))
                print(plan_path.shape)
                print("路径为：")
                print(plan_path)
                print(x_start,"---------------->",nbv_final)
                print("路径长度：",  rrt.get_pathlength(plan_path))

                x_start = nbv_final
            else:
                print("无规划视角")
