from django.shortcuts import render
from django.http import HttpResponse
import threading
import requests
from core.interface import path_plan
# from core.tsdf_online import *
from PIL import Image
import numpy as np
import json
import threading
import os
from imageio import *
import cv2

# Create your views here.

model = "drums_2"
log_base_dir = os.path.join("logs",model,"trainset")
if not os.path.exists(log_base_dir):
    os.makedirs(log_base_dir)

imgs = []
depth_imgs = []
poses = []

is_reconstruction_flag = False
reconstruction_imgs_index = 5
# for i in range(reconstruction_imgs_index):
#     print("Fusing frame %d/%d"%(i, reconstruction_imgs_index))

#     # Read RGB-D image and camera pose
#     img = cv2.cvtColor(cv2.imread(os.path.join(log_base_dir,"main_{:03d}.png".format(i))), cv2.COLOR_BGR2RGB)[::2,::2]
#     depth_img = cv2.imread(os.path.join(log_base_dir,"depth_{:03d}.png".format(i)),-1).astype(float)[::2,::2,0]
#     pose = np.loadtxt(os.path.join(log_base_dir,"pose_{:03d}.csv".format(i)),delimiter = ',')
#     imgs.append(img)
#     depth_imgs.append(depth_img)
#     poses.append(pose) 
#     tsdf_reconstruction(img,depth_img,pose)
# get_tsdf_model()
# view = np.array([-2.0,1.0,-2.0,0,1.57/2])
# uncer_all = get_all_uncertainty(view[0:3],view[3],view[4])
# print("uncer_all ",uncer_all)


#### 接受nerf更新模型完成信号，开始路径规划，规划完成后传递信息给render
def isfinsh(request):
    finsh = request.GET.get("finish")
    if finsh == 'yes':
        #################################### code here #############################
        plan_thread = threading.Thread(target=path_plan,args=())
        plan_thread.start()
    
    return HttpResponse('begin to path plan：{}'.format(finsh))

def get_picture(request):
    if request.method == 'POST':
        img = request.FILES.get("img", None)
        depth_img = request.FILES.get("depth_img",None)    
        pose = request.POST.get("pose",None)
        pose = str(pose)
        img = np.array(Image.open(img))
        print(img.shape)
        depth_img = np.array(Image.open(depth_img))
        print(depth_img.shape)
        pose = read_matrix(pose)  
        print(pose.shape)
        start_mission(img,depth_img,pose)
        
    return HttpResponse("get image")

def start_mission(img,depth_img,pose):
    global is_reconstruction_flag, reconstruction_imgs_index
    print("get images")
    imgs.append(img)
    depth_imgs.append(depth_img)
    poses.append(pose)  
    n = len(imgs)
    print("Fusing frame %d"%(n))
    imwrite(os.path.join(log_base_dir,"main_{:03d}.png".format(n-1)), img)
    imwrite(os.path.join(log_base_dir,"depth_{:03d}.png".format(n-1)), depth_img)
    np.savetxt(os.path.join(log_base_dir,"pose_{:03d}.csv".format(n-1)), pose, delimiter = ',')
    
    if n == 5:
        for i in range(5):
            tsdf_reconstruction(imgs[i][::2,::2],depth_imgs[i][::2,::2,0],poses[i])
        get_tsdf_model()
        view = np.array([-2.0,1.0,-2.0,0,1.57/2])
        uncer_all = get_all_uncertainty(view[0:3],view[3],view[4])
        print("uncer_all ",uncer_all)
        path_plan()

    if n >5:
        planned_view_num = np.loadtxt("core/results/planned_view_num.txt")[0]
        print(n,reconstruction_imgs_index,planned_view_num)
        if n - reconstruction_imgs_index == planned_view_num and n < 40:
            is_reconstruction_flag = True
            for j in range(reconstruction_imgs_index,n):
                tsdf_reconstruction(imgs[j][::2,::2],depth_imgs[j][::2,::2,0],poses[j])
            is_reconstruction_flag = False
            reconstruction_imgs_index = n
            get_tsdf_model()
            # view = np.array([-2.0,1.0,-2.0,0,1.57/2])
            # uncer_all = get_all_uncertainty(view[0:3],view[3],view[4])
            # print("uncer_all ",uncer_all)
            path_plan()
        if n - reconstruction_imgs_index>=4 and is_reconstruction_flag == False:
            reconstruction_imgs_index = n


def info_planner():
    
    # if len(self.add_img_set) >= 40 or self.NBV_step > 40:
    #     self.uncertainty_kargs = None
    #     return
    # self.NBV_step = self.NBV_step + 1
    # self.save_model()
    # self.uncertainty_kargs = self.copy_model(2)
    response = requests.get("http://localhost:6000/isfinish", "finish=yes")
    print("send info to planner")



def read_matrix(pose):
    a = []
    for line in pose.split('\n'):
        l = []
        if len(line) > 5:
            ll = line.split('\t')
            l.append(ll[0])
            l.append(ll[1])
            l.append(ll[2])
            l.append(ll[3])
            a.append(l)
    a = np.array(a).astype(np.float32)
    return a