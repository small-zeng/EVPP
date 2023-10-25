from django.shortcuts import render
from django.http import HttpResponse,JsonResponse
from django.http import request
import os
from PIL import Image
import numpy as np
import json
import threading

from numpy.core.numeric import count_nonzero

from core.nerf import *

# Create your views here.

# 接受图片并根据训练策略加入训练
# 向planner发送收敛完成信号
# 根据坐标方向返回到表面的距离
# 根据起始终点坐标返回是否直线通路
# 根据观测点及观测方向返回信息增益或不确定性
H = 400
W = 400
focal = 300

# H = 400
# W = 600
# focal = 600/36*29.088

running_flag = [False]
imgs = []
depth_imgs = []
poses = []
############################################################################  delete
global count
count = [0]
global controller
controller = [None]

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
        count[0] = count[0] + 1
        start_mission(img,depth_img,pose)

    return HttpResponse("get image")


def start_mission(img,depth_img,pose):
    
    if len(imgs) < 4:
        imgs.append(img)
        depth_imgs.append(depth_img)
        poses.append(pose)
    # elif len(imgs) == 4:
    #     running_flag[0] = True
    #     threading.Thread(target=controller.train,args=[]).start()
    #     controller.add_img(img,depth_img,pose)      
    else:
        if running_flag[0]:
            if count[0]%4 == 0:
                # threading.Thread(target=controller[0].add_img,args=(img,depth_img,pose,False)).start()
                controller[0].add_img(img,depth_img,pose,False)
            else:
                # threading.Thread(target=controller[0].add_img,args=(img,depth_img,pose,False)).start()
                controller[0].add_img(img,depth_img,pose,False)
            # controller[0].add_img(img,depth_img,pose)
        else:
            controller[0] = Controller(H,W,focal,imgs,depth_imgs,poses)
            controller[0].add_img(img,depth_img,pose,False)
            threading.Thread(target=controller[0].train,args=()).start()
            # threading.Thread(target=controller[0].add_img,args=(img,depth_img,pose,False)).start()
            # controller[0].add_img(img,depth_img,pose)
            running_flag[0] = True
        

def terminate_mission(request):
    controller[0].terminate_work()
    running_flag[0] = False
    return HttpResponse("terminate successfully")

def save_model(request):
    controller[0].save_model()
    return HttpResponse("save successfully")

def get_uncertainty(request):
    if request.method == 'POST':
        received_json_data = json.loads(request.body)
        locations = received_json_data['locations']
        us = received_json_data['us']
        vs = received_json_data['vs']
        poses = []
        # print("parse finished")
        for i in range(len(locations)):

            pose = get_pose(locations[i],us[i],vs[i])
            poses.append(pose)
        
        # uncer,avg_distance,ratio = controller[0].get_uncertainty(poses)
        uncers = controller[0].get_uncertainty(poses)
        # print("get finished")
        print(uncers)
        re = {"uncers":uncers.cpu().numpy().tolist()}
        # re = str(uncer.item()) + "," + str(avg_distance.item()) + "," + str(ratio)
        return JsonResponse(re)
    
def get_surface_points(request):
    if request.method == 'GET':
        x = float(request.GET['x'])
        y = float(request.GET["y"])
        z = float(request.GET["z"])
        radius = float(request.GET["radius"])
        step = float(request.GET["step"])
        # print(x,y,z,radius,step)
        points = controller[0].get_surface_points([x,y,z],radius,step)
        re = {"points":points.cpu().numpy().tolist()}
        # print(re)
        return JsonResponse(re)

def get_ray_depth(request):
    print("here")
    print(request.body)
    
    if request.method == 'POST':
        # print("parse data")
        received_json_data = json.loads(request.body)
        locations = received_json_data['locations']
        directions = received_json_data['directions']
        depths = controller[0].get_rays_depth(locations,directions)
        re = {'depths':depths.cpu().numpy().tolist()}
        return JsonResponse(re)

def get_pose(location,u,v):
    sx = np.sin(u)
    cx = np.cos(u)
    sy = np.sin(v)
    cy = np.cos(v)
    return [[cy, sy*sx, -sy*cx, location[0]],
                        [0, cx, sx, location[1]],
                        [-sy, cy*sx, -cy*cx, location[2]],
                        [0,0,0,1]]


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
