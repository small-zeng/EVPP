import threading
import requests
import numpy as np
import sys
import json
import torch
from torch.functional import Tensor
import time


device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

#### 请求视点延视线方向到无图的距离
#### location nparray [x,y,z]
#### direction nparray vector 3 [x,y,z]
#### return distance float 
# def get_distance(location,direction):
#     response = requests.get("http://localhost:8080/get_distance", "location="+str(location)+"&direction="+str(direction))
#     print(response.status_code)
#     print(response.text)
#     return float(response.text)

# def send_NBV(location,u,v):
#     u = u/np.pi*180.
#     v = v/np.pi*180.
#     response = requests.get("http://10.13.21.208:6200/", "x="+str(location[0])+"&y="+str(location[1])+"&z="+str(location[2])+"&u="+str(u)+"&v="+str(v))
#     nbv = {"x":str(location[0]),"y":str(location[1]),str(location[2]):"0",str(u):"0","v":str(v)}
#     #response = requests.post("http://10.15.196.86:9200/get_nbv/", data=json.dumps(nbv, ensure_ascii=False))
#     return response.text

def send_NBV(location,u,v):
    u = u/np.pi*180.
    v = v/np.pi*180.
    response = requests.get("http://10.13.21.209:6200/", "x="+str(location[0])+"&y="+str(location[1])+"&z="+str(location[2])+"&u="+str(u)+"&v="+str(v))
    nbv = {"x":str(location[0]),"y":str(location[1]),str(location[2]):"0",str(u):"0","v":str(v)}
    #response = requests.post("http://10.15.196.86:9200/get_nbv/", data=json.dumps(nbv, ensure_ascii=False))
    return response.text


def send_Path(path):
    data = {'path':path}
    headers = {'Content-Type': 'application/json'}
    response = requests.post("http://10.13.21.209:7300/get_path/", headers= headers,data=json.dumps(data))
    return response.text

def creat_window():
    data = {'path':"window"}
    headers = {'Content-Type': 'application/json'}
    response = requests.post("http://10.13.21.208:7300/get_window/", headers= headers,data=json.dumps(data))
    return response.text



#### 给定空间点及观测方向下的整幅图像的uncertainty
#### location nparray [x,y,z]
#### direction nparray vector 3 [x,y,z]
#### return uncertainty which normnized uncer of whole image
#### return distance float average(reached)
#### return ratio reached/unreachale float
def get_uncertainty(locations,us,vs):
    data = {'locations':locations,'us':us,'vs':vs}
    headers = {'Content-Type': 'application/json'}
    response = requests.post("http://localhost:6000/get_uncertainty/", headers= headers,data=json.dumps(data))
    re = response.json()
    # print(response.status_code)
    # print(response.text)
    # distances = re['distances']
    uncertaintys = re['uncers']
    # ratios = re['ratios']
    # distances = torch.Tensor(distances)
    # uncertaintys = torch.Tensor(uncertaintys)
    distances = []
    ratios = []
    return uncertaintys,distances,ratios

#### 给定空间点及观测方向下的整幅图像的uncertainty
#### location nparray [x,y,z]
#### radius float/int 
#### return a set of points 
def get_surface_points(location,radius,step):
    response = requests.get("http://localhost:7000/get_surface_points", "x="+str(location[0])+"&y="+str(location[1])+"&z="+str(location[2])+"&radius="+str(radius)+"&step="+str(step))
    # print(response.status_code)
    re = response.json()
    # print(re['points'])
    # print(re['points'].type)
    ## need to complete
    points = torch.Tensor(re['points'])
    return points

def get_ray_depth(locations,directions):
    data = {'locations':locations,'directions':directions}
    headers = {'Content-Type': 'application/json'}
    response = requests.post("http://localhost:7000/get_ray_depth", headers= headers,data=json.dumps(data))
    re = response.json()
    depths = re['depths']
    # print(depths)
    depths = torch.Tensor(depths)
    return depths

#### 给定起点终点，判断是否能够到达
#### begin nparray [x,y,z]
#### begin nparray [x,y,z]
#### return reachabe 1--yes -1--no 0--error
def is_reachable(begin,end):
    response = requests.get("http://localhost:8080/is_reachable", "begin="+str(begin)+"&end="+str(end))
    # print(response.status_code)
    # print(response.text)
    ## need to complete
    if response.text == 'yes':
        return 1
    if response.text == 'no':
        return -1
    return 0


if __name__=='__main__':
    # creat_window()
    # time.sleep(5)
    path = [[0,0,5],[0,0,6]]
    # send_NBV([1,2,3],1,2])
    send_Path(path)