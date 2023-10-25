"""Fuse 1000 RGB-D images from the 7-scenes dataset into a TSDF voxel volume with 2cm resolution.
"""

import time

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import open3d
import torch
import math
from imageio import *
from scipy.spatial import ConvexHull
import copy
import core.fusion as fusion

if not os.path.exists("saved_model"):
    os.mkdir("saved_model")
    
# ## IdahoStateCapitol
# version = 'v30_7'
# vol_bnds = np.array([[-6,6],[-0.1,4.0],[-6,6]])        ## 采样包围盒
# aabb_bnds = np.array([[-3,3],[-0.1,3.0],[-1.5,1.5]])   ## tsdf uncertainty包围盒
# print("采样包围盒；\n",vol_bnds)
# print("tsdf uncertainty包围盒: \n",aabb_bnds)

## cabin
version = 'v23_132'
vol_bnds = np.array([[-5,5],[0.01,5.0],[-5,6]])        ## 采样包围盒
aabb_bnds = np.array([[-1.0,1.0],[0.02,2.5],[-1.5,2.0]])   ## tsdf uncertainty包围盒
aoi_bnds = np.array([[-2.5,2.5],[0.1,3.5],[-2.5,3.0]]) 
print("采样包围盒；\n",vol_bnds)
print("tsdf uncertainty包围盒: \n",aabb_bnds)


device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
H = 400; W =400;K =np.array([[300,0,200],[0,300,200],[0,0,1]])
near = 0.5; far = 6.0
voxel_res = 0.1
N_samples = int((far - near)/voxel_res) + 1

# ======================================================================================================== #
# (Optional) This is an example of how to compute the 3D bounds
# in world coordinates of the convex hull of all camera view
# frustums in the dataset
# ======================================================================================================== #
# print("Estimating voxel volume bounds...")
base_dir = "../nerfServer/logs/unity_continue_depth_cabin_" + version + "/trainset"
# base_dir = "../tsdfServer/logs/unity_continue_depth_cabin_" + version + "/trainset"

train_data_far = 6.0
n_imgs = 5
noise = 0.00
cam_intr = np.array([[300,0,200],[0,300,200],[0,0,1]])

all_imgs_set = set()
add_imgs_set = set()
print("imgs_set: ",all_imgs_set)

coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H).to(device), torch.linspace(0, W-1, W).to(device)), -1)  # (H, W, 2)
coords = torch.reshape(coords, [-1, 2])  # (H * W, 2)

depth_coords = torch.stack(torch.meshgrid(torch.linspace(100, 300-1, 200).to(device), torch.linspace(100, 300-1, 200).to(device)), -1)  # (H, W, 2)
depth_coords = torch.reshape(coords, [-1, 2])  # (H * W, 2)

def isectSegAABB_gpu(rays_o, rays_d, amin, amax, tmin =0.0 ,  tmax =10000.0):
    ray_num = rays_o.shape[0]
    rays_o = rays_o.reshape(-1,3)
    rays_d = rays_d.reshape(-1,3)
    EPS = 1e-6
    amin = torch.tensor(amin).to(device)
    amax = torch.tensor(amax).to(device)
    view_bound = torch.ones(ray_num,2).to(device)*tmin
    view_bound[:,1] = tmax
    t1,t2,tmp = view_bound[:,0].clone(),view_bound[:,1].clone(),view_bound[:,0].clone()
    is_insert = torch.ones(ray_num).to(device)
    ## 分别与三个轴的两个分面求交点
    for i in range(3):
        # print("i = ",i)
        ## 如果某个轴的分量在两个点间没变，则说明与这个轴垂直，只需要判断是否在AABB盒内部，否则不可能相交
        # parallel2axis_index = torch.nonzero(torch.logical_and(torch.abs(rays_d[:,i])<EPS,torch.logical_and(rays_o[:,i]>=amin[i],rays_o[:,i]<=amax[i])))

        valid_index =  torch.nonzero(torch.abs(rays_d[:,i])>=EPS)[:,0].cpu()
        valid_ood = torch.div(1.0 , rays_d[valid_index,i])

        t1[valid_index] = (amin[i] -  rays_o[valid_index,i]) * valid_ood
        t2[valid_index] = (amax[i] -  rays_o[valid_index,i]) * valid_ood

        t1_over_t2_index =  torch.nonzero(torch.logical_and(torch.abs(rays_d[:,i])>=EPS,t1>t2))[:,0].cpu()
        tmp[t1_over_t2_index] = (t1[t1_over_t2_index]).clone()
        t1[t1_over_t2_index] = (t2[t1_over_t2_index]).clone()
        t2[t1_over_t2_index] = (tmp[t1_over_t2_index]).clone()
        # print("t1_over_t2_index = ",t1_over_t2_index)

        t1_over_tmin_index =  torch.nonzero(torch.logical_and(torch.abs(rays_d[:,i])>=EPS,t1>view_bound[:,0]))[:,0].cpu()
        view_bound[t1_over_tmin_index,0] = torch.clone(t1[t1_over_tmin_index])
        # print("t1_over_tmin_index = ",t1_over_tmin_index)

        t2_lower_tmax_index = torch.nonzero(torch.logical_and(torch.abs(rays_d[:,i])>=EPS,t2<view_bound[:,1]))[:,0].cpu()
        view_bound[t2_lower_tmax_index,1] = torch.clone(t2[t2_lower_tmax_index])
        # print("t1_over_tmin_index = ",t1_over_tmin_index)
        
        tmin_over_tmax_index = torch.nonzero(torch.logical_and(torch.abs(rays_d[:,i])>=EPS,view_bound[:,0]>view_bound[:,1]))[:,0].cpu()
        # print("tmin_over_tmin_index = ",torch.clone(tmin_over_tmax_index))
        is_insert[tmin_over_tmax_index] = 0

        # print(valid_index)

    # ## 不相交设置为[tmin,tmax]
    # not_insert_index = torch.nonzero(is_insert==0).cpu().numpy()
    # view_bound[(not_insert_index.T).tolist()] = torch.tensor([tmin,tmax]).to(device)

    return is_insert,view_bound


class TSDF:

    def __init__(self,ymin,ymax,object_center):

        self.ymin = ymin
        self.ymax = ymax
        self.object_center = object_center

        # Initialize voxel volume
        print("Initializing voxel volume...")
        self.tsdf_vol = fusion.TSDFVolume(vol_bnds, voxel_size=voxel_res, near= near,far =far,use_gpu=True)
        self.index_bnd = self.tsdf_vol._vol_dim
        ## 用于计算voxel索引
        self.vol_origin = torch.tensor(self.tsdf_vol._vol_origin).to(device)
        self.voxel_size = torch.tensor(self.tsdf_vol._voxel_size).to(device)

        # 统计所有voxel的状态，及表面voxel的 Nray(其他voxel Nray=0),color
        self.State_vol_origin = torch.zeros(self.tsdf_vol._tsdf_vol_gpu.shape).to(device)
        self.State_vol = torch.zeros(self.tsdf_vol._tsdf_vol_gpu.shape).to(device)
        self.Nray_vol = torch.zeros(self.tsdf_vol._tsdf_vol_gpu.shape).to(device)
        print(self.State_vol.shape)
        self.Color_vol = torch.ones(self.State_vol.shape[0],self.State_vol.shape[1],self.State_vol.shape[2],3).to(device)*255
        self.front_verts = torch.zeros(0,3).to(device)
        self.vol_bnds = vol_bnds

    ## 是否在tsdf框内
    def is_bounded(self,index):
        if 0<index[0,0]<self.index_bnd[0] and 0< index[1,0]<self.index_bnd[1] and  0<index[2,0]<self.index_bnd[2]:
            return True

        return False

    def get_pose(self,location,u,v):
        sx = np.sin(u)
        cx = np.cos(u)
        sy = np.sin(v)
        cy = np.cos(v)
        return [[cy, sy*sx, -sy*cx, location[0]],
                            [0, cx, sx, location[1]],
                            [-sy, cy*sx, -cy*cx, location[2]],
                            [0,0,0,1]]

    # Ray helpers
    def get_rays(self,H, W, K, c2w):
        i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
        i = i.t()
        j = j.t()
        dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1).to(device)
        # Rotate ray directions from camera frame to the world frame
        rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
        # Translate camera frame's origin to the world frame. It is the origin of all rays.
        rays_o = c2w[:3,-1].expand(rays_d.shape)
        return rays_o, rays_d
    
    def get_rays_test(self,H, W, K, c2w):
        i, j = torch.meshgrid(torch.linspace(-15, 345, W), torch.linspace(0, 400-1, H))  # pytorch's meshgrid has indexing='ij'
        i = i.t()
        j = j.t()
        i = i -180
        rays_d = torch.stack([torch.sin(i/180*math.pi), -(j-K[1][2])/K[1][1], torch.cos(i/180*math.pi)], -1).to(device)
        rays_d = torch.sum(rays_d[..., np.newaxis, :] * c2w[:3,:3], -1)
        # Translate camera frame's origin to the world frame. It is the origin of all rays.
        rays_o = c2w[:3,-1].expand(rays_d.shape)
        return rays_o, rays_d
    
    

    def batchify_rays(self,rays_flat, chunk=1024*32):
        """Render rays in smaller minibatches to avoid OOM.
        """
        all_ret = {}
        for i in range(0, rays_flat.shape[0], chunk):
            ret = self.render_rays(rays_flat[i:i+chunk])
            for k in ret:
                if k not in all_ret:
                    all_ret[k] = []
                all_ret[k].append(ret[k])

        all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
        return all_ret

    def render_rays(self,ray_batch):
        
        N_rays = ray_batch.shape[0]
        rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] # [N_rays, 3] each
        bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2])
        Near, Far = bounds[...,0], bounds[...,1] # [-1,1]

        t_vals = torch.linspace(0., 1., steps=N_samples).to(device)
        z_vals = Near * (1.-t_vals) + Far * (t_vals)
        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]
        

        pts_index = torch.cat(((pts[:,:,0])[:,:,None],pts[:,:,1:3]), axis = -1)  # [N_rays, N_samples, 3]
        pts_voxl_index =  torch.floor((pts_index-self.vol_origin)/self.voxel_size)
        pts_voxl_index = torch.flatten(pts_voxl_index,start_dim=0, end_dim=1)
        
        # print("before: ")
        # print(torch.min(pts_voxl_index[:,0]),torch.max(pts_voxl_index[:,0]))
        # print(torch.min(pts_voxl_index[:,1]),torch.max(pts_voxl_index[:,1]))
        # print(torch.min(pts_voxl_index[:,2]),torch.max(pts_voxl_index[:,2]))

        ##  越界则置为边界
        t0 = time.time()
        pts_voxl_index_x = pts_voxl_index[:,0].reshape(pts_voxl_index.shape[0],1)
        bound_max_x = torch.zeros(pts_voxl_index.shape[0],1).to(device)
        bound_min_x = torch.zeros(pts_voxl_index.shape[0],1).to(device)

        pts_voxl_index_y = pts_voxl_index[:,1].reshape(pts_voxl_index.shape[0],1)
        bound_max_y = torch.zeros(pts_voxl_index.shape[0],1).to(device)
        bound_min_y = torch.zeros(pts_voxl_index.shape[0],1).to(device)

        pts_voxl_index_z = pts_voxl_index[:,2].reshape(pts_voxl_index.shape[0],1)
        bound_max_z = torch.zeros(pts_voxl_index.shape[0],1).to(device)
        bound_min_z = torch.zeros(pts_voxl_index.shape[0],1).to(device)
      
        pts_voxl_index_x = torch.where(torch.logical_or(pts_voxl_index_x < 0,torch.logical_or(pts_voxl_index_y<0,
                            pts_voxl_index_z<0)),bound_min_x, pts_voxl_index_x)
        pts_voxl_index_y = torch.where(torch.logical_or(pts_voxl_index_x < 0,torch.logical_or(pts_voxl_index_y<0,
                            pts_voxl_index_z<0)),bound_min_y, pts_voxl_index_y)
        pts_voxl_index_z = torch.where(torch.logical_or(pts_voxl_index_x < 0,torch.logical_or(pts_voxl_index_y<0,
                            pts_voxl_index_z<0)),bound_min_z, pts_voxl_index_z)
    
        pts_voxl_index_x = torch.where(torch.logical_or(pts_voxl_index_x >=self.index_bnd[0]-1,torch.logical_or(pts_voxl_index_y
                           >=self.index_bnd[1]-1,pts_voxl_index_z>=self.index_bnd[2]-1)),bound_max_x, pts_voxl_index_x)
        pts_voxl_index_y = torch.where(torch.logical_or(pts_voxl_index_x >=self.index_bnd[0]-1,torch.logical_or(pts_voxl_index_y
                           >=self.index_bnd[1]-1,pts_voxl_index_z>=self.index_bnd[2]-1)),bound_max_y, pts_voxl_index_y)
        pts_voxl_index_z = torch.where(torch.logical_or(pts_voxl_index_x >=self.index_bnd[0]-1,torch.logical_or(pts_voxl_index_y
                           >=self.index_bnd[1]-1,pts_voxl_index_z>=self.index_bnd[2]-1)),bound_max_z, pts_voxl_index_z)

        pts_voxl_index = torch.cat((pts_voxl_index_x,pts_voxl_index_y,pts_voxl_index_z), axis = -1)
        # print("边界筛选用时 ", time.time() - t0)

        # print("after: ")
        # print(torch.min(pts_voxl_index[:,0]),torch.max(pts_voxl_index[:,0]))
        # print(torch.min(pts_voxl_index[:,1]),torch.max(pts_voxl_index[:,1]))
        # print(torch.min(pts_voxl_index[:,2]),torch.max(pts_voxl_index[:,2]))
        

        pts_voxl_index = pts_voxl_index.T.cpu().numpy()
        pts_voxl_state = self.State_vol_origin[pts_voxl_index].reshape(pts.shape[0],pts.shape[1])
        pts_voxl_Nray = self.Nray_vol[pts_voxl_index].reshape(pts.shape[0],pts.shape[1])
        pts_voxl_Color = self.Color_vol[pts_voxl_index].reshape(pts.shape[0],pts.shape[1],3)
        
        # print(pts_voxl_index.shape, pts_voxl_state.shape, pts_voxl_Nray.shape)

        
        ######################   pytorch implementation ####################

        N = pts_voxl_state.shape[0]
        rgb_map = torch.ones(N,3).to(device)*255
        depth_map = torch.ones(N).to(device)*far
        uncer_map = torch.zeros(N).to(device)

        pts_voxl_state_density =  torch.ones(pts_voxl_state.shape).to(device)
        pts_voxl_state_density[pts_voxl_state==0] = 0
        pts_voxl_state_density[pts_voxl_state==1] = 0
        
        ## 射线朝向未知区域或者表面(粗处理，包含了与物体相交部分射线)
        pts_voxl_state_buff =  torch.zeros(pts_voxl_state.shape).to(device)
        unknown_index = torch.nonzero(pts_voxl_state==0).cpu().numpy()
        pts_voxl_state_buff[unknown_index.T] = 1
        rays_unkown_num = torch.sum(pts_voxl_state_buff,axis = 1)
        rays_unkown_index = torch.nonzero(rays_unkown_num)[:,0].cpu().numpy()
        uncer_map[rays_unkown_index] = 0.2*(rays_unkown_num[rays_unkown_index]/N_samples)
        uncer_map[uncer_map>1]=1
        
        ## 射线与物体相交（表面细处理）
        pts_voxl_state_buff =  torch.ones(pts_voxl_state.shape).to(device)*10
        occ_index = torch.nonzero(pts_voxl_state==2).cpu().numpy()
        pts_voxl_state_buff[occ_index.T] = 3
        occ_pt_index = torch.min(pts_voxl_state_buff,1)[1].cpu().numpy().astype(np.int32)
        occ_ray_index = np.nonzero(occ_pt_index)[0]
        occ_pt_index = occ_pt_index[occ_ray_index]
        occ_surface_index = np.concatenate([occ_ray_index[np.newaxis,:],occ_pt_index[np.newaxis,:]],axis = 0).tolist()
        rgb_map[occ_ray_index] = pts_voxl_Color[occ_surface_index]
        depth_map[occ_ray_index] = torch.tensor(near + occ_pt_index/N_samples *(far-near)).to(device).float()
       
        is_insert,view_bound= isectSegAABB_gpu(rays_o,rays_d,aabb_bnds[:,0],aabb_bnds[:,1])
        view_bound_sign = view_bound[:,0]*view_bound[:,1]
        insert_invalid_index = torch.nonzero (is_insert==0).cpu().numpy()
        uncer_map[occ_ray_index] = torch.div(1.0 , 1.0 + 0.1*((depth_map[occ_ray_index]))*pts_voxl_Nray[occ_surface_index])
        uncer_map[(insert_invalid_index.T).tolist()] = 1e-10
        depth_map[(insert_invalid_index.T).tolist()] = far
        
        
        # is_insert,view_bound= isectSegAABB_gpu(rays_o,rays_d,aabb_bnds[:,0],aabb_bnds[:,1])
        # insert_invalid_index = torch.nonzero (is_insert==0).cpu().numpy()
        # uncer_map[(insert_invalid_index.T).tolist()] = 1e-10

       
        ######################################################################

        ######################   numpy implementation ########################
        # pts_voxl_state_arr = pts_voxl_state.cpu().numpy()
        # rgb = []
        # depth = []
        # uncer = []

        # for i in range(pts_voxl_state_arr.shape[0]):
        #     occ_index = np.argwhere(pts_voxl_state_arr[i]==2)
        #     unknown_index = np.argwhere(pts_voxl_state_arr[i]==0)
        #     empty_index = np.argwhere(pts_voxl_state_arr[i]==1)
        #     occ_index = occ_index.reshape(occ_index.shape[0])
        #     unknown_index = unknown_index.reshape(unknown_index.shape[0])
        #     empty_index = empty_index.reshape(empty_index.shape[0])
            
        #     ## unkown voxel 
        #     if occ_index.shape[0] == 0:
                
        #         depth.append(far)
        #         rgb.append([255,255,255])
        #         ratio = unknown_index.shape[0]/N_samples
        #         # print(ratio)
        #         value = 2* (ratio**2)
        #         if value >1:
        #             uncer.append(1)
        #         else:
        #             uncer.append(value)

        #     else:
        #         d = near + occ_index[0]/N_samples *(far-near)
        #         if d > far:
        #             print(occ_index[0],d)
        #         Nray = pts_voxl_Nray[i,occ_index[0]]
        #         uncer.append(1/(1+ 0.1*(d**2)*Nray))
        #         depth.append(d)
        #         rgb.append(pts_voxl_Color[i,occ_index[0]].cpu().numpy().tolist())

        # rgb_map = torch.tensor(rgb).to(device).reshape(pts_voxl_state_arr.shape[0],3)
        # depth_map = torch.tensor(depth).to(device).reshape(pts_voxl_state_arr.shape[0],1)
        # uncer_map = torch.tensor(uncer).to(device).reshape(pts_voxl_state_arr.shape[0],1)

        ######################################################################
        
        ret = {'rgb_map' : rgb_map, 'depth_map' : depth_map, 'uncer_map' : uncer_map}

        return ret

    def render(self,H, W, K, chunk=1024*32, rays=None):
        ret_list = []

        rays_o, rays_d = rays
        viewdirs = rays_d
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1,3]).float()

        # Create ray batch
        rays_o = torch.reshape(rays_o, [-1,3]).float()
        rays_d = torch.reshape(rays_d, [-1,3]).float()
        Near, Far = near * torch.ones_like(rays_d[...,:1]).to(device), far * torch.ones_like(rays_d[...,:1]).to(device)
        rays = torch.cat([rays_o, rays_d, Near, Far], -1)

        sh = rays_d.shape # [..., 3]
        
        # Render and reshape
        all_ret = self.batchify_rays(rays)
        for k in all_ret:
            k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
            all_ret[k] = torch.reshape(all_ret[k], k_sh)

        k_extract = ['rgb_map', 'depth_map', 'uncer_map']
        ret_list = [all_ret[k] for k in k_extract]
        ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}

        return ret_list + [ret_dict]

    def get_all_uncertainty(self,location,u,v,n):
        pose = self.get_pose(location,u,v)
        # pose = [[-0.99985,	0.00000,	0.01745,	0.00000],
        #         [0.00000,	1.00000,	0.00000,	1.00000],
        #         [0.01745,	0.00000,	0.99985,	0.00000],
        #         [0.00000,	0.00000,	0.00000,	1.00000]]
        # pose = [[-0.19081,-0.05137,-0.98028,-2.94085],[0,0.99863,-0.05234,0.84299],[-0.98163,0.00999,0.19055,0.57164],[0,0,0,1]]
        uncer = 0.0
        # print(pose.shape)

        pose = torch.tensor(pose).to(device)
        pose = pose[:3, :4]
        rays_o, rays_d = self.get_rays(H, W, K, pose) 
        # Create ray batch
        rays_o = torch.reshape(rays_o, [-1,3]).float()
        rays_d = torch.reshape(rays_d, [-1,3]).float()
        batch_rays = torch.stack([rays_o, rays_d], 0).to(device)
        rgb_map, depth_map, uncer_map, _ = self.render(H, W, K, rays=batch_rays)
        rgbs = rgb_map.cpu().numpy()
        rgbs = rgbs.reshape((400,400,3))
        depths = depth_map.cpu().numpy()
        depths = depths.reshape((400,400))
        uncers = uncer_map.cpu().numpy()
        uncers = uncers.reshape((400,400))

        print(rgbs[200,  200])
        # rgb32 = (np.clip(rgbs,0,255)).astype(np.uint32)
        rgbs = rgbs.astype(np.uint8)
        filename_img = os.path.join("dataset", 'rgb_{:03d}_{}.png'.format(0,n))
        imwrite(filename_img, rgbs)
        
        print(depths[200,  200])
        depths = depths/far*255
        rgb8 = (np.clip(depths,0,255)).astype(np.uint8)
        filename_img = os.path.join("dataset", 'depth_{:03d}_{}.png'.format(0,n))
        imwrite(filename_img, rgb8)

        # print(uncers[200, :])
        uncer = sum(map(sum,uncers)) /(400*400)
        uncers = 255-uncers/1.0*255
        rgb8 = (np.clip(uncers,0,255)).astype(np.uint8)
        filename_img = os.path.join("dataset", 'uncer_{:03d}_{}.png'.format(0,n))
        imwrite(filename_img, rgb8)
        
        return uncer
    
    
    def get_all_uncertainty_test(self,location,u,v,H=400 ,W=360,n =0):
        
        pose = self.get_pose(location,u,v)
        uncer = 0.0
        # print(pose.shape)

        pose = torch.tensor(pose).to(device)
        pose = pose[:3, :4]
        rays_o, rays_d = self.get_rays_test(H, W, K, pose) 
        # Create ray batch
        rays_o = torch.reshape(rays_o, [-1,3]).float()
        rays_d = torch.reshape(rays_d, [-1,3]).float()
        batch_rays = torch.stack([rays_o, rays_d], 0).to(device)
        rgb_map, depth_map, uncer_map, _ = self.render(H, W, K, rays=batch_rays)
        rgbs = rgb_map.cpu().numpy()
        rgbs = rgbs.reshape((H,W,3))
        depths = depth_map.cpu().numpy()
        depths = depths.reshape((H,W))
        uncers = uncer_map.cpu().numpy()
        uncers = uncers.reshape((H,W))

        
        # rgb32 = (np.clip(rgbs,0,255)).astype(np.uint32)
        rgbs = rgbs.astype(np.uint8)
        filename_img = os.path.join("dataset", 'rgb_{:03d}_{}.png'.format(0,n))
        imwrite(filename_img, rgbs)
        
        
        depths = depths/far*255
        rgb8 = (np.clip(depths,0,255)).astype(np.uint8)
        filename_img = os.path.join("dataset", 'depth_{:03d}_{}.png'.format(0,n))
        imwrite(filename_img, rgb8)

        uncer = sum(map(sum,uncers)) /(H*W)
        uncers = 255-uncers/1.0*255
        rgb8 = (np.clip(uncers,0,255)).astype(np.uint8)
        filename_img = os.path.join("dataset", 'uncer_{:03d}_{}.png'.format(0,n))
        imwrite(filename_img, rgb8)
        
        return uncer


    def get_uncertainty_tsdf(self,locations,us,vs):
        uncer_list =[]
        sample_num = 500
        poses = []
        for i in range(len(locations)):
            pose = self.get_pose(locations[i],us[i],vs[i])
            poses.append(pose)
        # print(pose.shape)
        
        view_num = len(poses)
        
        poses = torch.Tensor(poses).to(device)
        poses = poses[:,:3, :4]
        all_rays_o = torch.zeros((0,3)).to(device)
        all_rays_d = torch.zeros((0,3)).to(device)
        
        t1 = time.time()
        i = 0
        for pose in poses: 
            # ts = time.time()
            rays_o_img, rays_d_img = self.get_rays(H, W, K, pose)  # (H, W, 3), (H, W, 3)
            # print("ts = ",time.time()-ts)
            select_inds = np.random.choice(coords.shape[0], size=[sample_num], replace=False)  # (N_rand,)
            # select_depth_inds = np.random.choice(depth_coords.shape[0], size=[100], replace=False)  # (N_rand,)
            # select_inds = np.concatenate((select_inds,select_depth_inds),axis = -1)
            select_coords = coords[select_inds].long()  # (N_rand, 2)
            rays_o = rays_o_img[select_coords[:, 0], select_coords[:, 1]].clone()  # (N_rand, 3)
            rays_d = rays_d_img[select_coords[:, 0], select_coords[:, 1]].clone()  # (N_rand, 3)
            # print(len(rays_o))
            all_rays_o = torch.cat([all_rays_o, rays_o], 0)
            all_rays_d = torch.cat([all_rays_d, rays_d], 0)
            # print(len(all_rays_o))

            batch_rays = torch.stack([all_rays_o, all_rays_d], 0).to(device)
            # print(len(batch_rays), batch_rays.shape)
            i = i +1
        
        print("t1 = ",time.time()-t1)
        rgb_map, depth_map, uncer_map, _ = self.render(H, W, K, rays=batch_rays)
        print("t1 = ",time.time()-t1)
        uncers = uncer_map.cpu().numpy()
        uncers = uncers.reshape((view_num,sample_num))
        uncers = np.mean(uncers[:,0:sample_num],axis = 1)
        uncer_list = uncers.tolist()

        depths = depth_map.cpu().numpy()
        depths = depths.reshape((view_num,sample_num))
        depth_list = []
        for i in range(view_num):
            select_view_depth = depths[i].copy()
            select_view_depth =  select_view_depth[select_view_depth<far]
            select_view_depth = select_view_depth[select_view_depth>near]
            if select_view_depth.shape[0] == 0:
                depth_list.append(far)
            else:
                depth = np.mean(select_view_depth,axis = 0)
                depth_list.append(depth)
        return uncer_list,depth_list
    
    def get_uncertainty_tsdf_test(self,locations,H=100 ,W=100):
        uncer_list =[]
        poses = []
        for i in range(len(locations)):
            pose = self.get_pose(locations[i],0,0)
            poses.append(pose)
        # print(pose.shape)
        
        view_num = len(poses)
        
        poses = torch.Tensor(poses).to(device)
        poses = poses[:,:3, :4]
        all_rays_o = torch.zeros((0,3)).to(device)
        all_rays_d = torch.zeros((0,3)).to(device)
        
        t1 = time.time()
        i = 0
        for pose in poses: 
            # ts = time.time()
            rays_o_img, rays_d_img = self.get_rays_test(H, W, K, pose)  # (H, W, 3), (H, W, 3)
            # print("ts = ",time.time()-ts)
            rays_o = rays_o_img.clone().reshape(H*W,3)  # (N_rand, 3)
            rays_d = rays_d_img.clone().reshape(H*W,3)  # (N_rand, 3)
            # print(len(rays_o))
            all_rays_o = torch.cat([all_rays_o, rays_o], 0)
            all_rays_d = torch.cat([all_rays_d, rays_d], 0)
            # print(len(all_rays_o))

            batch_rays = torch.stack([all_rays_o, all_rays_d], 0).to(device)
            # print(len(batch_rays), batch_rays.shape)
            i = i +1
        
        print("t1 = ",time.time()-t1)
        rgb_map, depth_map, uncer_map, _ = self.render(H, W, K, rays=batch_rays)
        print("t1 = ",time.time()-t1)
        uncers = uncer_map.reshape((view_num,H,W))
        uncers = uncers.transpose(1,2)
        uncers = uncers.reshape((view_num,12,int(W/12*H)))
        uncers = uncers.cpu().numpy()
        uncers = np.mean(uncers,axis = 2)
        uncer_list = uncers.tolist()
        

        view_w = int(67.38013/360*W/2)
        print("view_w = ",view_w)
        depths = depth_map.reshape((view_num,H,W))
        depths = depths.transpose(1,2).cpu().numpy()
        print("depths = ",depths.shape)
        
        # depths = depths.reshape((view_num,12,int(W/12*H)))
        depth_list = []
        ratio_list = []
        var_list = []
        for i in range(depths.shape[0]):
            depth_list.append([])
            ratio_list.append([])
            var_list.append([])
            for j in range(12):
                start = int(((-15 + j*30)/360)*W) - view_w
                end = int(((-15 + j*30)/360)*W) + view_w
                if start < 0:
                    view_depth = np.concatenate((depths[i,start::],depths[i,0:end]),axis = 0)
                else :
                    view_depth = depths[i,start:end]
                # print(start,end,view_depth.shape)
                select_view_depth = view_depth[view_depth<far]
                select_view_depth = select_view_depth[select_view_depth>near]
                if select_view_depth.shape[0] == 0:
                    depth_list[i].append(0.0)
                    ratio_list[i].append(0.0)
                    var_list[i].append(10.0)
                else:
                    depth = np.mean(select_view_depth)
                    depth_list[i].append(depth)
                    var_list[i].append(np.maximum(0.001,np.var(select_view_depth)))
                    select_view_depth = view_depth[view_depth<4.0]
                    select_view_depth = select_view_depth[select_view_depth>2.0]
                    if select_view_depth.shape[0] == 0:
                        ratio_list[i].append(0.0)
                    else:
                        ratio_list[i].append(select_view_depth.shape[0]/view_depth.shape[0])
        # uncer_list = np.array(uncer_list)/(np.array(var_list)**0.5)
        # print(np.array(var_list))
        return uncer_list,depth_list,ratio_list,var_list
    
    

    ##  通过与栅格地图相交判断/筛选采样视角方向
    def get_tsdf_selectdir(self,location,u,v):
        
        pose = self.get_pose(location,u,v)
        pose = torch.Tensor(pose).to(device)   
        pose = pose[:3, :4]
        all_rays_o = torch.zeros((0,3)).to(device)
        all_rays_d = torch.zeros((0,3)).to(device)
        rays_o, rays_d = self.get_rays(H, W, K, pose)  # (H, W, 3), (H, W, 3)
        select_inds = np.random.choice(coords.shape[0], size=[1000], replace=False)  # (N_rand,)
        select_coords = coords[select_inds].long()  # (N_rand, 2)
        rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
        rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)

        pass


    def tsdf_test_1(self):
        # Loop through RGB-D images and fuse them together
        t0_elapse = time.time()

        for i in range(n_imgs):
            print("Fusing frame %d/%d"%(i, n_imgs))

            # Read RGB-D image and camera pose
            color_image = cv2.cvtColor(cv2.imread(os.path.join(base_dir,"%dmain.png"%(i))), cv2.COLOR_BGR2RGB)[::2,::2]
            depth_im = cv2.imread(os.path.join(base_dir,"%ddepth.png"%(i)),-1).astype(float)[::2,::2,0]
            cam_pose = np.loadtxt(os.path.join(base_dir,"%d.txt"%(i)))

            # print(depth_im)
            depth_im = train_data_far *depth_im/255.0
            depth_im[depth_im == train_data_far] = 0
            depth_im += np.random.randn(depth_im.shape[0],depth_im.shape[1]) * noise
            # print(depth_im[::40,::40])


            b = np.array([[1,0,0,0],
                        [0,-1,0,0],
                        [0,0,-1,0],
                        [0,0,0,1]])
            
            cam_pose = cam_pose@b
            
            # print(cam_pose)

            # Integrate observation into voxel volume (assume color aligned with depth)
            tsdf_vol.integrate(color_image, depth_im, cam_intr, cam_pose, obs_weight=1.0)

        fps = n_imgs / (time.time() - t0_elapse)
        print("Average FPS: {:.2f}".format(fps))

        # Get mesh from voxel volume and save to disk (can be viewed with Meshlab)
        print("Saving mesh to mesh.ply...")
        verts, faces, norms, colors, Nrays, verts_ind, Nray_vol, State_vol = tsdf_vol.get_mesh()
        fusion.meshwrite("dataset/cabin-0.5_noise_mesh_test.ply", verts, faces, norms, colors)
        
        
        ## 统计所有voxel的状态，及表面voxel的 Nray(其他voxel Nray=0),color
        State_vol = torch.tensor(State_vol).to(device)
        Nray_vol = torch.tensor(Nray_vol).to(device)
        Color_vol = torch.ones(State_vol.shape[0],State_vol.shape[1],State_vol.shape[2],3).to(device)*255
        
        
        ## 给表面voxel状态设置为2
        surface_verts = torch.tensor(verts).to(device)
        vol_origin = torch.tensor(tsdf_vol._vol_origin).to(device)
        voxel_size = torch.tensor(tsdf_vol._voxel_size).to(device)
        surface_verts_index = torch.floor((surface_verts-vol_origin)/voxel_size).T
        surface_verts_index_tensor = tuple([torch.LongTensor(index.cpu().numpy()) for index in surface_verts_index])
        surface_verts_value = torch.ones(surface_verts.shape[0]) * 2
        
        State_vol = State_vol.cpu()
        State_vol.index_put_(surface_verts_index_tensor, surface_verts_value)
        State_vol = State_vol.to(device)
        State_vol[0,0,0] = 1


        ## 给表面voxel设置color
        colors = torch.tensor(colors).float()
        Color_vol = Color_vol.cpu()
        Color_vol.index_put_(surface_verts_index_tensor, colors)
        Color_vol = Color_vol.to(device)

        return State_vol, Nray_vol, Color_vol  

    def tsdf_reconstruction(self,color_image, depth_im, cam_pose ):
        print("Fusing frame start")
        # print(depth_im)
        depth_im = train_data_far *depth_im/255.0
        # depth_im[depth_im == train_data_far] = 0
        depth_im += np.random.randn(depth_im.shape[0],depth_im.shape[1]) * noise
        # print(depth_im[::40,::40])

        # Convert unity coordinate to tsdf coordinate
        b = np.array([[1,0,0,0],
                    [0,-1,0,0],
                    [0,0,-1,0],
                    [0,0,0,1]])
        
        cam_pose = cam_pose@b
        
        # print(cam_pose)

        # Integrate observation into voxel volume (assume color aligned with depth)
        self.tsdf_vol.integrate(color_image, depth_im, cam_intr, cam_pose, obs_weight=1.0)
        print("Fusing frame finish")

        
    def get_tsdf_model(self):
        # global  index_bnd, State_vol, Nray_vol,Color_vol,State_vol_origin,front_verts
        # Get mesh from voxel volume and save to disk (can be viewed with Meshlab)
        print("Saving mesh to mesh.ply...")
        verts, faces, norms, colors, Nrays, verts_ind, nray_vol, state_vol = self.tsdf_vol.get_mesh()
        fusion.meshwrite("./saved_model/nerf_vpp_"+ version +".ply", verts, faces, norms, colors)
        
        # send_SamplePoints(verts.tolist())
        
        ## 统计所有voxel的状态，及表面voxel的 Nray(其他voxel Nray=0),color
        self.State_vol = torch.tensor(copy.deepcopy(state_vol)).to(device)
        self.Nray_vol = torch.tensor(copy.deepcopy(nray_vol)).to(device)

        # 给出先验empty space
        # box_min_index_low = np.ceil((aoi_bnds[:,0]-vol_bnds[:,0])/voxel_res).copy(order='C').astype(int)
        # box_min_index_high = np.ceil((aoi_bnds[:,1]-vol_bnds[:,0])/voxel_res).copy(order='C').astype(int)
        # box_max_index = np.ceil((vol_bnds[:,1]-vol_bnds[:,0])/voxel_res).copy(order='C').astype(int)
        # self.State_vol[0:box_min_index_low[0],:,:] = 1.0
        # self.State_vol[box_min_index_high[0]:box_max_index[0],:,:] = 1.0
        # self.State_vol[:,0:box_min_index_low[1],:] = 1.0
        # self.State_vol[:,box_min_index_high[1]:box_max_index[1],:] = 1.0
        # self.State_vol[:,:,0:box_min_index_low[2]] = 1.0
        # self.State_vol[:,:,box_min_index_high[1]:box_max_index[1]] = 1.0

        space_index_gpu = torch.nonzero(self.State_vol<3)
        space_index = space_index_gpu.cpu().numpy()
        center_index = torch.tensor(((self.object_center-self.tsdf_vol._vol_origin)/self.tsdf_vol._voxel_size)).to(device)
        center_index = center_index.repeat(space_index.shape[0],1)
        vox2center_dist = torch.norm(space_index_gpu.to(device)-center_index,p =2,dim=1)*voxel_res
        valid_index_sphere = space_index_gpu[vox2center_dist>3.0].cpu().numpy()
        print("valid_index_sphere ",valid_index_sphere.shape)
        self.State_vol[valid_index_sphere.T.tolist()] = 1
        
        ## 给表面voxel状态设置为2
        surface_verts = torch.tensor(verts).to(device)
        vol_origin = torch.tensor(self.tsdf_vol._vol_origin).to(device)
        voxel_size = torch.tensor(self.tsdf_vol._voxel_size).to(device)
        surface_verts_index = torch.floor((surface_verts-vol_origin)/voxel_size).T.long()
        self.State_vol[surface_verts_index.cpu().numpy().tolist()] = 2
        surface_verts_index_tensor = tuple([torch.LongTensor(index.cpu().numpy()) for index in surface_verts_index])


        ## 给表面voxel设置color
        # self.Color_vol[surface_verts_index.cpu().numpy().tolist()] =  torch.tensor(colors).float().to(device)
        colors = torch.tensor(colors).float()
        Color_vol = self.Color_vol.cpu()
        Color_vol.index_put_(surface_verts_index_tensor, colors)
        self.Color_vol = Color_vol.to(device)


        ## 膨胀前地图，留作备用
        self.State_vol_origin = torch.clone(self.State_vol)
        self.State_vol_origin[0,0,0] = 1
        ### 地图膨胀，便于empty space采样和路径规划
        ## 筛选出地图中occ voxel ,同时高度高度高于0.5的点进行膨胀（去除地面）
        height = 0.5
        height_index = (height-self.tsdf_vol._vol_bnds[1,0])/self.tsdf_vol._voxel_size
        occ_index=torch.nonzero(self.State_vol==2).T.cpu().numpy()
        occ_index = occ_index[np.concatenate([(occ_index[1]>height_index)[np.newaxis,:],(occ_index[1]>height_index)[np.newaxis,:],(occ_index[1]>height_index)[np.newaxis,:]],axis = 0)]
        occ_index = occ_index.reshape(3,int(occ_index.shape[0]/3))
        
        ## 地图边界
        xmax_index,ymax_index,zmax_index = self.State_vol.shape
        min_index = np.zeros(occ_index.shape[1])
        xmax_index = np.ones(occ_index.shape[1])*(xmax_index-1)
        ymax_index = np.ones(occ_index.shape[1])*(ymax_index-1)
        zmax_index = np.ones(occ_index.shape[1])*(zmax_index-1)
        for i in range(1,int(0.6/voxel_res+1),1):
            print(i)
            xsub_expand_index = np.concatenate([np.maximum((occ_index[0]-i),min_index)[np.newaxis,:],(occ_index[1])[np.newaxis,:],(occ_index[2])[np.newaxis,:]],axis = 0)
            xadd_expand_index = np.concatenate([np.minimum((occ_index[0]+i),xmax_index)[np.newaxis,:],(occ_index[1])[np.newaxis,:],(occ_index[2])[np.newaxis,:]],axis = 0)
            ysub_expand_index = np.concatenate([(occ_index[0])[np.newaxis,:],np.maximum((occ_index[1]-i),min_index)[np.newaxis,:],(occ_index[2])[np.newaxis,:]],axis = 0)
            yadd_expand_index = np.concatenate([(occ_index[0])[np.newaxis,:],np.minimum((occ_index[1]+i),ymax_index)[np.newaxis,:],(occ_index[2])[np.newaxis,:]],axis = 0)
            zsub_expand_index = np.concatenate([(occ_index[0])[np.newaxis,:],(occ_index[1])[np.newaxis,:],np.maximum((occ_index[2]-i),min_index)[np.newaxis,:]],axis = 0)
            zadd_expand_index = np.concatenate([(occ_index[0])[np.newaxis,:],(occ_index[1])[np.newaxis,:],np.minimum((occ_index[2]+i),zmax_index)[np.newaxis,:]],axis = 0)
            self.State_vol[xsub_expand_index] = 2
            self.State_vol[xadd_expand_index] = 2
            self.State_vol[ysub_expand_index] = 2
            self.State_vol[yadd_expand_index] = 2
            self.State_vol[zsub_expand_index] = 2
            self.State_vol[zadd_expand_index] = 2
        

    def is_in_empty(self,location):
        location = torch.tensor(location).to(device)
        ## 判断是否在边界框内部
        if vol_bnds[0,0]<=location[0]<vol_bnds[0,1] and vol_bnds[1,0]<=location[1]<vol_bnds[1,1] and vol_bnds[2,0]<=location[2]<vol_bnds[2,1]:
            location_voxl_index = torch.floor((location-self.vol_origin)/self.voxel_size).long().reshape(3,1).cpu().numpy()   
            ## 判断是否是empty voxel
            if self.State_vol[location_voxl_index] == 1:
                return True

        return False

    def is_valid(self,location):
        is_valid = True
        res = 0.1
        # ## 是否在uncertainty包围盒外
        # if aabb_bnds[0,0]<location[0]<aabb_bnds[0,1]-res and \
        #     aabb_bnds[1,0]<location[1]<aabb_bnds[1,1]-res and \
        #     aabb_bnds[2,0]<location[2]<aabb_bnds[2,1]-res:
        #     is_valid = False
        dist = np.linalg.norm(self.object_center-np.array(location),ord =2)
        if dist < 3.0:
            is_valid = False
        ## 是否满足一定高度
        if location[1]<self.ymin or location[1] > self.ymax:
            is_valid = False

        return is_valid
    
    def get_state(self,location):
        is_valid = False
        location_voxl_index = torch.ceil((location-self.vol_origin)/self.voxel_size).long().reshape(3,1).cpu().numpy()[:,0].tolist()
        if 0<location_voxl_index[0]<self.index_bnd[0]-1 and \
            0<location_voxl_index[1]<self.index_bnd[1]-1 and \
            0<location_voxl_index[2]<self.index_bnd[2]-1:
            is_valid = True
        if is_valid == False:
            return torch.tensor(0.0).to(device)
        state = self.State_vol[tuple(location_voxl_index)]
        return state
    
    def get_state_cpu(self,location):
        is_valid = True
        res = 0.1
        ## 是否在tsdf包围盒
        # if location[0]<=vol_bnds[0,0] or location[0]>=vol_bnds[0,1]-res or \
        #     location[1]<=vol_bnds[1,0] or location[1]>=vol_bnds[1,1]-res or \
        #     location[2]<=vol_bnds[2,0] or location[2]>=vol_bnds[2,1]-res:
        #     is_valid = False
        ## 是否在uncertainty包围盒外
        if aabb_bnds[0,0]<location[0]<aabb_bnds[0,1]-res and \
            aabb_bnds[1,0]<location[1]<aoi_bnds[1,1]-res and \
            aabb_bnds[2,0]<location[2]<aabb_bnds[2,1]-res:
            is_valid = False

        dist = np.linalg.norm(self.object_center-np.array(location),ord =2)
        if dist < 3.0:
            is_valid = False

        ## 是否满足一定高度
        if location[1]<self.ymin or location[1] > self.ymax:
            is_valid = False
        # print("location = ",location,is_valid)
        if is_valid == False:
            return 0.0,is_valid
        location = torch.tensor(location).to(device)
        location_voxl_index = torch.ceil((location-self.vol_origin)/self.voxel_size).long().reshape(3,1).cpu().numpy()[:,0].tolist()
        state = self.State_vol[tuple(location_voxl_index)]
        
        return state.cpu().numpy(),is_valid
    


    def tsdf_test(self):
        global all_imgs_set, add_imgs_set
        t0 = time.time()
        files_list = os.listdir(base_dir)
        for n in range(len(files_list)):
            if "main" in files_list[n]:
                index = int(files_list[n][5:8])
                if index not in all_imgs_set:
                    add_imgs_set.add(index)
                all_imgs_set.add(index)
        print("all_imgs_set: ",all_imgs_set)
        print("add_imgs_set: ",add_imgs_set)
                     
        for i in all_imgs_set:
            print("Fusing frame %d/%d"%(i, n_imgs))
            
            t0 = time.time()
            # Read RGB-D image and camera pose
            color_image = cv2.cvtColor(cv2.imread(os.path.join(base_dir,"main_{:03d}.png".format(i))), cv2.COLOR_BGR2RGB)[::2,::2]
            depth_im = cv2.imread(os.path.join(base_dir,"depth_{:03d}.png".format(i)),-1).astype(float)[::2,::2,0]
            cam_pose = np.loadtxt(os.path.join(base_dir,"depth_{:03d}.pngpose_{:03d}.csv".format(i,i)), delimiter = ',')
            print("read imgs time = ", time.time()-t0) 

            self.tsdf_reconstruction(color_image,depth_im,cam_pose)
        t1 = time.time()
        print("train time = ", t1-t0,"fps = ", n_imgs/(t1-t0))
        self.get_tsdf_model()
        t2 = time.time()
        print("get model state time = ", t2-t1)
        return self.State_vol



if __name__ == "__main__":

    tsdf = TSDF()
    tsdf.tsdf_test()
   
    view = np.array([0, 1, -4,0,0])
    t2 = time.time()
    h = 100
    w = (int(h*(360/67.38013))//12)*12
    print(h,w)
    uncer_all = tsdf.get_all_uncertainty_test(view[0:3],view[3],view[4],H=h ,W=w,n =0)
    # uncer_all = tsdf.get_all_uncertainty(view[0:3],view[3],view[4],n =0)
    print("render img  time = ",  time.time()-t2)
    
    
    t0 = time.time()
    views = []
    h = 40
    w = (int(h*(360/67.38013))//12)*12
    print(h,w)
    for n in range(0,1,1):
        yaw = n*30
        t0 = time.time()
        views.append([0,  1, -4,0,yaw/180.0*np.pi])
    views = np.array(views)
    uncer,depth = tsdf.get_uncertainty_tsdf_test(views[:,0:3].tolist(),H=h ,W=w)
    print(np.array(uncer))
    print(np.array(depth))
    
    print(" time ", time.time()-t0)

    
    

      

 