# Copyright (c) 2018 Andy Zeng

import numpy as np

from numba import njit, prange
from skimage import measure
import torch
from core.interface2 import *

FUSION_GPU_MODE = 1


class TSDFVolume:
  """Volumetric TSDF Fusion of RGB-D Images.
  """
  def __init__(self, vol_bnds, voxel_size,near,far, use_gpu=True):
    """Constructor.

    Args:
      vol_bnds (ndarray): An ndarray of shape (3, 2). Specifies the
        xyz bounds (min/max) in meters.
      voxel_size (float): The volume discretization in meters.
    """
    self.near = near
    self.far = far

    vol_bnds = np.asarray(vol_bnds)
    assert vol_bnds.shape == (3, 2), "[!] `vol_bnds` should be of shape (3, 2)."

    # Define voxel volume parameters
    self._vol_bnds = vol_bnds
    self._voxel_size = float(voxel_size)
    self._trunc_margin = 5 * self._voxel_size  # truncation on SDF
    self._color_const = 256 * 256

    # Adjust volume bounds and ensure C-order contiguous
    self._vol_dim = np.ceil((self._vol_bnds[:,1]-self._vol_bnds[:,0])/self._voxel_size).copy(order='C').astype(int)
    self._vol_bnds[:,1] = self._vol_bnds[:,0]+self._vol_dim*self._voxel_size
    self._vol_origin = self._vol_bnds[:,0].copy(order='C').astype(np.float32)

    print("Voxel volume size: {} x {} x {} - # points: {:,}".format(
      self._vol_dim[0], self._vol_dim[1], self._vol_dim[2],
      self._vol_dim[0]*self._vol_dim[1]*self._vol_dim[2])
    )

    # Initialize pointers to voxel volume in CPU memory
    # self._tsdf_vol_cpu = np.ones(self._vol_dim).astype(np.float32)
    self._tsdf_vol_gpu = torch.ones(tuple(self._vol_dim)).to(device)
    # for computing the cumulative moving average of observations per voxel
    # self._weight_vol_cpu = np.zeros(self._vol_dim).astype(np.float32)
    # self._color_vol_cpu = np.zeros(self._vol_dim).astype(np.float32)
    self._weight_vol_gpu = torch.zeros(tuple(self._vol_dim)).to(device)
    self._color_vol_gpu = torch.zeros(tuple(self._vol_dim)).to(device)
    
    ## 穿过某个voxel的射线数目
    # self._Nray_vol_cpu = np.zeros(self._vol_dim).astype(np.float32)
    self._Nray_vol_gpu = torch.zeros(tuple(self._vol_dim)).to(device)
    ## voxel state 
    # self._State_vol_cpu = np.zeros(self._vol_dim).astype(np.float32)
    self._State_vol_gpu = torch.zeros(tuple(self._vol_dim)).to(device)
    ## sdf
    # self._sdf_vol_cpu = np.ones(self._vol_dim).astype(np.float32)
    self._sdf_vol_gpu = torch.ones(tuple(self._vol_dim)).to(device)
    
    ## 选择运行模式
    self.gpu_mode = use_gpu and FUSION_GPU_MODE

    # Copy voxel volumes to CPU
    if self.gpu_mode:
          
      # self._tsdf_vol_gpu = torch.from_numpy(self._tsdf_vol_cpu).to(device)
      # self._weight_vol_gpu = torch.from_numpy(self._weight_vol_cpu).to(device)
      # self._color_vol_gpu = torch.from_numpy(self._color_vol_cpu).to(device)
      # self._Nray_vol_gpu = torch.from_numpy(self._Nray_vol_cpu).to(device)
      # self._State_vol_gpu = torch.from_numpy(self._State_vol_cpu).to(device)
      # self._sdf_vol_gpu = torch.from_numpy(self._sdf_vol_cpu).to(device)
      
      # Get voxel grid coordinates
      xv, yv, zv = torch.meshgrid(
        torch.arange(0,self._vol_dim[0]),
        torch.arange(0,self._vol_dim[1]),
        torch.arange(0,self._vol_dim[2]),
      ) 
      print(xv.shape)
      self.vox_coords_gpu = torch.cat([
        xv.reshape(1,-1),
        yv.reshape(1,-1),
        zv.reshape(1,-1)
      ], axis=0).T.to(device)


    else:
      # Get voxel grid coordinates
      xv, yv, zv = np.meshgrid(
        range(self._vol_dim[0]),
        range(self._vol_dim[1]),
        range(self._vol_dim[2]),
        indexing='ij'
      )
      self.vox_coords = np.concatenate([
        xv.reshape(1,-1),
        yv.reshape(1,-1),
        zv.reshape(1,-1)
      ], axis=0).astype(int).T

  @staticmethod
  @njit(parallel=True)
  def vox2world(vol_origin, vox_coords, vox_size):
    """Convert voxel grid coordinates to world coordinates.
    """
    vol_origin = vol_origin.astype(np.float32)
    vox_coords = vox_coords.astype(np.float32)
    cam_pts = np.empty_like(vox_coords, dtype=np.float32)
    for i in prange(vox_coords.shape[0]):
      for j in range(3):
        cam_pts[i, j] = vol_origin[j] + (vox_size * vox_coords[i, j])
    return cam_pts

  @staticmethod
  def vox2world_gpu(vol_origin, vox_coords, vox_size):
    """Convert voxel grid coordinates to world coordinates.
    """
    vol_origin =  vol_origin * torch.ones_like(vox_coords)
    cam_pts = torch.empty_like(vox_coords, dtype=torch.float32)
    cam_pts =  vol_origin + vox_size * vox_coords
    return cam_pts

  @staticmethod
  @njit(parallel=True)
  def cam2pix(cam_pts, intr):
    """Convert camera coordinates to pixel coordinates.
    """
    intr = intr.astype(np.float32)
    fx, fy = intr[0, 0], intr[1, 1]
    cx, cy = intr[0, 2], intr[1, 2]
    pix = np.empty((cam_pts.shape[0], 2), dtype=np.int64)
    for i in prange(cam_pts.shape[0]):
      pix[i, 0] = int(np.round((cam_pts[i, 0] * fx / cam_pts[i, 2]) + cx))
      pix[i, 1] = int(np.round((cam_pts[i, 1] * fy / cam_pts[i, 2]) + cy))
    return pix

  @staticmethod
  def cam2pix_gpu(cam_pts, intr):
    """Convert camera coordinates to pixel coordinates.
    """
    pix = torch.mm(intr.double(), cam_pts.T.double())
    pix = torch.div(pix,cam_pts.T.double()[2,:]).T
    pix = torch.round(pix)
    return pix[:,0:2].float()

  @staticmethod
  @njit(parallel=True)
  def integrate_tsdf(tsdf_vol, dist, w_old, obs_weight):
    """Integrate the TSDF volume.
    """
    tsdf_vol_int = np.empty_like(tsdf_vol, dtype=np.float32)
    w_new = np.empty_like(w_old, dtype=np.float32)
    for i in prange(len(tsdf_vol)):
      w_new[i] = w_old[i] + obs_weight
      tsdf_vol_int[i] = (w_old[i] * tsdf_vol[i] + obs_weight * dist[i]) / w_new[i]
    return tsdf_vol_int, w_new

  @staticmethod
  def integrate_tsdf_gpu(tsdf_vol, dist, w_old, obs_weight):
    """Integrate the TSDF volume.
    """
    tsdf_vol_int = torch.empty_like(tsdf_vol, dtype=torch.float32)
    w_new = torch.empty_like(w_old, dtype=torch.float32)

    obs_weight = obs_weight * torch.ones((tsdf_vol.shape[0]), dtype=torch.float32).to(device)
    w_new = w_old + obs_weight
    tsdf_vol_int = torch.div((w_old * tsdf_vol + obs_weight * dist),w_new) 

    return tsdf_vol_int, w_new

  def integrate(self, color_im, depth_im, cam_intr, cam_pose, obs_weight=1.):
    """Integrate an RGB-D frame into the TSDF volume.

    Args:
      color_im (ndarray): An RGB image of shape (H, W, 3).
      depth_im (ndarray): A depth image of shape (H, W).
      cam_intr (ndarray): The camera intrinsics matrix of shape (3, 3).
      cam_pose (ndarray): The camera pose (i.e. extrinsics) of shape (4, 4).
      obs_weight (float): The weight to assign for the current observation. A higher
        value
    """

    # print(depth_im.shape)
    im_h, im_w = depth_im.shape

    # Fold RGB color image into a single channel image
    color_im = color_im.astype(np.float32)
    color_im = np.floor(color_im[...,2]*self._color_const + color_im[...,1]*256 + color_im[...,0])

    if self.gpu_mode:  # GPU mode: integrate voxel volume (calls CUDA kernel)
      print("gpu")
      # Convert voxel grid coordinates to pixel coordinates
      cam_pts = self.vox2world_gpu(torch.tensor(self._vol_origin).to(device), self.vox_coords_gpu, self._voxel_size)
      cam_pts = rigid_transform_gpu(cam_pts, torch.tensor(np.linalg.inv(cam_pose)).to(device))
      pix_z = cam_pts[:, 2]
      pix = self.cam2pix_gpu(cam_pts, torch.from_numpy(cam_intr).to(device))
      pix_x, pix_y = pix[:, 0], pix[:, 1]
      pix = torch.cat([pix[:,1].reshape(pix.shape[0],1),pix[:,0].reshape(pix.shape[0],1)], axis = -1) ## 交换x,y顺序

      # Eliminate pixels outside view frustum
      valid_pix = torch.logical_and(pix_x >= 0,
                  torch.logical_and(pix_x < im_w,
                  torch.logical_and(pix_y >= 0,
                  torch.logical_and(pix_y < im_h,
                  pix_z > 0)))).cpu()
      depth_val = torch.zeros(pix_z.shape).float().to(device)
      depth_im = torch.from_numpy(depth_im).float().to(device)
      depth_val[valid_pix] = depth_im[pix[valid_pix].cpu().numpy().T]
      
      ## 把depth为far的射线上的voxel设为empty space
      far_pts =  torch.logical_and(depth_val==self.far,
                 torch.logical_and(pix_x >= 0,
                  torch.logical_and(pix_x < im_w,
                  torch.logical_and(pix_y >= 0,
                  torch.logical_and(pix_y < im_h,
                  pix_z > 0))))).cpu()
      far_vox_x = self.vox_coords_gpu[far_pts, 0]
      far_vox_y = self.vox_coords_gpu[far_pts, 1]
      far_vox_z = self.vox_coords_gpu[far_pts, 2]
      self._State_vol_gpu[far_vox_x, far_vox_y, far_vox_z] = 1
     
      
      ##depth=far设置为0，便于后续tsdf判断
      depth_im[depth_im==self.far]=0
      depth_val[valid_pix] = depth_im[pix[valid_pix].cpu().numpy().T]
      # Calculate SDF
      depth_diff = depth_val - pix_z
      

      ## 把达到物体上的射线上，并在物体前的voxel设为empty space
      hit_pts = torch.logical_and(depth_val > 0,  depth_diff >0)
      hit_vox_x = self.vox_coords_gpu[hit_pts, 0]
      hit_vox_y = self.vox_coords_gpu[hit_pts, 1]
      hit_vox_z = self.vox_coords_gpu[hit_pts, 2]
      self._State_vol_gpu[hit_vox_x, hit_vox_y, hit_vox_z] = 1

      ## Integrate TSDF
      valid_pts = torch.logical_and(depth_val > 0, depth_diff >= -self._trunc_margin)
      dist = torch.minimum(torch.tensor(1.0).float().to(device), depth_diff / self._trunc_margin)
      valid_vox_x = self.vox_coords_gpu[valid_pts, 0]
      valid_vox_y = self.vox_coords_gpu[valid_pts, 1]
      valid_vox_z = self.vox_coords_gpu[valid_pts, 2]
      w_old = self._weight_vol_gpu[valid_vox_x, valid_vox_y, valid_vox_z]
      tsdf_vals = self._tsdf_vol_gpu[valid_vox_x, valid_vox_y, valid_vox_z]
      valid_dist = dist[valid_pts] 

      # surface
      valid_surface_pts = torch.logical_and(depth_val > 0,
                          torch.logical_and(depth_diff >= -self._trunc_margin, depth_diff <= self._trunc_margin))
      valid_surface_vox_x = self.vox_coords_gpu[valid_surface_pts, 0]
      valid_surface_vox_y = self.vox_coords_gpu[valid_surface_pts, 1]
      valid_surface_vox_z= self.vox_coords_gpu[valid_surface_pts, 2]
      self._Nray_vol_gpu[valid_surface_vox_x, valid_surface_vox_y, valid_surface_vox_z] += 1;   ## surface 
      # self._State_vol_gpu[valid_surface_vox_x, valid_surface_vox_y, valid_surface_vox_z] = 2;   ## occupied voxel set as 2 
          
      tsdf_vol_new, w_new = self.integrate_tsdf_gpu(tsdf_vals, valid_dist, w_old, torch.tensor(obs_weight).float().to(device))
      self._weight_vol_gpu[valid_vox_x, valid_vox_y, valid_vox_z] = w_new
      self._tsdf_vol_gpu[valid_vox_x, valid_vox_y, valid_vox_z] = tsdf_vol_new

      # Integrate color
      old_color = self._color_vol_gpu[valid_vox_x, valid_vox_y, valid_vox_z]
      old_b = torch.floor(old_color / self._color_const)
      old_g = torch.floor((old_color-old_b*self._color_const)/256)
      old_r = old_color - old_b*self._color_const - old_g*256
      color_im = torch.from_numpy(color_im).float().to(device)
      new_color = color_im[pix[valid_pts].cpu().numpy().T]
      new_b = torch.floor(new_color / self._color_const)
      new_g = torch.floor((new_color - new_b*self._color_const) /256)
      new_r = new_color - new_b*self._color_const - new_g*256
      new_b = torch.minimum(torch.tensor(255.).float().to(device), torch.round((w_old*old_b + obs_weight*new_b) / w_new))
      new_g = torch.minimum(torch.tensor(255.).float().to(device), torch.round((w_old*old_g + obs_weight*new_g) / w_new))
      new_r = torch.minimum(torch.tensor(255.).float().to(device), torch.round((w_old*old_r + obs_weight*new_r) / w_new))
      self._color_vol_gpu[valid_vox_x, valid_vox_y, valid_vox_z] = new_b*self._color_const + new_g*256 + new_r

      
      print("gpu end")

     
    else:  # CPU mode: integrate voxel volume (vectorized implementation)
      # Convert voxel grid coordinates to pixel coordinates
      cam_pts = self.vox2world(self._vol_origin, self.vox_coords, self._voxel_size)
      cam_pts = rigid_transform(cam_pts, np.linalg.inv(cam_pose))
      pix_z = cam_pts[:, 2]
      pix = self.cam2pix(cam_pts, cam_intr)
      pix_x, pix_y = pix[:, 0], pix[:, 1]

      # Eliminate pixels outside view frustum
      valid_pix = np.logical_and(pix_x >= 0,
                  np.logical_and(pix_x < im_w,
                  np.logical_and(pix_y >= 0,
                  np.logical_and(pix_y < im_h,
                  pix_z > 0))))
      depth_val = np.zeros(pix_x.shape)
      depth_val[valid_pix] = depth_im[pix_y[valid_pix], pix_x[valid_pix]]

      # Integrate TSDF
      depth_diff = depth_val - pix_z
      valid_pts = np.logical_and(depth_val > 0, depth_diff >= -self._trunc_margin)
      dist = np.minimum(1, depth_diff / self._trunc_margin)
      valid_vox_x = self.vox_coords[valid_pts, 0]
      valid_vox_y = self.vox_coords[valid_pts, 1]
      valid_vox_z = self.vox_coords[valid_pts, 2]
      w_old = self._weight_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z]
      tsdf_vals = self._tsdf_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z]
      valid_dist = dist[valid_pts]   
      
      ## occupied voxel or empty voxel (surface) , set as 1 first
      self._State_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z] = 1;   

      ##surface
      valid_surface_pts = np.logical_and(depth_val > 0, depth_diff >= -self._trunc_margin, depth_diff <= self._trunc_margin)
      valid_surface_vox_x = self.vox_coords[valid_surface_pts, 0]
      valid_surface_vox_y = self.vox_coords[valid_surface_pts, 1]
      valid_surface_vox_z= self.vox_coords[valid_surface_pts, 2]
      self._Nray_vol_cpu[valid_surface_vox_x, valid_surface_vox_y, valid_surface_vox_z] += 1;   ## surface 
      self._State_vol_cpu[valid_surface_vox_x, valid_surface_vox_y, valid_surface_vox_z] = 2;   ## occupied voxel set as 2
          
      tsdf_vol_new, w_new = self.integrate_tsdf(tsdf_vals, valid_dist, w_old, obs_weight)
      self._weight_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z] = w_new
      self._tsdf_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z] = tsdf_vol_new

      # Integrate color
      old_color = self._color_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z]
      old_b = np.floor(old_color / self._color_const)
      old_g = np.floor((old_color-old_b*self._color_const)/256)
      old_r = old_color - old_b*self._color_const - old_g*256
      new_color = color_im[pix_y[valid_pts],pix_x[valid_pts]]
      new_b = np.floor(new_color / self._color_const)
      new_g = np.floor((new_color - new_b*self._color_const) /256)
      new_r = new_color - new_b*self._color_const - new_g*256
      new_b = np.minimum(255., np.round((w_old*old_b + obs_weight*new_b) / w_new))
      new_g = np.minimum(255., np.round((w_old*old_g + obs_weight*new_g) / w_new))
      new_r = np.minimum(255., np.round((w_old*old_r + obs_weight*new_r) / w_new))
      self._color_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z] = new_b*self._color_const + new_g*256 + new_r
  

  def get_volume(self):
    if self.gpu_mode:
      self._tsdf_vol_cpu = self._tsdf_vol_gpu.cpu().numpy()
      self._color_vol_cpu = self._color_vol_gpu.cpu().numpy()
      self._weight_vol_cpu = self._weight_vol_gpu.cpu().numpy()
      self._Nray_vol_cpu = self._Nray_vol_gpu.cpu().numpy()
      self._State_vol_cpu = self._State_vol_gpu.cpu().numpy()

    return self._tsdf_vol_cpu, self._color_vol_cpu, self._weight_vol_cpu,self._Nray_vol_cpu,self._State_vol_cpu

  def get_point_cloud(self):
    """Extract a point cloud from the voxel volume.
    """
    tsdf_vol, color_vol, weight_vol,Nray_vol, State_vol = self.get_volume()

    # Marching cubes
    verts = measure.marching_cubes_lewiner(tsdf_vol, level=0)[0]
    verts_ind = np.round(verts).astype(int)
    verts = verts*self._voxel_size + self._vol_origin

    # Get vertex colors
    rgb_vals = color_vol[verts_ind[:, 0], verts_ind[:, 1], verts_ind[:, 2]]
    colors_b = np.floor(rgb_vals / self._color_const)
    colors_g = np.floor((rgb_vals - colors_b*self._color_const) / 256)
    colors_r = rgb_vals - colors_b*self._color_const - colors_g*256
    colors = np.floor(np.asarray([colors_r, colors_g, colors_b])).T
    colors = colors.astype(np.uint8)

    # Get vertex weights
    _range = np.max(weight_vol)
    weight_vol_norm = (weight_vol - np.min(weight_vol))/_range
    weight_vals_norm = weight_vol_norm[verts_ind[:,0], verts_ind[:,1], verts_ind[:,2]]

    pc = np.hstack([verts, colors])
    return pc,weight_vals_norm

  def get_mesh(self):
    """Compute a mesh from the voxel volume using marching cubes.
    """
    tsdf_vol, color_vol, weight_vol,Nray_vol, State_vol = self.get_volume()

    # Marching cubes
    verts, faces, norms, vals = measure.marching_cubes_lewiner(tsdf_vol, level=0)
    verts_ind = np.round(verts).astype(int)
    verts = verts*self._voxel_size+self._vol_origin  # voxel grid coordinates to world coordinates
  

    # Get vertex colors
    rgb_vals = color_vol[verts_ind[:,0], verts_ind[:,1], verts_ind[:,2]]
    colors_b = np.floor(rgb_vals/self._color_const)
    colors_g = np.floor((rgb_vals-colors_b*self._color_const)/256)
    colors_r = rgb_vals-colors_b*self._color_const-colors_g*256
    colors = np.floor(np.asarray([colors_r,colors_g,colors_b])).T
    colors = colors.astype(np.uint8)
    Nrays =  Nray_vol[verts_ind[:,0], verts_ind[:,1], verts_ind[:,2]]

    return verts, faces, norms, colors,Nrays,verts_ind,self._Nray_vol_cpu,self._State_vol_cpu


def rigid_transform(xyz, transform):
  """Applies a rigid transform to an (N, 3) pointcloud.
  """
  xyz_h = np.hstack([xyz, np.ones((len(xyz), 1), dtype=np.float32)])
  xyz_t_h = np.dot(transform, xyz_h.T).T
  return xyz_t_h[:, :3]

def rigid_transform_gpu(xyz, transform):
    """Applies a rigid transform to an (N, 3) pointcloud.
    """ 
    xyz_h = torch.hstack([xyz, torch.ones((xyz.shape[0], 1), dtype=torch.float32).to(device)])
    xyz_t_h = torch.mm(transform.double(), xyz_h.T.double()).T
    return xyz_t_h[:, :3].float()


def get_view_frustum(depth_im, cam_intr, cam_pose):
  """Get corners of 3D camera view frustum of depth image
  """
  im_h = depth_im.shape[0]
  im_w = depth_im.shape[1]
  max_depth = np.max(depth_im)
  view_frust_pts = np.array([
    (np.array([0,0,0,im_w,im_w])-cam_intr[0,2])*np.array([0,max_depth,max_depth,max_depth,max_depth])/cam_intr[0,0],
    (np.array([0,0,im_h,0,im_h])-cam_intr[1,2])*np.array([0,max_depth,max_depth,max_depth,max_depth])/cam_intr[1,1],
    np.array([0,max_depth,max_depth,max_depth,max_depth])
  ])
  view_frust_pts = rigid_transform(view_frust_pts.T, cam_pose).T
  return view_frust_pts


def meshwrite(filename, verts, faces, norms, colors):
  """Save a 3D mesh to a polygon .ply file.
  """
  # Write header
  ply_file = open(filename,'w')
  ply_file.write("ply\n")
  ply_file.write("format ascii 1.0\n")
  ply_file.write("element vertex %d\n"%(verts.shape[0]))
  ply_file.write("property float x\n")
  ply_file.write("property float y\n")
  ply_file.write("property float z\n")
  ply_file.write("property float nx\n")
  ply_file.write("property float ny\n")
  ply_file.write("property float nz\n")
  ply_file.write("property uchar red\n")
  ply_file.write("property uchar green\n")
  ply_file.write("property uchar blue\n")
  ply_file.write("element face %d\n"%(faces.shape[0]))
  ply_file.write("property list uchar int vertex_index\n")
  ply_file.write("end_header\n")

  # Write vertex list
  for i in range(verts.shape[0]):
    ply_file.write("%f %f %f %f %f %f %d %d %d\n"%(
      -verts[i,0], verts[i,1], verts[i,2],
      norms[i,0], norms[i,1], norms[i,2],
      colors[i,0], colors[i,1], colors[i,2],
    ))

  # Write face list
  for i in range(faces.shape[0]):
    ply_file.write("3 %d %d %d\n"%(faces[i,0], faces[i,1], faces[i,2]))

  ply_file.close()



def pcwrite(filename, xyzrgb):
  """Save a point cloud to a polygon .ply file.
  """
  xyz = xyzrgb[:, :3]
  rgb = xyzrgb[:, 3:].astype(np.uint8)

  # Write header
  ply_file = open(filename,'w')
  ply_file.write("ply\n")
  ply_file.write("format ascii 1.0\n")
  ply_file.write("element vertex %d\n"%(xyz.shape[0]))
  ply_file.write("property float x\n")
  ply_file.write("property float y\n")
  ply_file.write("property float z\n")
  ply_file.write("property uchar red\n")
  ply_file.write("property uchar green\n")
  ply_file.write("property uchar blue\n")
  ply_file.write("end_header\n")

  # Write vertex list
  for i in range(xyz.shape[0]):
    ply_file.write("%f %f %f %d %d %d\n"%(
      -xyz[i, 0], xyz[i, 1], xyz[i, 2],
      rgb[i, 0], rgb[i, 1], rgb[i, 2],
    ))
