import os
import torch
import numpy as np
import imageio 
import json
import torch.nn.functional as F
import cv2


trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w


def load_blender_data(basedir, half_res=False, testskip=1):
    splits = ['train', 'val', 'test']
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)

    all_imgs = []
    all_poses = []
    counts = [0]
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        if s=='train' or testskip==0:
            skip = 1
        else:
            skip = testskip
            
        for frame in meta['frames'][::skip]:
            fname = os.path.join(basedir, frame['file_path'] + '.png')
            imgs.append(imageio.imread(fname))
            poses.append(np.array(frame['transform_matrix']))
        imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
        poses = np.array(poses).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)
    
    i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]
    
    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    
    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)
    
    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)
    
    if half_res:
        H = H//2
        W = W//2
        focal = focal/2.

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res
        # imgs = tf.image.resize_area(imgs, [400, 400]).numpy()

        
    return imgs, poses, render_poses, [H, W, focal], i_split

def pose_spherical_depth(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-2,0,0,0],[0,0,2,0],[0,2,0,0],[0,0,0,2]])) @ c2w
    return c2w

def read_matrix(filepath):
    a = []
    with open(filepath) as f:
        for line in f:
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



def load_blender_data_depth(basedir, half_res=False, testskip=1):
    s = 'train'
    all_imgs = []
    all_depth_imgs = []
    all_poses = []
    imgs = []
    poses = []
    depth_imgs = []
    if s=='train' or testskip==0:
        skip = 1
    else:
        skip = testskip
    for i in range(0,74,skip):
        depth_frame = os.path.join(basedir, str(i)+ 'depth.png')
        mian_fname = os.path.join(basedir, str(i)+ 'main.png')
        matrix_txt = os.path.join(basedir, str(i)+'.txt')
        depth_imgs.append(imageio.imread(depth_frame))
        imgs.append(imageio.imread(mian_fname))
        poses.append(read_matrix(matrix_txt))
            
    depth_imgs = (np.array(depth_imgs)/255.).astype(np.float32)
    imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
    poses = np.array(poses).astype(np.float32)
    all_imgs.append(imgs)
    all_depth_imgs.append(depth_imgs)
    all_poses.append(poses)
            
        
    
    i_split = [np.arange(0,74,3),np.arange(1, 74,3),np.arange(2,74,3)]
    
    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    
    H, W = imgs[0].shape[:2]
    focal = 600
    
    render_poses = torch.stack([pose_spherical_depth(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)
    
    if half_res:
        H = H//2
        W = W//2
        focal = focal/2.

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
        imgs_half_res_depth = np.zeros((depth_imgs.shape[0], H, W, 4))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res
        # imgs = tf.image.resize_area(imgs, [400, 400]).numpy()
        for i, img in enumerate(depth_imgs):
            imgs_half_res_depth[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        depth_imgs = imgs_half_res_depth

        depth_imgs = depth_imgs[...,0]
        
    return imgs, depth_imgs, poses, render_poses, [H, W, focal], i_split


def load_blender_data_cabin(basedir, half_res=False, testskip=1):
    s = 'train'
    all_imgs = []
    all_depth_imgs = []
    all_poses = []
    imgs = []
    poses = []
    depth_imgs = []
    if s=='train' or testskip==0:
        skip = 1
    else:
        skip = testskip
    for i in range(0,130,skip):
        depth_frame = os.path.join(basedir, str(i)+ 'depth.png')
        mian_fname = os.path.join(basedir, str(i)+ 'main.png')
        matrix_txt = os.path.join(basedir, str(i)+'.txt')
        depth_imgs.append(imageio.imread(depth_frame))
        imgs.append(imageio.imread(mian_fname))
        poses.append(read_matrix(matrix_txt))
            
    depth_imgs = (np.array(depth_imgs)/255.).astype(np.float32)
    imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
    poses = np.array(poses).astype(np.float32)
    all_imgs.append(imgs)
    all_depth_imgs.append(depth_imgs)
    all_poses.append(poses)
            
        
    
    i_split = [np.arange(0,41),np.arange(41, 125),np.arange(125,130)]
    
    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    
    H, W = imgs[0].shape[:2]
    focal = 600
    
    render_poses = torch.stack([pose_spherical_depth(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)
    
    if half_res:
        H = H//2
        W = W//2
        focal = focal/2.

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
        imgs_half_res_depth = np.zeros((depth_imgs.shape[0], H, W, 4))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res
        # imgs = tf.image.resize_area(imgs, [400, 400]).numpy()
        for i, img in enumerate(depth_imgs):
            imgs_half_res_depth[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        depth_imgs = imgs_half_res_depth

        depth_imgs = depth_imgs[...,0]
        # depth_imgs[depth_imgs > 0.99] = 0
        
    return imgs, depth_imgs, poses, render_poses, [H, W, focal], i_split


def load_blender_data_cabin3(basedir, half_res=False, testskip=1):
    s = 'train'
    all_imgs = []
    all_depth_imgs = []
    all_poses = []
    imgs = []
    poses = []
    depth_imgs = []
    if s=='train' or testskip==0:
        skip = 1
    else:
        skip = testskip
    for i in range(0,190,skip):
        depth_frame = os.path.join(basedir, str(i)+ 'depth.png')
        mian_fname = os.path.join(basedir, str(i)+ 'main.png')
        matrix_txt = os.path.join(basedir, str(i)+'.txt')
        depth_imgs.append(imageio.imread(depth_frame))
        imgs.append(imageio.imread(mian_fname))
        poses.append(read_matrix(matrix_txt))
            
    depth_imgs = (np.array(depth_imgs)/255.).astype(np.float32)
    imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
    poses = np.array(poses).astype(np.float32)
    all_imgs.append(imgs)
    all_depth_imgs.append(depth_imgs)
    all_poses.append(poses)
            
        
    
    i_split = [np.arange(0,150),np.arange(150, 170),np.arange(170,190)]
    
    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    
    H, W = imgs[0].shape[:2]
    focal = 1000
    
    render_poses = torch.stack([pose_spherical_depth(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)
    
    if half_res:
        H = H//2
        W = W//2
        focal = focal/2.

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
        imgs_half_res_depth = np.zeros((depth_imgs.shape[0], H, W, 4))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res
        # imgs = tf.image.resize_area(imgs, [400, 400]).numpy()
        for i, img in enumerate(depth_imgs):
            imgs_half_res_depth[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        depth_imgs = imgs_half_res_depth

        depth_imgs = depth_imgs[...,0]
        # depth_imgs[depth_imgs > 0.99] = 0
        
    return imgs, depth_imgs, poses, render_poses, [H, W, focal], i_split

def load_blender_data_testdata(basedir, half_res=False, testskip=1):
    s = 'train'
    all_imgs = []
    all_depth_imgs = []
    all_poses = []
    imgs = []
    poses = []
    depth_imgs = []
    if s=='train' or testskip==0:
        skip = 1
    else:
        skip = testskip
    for i in range(0,74,skip):
        depth_frame = os.path.join(basedir, str(i)+ 'depth.png')
        mian_fname = os.path.join(basedir, str(i)+ 'main.png')
        matrix_txt = os.path.join(basedir, str(i)+'.txt')
        depth_imgs.append(imageio.imread(depth_frame))
        imgs.append(imageio.imread(mian_fname))
        poses.append(read_matrix(matrix_txt))
            
    depth_imgs = (np.array(depth_imgs)/255.).astype(np.float32)
    imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
    poses = np.array(poses).astype(np.float32)
    all_imgs.append(imgs)
    all_depth_imgs.append(depth_imgs)
    all_poses.append(poses)
            
        
    
    i_split = [np.arange(0,74),np.arange(0, 74),np.arange(0,2)]
    
    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    
    H, W = imgs[0].shape[:2]
    focal = 1000
    
    render_poses = torch.stack([pose_spherical_depth(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)
    
    if half_res:
        H = H//2
        W = W//2
        focal = focal/2.

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
        imgs_half_res_depth = np.zeros((depth_imgs.shape[0], H, W, 4))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res
        # imgs = tf.image.resize_area(imgs, [400, 400]).numpy()
        for i, img in enumerate(depth_imgs):
            imgs_half_res_depth[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        depth_imgs = imgs_half_res_depth

        depth_imgs = depth_imgs[...,0]
        # depth_imgs[depth_imgs > 0.99] = 0
        
    return imgs, depth_imgs, poses, render_poses, [H, W, focal], i_split


def load_blender_data_self(basedir, length, half_res=False, testskip=1):
    s = 'train'
    all_imgs = []
    all_depth_imgs = []
    all_poses = []
    imgs = []
    poses = []
    depth_imgs = []
    if s=='train' or testskip==0:
        skip = 1
    else:
        skip = testskip
    for i in range(0,length,skip):
        depth_frame = os.path.join(basedir, 'depth_{:03d}.png'.format(i))
        mian_fname = os.path.join(basedir, 'main_{:03d}.png'.format(i))
        matrix_txt = os.path.join(basedir, 'depth_{:03d}.png'.format(i) + 'pose_{:03d}.csv'.format(i))
        depth_imgs.append(imageio.imread(depth_frame))
        imgs.append(imageio.imread(mian_fname))
        poses.append(np.loadtxt(matrix_txt,delimiter=","))
            
    depth_imgs = (np.array(depth_imgs)/255.).astype(np.float32)
    imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
    poses = np.array(poses).astype(np.float32)
    all_imgs.append(imgs)
    all_depth_imgs.append(depth_imgs)
    all_poses.append(poses)
            
        
    
    i_split = [np.arange(0,150),np.arange(150, 170),np.arange(0,200)]
    
    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    
    H, W = imgs[0].shape[:2]
    focal = 1000
    
    render_poses = torch.stack([pose_spherical_depth(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)
    
    if half_res:
        H = H//2
        W = W//2
        focal = focal/2.

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
        imgs_half_res_depth = np.zeros((depth_imgs.shape[0], H, W, 4))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res
        # imgs = tf.image.resize_area(imgs, [400, 400]).numpy()
        for i, img in enumerate(depth_imgs):
            imgs_half_res_depth[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        depth_imgs = imgs_half_res_depth

        depth_imgs = depth_imgs[...,0]
        # depth_imgs[depth_imgs > 0.99] = 0
        
    return imgs, depth_imgs, poses, render_poses, [H, W, focal], i_split
