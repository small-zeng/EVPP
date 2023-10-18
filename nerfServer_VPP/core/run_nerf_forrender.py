import os, sys
import numpy as np
import imageio
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange
import copy
from scipy.spatial.transform import Rotation as R

import matplotlib.pyplot as plt

from core.run_nerf_helpers_frorender import *

from skimage.metrics import structural_similarity as ssim
import lpips
import cv2 as cv

import logging
import colorlog
import open3d as o3d

# def get_pose(location,u,v):
#     sx = np.sin(u)
#     cx = np.cos(u)
#     sy = np.sin(v)
#     cy = np.cos(v)
#     return [[cy, sy*sx, -sy*cx, location[0]],
#                         [0, cx, sx, location[1]],
#                         [-sy, cy*sx, -cy*cx, location[2]],
#                         [0,0,0,1]]


log_colors_config = {
    'DEBUG': 'white',  # cyan white
    'INFO': 'green',
    'WARNING': 'yellow',
    'ERROR': 'red',
    'CRITICAL': 'bold_red',
}

logger = logging.getLogger('logger_name')

# 输出到控制台
console_handler = logging.StreamHandler()
# 输出到文件
file_handler = logging.FileHandler(filename='./logs/test.log', mode='a', encoding='utf8')

# 日志级别，logger 和 handler以最高级别为准，不同handler之间可以不一样，不相互影响
logger.setLevel(logging.INFO)
console_handler.setLevel(logging.INFO)
file_handler.setLevel(logging.INFO)

# 日志输出格式
file_formatter = logging.Formatter(
    fmt='[%(asctime)s.%(msecs)03d] %(filename)s -> %(funcName)s line:%(lineno)d [%(levelname)s] : %(message)s',
    datefmt='%Y-%m-%d  %H:%M:%S'
)
console_formatter = colorlog.ColoredFormatter(
    fmt='%(log_color)s[%(asctime)s.%(msecs)03d] %(filename)s -> %(funcName)s line:%(lineno)d [%(levelname)s] : %(message)s',
    datefmt='%Y-%m-%d  %H:%M:%S',
    log_colors=log_colors_config
)
console_handler.setFormatter(console_formatter)
file_handler.setFormatter(file_formatter)

# 重复日志问题：
# 1、防止多次addHandler；
# 2、loggername 保证每次添加的时候不一样；
# 3、显示完log之后调用removeHandler
if not logger.handlers:
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


device = torch.device(cuda_id if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False

loss_fn_alex = lpips.LPIPS(net='squeeze').to(device)


def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn
    def ret(inputs):
        logging.debug("applying fn and inputs length is  " + str(inputs.shape) + " , the size of chunk is " + str(chunk))
        return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret


def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024*64):
    """Prepares inputs and applies network 'fn'.
    """
    logging.debug("begin run network Prepares inputs and applies network 'fn'")
    logging.debug("inputs : " + str(inputs.shape) + " , " + "viewdirs : " + str(viewdirs.shape) + " , netchunk : " + str(netchunk))
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    logging.debug("inputs_flat : " + str(inputs_flat.shape))
    embedded = embed_fn(inputs_flat)

    if viewdirs is not None:
        logging.debug("viewdirs : " + str(viewdirs.shape))
        input_dirs = viewdirs[:,None].expand(inputs.shape)
        logging.debug("input_dirs : " + str(input_dirs.shape))
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        logging.debug("input_dirs_flat : " + str(input_dirs_flat.shape))
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        logging.debug("embedded_dirs : " + str(embedded_dirs.shape))
        embedded = torch.cat([embedded, embedded_dirs], -1)
        logging.debug("embedded : " + str(embedded.shape))

    logging.debug("after embedded inputs : " + str(embedded.shape))
    logging.debug("begin applying fn on embedded inputs")
    outputs_flat = batchify(fn, netchunk)(embedded)
    logging.debug("get ouputs from network : " + str(outputs_flat.shape))
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    logging.debug("get ouputs after reshape (final outputs) : " + str(outputs.shape))
    return outputs


def batchify_rays(views,rays_flat, chunk=1024*32, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(views,rays_flat[i:i+chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


def render(views,H, W, K, chunk=1024*32, rays=None, c2w=None, ndc=True,
                  near=0., far=1.,
                  use_viewdirs=False, c2w_staticcam=None,
                  **kwargs):
    """Render rays
    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for 
       camera while using other c2w argument for viewing directions.
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """
    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(H, W, K, c2w)
    else:
        # use provided ray batch
        rays_o, rays_d = rays

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, K, c2w_staticcam)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1,3]).float()

    sh = rays_d.shape # [..., 3]
    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1,3]).float()
    rays_d = torch.reshape(rays_d, [-1,3]).float()

    near, far = near * torch.ones_like(rays_d[...,:1]).to(device), far * torch.ones_like(rays_d[...,:1]).to(device)
    rays = torch.cat([rays_o, rays_d, near, far], -1)
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)

    # Render and reshape
    all_ret = batchify_rays(views,rays, chunk, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map', 'uncertainty_map','depth_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_list.append(rays_o)
    ret_list.append(rays_d)
    ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]


def render_path(views,gt,start,basedir,expname,testdata,render_poses,offset, hwf, K, chunk, render_kwargs, gt_imgs=None,depth_imgs=None,index = None, far = None, savedir=None, render_factor=0):

    H, W, focal = hwf
    print(H,W,focal)

    if render_factor!=0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor

    rgbs = []
    disps = []
    uncers = []
    depths = []

    allpoints = torch.zeros((0,3)).to(device)
    allpoints_gt = torch.zeros((0,3)).to(device)

    testsavedir = os.path.join(basedir, expname,'atestdata',testdata + '_{:06d}'.format(start))
    os.makedirs(testsavedir, exist_ok=True)

    t = time.time()
    for i, c2w in enumerate(tqdm(render_poses)):
        
        print(i, time.time() - t)
        t = time.time()
        rgb, disp, acc, uncer, depth, rays_o, rays_d, _ = render(views,H, W, K, chunk=chunk, c2w=c2w[:3,:4], **render_kwargs)

        

        target_depth_un = depth_imgs[index[i]]
        img_loss_perray_un = img2mesrespective(rgb, gt_imgs[index[i]])
        img_loss_un = torch.sum(img_loss_perray_un)
        delta_un = torch.sum(uncer)
        depth_diff_un = target_depth_un - (depth/far)
        depth_loss_un = torch.sum(depth_diff_un**2)
        ####################################################
        depth_diff_un = torch.abs(depth_diff_un)
        depth_calc = depth_diff_un[target_depth_un != 1]
        depth_avg = torch.sum(depth_calc)/len(depth_calc)
        depth_avg = depth_avg*far
        ratio3 = len(depth_calc[depth_calc<(0.03/far)])/len(depth_calc)

        loss_un = torch.log(delta_un) + img_loss_un/(delta_un ** 2) + depth_loss_un

        psnr_un = mse2psnr(img2msepsnr(rgb, gt_imgs[index[i]]))
        

        in1  = rgb.T[None,...]
        in2  = gt_imgs[index[i]].T[None,...]
        lpips_value = loss_fn_alex(in1, in2)

        if savedir is not None:
            # print("begin rgb")
            rgb8 = to8b(rgb.cpu().numpy())
            filename = os.path.join(savedir, '{:03d}.png'.format(i+offset))
            imageio.imwrite(filename, rgb8)
            
            depth1 = depth/far*255
            rgb8 = (np.clip(depth1.cpu().numpy(),0,255)).astype(np.uint8)
            filename_img = os.path.join(savedir, 'depth_{:03d}.png'.format(i+offset))
            imageio.imwrite(filename_img, rgb8)

        ssim_value = ssim(rgb.cpu().numpy(), gt_imgs[index[i]].cpu().numpy(),data_range = 1,multichannel=True)

        ####################################### sample surface points ###############################################
        depth = depth.reshape((H*W),1)
        target_depth_un = target_depth_un.reshape((H*W))
  
        rays_o = rays_o[target_depth_un != 1]
        rays_d = rays_d[target_depth_un != 1]
        target_depth_un = target_depth_un.reshape((H*W,1))
        points = rays_o + rays_d * depth[target_depth_un != 1][...,None]
        if gt:
            points_gt = rays_o + rays_d * target_depth_un[target_depth_un != 1][...,None] * far
            allpoints_gt = torch.cat([allpoints_gt,points_gt[::30]],0)

        allpoints = torch.cat([allpoints,points[::30]],0)

        uncer_data = torch.sum(uncer,-1)
        uncer_data = uncer_data.reshape((H,W))
        uncer_data = 1/uncer_data

        uncer_data = (uncer_data*2)**2
        uncer_data = uncer_data.cpu().numpy()
        rgb8 = (np.clip(uncer_data,0,255)).astype(np.uint8)
        filename_img = os.path.join(testsavedir, 'uncer_{:03d}.png'.format(i + offset))
        # imageio.imwrite(filename_img, rgb8)
        filename = os.path.join(basedir, expname,'atestdata', testdata+'.txt')
        with open(filename,'a') as f: 
            f.write("current step is + " + str(start)+"\n")
            f.write("current img is + " + str(index[i]) +"\n")
            f.write("img loss is + " + str(img_loss_un.item()) +"\n")
            f.write("uncertainty sum is + " + str(delta_un.item()) +"\n")
            f.write("loss_depth_un is + " + str(depth_loss_un.item()) +"\n")
            f.write("final loss is + " + str(loss_un.item()) +"\n")
            f.write("psnr is + " + str(psnr_un.item()) +"\n")
            f.write("ssim is + " + str(ssim_value)+"\n")
            f.write("lpips is + " + str(lpips_value.item())+"\n")
            f.write("depth average is + " + str(depth_avg.item())+"\n")
            f.write("ratio3 is + " + str(ratio3)+"\n")


        if i==0:
            print(rgb.shape, disp.shape)

        """
        if gt_imgs is not None and render_factor==0:
            p = -10. * np.log10(np.mean(np.square(rgb.cpu().numpy() - gt_imgs[i])))
            print(p)
        """
    ####################################### sample surface points ###############################################
    filename_p = os.path.join(basedir, expname,'atestdata', testdata+'_points.csv')
    np.savetxt(filename_p,allpoints.cpu().numpy(),delimiter=',',fmt='%.04f')
    
    pcd_allpoints = o3d.geometry.PointCloud()
    pcd_allpoints.points = o3d.utility.Vector3dVector(allpoints.cpu().numpy())
    filename_pcd = os.path.join(basedir, expname,'atestdata', testdata+'_points.xyz')
    o3d.io.write_point_cloud(filename_pcd,pcd_allpoints)
    if gt:
        filename_p = os.path.join(basedir, expname,'atestdata', testdata+'_points_gt.csv')
        np.savetxt(filename_p,allpoints_gt.cpu().numpy(),delimiter=',',fmt='%.04f')
        
        pcd_allpoints_gt = o3d.geometry.PointCloud()
        pcd_allpoints_gt.points = o3d.utility.Vector3dVector(allpoints_gt.cpu().numpy())
        filename_pcd = os.path.join(basedir, expname,'atestdata', testdata+'_points_gt.xyz')
        o3d.io.write_point_cloud(filename_pcd,pcd_allpoints_gt)
        


    # rgbs = np.stack(rgbs, 0)
    # disps = np.stack(disps, 0)

    return rgbs, disps, uncers, depths

def render_path_train(views,gt,start,basedir,expname,testdata,render_poses,offset, hwf, K, chunk, render_kwargs, gt_imgs=None,depth_imgs=None,index = None, far = None, savedir=None, render_factor=0):
    
    H, W, focal = hwf
    print(H,W,focal)

    if render_factor!=0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor

    rgbs = []
    disps = []
    uncers = []
    depths = []

    allpoints = torch.zeros((0,3)).to(device)
    allpoints_gt = torch.zeros((0,3)).to(device)

    testsavedir = os.path.join(basedir, expname,'atestdata',testdata + '_{:06d}'.format(start))
    os.makedirs(testsavedir, exist_ok=True)

    t = time.time()
    k = -1
    for i, c2w in enumerate(tqdm(render_poses)):
        
        print(i, time.time() - t)
        t = time.time()
        for j in range(5):
            k = k + 1
            u_noise = np.random.randn()*5.0
            v_noise = np.random.randn()*10.0
            r = R.from_euler('ZYX', [u_noise, v_noise, 0], degrees=True)
            pose_uv_noise = torch.Tensor(r.as_matrix()).to(device)
            print("pose_noise = ",pose_uv_noise)
            # c2w[:3,:3] = pose_uv_noise @ c2w[:3,:3]
            c2w[:3,3] = c2w[:3,3] + torch.randn((3)).to(device) * 0.15
            filename_img = os.path.join(savedir, 'pose_{:03d}.csv'.format(k+offset))
            np.savetxt(filename_img,c2w.cpu().numpy(),delimiter=",")

            rgb, disp, acc, uncer, depth, rays_o, rays_d, _ = render(views,H, W, K, chunk=chunk, c2w=c2w[:3,:4], **render_kwargs)

            target_depth_un = depth_imgs[index[i]]
            img_loss_perray_un = img2mesrespective(rgb, gt_imgs[index[i]])
            img_loss_un = torch.sum(img_loss_perray_un)
            delta_un = torch.sum(uncer)
            depth_diff_un = target_depth_un - (depth/far)
            depth_loss_un = torch.sum(depth_diff_un**2)
            ####################################################
            depth_diff_un = torch.abs(depth_diff_un)
            depth_calc = depth_diff_un[target_depth_un != 1]
            depth_avg = torch.sum(depth_calc)/len(depth_calc)
            depth_avg = depth_avg*far
            ratio3 = len(depth_calc[depth_calc<(0.03/far)])/len(depth_calc)

            loss_un = torch.log(delta_un) + img_loss_un/(delta_un ** 2) + depth_loss_un

            psnr_un = mse2psnr(img2msepsnr(rgb, gt_imgs[index[i]]))
            

            in1  = rgb.T[None,...]
            in2  = gt_imgs[index[i]].T[None,...]
            lpips_value = loss_fn_alex(in1, in2)

            if savedir is not None:
                # print("begin rgb")
                rgb8 = to8b(rgb.cpu().numpy())
                filename = os.path.join(savedir, '{:03d}.png'.format(k+offset))
                imageio.imwrite(filename, rgb8)
                
                depth1 = depth/far*255
                rgb8 = (np.clip(depth1.cpu().numpy(),0,255)).astype(np.uint8)
                filename_img = os.path.join(savedir, 'depth_{:03d}.png'.format(k+offset))
                imageio.imwrite(filename_img, rgb8)

            ssim_value = ssim(rgb.cpu().numpy(), gt_imgs[index[i]].cpu().numpy(),data_range = 1,multichannel=True)

            ####################################### sample surface points ###############################################
            depth = depth.reshape((H*W),1)
            target_depth_un = target_depth_un.reshape((H*W))

            rays_o = rays_o[target_depth_un != 1]
            rays_d = rays_d[target_depth_un != 1]
            target_depth_un = target_depth_un.reshape((H*W,1))
            points = rays_o + rays_d * depth[target_depth_un != 1][...,None]
            if gt:
                points_gt = rays_o + rays_d * target_depth_un[target_depth_un != 1][...,None] * far
                allpoints_gt = torch.cat([allpoints_gt,points_gt[::30]],0)

            allpoints = torch.cat([allpoints,points[::30]],0)

            uncer_data = torch.sum(uncer,-1)
            uncer_data = uncer_data.reshape((H,W))
            uncer_data = 1/uncer_data

            uncer_data = (uncer_data*2)**2
            uncer_data = uncer_data.cpu().numpy()
            rgb8 = (np.clip(uncer_data,0,255)).astype(np.uint8)
            filename_img = os.path.join(testsavedir, 'uncer_{:03d}.png'.format(k + offset))
            # # imageio.imwrite(filename_img, rgb8)
            # filename = os.path.join(basedir, expname,'atestdata', testdata+'.txt')
            # with open(filename,'a') as f: 
            #     f.write("current step is + " + str(start)+"\n")
            #     f.write("current img is + " + str(index[i]) +"\n")
            #     f.write("img loss is + " + str(img_loss_un.item()) +"\n")
            #     f.write("uncertainty sum is + " + str(delta_un.item()) +"\n")
            #     f.write("loss_depth_un is + " + str(depth_loss_un.item()) +"\n")
            #     f.write("final loss is + " + str(loss_un.item()) +"\n")
            #     f.write("psnr is + " + str(psnr_un.item()) +"\n")
            #     f.write("ssim is + " + str(ssim_value)+"\n")
            #     f.write("lpips is + " + str(lpips_value.item())+"\n")
            #     f.write("depth average is + " + str(depth_avg.item())+"\n")
            #     f.write("ratio3 is + " + str(ratio3)+"\n")


            # if i==0:
            #     print(rgb.shape, disp.shape)

        # ####################################### sample surface points ###############################################
        # filename_p = os.path.join(basedir, expname,'atestdata', testdata+'_points.csv')
        # np.savetxt(filename_p,allpoints.cpu().numpy(),delimiter=',',fmt='%.04f')
        
        # pcd_allpoints = o3d.geometry.PointCloud()
        # pcd_allpoints.points = o3d.utility.Vector3dVector(allpoints.cpu().numpy())
        # filename_pcd = os.path.join(basedir, expname,'atestdata', testdata+'_points.xyz')
        # o3d.io.write_point_cloud(filename_pcd,pcd_allpoints)
        # if gt:
        #     filename_p = os.path.join(basedir, expname,'atestdata', testdata+'_points_gt.csv')
        #     np.savetxt(filename_p,allpoints_gt.cpu().numpy(),delimiter=',',fmt='%.04f')
            
        #     pcd_allpoints_gt = o3d.geometry.PointCloud()
        #     pcd_allpoints_gt.points = o3d.utility.Vector3dVector(allpoints_gt.cpu().numpy())
        #     filename_pcd = os.path.join(basedir, expname,'atestdata', testdata+'_points_gt.xyz')
        #     o3d.io.write_point_cloud(filename_pcd,pcd_allpoints_gt)
        
    return rgbs, disps, uncers, depths


def create_nerf(args):
    """Instantiate NeRF's MLP model.
    """
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)
    output_ch = 5 if args.N_importance > 0 else 4
    skips = [4]
    model = NeRF(D=args.netdepth, W=args.netwidth,
                 input_ch=input_ch, output_ch=output_ch, skips=skips,
                 input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
    grad_vars = list(model.parameters())

    logging.info("corase model grad_vars length :  " + str(len(grad_vars)))

    model_fine = None
    if args.N_importance > 0:
        model_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                          input_ch=input_ch, output_ch=output_ch, skips=skips,
                          input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
        grad_vars += list(model_fine.parameters())

    logging.info("both model grad_vars length :  " + str(len(grad_vars)))

    network_query_fn = lambda inputs, viewdirs, network_fn : run_network(inputs, viewdirs, network_fn,
                                                                embed_fn=embed_fn,
                                                                embeddirs_fn=embeddirs_fn,
                                                                netchunk=args.netchunk)

    # Create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################

    logging.info("Load checkpoints")
    # Load checkpoints
    if args.ft_path is not None and args.ft_path!='None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if 'tar' in f]

    logging.info('Found ckpts' + str(ckpts))
    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        logging.info('Reloading from' + str(ckpt_path))
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        logging.info("check point : " + str(ckpt))
        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])
        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])

    ##########################

    render_kwargs_train = {
        'network_query_fn' : network_query_fn,
        'perturb' : args.perturb,
        'N_importance' : args.N_importance,
        'network_fine' : model_fine,
        'N_samples' : args.N_samples,
        'network_fn' : model,
        'use_viewdirs' : args.use_viewdirs,
        'white_bkgd' : args.white_bkgd,
        'raw_noise_std' : args.raw_noise_std,
    }

    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != 'llff' or args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    logging.info("render_kwargs_train : " + str(render_kwargs_train) + " , render_kwargs_test : " + str(render_kwargs_test) + " , start : " + str(start))
    logging.info("grad_vars lengths: " + str(len(grad_vars)))
    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer


def raw2outputs(mask, raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)

    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).to(device).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

    rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]

    ############################### draw path and views ##############################
    # color = torch.Tensor([0,0,0]).to(device)
    # density = torch.Tensor([20]).to(device)
    # rgb[mask] = color
    # raw[...,3][mask] = density
    ############################### draw path and views ##############################
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[...,3].shape).to(device) * raw_noise_std

        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[...,3].shape)) * raw_noise_std
            noise = torch.Tensor(noise).to(device)

    alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_samples]
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)).to(device), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map).to(device), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map[...,None])

    return rgb_map, disp_map, acc_map, weights, depth_map

def get_pose(u,v):
    # u = np.pi * u / 180
    # v = np.pi * v / 180
    sx = np.sin(u)
    cx = np.cos(u)
    sy = np.sin(v)
    cy = np.cos(v)
    return torch.Tensor([[cy, sy*sx, -sy*cx],
                        [0, cx, sx],
                        [-sy, cy*sx, -cy*cx]]).to(device)


def render_rays(views,
                ray_batch,
                network_fn,
                network_query_fn,
                N_samples,
                retraw=False,
                lindisp=False,
                perturb=0.,
                N_importance=0,
                network_fine=None,
                white_bkgd=False,
                raw_noise_std=0.,
                verbose=False,
                pytest=False):
    """Volumetric rendering.
    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_fn: function. Model for predicting RGB and density at each point
        in space.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      network_fine: "fine" network with same spec as network_fn.
      white_bkgd: bool. If True, assume a white background.
      raw_noise_std: ...
      verbose: bool. If True, print more debugging info.
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] # [N_rays, 3] each
    viewdirs = ray_batch[:,-3:] if ray_batch.shape[-1] > 8 else None
    bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2])
    near, far = bounds[...,0], bounds[...,1] # [-1,1]

    t_vals = torch.linspace(0., 1., steps=N_samples).to(device)
    if not lindisp:
        z_vals = near * (1.-t_vals) + far * (t_vals)
    else:
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

    z_vals = z_vals.expand([N_rays, N_samples]).to(device)

    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = torch.cat([mids, z_vals[...,-1:]], -1).to(device)
        lower = torch.cat([z_vals[...,:1], mids], -1).to(device)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape).to(device)

        # Pytest, overwrite u with numpy's fixed random numbers
        if pytest:
            np.random.seed(0)
            t_rand = np.random.rand(*list(z_vals.shape))
            t_rand = torch.Tensor(t_rand).to(device)

        z_vals = lower + (upper - lower) * t_rand

    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]


#     raw = run_network(pts)
    raw = network_query_fn(pts, viewdirs, network_fn)


    

    ################################################ mask #######################################################
    # mask =None
    # for path in views[:3]:
    #     print("path length:",len(path))
    #     # for i in range(1,len(path)):
    #     start = path[0]
    #     end = path[-1]
    #     print("statr:",start,"end:",end)
    #     dirr = end - start
    #     length = (dirr[0] ** 2 + dirr[1] ** 2 + dirr[2] ** 2)** (1/2)
    #     if dirr[1] > 0:
    #         u = np.arccos((dirr[2])/( (dirr[1]**2+dirr[2]**2)**(1/2) ) )
    #     else:
    #         u = -np.arccos((dirr[2])/( (dirr[1]**2+dirr[2]**2)**(1/2) ) )
    #     if dirr[0] > 0:
    #         v = -np.arccos((-dirr[2])/( (dirr[0]**2+dirr[2]**2)**(1/2) ) )
    #     else:
    #         v = np.arccos((-dirr[2])/( (dirr[0]**2+dirr[2]**2)**(1/2) ) )
    #     print("u:",u/np.pi*180)
    #     print("v:",v/np.pi*180)
    #     rot = get_pose(u,v)
    #     tmp = copy.deepcopy(pts)
    #     tmp = tmp + torch.Tensor(-start).to(device)
    #     tmp = torch.matmul(tmp,rot)
    #     x = tmp[...,0]
    #     y = tmp[...,1]
    #     z = tmp[...,2]
    #     if mask != None:
    #         mask = ((x**2+y**2 <= 0.005) & (z < length) & (z > 0)) | mask
    #     else:
    #         mask = (x**2+y**2 <= 0.005) & (z < length) & (z > 0)
    #     print("current mask:",len(mask == True))


    mask = pts != None
    # start = np.array('-2.463548204044025,0.4582530004146295,-1.9698651998438088,'.split(',')[:3]).astype(np.float)
    # end = np.array('-3.7016431721286676,0.6852894856235149,0.07698822247872664,'.split(',')[:3]).astype(np.float)
    # dirr = end - start
    # print("dirr:",dirr)
    # length = (dirr[0] ** 2 + dirr[1] ** 2 + dirr[2] ** 2)** (1/2)
    # print("length:",length)
    # if dirr[1] > 0:
    #     u = np.arccos((dirr[2])/( (dirr[1]**2+dirr[2]**2)**(1/2) ) )
    # else:
    #     u = -np.arccos((dirr[2])/( (dirr[1]**2+dirr[2]**2)**(1/2) ) )
    # if dirr[0] > 0:
    #     v = -np.arccos((-dirr[2])/( (dirr[0]**2+dirr[2]**2)**(1/2) ) )
    # else:
    #     v = np.arccos((-dirr[2])/( (dirr[0]**2+dirr[2]**2)**(1/2) ) )
    # print("u:",u/np.pi*180)
    # print("v:",v/np.pi*180)
    # rot = get_pose(u,v)
    # tmp = pts + torch.Tensor(-start).to(device)
    # tmp = torch.matmul(tmp,rot)
    # x = tmp[...,0]
    # y = tmp[...,1]
    # z = tmp[...,2]
    # mask = (x**2+y**2 <= 0.005) & (z < length) & (z > 0) & mask
    # print("mask:",len(mask==True))


    # u = 90
    # v = 90
    # position = [0,-1,0]
    # rot = get_pose(u,v)
    # pts += torch.Tensor(position).to(device)
    # pts = torch.matmul(pts,rot)
    
    # x = pts[...,0]
    # y = pts[...,1]
    # z = pts[...,2]
    # # mask = (pts[...,0] > -1) & (pts[...,0] < 1) & (pts[...,1] > 1) & (pts[...,1] < 4) & (pts[...,2] > -1) & (pts[...,2] < 1)
    # mask = ((x**2+y**2 <= 0.005) & (z < 2.72) & (z > 0)) 
    # # mask = pts != None
    
 
    uncertainty_map = raw[...,4]
    logging.debug("get from coarse model raw's shape : " + str(raw.shape) + " uncertainty shape :" + str(uncertainty_map.shape))
    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(mask,raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

    


    if N_importance > 0:

        rgb_map_0, disp_map_0, acc_map_0, uncertainty_map0, depth_map0 = rgb_map, disp_map, acc_map, uncertainty_map, depth_map

        z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], N_importance, det=(perturb==0.), pytest=pytest)
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_importance, 3]

        run_fn = network_fn if network_fine is None else network_fine
#         raw = run_network(pts, fn=run_fn)
        raw = network_query_fn(pts, viewdirs, run_fn)
        
        ################################################ mask #######################################################
        # mask = None
        # for path in views[:3]:
        #     # print("path length:",len(path))
        #     # for i in range(1,len(path)):
        #     start = path[0]
        #     end = path[-1]
        #     dirr = end - start
        #     length = (dirr[0] ** 2 + dirr[1] ** 2 + dirr[2] ** 2)** (1/2)
        #     if dirr[1] > 0:
        #         u = np.arccos((dirr[2])/( (dirr[1]**2+dirr[2]**2)**(1/2) ) )
        #     else:
        #         u = -np.arccos((dirr[2])/( (dirr[1]**2+dirr[2]**2)**(1/2) ) )
        #     if dirr[0] > 0:
        #         v = -np.arccos((-dirr[2])/( (dirr[0]**2+dirr[2]**2)**(1/2) ) )
        #     else:
        #         v = np.arccos((-dirr[2])/( (dirr[0]**2+dirr[2]**2)**(1/2) ) )
        #     tmp = copy.deepcopy(pts)
        #     tmp = tmp + torch.Tensor(-start).to(device)
        #     tmp = torch.matmul(tmp,rot)
        #     x = tmp[...,0]
        #     y = tmp[...,1]
        #     z = tmp[...,2]
        #     if mask != None:
        #         mask = ((x**2+y**2 <= 0.005) & (z < length) & (z > 0)) | mask
        #     else:
        #         mask = (x**2+y**2 <= 0.005) & (z < length) & (z > 0)
        #     # print("current mask:",len(mask == True))
        

        mask = pts != None
        # # start = np.array('0.0,1.0,-3.0,'.split(',')[:3]).astype(np.float)
        # # end = np.array('-2.463548204044025,0.4582530004146295,-1.9698651998438088,'.split(',')[:3]).astype(np.float)
        # dirr = end - start
        # # print("dir:",dir)
        # length = (dirr[0] ** 2 + dirr[1] ** 2 + dirr[2] ** 2)** (1/2)
        # # print("length:",length)
        # if dirr[1] > 0:
        #     u = np.arccos((dirr[2])/( (dirr[1]**2+dirr[2]**2)**(1/2) ) )
        # else:
        #     u = -np.arccos((dirr[2])/( (dirr[1]**2+dirr[2]**2)**(1/2) ) )
        # if dirr[0] > 0:
        #     v = -np.arccos((-dirr[2])/( (dirr[0]**2+dirr[2]**2)**(1/2) ) )
        # else:
        #     v = np.arccos((-dirr[2])/( (dirr[0]**2+dirr[2]**2)**(1/2) ) )
        # # print("u:",u/np.pi*180)
        # # print("v:",v/np.pi*180)
        # rot = get_pose(u,v)
        # tmp = pts + torch.Tensor(-start).to(device)
        # tmp = torch.matmul(tmp,rot)
        # x = tmp[...,0]
        # y = tmp[...,1]
        # z = tmp[...,2]
        # mask = (x**2+y**2 <= 0.005) & (z < length) & (z > 0) & mask
        # # print("mask:",len(mask==True))


        # rot = get_pose(u,v)
        # pts += torch.Tensor(position).to(device)
        # pts = torch.matmul(pts,rot)
        # x = pts[...,0]
        # y = pts[...,1]
        # z = pts[...,2]
        # # mask = (pts[...,0] > -1) & (pts[...,0] < 1) & (pts[...,1] > 0.5) & (pts[...,1] < 2) & (pts[...,2] > -1) & (pts[...,2] < 1)
        # mask = ((x**2+y**2 <= 0.005) & (z < 2.72) & (z > 0)) 
        # # mask = pts != None

        uncertainty_map = raw[...,4]
        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(mask, raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)
        logging.debug("get from fine model raw's shape : " + str(raw.shape) + " uncertainty shape :" + str(uncertainty_map.shape))
    ret = {'rgb_map' : rgb_map, 'disp_map' : disp_map, 'acc_map' : acc_map, 'uncertainty_map' : uncertainty_map, 'depth_map': depth_map}
    if retraw:
        ret['raw'] = raw
    if N_importance > 0:
        ret['rgb0'] = rgb_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0
        ret['uncertainty_map0'] = uncertainty_map0
        ret['depth_map0'] = depth_map0
        ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

        

    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret


def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, 
                        help='config file path')
    parser.add_argument("--expname", type=str, 
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/', 
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern', 
                        help='input data directory')

    # training options
    parser.add_argument("--netdepth", type=int, default=8, 
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256, 
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8, 
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256, 
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32*32*4, 
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4, 
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250, 
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024*32, 
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*64, 
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true', 
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true', 
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None, 
                        help='specific weights npy file to reload for coarse network')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64, 
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true', 
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0, 
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10, 
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4, 
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0., 
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    parser.add_argument("--render_only", action='store_true', 
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true', 
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_factor", type=int, default=0, 
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # training options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops') 

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff', 
                        help='options: llff / blender / deepvoxels')
    parser.add_argument("--testskip", type=int, default=8, 
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    ## deepvoxels flags
    parser.add_argument("--shape", type=str, default='greek', 
                        help='options : armchair / cube / greek / vase')

    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true', 
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true', 
                        help='load blender synthetic data at 400x400 instead of 800x800')

    ## llff flags
    parser.add_argument("--factor", type=int, default=8, 
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true', 
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true', 
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true', 
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8, 
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=100, 
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img",     type=int, default=500, 
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=10000, 
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=50000, 
                        help='frequency of testset saving')
    parser.add_argument("--i_video",   type=int, default=50000, 
                        help='frequency of render_poses video saving')
    parser.add_argument("--i_uncertainty",   type=int, default=2000, 
                        help='frequency of render_poses img and corresponding uncertainty value')

    return parser


def train():

    parser = config_parser()
    args = parser.parse_args()

    logging.info(str(args))

    # Load data
    K = None
    if args.dataset_type == 'llff':
        images, poses, bds, render_poses, i_test = load_llff_data(args.datadir, args.factor,
                                                                  recenter=True, bd_factor=.75,
                                                                  spherify=args.spherify)
        hwf = poses[0,:3,-1]
        poses = poses[:,:3,:4]
        print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)
        if not isinstance(i_test, list):
            i_test = [i_test]

        if args.llffhold > 0:
            print('Auto LLFF holdout,', args.llffhold)
            i_test = np.arange(images.shape[0])[::args.llffhold]

        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                        (i not in i_test and i not in i_val)])

        print('DEFINING BOUNDS')
        if args.no_ndc:
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.
            
        else:
            near = 0.
            far = 1.
        print('NEAR FAR', near, far)

    elif args.dataset_type == 'blender':
        images, depth_imgs, poses, render_poses, hwf, i_split = load_blender_data_depth(args.datadir, args.half_res, args.testskip)
        logging.info("depth images shape is : " + str(depth_imgs.shape))
        logging.info('Loaded blender image shape is ' + str(images.shape) + " render_poses shape is " + str(render_poses.shape)+ " hwf are " + str(hwf) + " args.datadir  " + str(args.datadir))
        print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        near = 0.5
        far = 80

        if args.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]

    elif args.dataset_type == 'LINEMOD':
        images, poses, render_poses, hwf, K, i_split, near, far = load_LINEMOD_data(args.datadir, args.half_res, args.testskip)
        print(f'Loaded LINEMOD, images shape: {images.shape}, hwf: {hwf}, K: {K}')
        print(f'[CHECK HERE] near: {near}, far: {far}.')
        i_train, i_val, i_test = i_split

        if args.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]

    elif args.dataset_type == 'deepvoxels':

        images, poses, render_poses, hwf, i_split = load_dv_data(scene=args.shape,
                                                                 basedir=args.datadir,
                                                                 testskip=args.testskip)

        print('Loaded deepvoxels', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        hemi_R = np.mean(np.linalg.norm(poses[:,:3,-1], axis=-1))
        near = hemi_R-1.
        far = hemi_R+1.

    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    logging.info("Cast intrinsics to right types")
    logging.info("hwf: " + str(hwf))

    if K is None:
        K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])

    if args.render_test:
        render_poses = np.array(poses[i_test])

    logging.info("begin Create log dir and copy the config file")
    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    logging.info("begin Create nerf model")
    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)

    logging.info("nerf created")

    global_step = start

    bds_dict = {
        'near' : near,
        'far' : far,
    }

    logging.info("near and far bound : " + str(bds_dict))

    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # Move testing data to GPU
    render_poses = torch.Tensor(render_poses).to(device)

    # Short circuit if only rendering out from trained model
    if args.render_only:
        logging.info("RENDER")
        print('RENDER ONLY')
        with torch.no_grad():
            if args.render_test:
                # render_test switches to test poses
                images = images[i_test]
            else:
                # Default is smoother render_poses path
                images = None

            testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:06d}'.format('test' if args.render_test else 'path', start))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', render_poses.shape)

            rgbs_u, _ ,uncers, depths  = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test, gt_imgs=images, savedir=testsavedir, render_factor=args.render_factor)
            print('Done rendering', testsavedir)
            imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'), to8b(rgbs_u), fps=30, quality=8)

            return

    # Prepare raybatch tensor if batching random rays
    N_rand = args.N_rand
    use_batching = not args.no_batching
    if use_batching:
        # For random ray batching
        print('get rays')
        rays = np.stack([get_rays_np(H, W, K, p) for p in poses[:,:3,:4]], 0) # [N, ro+rd, H, W, 3]
        print('done, concats')
        rays_rgb = np.concatenate([rays, images[:,None]], 1) # [N, ro+rd+rgb, H, W, 3]
        rays_rgb = np.transpose(rays_rgb, [0,2,3,1,4]) # [N, H, W, ro+rd+rgb, 3]
        rays_rgb = np.stack([rays_rgb[i] for i in i_train], 0) # train images only
        rays_rgb = np.reshape(rays_rgb, [-1,3,3]) # [(N-1)*H*W, ro+rd+rgb, 3]
        rays_rgb = rays_rgb.astype(np.float32)
        print('shuffle rays')
        np.random.shuffle(rays_rgb)

        print('done')
        i_batch = 0
        logging.info("ras_rgb shape : " + str(rays_rgb.shape))
        logging.debug("get rays : " + str(rays_rgb.head()))

    # Move training data to GPU
    if use_batching:
        images = torch.Tensor(images).to(device)
    images = torch.Tensor(images).to(device)
    depth_imgs = torch.Tensor(depth_imgs).to(device)
    poses = torch.Tensor(poses).to(device)
    if use_batching:
        rays_rgb = torch.Tensor(rays_rgb).to(device)

    all_trian_views = i_train
    i_train = all_trian_views[:4]
    i_trianed = []
    all_train_index = 4
    N_iters = 200000 + 1
    print('Begin')
    print('ALL TRAIN views are', all_trian_views)
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)

    logging.info('TRAIN views are' + str(i_train))
    logging.info('TEST views are' + str(i_test))
    logging.info('VAL views are'+ str(i_val))
    # Summary writers
    # writer = SummaryWriter(os.path.join(basedir, 'summaries', expname))
    rays_loss = None
    # indicate whether this img have involved in train
    img_flag = [None,None,None,None]
    select_coords = None
    current_select_coords = None
    coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H).to(device), torch.linspace(0, W-1, W).to(device)), -1)  # (H, W, 2)
    coords = torch.reshape(coords, [-1,2])  # (H * W, 2)
    start = start + 1
    for i in trange(start, N_iters):
        time0 = time.time()

        if rays_loss is not None:
            rays_loss = rays_loss.reshape((len(i_train),int(N_rand/len(i_train)))) # N_img, N_sample_per_img
            old_select_coords = all_select
            old_select_coords = old_select_coords.reshape((len(i_train),int(N_rand/len(i_train)),2)) # N_img, N_sample_per_img,2
            select_coords = torch.zeros((0,2))

        all_rays_o = torch.zeros((0,3)).to(device)
        all_rays_d = torch.zeros((0,3)).to(device)
        all_target_depth = torch.zeros((0)).to(device)
        all_target_img = torch.zeros((0,3)).to(device)
        all_select = torch.zeros(0,2).to(device)
        select_coords = None
        for i_img in np.arange(0,len(i_train),1):
            N_per_img = int(N_rand/len(i_train))
            img = i_train[i_img]
            origin_img = images[img]
            depth_img = depth_imgs[img]
            pose = poses[img, :3,:4]
            rays_o, rays_d = get_rays(H, W, K, torch.Tensor(pose).to(device))  # (H, W, 3), (H, W, 3)
            # 该图第一次采样，随机采样
            if img_flag[i_img] is None:
                select_inds = np.random.choice(coords.shape[0], size=[N_per_img], replace=False)  # (N_per_img,)
                select_coords = coords[select_inds].long()  # (N_per_img, 2)
            # 该图已参与训练，重采样
            else:
                current_select_coords = old_select_coords[i_img]
                select_coords = torch.zeros((0,2)).to(device)
                ## 75%重采样，其余随机分布
                N_resample = int(N_per_img/4 * 3)
                particles_distribution_num = rays_loss[i_img] / torch.sum(rays_loss[i_img]) * N_resample
                particles_distribution_num = particles_distribution_num.long()
                for particle_index in np.arange(0,len(particles_distribution_num),1):
                    if particles_distribution_num[particle_index] > 0:
                        bias_h = torch.randn((particles_distribution_num[particle_index]))*3
                        bias_w = torch.randn((particles_distribution_num[particle_index]))*3
                        generate_coords_h = bias_h + current_select_coords[particle_index][0]
                        generate_coords_w = bias_w + current_select_coords[particle_index][1]
                        generate_coords = torch.stack([generate_coords_h,generate_coords_w],-1)
                        select_coords = torch.cat([select_coords,generate_coords],0)
                # 去除超出边界的点
                select_coords = select_coords[select_coords[0] > 0]
                select_coords = select_coords[select_coords[0] < H]
                select_coords = select_coords[select_coords[1] > 0]
                select_coords = select_coords[select_coords[1] < W]
                N_randsample = N_randsample - len(select_coords) + int(N_rand/4)
                select_inds = np.random.choice(coords.shape[0], size=[N_randsample], replace=False)  # (N_randsample,)
                select_coords = torch.cat([select_coords,coords[select_inds].long()],0)  # (N_per_img, 2)
            rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_per_img, 3)
            rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_per_img, 3)
            target_depth = depth_img[select_coords[:, 0], select_coords[:, 1]] # N_per_img,4
            target_s = origin_img[select_coords[:, 0], select_coords[:, 1]]  # (N_per_img, 3)
            all_select = torch.cat([all_select,select_coords],0)
            all_rays_o = torch.cat([all_rays_o,rays_o],0)
            all_rays_d = torch.cat([all_rays_d,rays_d],0)
            all_target_depth = torch.cat([all_target_depth,target_depth],0)
            all_target_img = torch.cat([all_target_img,target_s],0)
        batch_rays = torch.stack([all_rays_o, all_rays_d], 0).to(device)


        #####  Core optimization loop  #####
        logging.info("begin render image and core optimization")
        rgb, disp, acc, uncertainty,depth, extras = render(H, W, K, chunk=args.chunk, rays=batch_rays,
                                                verbose=i < 10, retraw=True,
                                                **render_kwargs_train)
        logging.info("get from outputs and rendered resukts")
        logging.info("rgb shape is " + str(rgb.shape) + " ,disp shape is" + str(disp.shape) + " , acc shape is " + str(acc.shape) + " ,extras are " + str(extras.keys()))
        logging.info("uncertainty shape is " + str(uncertainty.shape) + " , targets shape is " + str(target_s.shape))
        logging.debug("uncertainty datas like : " + str(uncertainty[0]))
        optimizer.zero_grad()

        logging.info("############################## current global step is " + str(global_step) + "  ###############################")
        
        img_loss_perray = img2mesrespective(rgb, all_target_img)
        img_loss = torch.sum(img_loss_perray)
        depth_diff = all_target_depth - (depth/8.94)
        depth_loss = torch.sum(depth_diff**2)
        delta = torch.sum(uncertainty)
        ems_loss = img_loss + 20*depth_loss
        loss = torch.log(delta) + img_loss/(delta ** 2) + depth_loss
        # loss = ems_loss
        
        psnr = mse2psnr(img2msepsnr(rgb,all_target_img))
        logging.info("img_loss is " + str(img_loss.item()) + " , delta is " + str(delta.item())+ " , device type : " + str(delta.device))
        logging.info("depth loss is ," + str(depth_loss.item())+", ems loss is ," + str(ems_loss.item()))
        logging.info("fine loss : " + str(loss.item()) + " fine img_psnr is " + str(psnr.item()))

        psnrs = mse2psnr(img2msepsnr(rgb.reshape((4,256,3)),all_target_img.reshape((4,256,3))))
        logging.info("psnrs of trained imgs is " + str(psnrs))
        for psnr_index in np.arange(0,len(psnrs),1):
            if psnrs[psnr_index] > 29:
                i_trianed.append(i_train[psnr_index])
                if (torch.rand((1)) < 0.5 and all_train_index < len(all_trian_views) - 1):
                    i_train[psnr_index] = all_trian_views[all_train_index]
                    all_train_index = all_train_index + 1
                else:
                    i_train[psnr_index] = np.random.choice(i_trianed)
                print('TRAIN views are', i_train)
                print('TRAINED views are', i_trianed)
                logging.info('TRAIN views are' + str(i_train))    
                logging.info('TRAINED views are' + str(i_trianed))  
        

        if 'rgb0' in extras:
            
            img_loss_perray0 = img2mesrespective(extras['rgb0'], all_target_img)
            img_loss0 = torch.sum(img_loss_perray0)
            uncertainty0 = extras['uncertainty_map0']
            delta0 = torch.sum(uncertainty0)

            loss0 = torch.log(delta0) + img_loss0/(delta0 ** 2)
            # loss0 = img_loss0
            loss = loss + loss0
            psnr0 = mse2psnr(img2msepsnr(extras['rgb0'], all_target_img))
            logging.info("img_loss0 is " + str(img_loss0.item()) + " , delta0 is " + str(delta0.item())+ " , device type : " + str(delta0.device))
            logging.info("corase img_loss : " + str(loss0.item()) + " coarse img_psnr0 is " + str(psnr0.item()))
        logging.info("final loss is  " + str(loss) + " begin backward optimize")
        loss.backward()
        optimizer.step()

        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        logging.info("decay_Rate : " + str(decay_rate)+ " , decay_steps : " + str(decay_steps) + " , new_lrate is " + str(new_lrate) + " , current step is " + str(global_step))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        ################################

        dt = time.time()-time0
        # print(f"Step: {global_step}, Loss: {loss}, Time: {dt}")
        #####           end            #####

        # Rest is logging
        # logging.info("begin saving *******************************************")
        if i%args.i_weights==0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            torch.save({
                'global_step': global_step,
                'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
            print('Saved checkpoints at', path)

        if i%args.i_video==0 and i > 0:
            # Turn on testing mode
            with torch.no_grad():
                rgbs_u, disps, uncer,depth = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test)
            print('Done, saving', rgbs_u.shape, disps.shape)
            moviebase = os.path.join(basedir, expname, '{}_spiral_{:06d}_'.format(expname, i))
            imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs_u), fps=30, quality=8)
            imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps / np.max(disps)), fps=30, quality=8)

            # if args.use_viewdirs:
            #     render_kwargs_test['c2w_staticcam'] = render_poses[0][:3,:4]
            #     with torch.no_grad():
            #         rgbs_still, _ = render_path(render_poses, hwf, args.chunk, render_kwargs_test)
            #     render_kwargs_test['c2w_staticcam'] = None
            #     imageio.mimwrite(moviebase + 'rgb_still.mp4', to8b(rgbs_still), fps=30, quality=8)

        if i%args.i_testset==0 and i > 0:
            testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', poses[i_test].shape)
            with torch.no_grad():
                render_path(torch.Tensor(poses[i_test]).to(device), hwf, K, args.chunk, render_kwargs_test, gt_imgs=images[i_test], savedir=testsavedir)
            print('Saved test set')
        
        if i%args.i_uncertainty==0 and i > 0 :
            testsavedir = os.path.join(basedir, expname, 'uncertaintyset_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            index = i_test[::4]
            print('test poses shape', poses[index].shape)
            with torch.no_grad():
                rgbs_u , __, uncer,depths = render_path(torch.Tensor(poses[index]).to(device), hwf, K, args.chunk, render_kwargs_test, gt_imgs=images[index], savedir=testsavedir)
                rgbs_u = torch.from_numpy(rgbs_u).to(device)
                uncer = torch.Tensor(uncer).to(device)
                depths = torch.Tensor(depths).to(device)
                
                for img_index in range(0,len(rgbs_u)):
                    target_depth_un = depth_imgs[index[img_index]]
                    img_loss_perray_un = img2mesrespective(rgbs_u[img_index], images[index[img_index]])
                    img_loss_un = torch.sum(img_loss_perray_un)
                    delta_un = torch.sum(uncer[img_index])
                    depth_diff_un = target_depth_un - (depths[img_index]/8.94)
                    depth_loss_un = torch.sum(depth_diff_un**2)
                    ems_loss_un = img_loss_un + 20*depth_loss_un
                    loss_un = torch.log(delta_un) + img_loss_un/(delta_un ** 2) + depth_loss_un
                    psnr_un = mse2psnr(img2msepsnr(rgbs_u[img_index], images[index[img_index]]))
                    filename = os.path.join(basedir, expname, 'testdata.txt'.format(i))
                    with open(filename,'a') as f: 
                        f.write("current step is + " + str(i)+"\n")
                        f.write("current img is + " + str(index[img_index]) +"\n")
                        f.write("img loss is + " + str(img_loss_un.item()) +"\n")
                        f.write("uncertainty sum is + " + str(delta_un.item()) +"\n")
                        f.write("loss_depth_un is + " + str(depth_loss_un.item()) +"\n")
                        f.write("ems_loss_un is + " + str(ems_loss_un.item()) +"\n")
                        f.write("final loss is + " + str(loss_un.item()) +"\n")
                        f.write("psnr is + " + str(psnr_un.item()) +"\n")
            print('Saved test set')


    
        if i%args.i_print==0:
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}")

        if psnr > 30 and all_train_index == len(all_trian_views):
            print("mission complete, quitting")
            logging.info("mission complete, quitting")
            testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', poses[i_test].shape)
            with torch.no_grad():
                render_path(torch.Tensor(poses[i_test]).to(device), hwf, K, args.chunk, render_kwargs_test, gt_imgs=images[i_test], savedir=testsavedir)
            print('Saved test set')
            print("mission complete, quitting")
            logging.info("mission complete, quitting")
            return
        
        global_step += 1


if __name__=='__main__':
    logging.basicConfig(level=logging.INFO,#控制台打印的日志级别
                    filename='./logs/depth_uncer_v20.log',
                    filemode='a',##模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志
                    #a是追加模式，默认如果不写的话，就是追加模式
                    format=
                    '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                    #日志格式
                    )
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    train()
