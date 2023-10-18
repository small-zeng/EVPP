from operator import pos
from re import T
from numpy.core.numeric import argwhere


from numpy.lib.function_base import delete

from core.run_nerf_forrender import *

from core.run_nerf_helpers_frorender import *
from core.nerf_configs import *
from core.load_blender import *
import numpy as np
import threading
import copy
import os
import requests
import colorlog

parser = config_parser()
args, argv = parser.parse_known_args()

device_id = 'cuda:7'

os.environ["CUDA_VISIBLE_DEVICES"] = '7'
device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")

def create_nerf(args,index):
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



    model_fine = None
    if args.N_importance > 0:
        model_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                          input_ch=input_ch, output_ch=output_ch, skips=skips,
                          input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
        grad_vars += list(model_fine.parameters())



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


    # Load checkpoints
    if args.ft_path is not None and args.ft_path!='None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if 'tar' in f]


    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        if os.path.join(basedir, expname, '{:06d}.tar'.format(index)) in ckpts:
            ckpt_index = ckpts.index(os.path.join(basedir, expname, '{:06d}.tar'.format(index)))
            ckpt_path = ckpts[ckpt_index]

        print('Reloading from', ckpt_path)
        # print(torch.ones((1000,10000)).to(device).device)
        ckpt = torch.load(ckpt_path, map_location=device_id)
        # torch.load()


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


    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer

def rendertest():
    
    # testdata = 'testdata-3'
    # testdata = 'drums-test-34'
    basedir = "../logs/"
    expname = 'unity_uncertainty_depth_cabin_v42_1'

    # testdata = 'drums-test-34'
    # testdata = 'testdata-34'
    # testdata = 'cabin-34'
    # testdata = 'engine-34-0.5'
    # testdata = 'tank-34-0.5'
    # testdata = 'elites-34-0.5'
    # testdata = 'cabin-34-0.5'
    # testdata = 'drums-34-0.5'
    # testdata = "cabin-j-3"
    # testdata = "drums-j-3"
    # testdata = "engine-j-3"
    # testdata = "tank-j-3"
    testdata = "elites-j-3"
    # basedir = '../nerfServer/logs/'
    # expname = 'unity_continue_depth_cabin_v20_1'
    gt = False
    num = 100000
    

    renderall = False
    only5 = False
    length = 100

    args.basedir = basedir
    args.expname = expname
    dir = "../data/" + testdata

    print(testdata)
    print(expname)

    H = 400
    W = 400
    focal = 300
    near = 0.5
    far = 6

    bds_dict = {
        'near' : near,
        'far' : far,
    }

    K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])
    hwf = [H, W, focal]
    images, depth_imgs, poses, render_poses, hwf, i_split = load_blender_data_testdata(dir,args.half_res, args.testskip)
    images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
    images = torch.Tensor(images).to(device)
    depth_imgs = torch.Tensor(depth_imgs).to(device)
    poses = torch.Tensor(poses).to(device)
    _i_train, _i_val, i_test = i_split
    print("i_test",i_test)
    if args.ft_path is not None and args.ft_path!='None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if 'tar' in f]


    if renderall:
        gap = 2000
    else:
        gap = 2000000

    # 38
    for index in range(num,400000,gap):
        render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args,index)
        render_kwargs_train.update(bds_dict)
        render_kwargs_test.update(bds_dict)
        print("current model step is " + str(start))
        testsavedir = os.path.join(basedir, expname,'atestdata',testdata + '_{:06d}'.format(start))
        os.makedirs(testsavedir, exist_ok=True)
        if renderall or only5:
            index = i_test[::20]
        else:
            index = i_test

        

        for i in range(len(index)+1):
            
            iter_index = index[i*length:(i+1)*length]
            if len(iter_index) == 0:
                continue
            with torch.no_grad():
                render_path(gt,start,basedir,expname,testdata,poses[iter_index],i*length, hwf, K, args.chunk, render_kwargs_test, gt_imgs=images, depth_imgs = depth_imgs,index = iter_index, far = far, savedir=testsavedir)
                # rgbs_u , __, uncer,depths = render_path(start,basedir,expname,testdata,poses[iter_index],i*length, hwf, K, args.chunk, render_kwargs_test, gt_imgs=images,gt_depths = depth_imgs,index = iter_index, far = far, savedir=testsavedir)
                # rgbs_u = torch.Tensor(rgbs_u).to(device)
                # uncer = torch.Tensor(uncer).to(device)
                # depths = torch.Tensor(depths).to(device)
                # for img_index in range(0,len(rgbs_u)):
                #     target_depth_un = depth_imgs[iter_index[img_index]]
                #     img_loss_perray_un = img2mesrespective(rgbs_u[img_index], images[iter_index[img_index]])
                #     img_loss_un = torch.sum(img_loss_perray_un)
                #     delta_un = torch.sum(uncer[img_index])
                #     depth_diff_un = target_depth_un - (depths[img_index]/far)
                #     depth_loss_un = torch.sum(depth_diff_un**2)
                #     loss_un = torch.log(delta_un) + img_loss_un/(delta_un ** 2) + depth_loss_un
                #     psnr_un = mse2psnr(img2msepsnr(rgbs_u[img_index], images[iter_index[img_index]]))
                #     filename = os.path.join(basedir, expname,'atestdata', testdata+'.txt')

                #     ###
                #     uncer_data = torch.sum(uncer[img_index],-1)
                #     uncer_data = uncer_data.reshape((400,400))
                #     uncer_data = 1/uncer_data
                #     uncer_data = uncer_data.cpu().numpy()
                #     rgb8 = (np.clip(uncer_data,0,255)).astype(np.uint8)
                #     filename_img = os.path.join(testsavedir, 'uncer_{:03d}.png'.format(i*length+img_index))
                #     imageio.imwrite(filename_img, rgb8)
                #     with open(filename,'a') as f: 
                #         f.write("current step is + " + str(start)+"\n")
                #         f.write("current img is + " + str(iter_index[img_index]) +"\n")
                #         f.write("img loss is + " + str(img_loss_un.item()) +"\n")
                #         f.write("uncertainty sum is + " + str(delta_un.item()) +"\n")
                #         f.write("loss_depth_un is + " + str(depth_loss_un.item()) +"\n")
                #         f.write("final loss is + " + str(loss_un.item()) +"\n")
                #         f.write("psnr is + " + str(psnr_un.item()) +"\n")
        print('Saved test set')
        

if __name__=='__main__':
    
    rendertest()