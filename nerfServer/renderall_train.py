from operator import pos
from re import T
from numpy.core.numeric import argwhere


from numpy.lib.function_base import delete

from core.run_nerf_forrender import *

from core.run_nerf_helpers_frorender import *
from core.nerf_configs import *
from core.load_blender_forrender import *
import numpy as np
import threading
import copy
import os
import requests
import colorlog

parser = config_parser()
args, argv = parser.parse_known_args()

device_id = cuda_id

device = torch.device(cuda_id if torch.cuda.is_available() else "cpu")

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


    # print('Found ckpts', ckpts)
    ckpt_path = None
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
    basedir = "logs/"
    expname = 'unity_continue_depth_cabin_v23_111'

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
    # testdata = "elites-j-3"
    
    testdata = 'room_render'

    # basedir = '../nerfServer/logs/'
    # expname = 'unity_continue_depth_cabin_v17_2'
    gt = False
    num = 1000

    video = False
    renderall = False
    only5 = False
    length = 100

    args.basedir = basedir
    args.expname = expname
    # testdata = "engine"
    # dir = "../data/circle/" + testdata
    # dir = "../dataset/" + testdata
    dir = "logs/unity_continue_depth_cabin_v23_111/trainset"

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
    images, depth_imgs, poses, render_poses, _, i_split = load_blender_data_self(dir,40,args.half_res, args.testskip)
    images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
    images = torch.Tensor(images).to(device)
    depth_imgs = torch.Tensor(depth_imgs).to(device)
    poses = torch.Tensor(poses).to(device)
    print("images shape",images.shape)
    _i_train, _i_val, i_test = i_split
    print("i_test",i_test)
    if args.ft_path is not None and args.ft_path!='None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if 'tar' in f]

    print("num of models :",len(ckpts))

    views = []

    with open('./path/path_rrt_V22_1.txt') as f:
        path = []
        for line in f:
            if "规划路径" in line:
                if len(path) >= 2:
                    views.append(path)
                path = []
            else:
                path.append(np.array(line.split(',')[:3]).astype(np.float))
    
    views = np.array(views)

    print("load path, shape is :",views.shape)

    
    gap = 200
    iter_set = [300]

    for n in range(700,17000,700):
        iter_set.append(n)

    print("iter_set size:",len(iter_set))

    iter_index = 0
    iter_len = 5

    path_index = -1

    with open('./path/path_rrt_V22_1.txt') as f:
        f =  f.readlines()
    if video:
        index = i_test
        for iteration in iter_set:
            if iter_index% 10 == 0:
                print("iter_index:",iter_index)
            print("iteration:",iteration)
            render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args,iteration)
            render_kwargs_train.update(bds_dict)
            render_kwargs_test.update(bds_dict)
            print("current model step is " + str(start))
            testsavedir = os.path.join(basedir, expname,'btestdata', testdata,testdata + '_{:06d}'.format(start))
            os.makedirs(testsavedir, exist_ok=True)
            if testdata == 'track':
                tmp = path_index
                for i in range(tmp,len(f)):
                    path_index += 1
                    line = f[path_index]
                    print(line)
                    print("iter_index:",iter_index)
                    pose_index = index[iter_index*iter_len:(iter_index+1)*iter_len]
                    if len(pose_index) == 0:
                        continue
                    with torch.no_grad():
                        render_path(views,gt,start,basedir,expname,testdata,poses[pose_index],iter_index*iter_len, hwf, K, args.chunk, render_kwargs_test, far = far, savedir=testsavedir)
                        iter_index += 1
                    if "规划路径" in line:
                        break
            else:
                with torch.no_grad():
                    render_path(views,None,start,basedir,expname,testdata,poses,iter_index*iter_len, hwf, K, args.chunk, render_kwargs_test, far = far, savedir=testsavedir)
            
    else:
        # 38
        for index in np.array([168000]):
            render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args,index)
            render_kwargs_train.update(bds_dict)
            render_kwargs_test.update(bds_dict)
            print("current model step is " + str(start))
            testsavedir = os.path.join(basedir, expname,'atraindata',testdata + '_{:06d}'.format(start))
            os.makedirs(testsavedir, exist_ok=True)
            if renderall or only5:
                index = i_test[::10]
            else:
                index = i_test
 
            

            for i in range(len(index)+1):
                
                iter_index = index[i*length:(i+1)*length]
                if len(iter_index) == 0:
                    continue
                with torch.no_grad():
                    render_path_train(views,gt,start,basedir,expname,testdata,poses[iter_index],i*length, hwf, K, args.chunk, render_kwargs_test, gt_imgs=images, depth_imgs = depth_imgs,index = iter_index, far = far, savedir=testsavedir)
                   
            print('Saved test set')
        

if __name__=='__main__':
    
    rendertest()