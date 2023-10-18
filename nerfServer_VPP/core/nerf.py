
from operator import pos
from re import T

from numpy.lib.function_base import delete
from core.run_nerf import *

from core.run_nerf_helpers import *
from core.nerf_configs import *
from core.load_blender import *
import numpy as np
import threading
import cv2
import copy
import os
import requests
import colorlog

device = torch.device(cuda_id if torch.cuda.is_available() else "cpu")
device2 = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
version = '25_132'
    
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
file_handler = logging.FileHandler(
    filename='./logs/test.log', mode='a', encoding='utf8')

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

logging.basicConfig(level=logging.INFO,  # 控制台打印的日志级别
                    filename='logs/NerfVPP_v'+str(version)+'.log',
                    filemode='a',  # 模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志
                    # a是追加模式，默认如果不写的话，就是追加模式
                    format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                    # 日志格式
                    )

class Controller():

    # prepare condition HWF and at least 4 imgs
    def __init__(self, H, W, focal, imgs, depth_imgs, poses):
        self.add_img_set = torch.zeros((0,H,W,3)).to(device)
        self.add_depth_img_set = torch.zeros((0,H,W)).to(device)
        self.add_pose_set = torch.zeros((0,4,4)).to(device)
        self.img_set = []
        self.depth_img_set = []
        self.pose_set = []
        self.add_img_index = 0

        self.uncertainty_kargs = None
        self.sapmle_model_index = 0
        self.devices = [device2]

        self.i_train = []
        self.i_test = []
        self.i_trained = set()
        
        self.change_index = 0

        self.H = H
        self.W = W
        self.focal = focal
        # self.near = 0.5
        # self.far = 6
        self.near = 0.5
        self.far = 80

        self.K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])
        self.hwf = [H, W, focal]

        self.N_rand = 1024
        
        self.NBV_step = 0
        self.terminate = False

        self.rays_loss = None
        self.all_rays_o = torch.zeros((0, 3))
        self.all_rays_d = torch.zeros((0, 3))
        self.all_target_depth = torch.zeros((0, 1))
        self.all_target_img = torch.zeros((0, 3))
        self.all_select = torch.zeros(0, 2)
        self.img_flag = [None, None, None, None]
        self.select_coords = None
        self.current_select_coords = None
        self.batch_rays = None
        self.coords = torch.stack(torch.meshgrid(torch.linspace(
            0, self.H-1, self.H).to(device), torch.linspace(0, self.W-1, self.W).to(device)), -1)  # (H, W, 2)
        self.coords = torch.reshape(self.coords, [-1, 2])  # (H * W, 2)

        parser = config_parser()
        self.args, self.argv = parser.parse_known_args()
        self.basedir = self.args.basedir
        self.expname = self.args.expname
        os.makedirs(os.path.join(self.basedir, self.expname), exist_ok=True)
        f = os.path.join(self.basedir, self.expname, 'args.txt')
        with open(f, 'w') as file:
            for arg in sorted(vars(self.args)):
                attr = getattr(self.args, arg)
                file.write('{} = {}\n'.format(arg, attr))
        if self.args.config is not None:
            f = os.path.join(self.basedir, self.expname, 'configs.txt')
            with open(f, 'w') as file:
                file.write(open(self.args.config, 'r').read())

        self.add_all_trian_views = []
        self.all_trian_views = []
        for i in np.arange(0, len(imgs), 1):
            self.add_img(imgs[i], depth_imgs[i], poses[i],False)
        self.i_train = np.arange(0,len(self.add_img_set))
        self.all_train_index = len(self.add_img_set)

        # images, _depth_imgs, _poses, render_poses, hwf, i_split = load_blender_data_cabin(self.args.datadir, self.args.half_res, self.args.testskip)
        # _i_train, _i_val, self.i_test = i_split
        # images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        # self.test_imgs = torch.Tensor(images).to(device2)
        # self.test_depth_imgs = torch.Tensor(_depth_imgs).to(device2)
        # self.test_poses = torch.Tensor(_poses).to(device2)
        self.create_nerf_model()

    def create_nerf_model(self):
        # self.args.N_importance = -1
        render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(
            self.args)
        self.render_kwargs_train = render_kwargs_train
        self.render_kwargs_test = render_kwargs_test
        self.i = start
        self.optimizer = optimizer
        
        bds_dict = {
        'near' : self.near,
        'far' : self.far,
        }

        self.render_kwargs_train.update(bds_dict)
        self.render_kwargs_test.update(bds_dict)

    def continue_mission(self):
        data_dir = os.join(self.basedir,self.expname,'trainset')
        self.NBV_step = 100
        images, _depth_imgs, _poses, render_poses, hwf, i_split = load_blender_data_cabin(data_dir, self.args.half_res, self.args.testskip)
        _i_train, _i_val, self.i_test = i_split
        images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        self.test_imgs = torch.Tensor(images).to(device2)
        self.test_depth_imgs = torch.Tensor(_depth_imgs).to(device2)
        self.test_poses = torch.Tensor(_poses).to(device2)

    def add_img(self, img, depth_img, pose, test):
        
        img = (np.array(img) / 255.).astype(np.float32)
        depth_img = (np.array(depth_img)/255.).astype(np.float32)
        pose = np.array(pose).astype(np.float32)

        basename = self.basedir +"/"+self.expname + "/"+"trainset/"
        os.makedirs(basename, exist_ok=True)
        rgb8 = to8b(img)
        filename = os.path.join(basename, 'main_{:03d}.png'.format(self.add_img_index))
        imageio.imwrite(filename, rgb8)
        rgb8 = to8b(depth_img)
        filename = os.path.join(basename, 'depth_{:03d}.png'.format(self.add_img_index))
        imageio.imwrite(filename, rgb8)
        np.savetxt(filename + 'pose_{:03d}.csv'.format(self.add_img_index),pose,delimiter=',')
        self.add_img_index += 1

        ## half_res 
        imgs_half_res = np.zeros((self.H, self.W, 4))
        imgs_half_res_depth = np.zeros(( self.H, self.W, 4))
        imgs_half_res = cv2.resize(img, (self.W, self.H), interpolation=cv2.INTER_AREA)
        img = imgs_half_res
        # imgs = tf.image.resize_area(imgs, [H,W]).numpy()

        imgs_half_res_depth = cv2.resize(depth_img, (self.W, self.H), interpolation=cv2.INTER_AREA)
        depth_img = imgs_half_res_depth

        img = img[...,:3]*img[...,-1:] + (1.-img[...,-1:])
        depth_img = depth_img[...,0]

        img = torch.Tensor(img).to(device)
        depth_img = torch.Tensor(depth_img).to(device)
        pose = torch.Tensor(pose).to(device)

        ##22_2 remove depth
        # bias = torch.randn((self.H,self.W)) / self.far * 0.03
        # bias = bias.to(device)
        # depth_img = depth_img + bias

        # #### new depth from L515 and simulate z^2 relationship
        # means = 0.1125 * (depth_img * self.far)**2 + 4.8875
        # std = 2.925 * (depth_img * self.far)**2 + 3.325
        # # means = 0.005 + depth_imgs[i] * far / 9 * (0.014 - 0.005)
        # # std = 0.0025 + depth_imgs[i] * far / 9 * (0.0155 - 0.0025)
        # bias = torch.normal(mean = 0,std = std) / self.far / 1000
        # bias[depth_img==1] = 0
        # bias = bias.to(device)
        # depth_img = depth_img + bias

        #### new depth from Lidar and simulate z^2 relationship
        means = 1.23464888e-02 * (depth_img * self.far)**2 + 4.65098301e-02
        std = 1.22884446e-02 * (depth_img * self.far)**2 +1.57083689e-02
        bias = torch.normal(mean = 0,std = std/12.0) /self.far / 1000
        bias[depth_img==1] = 0
        bias = bias.to(device)
        depth_img = depth_img + bias

        print(self.add_img_set.shape,img[None,...].shape)
        self.add_img_set = torch.cat([self.add_img_set,img[None,...]],0)
        self.add_depth_img_set = torch.cat([self.add_depth_img_set,depth_img[None,...]],0)
        self.add_pose_set = torch.cat([self.add_pose_set,pose[None,...]],0)
        if test:
            self.i_test.append(len(self.add_img_set)-1)
        ##### need else
        self.add_all_trian_views.append(len(self.add_img_set)-1)

        print("added img and length of img set is " + str(len(self.img_set)) + " , " + str(len(self.add_img_set)) + " , " + str(self.all_trian_views) + " , " + str(self.i_test))

    def push_img(self):
        if len(self.img_set) == len(self.add_img_set):
            return  
        self.img_set = self.add_img_set
        self.depth_img_set = self.add_depth_img_set
        self.pose_set = self.add_pose_set
        self.all_trian_views = self.add_all_trian_views
        print("push img and length of img set is " + str(len(self.img_set)) + " , " + str(len(self.add_img_set)) + " , " + str(self.all_trian_views) + " , " + str(self.i_test))

    def terminate_work(self):
        self.terminate = True

    def get_uncertainty(self,poses):
        # print("self.uncertainty_kargs length is ",len(self.uncertainty_kargs))
        # print("samlpe dev index ", self.sapmle_model_index)
        kwargs = self.uncertainty_kargs[self.sapmle_model_index]
        self.sapmle_model_index += 1
        self.sapmle_model_index %= len(self.devices)

        dev_sample = kwargs['device']
        # print("sample dev is ", dev_sample)
        # print("model dev is ", next(kwargs['network_fine'].parameters()).device)
        sample_num = 1000
        center_bias = 5
        poses = torch.Tensor(poses).to(dev_sample)
        poses = poses[:,:3, :4]
        all_rays_o = torch.zeros((0,3)).to(dev_sample)
        all_rays_d = torch.zeros((0,3)).to(dev_sample)
        for pose in poses:

            rays_o, rays_d = get_rays(self.H, self.W, self.K, pose,dev_sample)  # (H, W, 3), (H, W, 3)
            # rays_os = rays_o[int(self.H/2-center_bias):int(self.H/2+center_bias),int(self.W/2-center_bias):int(self.W/2+center_bias)].reshape((100,3))
            # rays_ds = rays_d[int(self.H/2-center_bias):int(self.H/2+center_bias),int(self.W/2-center_bias):int(self.W/2+center_bias)].reshape((100,3))
            select_inds = np.random.choice(self.coords.shape[0], size=[sample_num], replace=False)  # (N_rand,)
            select_coords = self.coords[select_inds].long()  # (N_rand, 2)
            rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
            rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
            # print(len(rays_o))
            all_rays_o = torch.cat([all_rays_o, rays_o], 0)
            all_rays_d = torch.cat([all_rays_d, rays_d], 0)
            # print(len(all_rays_o))
            # rays_o = torch.cat([rays_o,rays_os],0)
            # rays_d = torch.cat([rays_d,rays_ds],0)
        batch_rays = torch.stack([all_rays_o, all_rays_d], 0).to(dev_sample)
        print(len(batch_rays))


        with torch.no_grad():

            rgb, disp, acc, uncertainty,depth, extras = render(self.H, self.W, self.K, chunk=self.args.chunk, rays=batch_rays,
                                                    verbose=self.i < 10, retraw=True,dev = dev_sample,
                                                    **kwargs)
            # print(depth[::10])
            uncertainty = uncertainty**2
            uncers = uncertainty.reshape( (len(poses),sample_num*192) )
            uncers = torch.sum(uncers,-1)/sample_num
            # rgbs = torch.mean(rgb,-1)*255
            # # print(rgbs[::100])
            # depth[rgbs > 250] = 0
            
            
            # depth_center = depth[sample_num:]
            # depth = depth[:sample_num]
            # reached_num = len(depth[depth>self.near])
            # # print(depth)
            # reached_num_center = len(depth_center[depth_center>self.near])
            # # print(reached_num_center)
            # avg_distance = torch.sum(depth_center[depth_center>self.near])/reached_num_center
            # ratio = reached_num/sample_num
            return uncers#,avg_distance,ratio

    def get_surface_points(self,location,radius,step):
        kwargs = self.uncertainty_kargs[self.sapmle_model_index]
        self.sapmle_model_index += 1
        self.sapmle_model_index %= len(self.devices)

        dev_sample = kwargs['device']

        rays_d = []
        for u in np.arange(0,2*np.pi,step):
            for v in np.arange(0,np.pi,step):
                rays_d.append([np.cos(u),np.sin(u)*np.cos(v),np.sin(u)*np.sin(v)])
        rays_d = torch.Tensor(rays_d).to(dev_sample)
        location = torch.Tensor(location).to(dev_sample)
        rays_o = location.expand([rays_d.shape[0],rays_d.shape[1]])
        batch_rays = torch.stack([rays_o, rays_d], 0).to(dev_sample)
        with torch.no_grad():
            rgb, disp, acc, uncertainty,depth, extras = render(self.H, self.W, self.K, chunk=self.args.chunk, rays=batch_rays,
                                                    verbose=self.i < 10, retraw=True,dev = dev_sample,
                                                    **kwargs)
            rgbs = torch.mean(rgb,-1)*255
            # print(rgbs[::100])
            depth[rgbs > 250] = 0
            points = rays_o + rays_d * depth[...,None]
            points = points[depth < radius]
            depth = depth[depth < radius]
            points = points[depth > self.near]
            return points

    def get_rays_depth(self,locations,dirs):
        inputs = torch.Tensor(locations).to(device)
        dirs = torch.Tensor(dirs).to(device)

        with torch.no_grad():
            
            sh = dirs.shape
            rays_o = torch.reshape(inputs, [-1,3]).float()
            rays_d = torch.reshape(dirs, [-1,3]).float()

            near, far = self.near * torch.ones_like(rays_d[...,:1]).to(device), self.far * torch.ones_like(rays_d[...,:1]).to(device)
            rays = torch.cat([rays_o, rays_d, near, far], -1)
            rays = torch.cat([rays, dirs], -1)

            kwargs = self.uncertainty_kargs[0].copy()
            kwargs.pop('use_viewdirs')
            kwargs.pop('ndc')
            kwargs.pop('near')
            kwargs.pop('far')

            # Render and reshape
            all_ret = batchify_rays(rays, self.args.chunk, **kwargs)

            for k in all_ret:
                k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
                all_ret[k] = torch.reshape(all_ret[k], k_sh)

            k_extract = ['rgb_map', 'disp_map', 'acc_map', 'uncertainty_map','depth_map']
            ret_list = [all_ret[k] for k in k_extract]
            ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}

            rgb_map, disp_map, acc_map, weights, depth_map = ret_list

            rgbs = torch.mean(rgb_map,-1) * 255

            depth_map[rgbs > 250] = 0

            
        depth_map[depth_map < self.near] = 0
        return depth_map

    def info_planner(self):
        
        if len(self.add_img_set) >= 28 or self.NBV_step > 100:
            self.uncertainty_kargs = None
            return
        self.NBV_step = self.NBV_step + 1
        self.save_model()
        self.uncertainty_kargs = self.copy_model(2)
        response = requests.get("http://localhost:6100/isfinish", "finish=yes")
        print("send info to planner")
        

    def train(self):

        while True:
            # if self.i < 1000:
            #     self.uncertainty_kargs = self.copy_model(2)
            #     self.info_planner()
            #     return
            # if self.i % 300 == 0 and self.i > 0 :
            if (self.i % 400 == 0 and self.i > 0) or self.i == 10:
                torch.cuda.empty_cache()
                # threading.Thread(target=self.info_planner,args=()).start
                self.info_planner()
                

            time0 = time.time()
            
            self.push_img()
            self.load_data()

            ## lock must be released before train exit
            

            rgb, disp, acc, uncertainty, depth, extras = render(self.H, self.W, self.K, chunk=self.args.chunk, rays=self.batch_rays,
                                                                verbose=self.i < 10, retraw=True,dev = self.render_kwargs_train['device'],
                                                                **self.render_kwargs_train)
            self.optimizer.zero_grad()

            logging.info("############################## current global step is " + str(self.i) + "  ###############################")

            img_loss_perray = img2mesrespective(rgb, self.all_target_img)
            img_loss = torch.sum(img_loss_perray)
            depth_diff = self.all_target_depth - (depth/self.far)
            depth_loss = torch.sum(depth_diff**2)
            delta = torch.sum(uncertainty**2)

            ## img based
            img_loss = img_loss/self.N_rand
            depth_loss = depth_loss/self.N_rand
            delta = delta/self.N_rand

            ## ray based
            # delta_perray = torch.sum(uncertainty**2,-1)

            ## 128 ray based
            # img_loss_four = torch.sum(img_loss_perray.reshape((8,128)),-1)/128
            # delta_four = torch.sum(delta_perray.reshape((8,128)),-1)/128

            if self.i < 10000:
                self.rays_loss = depth_diff**2
                # loss = torch.log(delta) + img_loss/(delta ** 2) + depth_loss
                loss = 1/2*torch.log(delta) + 1/2*img_loss/(delta) + depth_loss
                # loss = 1/2*torch.sum(torch.log(delta_perray)) + 1/2*torch.sum((img_loss_perray + torch.Tensor([3/1024]).to(device)) / (delta_perray)) + depth_loss
                # loss = 1/2*torch.sum(torch.log(delta_four)) + 1/2*torch.sum((img_loss_four + torch.Tensor([3/1024]).to(device)) / (delta_four)) + depth_loss
            else:
                self.rays_loss = img_loss_perray
                # loss = torch.log(delta) + img_loss/(delta ** 2) + depth_loss/10
                loss = 1/2*torch.log(delta) + 1/2*img_loss/(delta) + depth_loss/10
                # loss = 1/2*torch.sum(torch.log(delta_perray)) + 1/2*torch.sum((img_loss_perray + torch.Tensor([3/1024]).to(device)) / (delta_perray)) + depth_loss/10
                # loss = 1/2*torch.sum(torch.log(delta_four)) + 1/2*torch.sum((img_loss_four + torch.Tensor([3/1024]).to(device)) / (delta_four)) + depth_loss/10

            psnr = mse2psnr(img2msepsnr(rgb, self.all_target_img))
            logging.info("img_loss is " + str(img_loss.item()) + " , delta is " +
                         str(delta.item()) + " , device type : " + str(delta.device))
            logging.info("depth loss is ," + str(depth_loss.item()))
            logging.info("fine loss : " + str(loss.item()) +
                         " fine img_psnr is " + str(psnr.item()))
            
            all_loss = self.rays_loss
            loss_per_img = all_loss.reshape( (len(self.i_train), int(self.N_rand/len(self.i_train))) )
            loss_per_img = torch.sum(loss_per_img,-1)
            self.change_index = torch.argmin(loss_per_img)
            self.i_trained.add(self.i_train[self.change_index])
            if (self.all_train_index < len(self.all_trian_views) ):
                self.i_train[self.change_index] = self.all_trian_views[self.all_train_index]
                self.all_train_index = self.all_train_index + 1
            else:
                self.i_train[self.change_index] = np.random.choice(list(self.i_trained))
                self.img_flag[self.change_index] = None
            if self.i%20 == 0 and self.i > 0:   
                print('TRAIN views are', self.i_train)
                print('TRAINED views are', self.i_trained)
                logging.info('TRAIN views are' + str(self.i_train))    
                logging.info('TRAINED views are' + str(self.i_trained))  

            if 'rgb0' in extras:

                img_loss_perray0 = img2mesrespective(
                    extras['rgb0'], self.all_target_img)
                img_loss0 = torch.sum(img_loss_perray0)
                uncertainty0 = extras['uncertainty_map0']
                delta0 = torch.sum(uncertainty0**2)

                ## img based
                img_loss0 = img_loss0/self.N_rand
                delta0 = delta0/self.N_rand

                ## ray based
                # delta0_perray = torch.sum(uncertainty0**2,-1)

                ## 128 ray based
                # img_loss_four0 = torch.sum(img_loss_perray0.reshape((8,128)),-1)/128
                # delta_four0 = torch.sum(delta0_perray.reshape((8,128)),-1)/128

                # loss0 = torch.log(delta0) + img_loss0/(delta0 ** 2)
                loss0 = 1/2*torch.log(delta0) + 1/2*img_loss0/(delta0)
                # loss0 = 1/2*torch.sum(torch.log(delta0_perray)) + 1/2*torch.sum((img_loss_perray0 + torch.Tensor([3/1024]).to(device)) / (delta0_perray))
                # loss0 = 1/2*torch.sum(torch.log(delta_four0)) + 1/2*torch.sum((img_loss_four0 + torch.Tensor([3/1024]).to(device)) / (delta_four0)) 

                # loss0 = img_loss0
                loss = loss + loss0
                psnr0 = mse2psnr(img2msepsnr(
                    extras['rgb0'], self.all_target_img))
                logging.info("img_loss0 is " + str(img_loss0.item()) + " , delta0 is " +
                             str(delta0.item()) + " , device type : " + str(delta0.device))
                logging.info("corase img_loss : " + str(loss0.item()) +
                             " coarse img_psnr0 is " + str(psnr0.item()))
            logging.info("final loss is  " + str(loss) +
                         " begin backward optimize")
            loss.backward()
            self.optimizer.step()

            # NOTE: IMPORTANT!
            ###   update learning rate   ###
            decay_rate = 0.1
            decay_steps = self.args.lrate_decay * 1000
            new_lrate = self.args.lrate * \
                (decay_rate ** (self.i / decay_steps))
            logging.info("decay_Rate : " + str(decay_rate) + " , decay_steps : " + str(decay_steps) + " , new_lrate is " + str(new_lrate) + " , current step is " + str(self.i))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lrate
            ################################

            dt = time.time()-time0

            # if self.i % self.args.i_weights == 0 and self.i > 0:
            #     print(self.i)
            #     self.save_model()

            if (self.i%self.args.i_uncertainty==0 and self.i > 2000) or (self.i%(int(self.args.i_uncertainty/10)) == 0 and self.i > 0 and self.i <= 2000):
                # pass
                # self.test_data()
                self.save_model()
            

            if self.terminate:
                self.terminate = False
                self.save_model()
                return
                
            self.i = self.i+1
            if self.i%self.args.i_print==0:
                tqdm.write(f"[TRAIN] Iter: {self.i} Loss: {loss.item()}  PSNR: {psnr.item()}")
            if self.i % 20 == 0 :
                print("current step is + " +str(self.i) + "speed : " + str(1/dt))

    def test_data(self):
        render_kwargs_test = self.copy_model(1)
        threading.Thread(target=self.render_test_data,kwargs=render_kwargs_test).start()

    def render_test_data(self,**render_kwargs_test):
        dev_test = render_kwargs_test['device']
        testsavedir = os.path.join(self.basedir, self.expname, 'uncertaintyset_{:06d}'.format(self.i))
        os.makedirs(testsavedir, exist_ok=True)
        if len(self.i_test) > 6:
            index = self.i_test[:20:2]
        else:
            index = self.i_test
        if len(index) == 0:
            return
        print('test poses shape', self.test_poses[index].shape)
        render_kwargs_test['network_fine'].to(dev_test)
        render_kwargs_test['network_fn'].to(dev_test)
        with torch.no_grad():
            rgbs_u , __, uncer,depths = render_path(self.test_poses[index], self.hwf, self.K, self.args.chunk, render_kwargs_test, gt_imgs=None, savedir=testsavedir,dev = dev_test)
            rgbs_u = torch.Tensor(rgbs_u).to(dev_test)
            uncer = torch.Tensor(uncer).to(dev_test)
            depths = torch.Tensor(depths).to(dev_test)
            for img_index in range(0,len(rgbs_u)):
                target_depth_un = self.test_depth_imgs[index[img_index]]
                img_loss_perray_un = img2mesrespective(rgbs_u[img_index], self.test_imgs[index[img_index]])
                img_loss_un = torch.sum(img_loss_perray_un)
                delta_un = torch.sum(uncer[img_index])
                depth_diff_un = target_depth_un - (depths[img_index]/self.far)
                depth_loss_un = torch.sum(depth_diff_un**2)
                # loss_un = torch.log(delta_un) + img_loss_un/(delta_un ** 2) + depth_loss_un
                loss_un = torch.log(delta_un) + img_loss_un/(delta_un) + depth_loss_un
                psnr_un = mse2psnr(img2msepsnr(rgbs_u[img_index], self.test_imgs[index[img_index]]))
                filename = os.path.join(self.basedir, self.expname, 'testdata.txt'.format(self.i))
                uncer_data = torch.sum(uncer[img_index],-1)
                uncer_data = uncer_data.reshape((self.H,self.W))
                uncer_data = 1/uncer_data
                uncer_data = uncer_data.cpu().numpy()
                rgb8 = (np.clip(uncer_data,0,255)).astype(np.uint8)
                filename_img = os.path.join(testsavedir, 'uncer_{:03d}.png'.format(img_index))
                imageio.imwrite(filename_img, rgb8)
                with open(filename,'a') as f: 
                    f.write("current step is + " + str(self.i)+"\n")
                    f.write("current img is + " + str(index[img_index]) +"\n")
                    f.write("img loss is + " + str(img_loss_un.item()) +"\n")
                    f.write("uncertainty sum is + " + str(delta_un.item()) +"\n")
                    f.write("loss_depth_un is + " + str(depth_loss_un.item()) +"\n")
                    f.write("final loss is + " + str(loss_un.item()) +"\n")
                    f.write("psnr is + " + str(psnr_un.item()) +"\n")
        print('Saved test set')


    def save_model(self):
        path = os.path.join(self.basedir, self.expname,
                            '{:06d}.tar'.format(self.i))
        torch.save({
            'global_step': self.i,
            'network_fn_state_dict': self.render_kwargs_train['network_fn'].state_dict(),
            'network_fine_state_dict': self.render_kwargs_train['network_fine'].state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        print('Saved checkpoints at', path)

    def load_data(self):
        if self.rays_loss is not None:
            self.rays_loss = self.rays_loss.reshape((len(self.i_train), int(
                self.N_rand/len(self.i_train))))  # N_img, N_sample_per_img
            self.old_select_coords = self.all_select
            self.old_select_coords = self.old_select_coords.reshape((len(self.i_train), int(
                self.N_rand/len(self.i_train)), 2))  # N_img, N_sample_per_img,2
            self.select_coords = torch.zeros((0, 2))
        self.all_rays_o = torch.zeros((0,3)).to(device)
        self.all_rays_d = torch.zeros((0,3)).to(device)
        self.all_target_depth = torch.zeros((0)).to(device)
        self.all_target_img = torch.zeros((0,3)).to(device)
        self.all_select = torch.zeros(0,2).to(device)
        self.select_coords = None
        for i_img in np.arange(0, len(self.i_train), 1):
            N_per_img = int(self.N_rand/len(self.i_train))
            img = self.i_train[i_img]
            origin_img = self.img_set[img]
            depth_img = self.depth_img_set[img]
            pose = self.pose_set[img, :3, :4]
            rays_o, rays_d = get_rays(self.H, self.W, self.K, pose,device)  # (H, W, 3), (H, W, 3)
            # 该图第一次采样，随机采样
            if self.img_flag[i_img] is None:
                select_inds = np.random.choice(self.coords.shape[0], size=[
                                               N_per_img], replace=False)  # (N_per_img,)
                # (N_per_img, 2)
                self.select_coords = self.coords[select_inds].long()
                ########################## use or not particle filter ############################
                if self.i > 5000:
                    self.img_flag[i_img] = 1
            # 该图已参与训练，重采样
            else:
                self.current_select_coords = self.old_select_coords[i_img]
                self.select_coords = torch.zeros((0, 2)).to(device)
                # 50%重采样，其余随机分布
                N_resample = int(N_per_img/4)
                particles_distribution = self.rays_loss[i_img] / torch.sum(self.rays_loss[i_img])
                particles_distribution_num = particles_distribution * N_resample
                particles_distribution_num = particles_distribution_num.int()
                for particle_index in np.arange(0, len(particles_distribution_num), 1):
                    if particles_distribution_num[particle_index] > 0:
                        bias_h = torch.randn((particles_distribution_num[particle_index])).to(device)*10
                        bias_w = torch.randn((particles_distribution_num[particle_index])).to(device)*10
                        generate_coords_h = bias_h + self.current_select_coords[particle_index][0]
                        generate_coords_w = bias_w + self.current_select_coords[particle_index][1]
                        generate_coords = torch.stack([generate_coords_h, generate_coords_w], -1)
                        self.select_coords = torch.cat([self.select_coords, generate_coords], 0)
                # 去除超出边界的点
                self.select_coords = self.select_coords[self.select_coords[:,0] > 0]
                self.select_coords = self.select_coords[self.select_coords[:,0] < self.H]
                self.select_coords = self.select_coords[self.select_coords[:,1] > 0]
                self.select_coords = self.select_coords[self.select_coords[:,1] < self.W]
                # 剩余部分随机采样
                N_randsample = N_resample - len(self.select_coords) + int(N_per_img - N_resample)
                select_inds = np.random.choice(self.coords.shape[0], size=[N_randsample], replace=False)  # (N_randsample,)
                self.select_coords = torch.cat(
                    [self.select_coords, self.coords[select_inds].long()], 0)  # (N_per_img, 2)
            rays_o = rays_o[self.select_coords[:, 0].long(),self.select_coords[:, 1].long()]  # (N_per_img, 3)
            rays_d = rays_d[self.select_coords[:, 0].long(),self.select_coords[:, 1].long()]  # (N_per_img, 3)
            target_depth = depth_img[self.select_coords[:,0].long(), self.select_coords[:, 1].long()]  # N_per_img
            target_s = origin_img[self.select_coords[:, 0].long(),self.select_coords[:, 1].long()]  # (N_per_img, 3)
            self.all_select = torch.cat([self.all_select, self.select_coords], 0)
            self.all_rays_o = torch.cat([self.all_rays_o, rays_o], 0)
            self.all_rays_d = torch.cat([self.all_rays_d, rays_d], 0)
            self.all_target_depth = torch.cat([self.all_target_depth, target_depth], 0)
            self.all_target_img = torch.cat([self.all_target_img, target_s], 0)
        self.batch_rays = torch.stack([self.all_rays_o, self.all_rays_d], 0).to(device)

    def load_data_one_img(self):
        # Random from one image
            img_i = np.random.choice(self.i_train)
            target = self.img_set[img_i]
            target_depth = self.depth_img_set[img_i]
            pose = self.pose_set[img_i, :3,:4]

            rays_o, rays_d = get_rays(self.H, self.W, self.K, pose,device)  # (H, W, 3), (H, W, 3)
            coords = torch.stack(torch.meshgrid(torch.linspace(0, self.H-1, self.H).to(device), torch.linspace(0, self.W-1, self.W).to(device)), -1)  # (H, W, 2)
            
            coords = torch.reshape(coords, [-1,2])  # (H * W, 2)
            select_inds = np.random.choice(coords.shape[0], size=[self.N_rand], replace=False)  # (N_rand,)
            select_coords = coords[select_inds].long()  # (N_rand, 2)
            rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
            rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
            self.batch_rays = torch.stack([rays_o, rays_d], 0).to(device)
            self.all_target_depth = target_depth[select_coords[:, 0], select_coords[:, 1]]
            self.all_target_img = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
            

    def set_camera_pare(self, H, W, focal):
        self.H = H
        self.W = W
        self.focal = focal
    
    ##
    # target: indicate the purpose of this copy, 1 for test and 2 for sample uncertainty
    def copy_model(self,target):
        kwargss = []
        devs = None
        if target == 1:
            devs = [self.devices[0]]
        if target == 2:
            devs = self.devices
        for dev in devs:
            # print("current copy into device " , dev)
            render_kwargs_test = {k : self.render_kwargs_train[k] for k in self.render_kwargs_train}
            mdf = copy.deepcopy(self.render_kwargs_train['network_fine']).to(dev)
            mdc = copy.deepcopy(self.render_kwargs_train['network_fn']).to(dev)
            
            embed_fn, input_ch, embedder_obj = get_embedder(self.args.multires, self.args.i_embed,dev)
        

            input_ch_views = 0
            embeddirs_fn = None
            if self.args.use_viewdirs:
                embeddirs_fn, input_ch_views, embedder_obj_views = get_embedder(self.args.multires_views, self.args.i_embed,dev)
                

            network_query_fn = lambda inputs, viewdirs, network_fn : run_network(inputs, viewdirs, network_fn,
                                                                embed_fn=embed_fn,
                                                                embeddirs_fn=embeddirs_fn,
                                                                netchunk=self.args.netchunk)
            # print("model fine in dev ",next(mdf.parameters()).device)
            # print("embedder in dev ",embedder_obj.device)
            render_kwargs_test['network_query_fn'] = network_query_fn
            render_kwargs_test['network_fine'] = mdf
            render_kwargs_test['network_fn'] = mdc
            render_kwargs_test['device'] = dev
            render_kwargs_test['perturb'] = False
            render_kwargs_test['raw_noise_std'] = 0.
            kwargss.append(render_kwargs_test)
        if target == 1:
            return kwargss[0]
        if target == 2:
            return kwargss
        return kwargss
    
   
