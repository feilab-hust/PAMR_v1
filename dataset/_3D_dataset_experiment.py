import torch
from pathlib import Path
import cv2
import numpy as np
import tifffile as tif

from model.Multi_slices import Multi_slices_forward
from util.util import load_hparam, dwt_
from util.experiment import get_coefficient_index, get_kxky_orig_, set_OTF_High_freq, get_planewaves, get_oblique_factor

class multiLayerExperiment():
    def __init__(self, hp_fpm_ = None, hp_fpm = None):
        # load config
        hp_fpm = load_hparam(hp_fpm) if hp_fpm_ == None else hp_fpm_

        # system config
        self.device = 'cpu'
        self.Lambda = hp_fpm.system.Lambda
        self.delta_z = hp_fpm.LEDs.delta_z
        self.R = hp_fpm.LEDs.R
        self.oNA = hp_fpm.system.oNA
        self.rotate_degree = hp_fpm.recon.rotate_degree
        self.n_media = hp_fpm.system.n_media
        self.LEDs_per_circle = hp_fpm.LEDs.LEDs_per_circle
        self.sin_list = hp_fpm.LEDs.sin_list
        self.NC_all = hp_fpm.LEDs.NC_all

        # recon config
        self.forward_mod = str(hp_fpm.recon.forward_mod)
        self.orig_size = hp_fpm.recon.orig_size
        self.cut_index = hp_fpm.recon.cut_index
        self.imsize = hp_fpm.recon.imsize
        self.spsize = hp_fpm.system.spsize / hp_fpm.system.Mag
        self.pad = hp_fpm.recon.pad
        self.dz = hp_fpm.system.dz

        self.recon_depth = hp_fpm.system.recon_depth
        self.stack_num = hp_fpm.recon.stack_num
        self.focus_slice = int(hp_fpm.recon.stack_num) - int(hp_fpm.system.focus_slice)
        self.start_index = hp_fpm.recon.start_index
        self.delta_h = hp_fpm.recon.delta_h
        self.z_plane = hp_fpm.recon.z_plane
        self.grad_mode = hp_fpm.recon.grad
        self.wether_OF = hp_fpm.recon.oblique_factor

        if hp_fpm.dataset.name_head is not None:
            self.img_stack_path_list = [Path(hp_fpm.dataset.path_dir) / hp_fpm.dataset.name_head]
        else:
            assert hp_fpm.dataset.recon_frames is not None
            img_stack_path_list = list(filter(lambda x: x.suffix == '.tif', Path(hp_fpm.dataset.path_dir).iterdir()))
            img_stack_path_list = sorted(img_stack_path_list, key = lambda x: eval(x.name.split('_')[1] + '.' + x.name.split('_')[-1].split('.')[0].zfill(5)))
            if hp_fpm.dataset.recon_frames == 'all':
                self.img_stack_path_list = img_stack_path_list
            else:
                if hp_fpm.dataset.recon_frames >= len(img_stack_path_list):
                    print('The recon_frames is larger than the number of all frames, recon all frames')
                self.img_stack_path_list = img_stack_path_list[0: hp_fpm.dataset.recon_frames]

        # preparing
        self.get_configuration()
        self.get_OTF()
        self.get_planewaves_tensor()
        
    def get_configuration(self):
        self.imsize = self.imsize
        self.k0 = 2 * np.pi / self.Lambda
        self.Num_LED_used = sum(self.LEDs_per_circle) - 1 # center LED is not used for reconstruction
        self.coefficient_index, self.LED_height = \
            get_coefficient_index(sin_a_list = self.sin_list[: self.NC_all],
                                  circle_num = self.NC_all, delta_z = self.delta_z,
                                  R = self.R, LEDs_per_circle = self.LEDs_per_circle)
        self.Upsample = 1.95
        self.HR_pad = int(np.round(self.pad * self.Upsample)) 
        self.hr_imsize = int(np.round(self.imsize * self.Upsample)) 
        self.HR_imsize = self.hr_imsize + self.HR_pad * 2
        self.HR_center = (self.HR_imsize) / 2
        self.Upsample = self.hr_imsize / self.imsize

        self.dz = self.recon_depth / self.stack_num if self.dz == None else self.dz * 1e-3
        self.psize = self.spsize
        self.dkxy = 2 * np.pi / (self.psize * self.HR_imsize)
        self.cutoff = self.oNA * self.k0 / self.dkxy

        print('delta_XY: ', self.psize)
        print('delta_Z', self.dz)

    def get_OTF(self):
        OTF_High = set_OTF_High_freq(HR_s = self.HR_imsize, stack_num = self.stack_num, cutoff = self.cutoff)
        self.OTF_High_complex_np = OTF_High[0]
        self.CTF_High_complex_np = OTF_High[-1]
        self.Uxx = OTF_High[1]
        self.Uyy = OTF_High[2]
        self.Uzz = OTF_High[3]
        self.kxx = self.dkxy * self.Uxx[...,0]
        self.kyy = self.dkxy * self.Uyy[...,0]
        self.Ps = np.fft.ifftshift(self.OTF_High_complex_np)
        self.Ps_tensor = torch.from_numpy(self.Ps).type(torch.complex64)
        self.Ps_tensor_CTF = torch.from_numpy(np.fft.ifftshift(self.CTF_High_complex_np)).type(torch.complex64)

    def get_planewaves_tensor(self):
        self.kxkyIllu_orig = get_kxky_orig_(sin_a_list = self.sin_list, circle_num = self.NC_all,
                                            delta_z = self.delta_z, k0 = self.k0,
                                            dkxy = self.dkxy, rotate = self.rotate_degree,
                                            R = self.R, LEDs_per_circle = self.LEDs_per_circle)


        planewaves = get_planewaves(self.kxkyIllu_orig, self.dkxy, self.Uxx[...,0], self.Uyy[...,0], self.psize) 
        self.planewaves_tensor = torch.from_numpy(planewaves).type(torch.complex64)
        
        OF_factor = np.fft.ifftshift(get_oblique_factor(self.kxkyIllu_orig, self.dkxy, self.kxx, self.kyy, self.k0))
        self.OF_factor = torch.from_numpy(OF_factor).type(torch.float32)
        
    def multi_layer_forward(self):
        kz_tmp = np.fft.ifftshift((self.n_media ** 2 * self.k0 ** 2 - self.kxx ** 2 - self.kyy ** 2 + 0j) ** 0.5) 
        prop_kernel_pha = 1j * kz_tmp
        kz_tmp_tensor = torch.from_numpy(kz_tmp).type(torch.complex64)
        prop_kernel_pha_tensor = torch.from_numpy(prop_kernel_pha).type(torch.complex64)
        Green_kernal_mask = np.fft.ifftshift(self.n_media ** 2 * self.k0 ** 2 >= 1.001 * (self.kxx ** 2 + self.kyy ** 2))
        Green_kernal_mask_tensor = torch.from_numpy(Green_kernal_mask)

        # propagating
        Multi_slice = Multi_slices_forward(prop_kernel_pha_tensor, self.Ps_tensor_CTF, self.dz,
                                           self.focus_slice, self.stack_num, self.z_plane, self.HR_pad,
                                           kz_tmp_tensor, self.forward_mod, Green_kernal_mask_tensor, self.device)

        return prop_kernel_pha_tensor, kz_tmp_tensor, Green_kernal_mask_tensor, Multi_slice

    def get_LR2HR(self, idx):
        img_path = self.img_stack_path_list[idx]

        LR_imgs = tif.imread(str(img_path))
        LR_imgs = np.array(LR_imgs).astype(np.float32)

        imgs_LR2HR = list(
            map(lambda x: torch.from_numpy(cv2.resize(x[self.cut_index[0]: self.cut_index[0] + self.imsize,
                                                      self.cut_index[1]: self.cut_index[1] + self.imsize],
                                                      dsize=(self.hr_imsize, self.hr_imsize),
                                                      interpolation=cv2.INTER_LINEAR)), LR_imgs))
        imgs_LR2HR = torch.stack(imgs_LR2HR, dim=0).type(torch.float32) - 500
        print(f'LED_num_used: {len(imgs_LR2HR)}')

        imgs_LR2HR_selected = torch.clip(imgs_LR2HR, min=0.) ** 0.5
        imgs_LR2HR_selected_max = imgs_LR2HR_selected.max()
        imgs_LR2HR_selected = imgs_LR2HR_selected / imgs_LR2HR_selected_max
        imgs_LR2HR_selected_mean = imgs_LR2HR_selected.mean([-1, -2], keepdim=True)

        if self.grad_mode == 'dwt':
            imgs_LR2HR_selected_grad = dwt_(imgs_LR2HR_selected.unsqueeze(0))[-1]
        else:
            raise NotImplementedError

        imgs_LR2HR_selected_grad_max = imgs_LR2HR_selected_grad.max()
        torch.cuda.empty_cache()

        return [imgs_LR2HR_selected, imgs_LR2HR_selected_grad, \
            self.planewaves_tensor[1:] * imgs_LR2HR_selected_mean, \
            self.OF_factor[1:] if self.wether_OF else None,
            imgs_LR2HR_selected_max, imgs_LR2HR_selected_grad_max], img_path.name

    def post_process(self):
            # print config
            print('\n+'+10*'--'+'+\n')
            print('LowRes size: {}'.format(self.imsize))
            print('HighRes size: {}'.format(self.HR_imsize))
            print('forward mod: {}'.format(self.forward_mod))
            print('+'+10*'--'+'+\n')

if __name__ == '__main__':
    Exp = multiLayerExperiment()