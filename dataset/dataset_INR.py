import torch
from torch.utils.data import Dataset
from dataset.utils import DataloaderMode
from dataset._3D_dataset_experiment import multiLayerExperiment

class Dataset_INR(Dataset):
    def __init__(self, hp_fpm, hp_net, mode):
        self.mode = mode
        if mode is DataloaderMode.train:
            self.Data_ = multiLayerExperiment(hp_fpm)
        else:
            raise ValueError(f"invalid dataloader mode {mode}")

        if hp_fpm.LEDs.LED_mode == 'single':
            self.Num_LED = self.Data_.Num_LED_used
            elements_need = self.Data_.multi_layer_forward()
        else:
            raise ValueError 

        # add arguments for model class in model_INR.py
        if hp_net.model.get('hash_table_length') is None:
            hp_net.model['hash_table_length'] = [self.Data_.HR_imsize, self.Data_.HR_imsize, self.Data_.stack_num]
        if hp_net.model.get('LR2HR_size') is None:
            hp_net.model['LR2HR_size'] = [self.Num_LED, self.Data_.HR_imsize, self.Data_.HR_imsize ]
        if hp_net.model.get('pad') is None:
            hp_net.model['pad'] = self.Data_.HR_pad
        if hp_net.model.get('k0') is None:
            hp_net.model['k0'] = self.Data_.k0
        if hp_net.model.get('n_media') is None:
            hp_net.model['n_media'] = self.Data_.n_media
        if hp_net.model.get('cutoff') is None:
            hp_net.model['cutoff'] = int(self.Data_.cutoff)
        if hp_net.model.get('LED_used') is None:
            hp_net.model['LED_used'] = self.Data_.Num_LED_used
        if hp_net.model.get('grad_mode') is None:
            hp_net.model['grad_mode'] = self.Data_.grad_mode

        self.forward_function = elements_need[-1]

        ''' load dataset used to recon device: gpu '''
        self.forward_function.to_device(hp_net.model.device)

    def __len__(self):
            return len(self.Data_.img_stack_path_list)

    def __getitem__(self, idx):
        img_LR2HR_list, img_name = self.Data_.get_LR2HR(idx) # TODO: need optimize for opensourced dataset
        self.LR2HR_gt, self.LR2HR_grad_gt, self.plane_waves, self.OF_factor, self.imgs_LR2HR_max, self.imgs_LR2HR_grad_max = img_LR2HR_list
        return self.LR2HR_gt.cpu(), self.LR2HR_grad_gt, \
            self.plane_waves, \
            self.OF_factor if self.OF_factor is not None else torch.zeros(1), \
            img_name 
