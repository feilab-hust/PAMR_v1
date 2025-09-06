import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, MultiStepLR
import numpy as np
from pathlib2 import Path

from util.util import DotDict, TV_3d, dwt_

class Model:
    def __init__(self, hp_net, net_arch, forward_function, Num_LEDs,loss_f, rank=0, world_size=1):
        self.hp_net = hp_net
        self.device = self.hp_net.model.device
        self.net = net_arch.to(self.device)
        self.forward_function = forward_function
        self.Num_LEDs = Num_LEDs
        self.HR_size_list = hp_net.model.hash_table_length
        self.rank = rank
        self.world_size = world_size
        if self.device != "cpu" and self.world_size != 0:
            self.net = DDP(self.net, device_ids=[self.rank])
        self.GT = None
        self.plane_waves = None
        self.step = 0
        self.sub_step = 0
        self.epoch = -1
        self.grad_mode = hp_net.model.grad_mode

        self.l1_loss_list = []
        
        # init optimizer
        optimizer_mode = self.hp_net.train.optimizer.mode
        if optimizer_mode == 'adamw':
            self.optimizer = torch.optim.AdamW(
                self.net.parameters(), **(self.hp_net.train.optimizer['param'])
            )
        else:
            raise Exception("%s optimizer not supported" % optimizer_mode)

        # init optimizer lr_scheduler
        scheduler = self.hp_net.train.optimizer.scheduler
        scheduler_mode = scheduler.mode
        if scheduler_mode == 'CosineAnnealingWarm':
            self.lr_scheduler = CosineAnnealingWarmRestarts(
                self.optimizer, T_0 = scheduler.T_0,
                T_mult = scheduler.T_mult,
                eta_min = scheduler.min_lr_ratio * self.hp_net.train.optimizer['param']['lr'])
        elif scheduler_mode == 'MultiStep':
            self.lr_scheduler = MultiStepLR(
                self.optimizer, milestones = scheduler.decay_step, gamma = scheduler.decay_gamma)
        else:
            self.lr_scheduler = None

        # init loss
        self.loss_f = loss_f
        self.log = DotDict()

        # init lr
        self.log.lr = self.optimizer.param_groups[0]['lr']

    def feed_data(self, **data): 
        for k, v in data.items():
            if k not in ['GT_sample', 'input']: 
                data[k] = v.to(self.device)
            else:
                data[k] = v
        self.GT = data.get("GT")
        self.GT_grad = data.get("GT_grad")
        self.plane_waves = data.get('planewaves')
        self.OF_factor = data.get('OF_factor')

    def optimize_parameters(self):
        self.net.train()
        self.optimizer.zero_grad()
        self.output = self.run_network()
        loss_v, loss_v_state = self.loss_f(self.output, self.GT, self.GT_grad)

        if self.hp_net.train.loss.TV:
            loss_tv = TV_3d(torch.real(self.output['recon_RI_media']), 1)
            loss_v = loss_v + loss_tv * self.hp_net.train.loss.TV_weight

        loss_v.backward()
        self.optimizer.step()

        # set log
        self.log.loss_v = loss_v.item()
        self.log.loss_v_state = loss_v_state
        self.l1_loss_list.append(loss_v_state['smooth_l1'])

    def run_lr_scheduler(self):
        if self.lr_scheduler:
            self.lr_scheduler.step()
            self.log.lr = self.optimizer.param_groups[0]['lr']
        else:
            pass

    def model_test(self):
        self.output = self.inference()
        evaluate_state = None

        return self.output, evaluate_state

    def inference(self):
        self.net.eval()
        return self.run_network()

    def run_network(self):
        output, output_aberration = self.net()
        output_RI = output.view(self.hp_net.model.hash_table_length + [1])# TODO: rescale of the RI output: max_RI
        output_RI_media = output_RI[..., 0] + 1j * 0.
        output_RI_function = self.hp_net.model.k0 ** 2 * (output_RI_media ** 2 + output_RI_media * self.hp_net.model.n_media * 2)

        ''' output_aberration -> OTF '''
        if output_aberration is not None:

            OTF_phase = output_aberration.reshape(self.hp_net.model.hash_table_length[0] // 4, self.hp_net.model.hash_table_length[1] // 4).unsqueeze(0).unsqueeze(0)
            OTF_phase = F.interpolate(OTF_phase, size = self.hp_net.model.LR2HR_size[1:],
                                        mode = 'bilinear', align_corners = True)
            OTF_phase = OTF_phase[0,0] * (np.pi / 2)
            OTF_phase = torch.fft.fftshift(OTF_phase)

        ''' get output LR2HR  '''
        imgs_LR2HR = self.forward_function(output_RI_function.unsqueeze(0), self.plane_waves.to(self.device), 
                                           OTF_phase.to(self.device) if output_aberration is not None else None,
                                           oblique_factor = self.OF_factor.to(self.device) if self.OF_factor is not None else None)
        
        imgs_LR2HR = imgs_LR2HR.unsqueeze(0)
        if self.grad_mode == 'dwt':
            imgs_LR2HR_grad = dwt_(imgs_LR2HR)[-1].unsqueeze(0)
        else:
            raise NotImplementedError

        return {'output': imgs_LR2HR, 'output_grad': imgs_LR2HR_grad,
                'recon_RI_media': output_RI_media[self.hp_net.model.pad: -self.hp_net.model.pad, self.hp_net.model.pad: -self.hp_net.model.pad],
                'recon_aberration': OTF_phase if output_aberration is not None else None}

    def save_network(self, logger, save_file=True):
        if self.rank == 0:
            net = self.net.module if isinstance(self.net, DDP) else self.net
            state_dict = net.state_dict()
            for key, param in state_dict.items():
                state_dict[key] = param.to("cpu")
            if save_file:
                network_path = Path(self.hp_net.log.chkpt_dir) / 'network'
                network_path.mkdir(parents = True, exist_ok = True)
                save_filename = "%s_epoch_%d_step_%d.pth" % (self.hp_net.log.name, self.epoch, self.step)
                save_path = network_path / save_filename
                torch.save(state_dict, str(save_path))
                if logger is not None:
                    logger.info("Saved network checkpoint to: %s" % save_path)
            return state_dict

