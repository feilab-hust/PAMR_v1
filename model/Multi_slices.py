import torch
import torch.nn as nn
import torch.nn.functional as f
from util.register import MODEL_REGISTRY

@MODEL_REGISTRY.register()
class Multi_slices_forward(nn.Module):
    def __init__(self, prop_kernel_pha, OTF, dz, focus_slice,  stack_num, z_plane, pad,
                 kz_tmp, mod, Green_kernal_mask_tensor, device = 'cpu'):
        super(Multi_slices_forward, self).__init__()
        self.prop_kernel_pha = prop_kernel_pha.to(device).type(torch.complex64)
        self.OTF = OTF.to(device).type(torch.complex64)
        self.OTF_update = None
        self.dz = dz
        self.focus_slice = focus_slice
        self.stack_num = stack_num
        self.z_plane = z_plane
        self.pad = pad
        self.kz_tmp = kz_tmp.to(device).type(torch.complex64)
        self.mod = mod

        self.sample = None
        self.prop_kernel = torch.exp(self.prop_kernel_pha * self.dz)

        self.Green_tmp = 1j * self.prop_kernel / (2 * self.kz_tmp) 
        self.Green_tmp *= Green_kernal_mask_tensor.to(device)
        self.Green_tmp[torch.isnan(self.Green_tmp) == 1] = 0

        self.prop_kernel = self.prop_kernel.to(device)
        self.Green_tmp = self.Green_tmp.to(device)

        self.device = device

    def to_device(self, device_inp):
        self.prop_kernel = self.prop_kernel.to(device_inp).unsqueeze(0)
        self.Green_tmp = self.Green_tmp.to(device_inp).unsqueeze(0)
        self.OTF = self.OTF.to(device_inp)

    def forward(self, sample, planewave, OTF_input = None, oblique_factor = None):
        self.sample = sample

        if OTF_input is not None:
            OTF_trigger = True
            self.OTF_update = torch.abs(self.OTF) * torch.exp(1j * OTF_input)
        else:
            OTF_trigger = False

        if self.mod == 'Multi_Born':
            incidence_last_slice = self._propegate_to_last_slice_Multi_Born(planewave, oblique_factor)
        else:
            raise Exception('invalid forward mod')

        incidence_center_focus_slice = self._propegate_to_center_focus_slice(incidence_last_slice)

        output = self._pupil_constrain_abs_square_output(incidence_center_focus_slice, OTF_input = OTF_trigger)
        return output

    def _propegate_to_last_slice_Multi_Born(self, planwave_each, oblique_factor):
        incidence = torch.fft.fft2(planwave_each)

        for i in range(int(self.stack_num)):
            U_tmp = incidence * self.prop_kernel
            scater_born = self.Green_tmp * torch.fft.fft2(torch.fft.ifft2(incidence) * self.sample[...,i] * self.dz) 
           
            if oblique_factor is not None:
                scater_born = scater_born * oblique_factor
                
            incidence = U_tmp + scater_born

        return incidence 

    def _propegate_to_center_focus_slice(self, incidence_last_slice):
        prop_kernel_center = torch.conj(self.prop_kernel ** (self.focus_slice))
        prop_kernel_focus = self.prop_kernel ** (self.z_plane / self.dz)
        return incidence_last_slice * prop_kernel_center * prop_kernel_focus

    def _pupil_constrain_abs_square_output(self, incidence_focus_slice, OTF_input = True):
        if OTF_input:
            output = torch.fft.ifft2(incidence_focus_slice * self.OTF_update.unsqueeze(0))
        else:
            output = torch.fft.ifft2(incidence_focus_slice * self.OTF)

        if self.pad > 0:
            return torch.abs(output)[:, self.pad: -self.pad, self.pad: -self.pad]
        else:
            return torch.abs(output)








