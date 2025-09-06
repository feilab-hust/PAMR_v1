import subprocess
import yaml
import random
import numpy as np
import torch
from copy import deepcopy
from datetime import datetime
import os

def set_random_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False 
    torch.backends.cudnn.deterministic = True

def get_timestamp():
    return datetime.now().strftime("%y%m%d-%H%M%S")

def load_hparam(filename):
    stream = open(filename, "r", encoding = 'utf-8')
    docs = yaml.load_all(stream, Loader=yaml.Loader)
    hparam_dict = DotDict()
    for doc in docs:
        for k, v in doc.items():
            hparam_dict[k] = DotDict(v)
    return hparam_dict

def Nor(X):
    tmp = X - X.min()
    tmp = tmp / tmp.max()
    return tmp

def TV_z(img, h):
    pixel_diff = torch.abs(img[...,h::h] - img[...,:-h:h])
    return torch.mean(pixel_diff)

def TV_xy(img, h):
    pixel_diff_y = torch.mean(torch.abs(img[h::h] - img[:-h:h]))
    pixel_diff_x = torch.mean(torch.abs(img[:,h::h] - img[:,:-h:h]))
    return pixel_diff_y + pixel_diff_x

def TV_3d(img, h):
    tv_z = TV_z(img, h) + 0.5 * TV_z(img, h * 2)
    tv_xy = TV_xy(img, h) + 0.5 * TV_xy(img, h * 2)
    return tv_z + tv_xy

def real_constrain(X, n):
    real_X = torch.real(X)
    imag_X = torch.imag(X)
    real_X[real_X < n] = n
    return real_X + 1j * imag_X

def dwt_(x):
    x = x.transpose(1, 0)
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4
    return x_LL, torch.cat([x_LH, x_HL], dim = 1)
    # return x_LL, torch.cat([x_LH, x_HL, x_HH], dim = 1)

class loss_moniter_dict():
    def __init__(self):
        super().__init__()
        self.n = 0
        self.dict = {}
    def update(self, loss_d: dict):
        if self.n == 0:
            self.dict.update(loss_d)
        else:
            for k, v in loss_d.items():
                try:
                    self.dict[k] += deepcopy(v)
                except:
                    raise ValueError('something wrong with the loss_state update')
        self.n += 1
    def average(self):
        for k, v in self.dict.items():
            self.dict[k] /= self.n
        return self.dict

class DotDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, dict_=None):
        super().__init__()
        if dict_ is not None:
            if not isinstance(dict_, dict):
                raise ValueError
            for k, v in dict_.items():
                if isinstance(v, dict):
                    self[k] = DotDict(v)
                else:
                    self[k] = v

    def __copy__(self):
        copy = type(self)()
        for k, v in self.items():
            copy[k] = v
        return copy

    def __deepcopy__(self, memodict={}):
        copy = type(self)()
        memodict[id(self)] = copy
        for k, v in self.items():
            copy[k] = deepcopy(v, memodict)
        return copy

    def __getstate__(self):
        return self.to_dict()

    def __setstate__(self, state):
        self.__init__(state)

    def to_dict(self):
        output_dict = dict()
        for k, v in self.items():
            if isinstance(v, DotDict):
                output_dict[k] = v.to_dict()
            else:
                output_dict[k] = v
        return output_dict

def dict_update(raw, new):
    dict_update_iter(raw, new)
    dict_add(raw, new)

def dict_update_iter(raw, new):
    for k in raw:
        if k not in new.keys():
            continue
        if isinstance(raw[k], DotDict) and isinstance(new[k], DotDict):
            dict_update_iter(raw[k], new[k])
        else:
            raw[k] = new[k]

def dict_add(raw, new):
    update_dict = DotDict()
    for k in new:
        if k not in raw.keys():
            update_dict[k] = new[k]

    raw.update(update_dict)


def deep_dict_update(main_dict: dict, update_dict: dict) -> None:
    for key in update_dict:
        if (
                key in main_dict
                and isinstance(main_dict[key], DotDict)
                and isinstance(update_dict[key], DotDict)
        ):
            deep_dict_update(main_dict[key], update_dict[key])
        elif (
                key in main_dict
                and isinstance(main_dict[key], list)
                and isinstance(update_dict[key], list)
        ):
            main_dict[key] = update_dict[key]
        else:
            main_dict[key] = update_dict[key]

