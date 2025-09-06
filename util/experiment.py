import numpy as np
from numpy import pi as pi

def get_kxky_orig_(sin_a_list, circle_num, delta_z, k0, dkxy, rotate, R, LEDs_per_circle):
    xy_diff = np.array([0., 0.])
    rotate_rad = rotate / 180 * pi
    XYZ = [[-xy_diff[0], xy_diff[1], R + delta_z]]
    kxky = {}

    def XY_spatial(index, theta_0):
        theta = index * 2 * pi / LEDs_per_circle[n] + theta_0
        return [r_tmp * np.cos(theta) - xy_diff[0], r_tmp * np.sin(theta) + xy_diff[1], z_tmp]

    def kxky_(xyz):
        r = np.sqrt(np.square(np.array(xyz)).sum())
        sin_x_tmp = -xyz[0] / r
        sin_y_tmp = xyz[1] / r
        sin_x = np.cos(rotate_rad) * sin_x_tmp + np.sin(rotate_rad) * sin_y_tmp
        sin_y = -1 * np.sin(rotate_rad) * sin_x_tmp + np.cos(rotate_rad) * sin_y_tmp
        kx_tmp = sin_x * k0 / dkxy
        ky_tmp = sin_y * k0 / dkxy
        return {'kx': kx_tmp, 'ky': ky_tmp}

    theta_0 = [0, pi / 6, 0, 0, pi / 18, pi / 18]
    for n in range(1, circle_num):
        z_tmp = ((1 - sin_a_list[n] ** 2) ** 0.5) * R + delta_z
        r_tmp = sin_a_list[n] * R
        xyz_spatial = list(map(lambda x: XY_spatial(x, theta_0[n]), range(LEDs_per_circle[n])))
        XYZ += xyz_spatial

    kxky_tmp = list(map(kxky_, XYZ))
    for i in range(len(kxky_tmp)):
        kxky[i] = kxky_tmp[i]

    return kxky

def get_coefficient_index(sin_a_list, circle_num, delta_z, R, LEDs_per_circle):
    coefficient_index = {}
    LED_height = []
    for n in range(1, circle_num):
        r_tmp = sin_a_list[n] * R
        LED_h = np.cos(np.arcsin(sin_a_list[n])) * R + delta_z
        sin_a_tmp = r_tmp / ((LED_h) ** 2 + r_tmp ** 2) ** (0.5)

        for i in range(sum(LEDs_per_circle[:n]), sum(LEDs_per_circle[:n]) + LEDs_per_circle[n]):
            coefficient_index[i] = dict(sin_a = sin_a_tmp, theta = -1 * (i - sum(LEDs_per_circle[:n])) * 2 * pi / LEDs_per_circle[n])
            LED_height.append(LED_h)

    coefficient_index[0] = dict(sin_a = sin_a_list[0], theta = 0)

    return coefficient_index, LED_height

def set_OTF_High_freq(HR_s, stack_num, cutoff = None):
    ux = np.arange(-1 * int(HR_s / 2), int(HR_s / 2) + (int(HR_s) % 2), step = 1)
    uy = np.arange(-1 * int(HR_s / 2), int(HR_s / 2) + (int(HR_s) % 2), step = 1)
    uz = np.arange(-1 * int(stack_num / 2 - 1), int(stack_num / 2 + 1) + (stack_num % 2), step = 1)
    uxx, uyy, uzz = np.meshgrid(ux, uy, uz)
    OTF_pupil = 1.0 * ((uxx[:,:,0]**2 + uyy[:,:,0]**2) <= cutoff ** 2)

    OTF_complex_amp_abs = np.abs(OTF_pupil).astype(np.complex64)
    OTF_complex_amp_phase = np.zeros_like(np.abs(OTF_pupil))
    OTF_complex_amp = OTF_complex_amp_abs * np.exp(1j * OTF_complex_amp_phase)

    return OTF_complex_amp, uxx, uyy, uzz, OTF_pupil 

def get_planewaves(kxky, dkxy, xx, yy, psize):
    planewaves_tmp = []
    for i in range(len(kxky)):
        tmp = np.exp(1j * (-np.round(kxky[i]['kx']) * dkxy * xx * psize + -np.round(kxky[i]['ky']) * dkxy * yy * psize))
        planewaves_tmp.append(tmp)

    return np.array(planewaves_tmp).astype(np.complex64)

def get_oblique_factor(kxky, dkxy, kxx, kyy, k0):
    oblique_factor_tmp = []
    for i in range(len(kxky)):
        tmp = k0 ** 2 - np.square(kxx + kxky[i]['kx'] * dkxy) - np.square(kyy + kxky[i]['ky'] * dkxy)
        tmp[tmp < 0] = 0
        tmp += 1e-6
        tmp = np.sqrt(tmp) / k0
        oblique_factor_tmp.append(tmp)
    return np.array(oblique_factor_tmp).astype(np.float32)
