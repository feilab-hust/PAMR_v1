import torch
from tensorboardX import SummaryWriter
from util.util import Nor


class Writer(SummaryWriter):
    def __init__(self, hp, logdir, vis = None):
        super().__init__()
        self.hp = hp
        self.vis = vis
        if hp.log.use_tensorboard:
            self.tensorboard = SummaryWriter(logdir)
        if hp.log.use_visdom:
            assert vis.check_connection, Exception('Visdom server connect problems')
            pass

    def train_logging(self, train_loss, train_loss_state: dict, model_output, gt_imgs, gt_grad_imgs, lr, epoch):
        if self.hp.log.use_tensorboard:
            self.tensorboard.add_scalar("train_loss", train_loss, epoch)
            self.tensorboard.add_scalar("lr", lr, epoch)
            for k in train_loss_state.keys():
                self.tensorboard.add_scalar('train' + str(k), train_loss_state[k], epoch)

        if self.hp.log.use_visdom:
            self.visdom_show_train(train_loss, train_loss_state, model_output, gt_imgs, gt_grad_imgs, epoch)

    def test_logging(self, test_loss, test_loss_state: dict, model_output, gt_imgs, epoch):
        if self.hp.log.use_tensorboard:
            self.tensorboard.add_scalar("test_loss", test_loss, epoch)
            for k in test_loss_state.keys():
                self.tensorboard.add_scalar('test' + str(k), test_loss_state[k], epoch)

        if self.hp.log.use_visdom:
            self.visdom_show_test(test_loss, test_loss_state, model_output, gt_imgs, epoch)

    def get_show_img_LR2HR(self, output_imgs, gt_imgs):
        show_img_idx = [0, 2, 8, -2]
        if output_imgs.shape[1] < max(show_img_idx):
            show_img_idx = list(range(output_imgs.shape[1]))

        show_img_infer = output_imgs[:,show_img_idx,...].permute(1,0,2,3)
        show_img_gt = gt_imgs[:,show_img_idx,...].permute(1,0,2,3)
        for img_i in range(len(show_img_idx)):
            show_img_infer[img_i] = Nor(show_img_infer[img_i])
            show_img_gt[img_i] = Nor(show_img_gt[img_i])
        return show_img_infer, show_img_gt, len(show_img_idx)

    def get_show_RI(self, recon_RI, post_desc = ''):
        recon_RI_real = torch.real(recon_RI)
        recon_RI_imag = torch.imag(recon_RI)

        H, W, D = recon_RI_real.shape
        img_real_XY = recon_RI_real[:,:,D//2]
        img_real_XZ = recon_RI_real[H//2,:,:]
        img_real_YZ = recon_RI_real[:,W//2,:]
        img_imag_XY = recon_RI_imag[:,:,D//2]
        img_imag_XZ = recon_RI_imag[H//2,:,:]
        img_imag_YZ = recon_RI_imag[:,W//2,:]

        self.vis.images(Nor(img_real_XY) * 0.5, win = 'recon_RI_XY_real' + post_desc, opts = dict(title = 'recon_RI_XY_real' + post_desc))
        self.vis.images(Nor(img_real_XZ) * 0.5, win = 'recon_RI_XZ_real' + post_desc, opts = dict(title = 'recon_RI_XZ_real' + post_desc))
        self.vis.images(Nor(img_real_YZ) * 0.5, win = 'recon_RI_YZ_real' + post_desc, opts = dict(title = 'recon_RI_YZ_real' + post_desc))

        self.vis.images(Nor(img_imag_XY) * 0.5, win = 'recon_RI_XY_imag' + post_desc, opts = dict(title = 'recon_RI_XY_imag' + post_desc))
        self.vis.images(Nor(img_imag_XZ) * 0.5, win = 'recon_RI_XZ_imag' + post_desc, opts = dict(title = 'recon_RI_XZ_imag' + post_desc))
        self.vis.images(Nor(img_imag_YZ) * 0.5, win = 'recon_RI_YZ_imag' + post_desc, opts = dict(title = 'recon_RI_YZ_imag' + post_desc))

        img_real_imag_XY = torch.cat([img_real_XY, img_imag_XY], dim = -1)
        self.vis.images(Nor(img_real_imag_XY) * 0.5, win = 'recon_RI_XY_real_imag' + post_desc, opts = dict(title = 'recon_RI_XY_real_imag' + post_desc))

    @ torch.no_grad()
    def visdom_show_train(self, loss, loss_state: dict, model_output, gt_imgs, gt_imgs_grad, epoch):
        ''' vis show Loss '''

        for k, v in loss_state.items():
            self.vis.line([v], [epoch], win = f'loss_{k}', opts = dict(title = f'{k}_Loss'), update = 'append')

        output_imgs = model_output['output']
        show_img_infer, show_img_gt, len_num = self.get_show_img_LR2HR(output_imgs, gt_imgs)
        self.vis.images(show_img_infer, nrow = len_num // 2, win = 'imgs_infer', opts = dict(title = 'imgs_infer'))
        self.vis.images(show_img_gt, nrow = len_num // 2, win = 'imgs_gt', opts = dict(title = 'imgs_gt'))

        ''' vis show output_LR2HR_grad & GT_grad '''
        try:
            output_imgs_grad = model_output['output_grad']
            show_img_grad_infer, show_img_grad_gt, len_num = self.get_show_img_LR2HR(output_imgs_grad.sum(2), gt_imgs_grad.sum(2))
            self.vis.images(show_img_grad_infer, nrow = len_num // 2, win = 'imgs_grad_infer', opts = dict(title = 'imgs_grad_infer'))
            self.vis.images(show_img_grad_gt, nrow = len_num // 2, win = 'imgs_grad_gt', opts = dict(title = 'imgs_grad_gt'))
        except:
            pass
        ''' vis show recon RI & GT(if has) '''
        try:
            recon_RI = model_output['recon_RI_media'] + self.hp.model.n_media
            self.get_show_RI(recon_RI)
        except:
            pass

        ''' vis show recon aberration (if has) '''
        try:
            recon_aberration = model_output['recon_aberration']
            if recon_aberration is not None:
                self.vis.images(torch.fft.fftshift(Nor(recon_aberration) * 0.5), win = 'recon aberration', opts = dict(title = 'recon aberration'))
        except:
            pass

    @ torch.no_grad()
    def visdom_show_test(self, loss, loss_state: dict, model_output, gt_imgs, epoch):
        ''' vis show Loss '''
        for k, v in loss_state.items():
            self.vis.line([loss_state[k]], [epoch], win = f'loss_{k}', opts = dict(title = f'{k}_Loss'), update = 'append')

        ''' vis show output_LR2HR & GT '''
        output_imgs = model_output['output']
        show_img_infer, _, len_num = self.get_show_img_LR2HR(output_imgs, gt_imgs)
        self.vis.images(show_img_infer, nrow = len_num // 2, win = 'imgs_infer_test', opts = dict(title = 'imgs_infer_test'))

        ''' vis show recon RI & GT(if has) '''
        recon_RI = model_output['recon_RI_media'] - self.hp.model.n_media
        self.get_show_RI(recon_RI, post_desc = '_test')

        ''' vis show recon aberration (if has) '''
        recon_aberration = model_output['recon_aberration']
        if recon_aberration is not None:
            self.vis.images(torch.fft.fftshift(Nor(recon_aberration) * 0.5), win = 'recon aberration_test', opts = dict(title = 'recon aberration_test'))
