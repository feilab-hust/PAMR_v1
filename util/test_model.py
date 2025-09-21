import numpy as np
import torch
from pathlib2 import Path
import tifffile as tiff

def test_model(hp, model, test_loader, writer, logger, img_name):
    total_test_loss = 0
    total_test_loss_state = {}
    with torch.no_grad():

        target, target_grad, planewaves, _= test_loader 
        model.feed_data(GT=target, GT_grad = target_grad, GT_sample = -1, planewaves = planewaves)
        output, evaluate_state = model.model_test()
        loss_v, loss_v_state = model.loss_f(output, model.GT, model.GT_grad)
        total_test_loss += loss_v.to("cpu").item()
        for k in loss_v_state.keys():
            if k in total_test_loss_state.keys():
                total_test_loss_state[k] += loss_v_state[k]
            else:
                total_test_loss_state.update({k: loss_v_state[k]})


        if writer is not None:
            writer.test_logging(total_test_loss, total_test_loss_state, output, model.GT, model.epoch + 1)
        if logger is not None:
            logger.info("^_^ Test Loss %.04f at step %d ^_^" % (total_test_loss, model.step + 1))
            logger.info("---> Evaluate with GT_sample: {}".format(evaluate_state))
            logger.info("----> Test: {}".format(total_test_loss_state))

        output_RI = torch.real(output['recon_RI_media'].cpu()).numpy()[..., 1:-1]
        output_Ab = torch.imag(output['recon_RI_media'].cpu()).numpy()[..., 1:-1]
        output_RI = output_RI + model.hp_net.model.n_media

        recon_aberration = torch.fft.fftshift(output['recon_aberration']).cpu().numpy() if output['recon_aberration'] is not None else np.zeros_like(output_RI)

        save_p_ri = Path(hp.recon.save_path) / f'epoch_{model.epoch}' / 'RI'
        save_p_P = Path(hp.recon.save_path) / f'epoch_{model.epoch}' / 'Pupil'
        save_p_ri.mkdir(parents = True, exist_ok = True)
        save_p_P.mkdir(parents = True, exist_ok = True)
        tiff.imwrite(save_p_ri / f'RI_{img_name[:-4]}{hp.recon.post_suffix}.tiff', output_RI.transpose(2,0,1))
        tiff.imwrite(save_p_P / f'Pupil_{img_name[:-4]}{hp.recon.post_suffix}.tiff', recon_aberration)