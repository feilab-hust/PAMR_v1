import argparse
import yaml
import itertools
import traceback
import random
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def setup(hp_net, rank, world_size):
    os.environ["MASTER_ADDR"] = hp_net.train.dist.master_addr
    os.environ["MASTER_PORT"] = hp_net.train.dist.master_port

    # initialize the process group
    dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo",
                            rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def distributed_run(fn, hp_fpm, hp_net, world_size, vis = None):
    mp.spawn(fn, args=(hp_fpm, hp_net, world_size, vis), nprocs=world_size, join=True)

def train_loop(rank, hp_fpm, hp_net, world_size=0, vis = None):
    if hp_net.model.device == "cuda" and world_size != 0:
        hp_net.model.device = rank
        setup(hp_net, rank, world_size)
        torch.cuda.set_device(hp_net.model.device)

    # setup logger / writer
    if rank != 0:
        logger = None
        writer = None
    else:
        # set logger
        logger = make_logger(hp_net, hp_fpm)
        # set writer (tensorboard, visdom)
        writer = Writer(hp_net, os.path.join(hp_net.log.log_dir, "tensorboard"), vis)
        hp_str = yaml.dump(hp_net.to_dict())
        if hp_net.log.print_yaml:
            logger.info("Config:")
            logger.info(hp_str)
        if hp_net.data.train_dir == "" or hp_net.data.test_dir == "":
            logger.error("train or test data directory cannot be empty.")
            raise Exception("Please specify directories of data")
        logger.info("Set up train process")

    # Sync dist processes (because of download MNIST Dataset)
    if hp_net.model.device == "cuda" and world_size != 0:
        dist.barrier()

    # make dataloaders
    if logger is not None:
        logger.info("Making train dataloader...")
    train_loader = create_dataloader(hp_fpm, hp_net, DataloaderMode.train, rank, world_size)

    # init Model
    net_arch = Net_arch(hp_net, post_name = 'opensourced')
    loss_f = loss_used(hp_net)
    forward_function = train_loader.dataset.forward_function
    Num_LEDs = train_loader.dataset.Num_LED

    model = Model(hp_net, net_arch, forward_function, Num_LEDs, loss_f, rank, world_size)
    logger.info("Starting new training run.")

    try:
        if world_size == 0 or hp_net.data.divide_dataset_per_gpu:
            epoch_step = 1
        else:
            epoch_step = world_size

        for i, input_list in enumerate(train_loader):
            target, target_grad, planewaves, OF_factor, img_name = input_list
            input_list_ = [target, target_grad, planewaves[0], OF_factor[0]]
            if logger is not None:
                logger.info(f"Training the {i + 1}th / [{len(train_loader)}] sample; name: {img_name[0]}")

            with tqdm(total = hp_net.train.num_epoch) as t:
                t.set_description('training ---> ')
                model.epoch = -1

                for model.epoch in itertools.count(model.epoch + 1, epoch_step):
                    if model.epoch > hp_net.train.num_epoch:
                        break
                    train_model(hp_net, model, train_loader = input_list_, writer = writer, logger = logger)
                    model.run_lr_scheduler()
                    if hp_net.log.val_interval and model.epoch > 0 and model.epoch % hp_net.log.val_interval == 0:
                        test_model(hp_fpm, model, test_loader = train_loader, writer = writer, logger = logger)
                    if model.epoch == hp_net.train.num_epoch or (hp_net.log.chkpt_interval and model.epoch > 0 and model.epoch % hp_net.log.chkpt_interval == 0):
                        model.save_network(logger)
                    t.update(n = 1)

            if logger is not None:
                logger.info(f"End of Train")

            if hp_net.log.show_last:
                test_model(hp_fpm, model, test_loader = input_list_, writer = writer, logger = logger, img_name = img_name[0])

                if logger is not None:
                    logger.info(f"Save the reconstructed RI, name: {img_name[0]}")

            torch.cuda.empty_cache()

    except Exception as e:
        if logger is not None:
            logger.error(traceback.format_exc())
        else:
            traceback.print_exc()
    finally:
        if hp_net.model.device == "cuda" and world_size != 0:
            cleanup()

def main():
    import os
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-Net_c", "--Net_config", type=str, default = r'config/Net_INR_experience_Bio.yaml',
        help="Net yaml file for config."
    )
    parser.add_argument(
        "-3D_FPM_c", "--FPM_config", type=str, default = r'config/3D_FPM_experience_Bio.yaml',
        help="3D_FPM yaml file for config."
    )
    args = parser.parse_args()

    ''' default config (change less often) '''
    hp_net = load_hparam(r'config_default/Net_INR_experience_Bio_default.yaml')
    hp_fpm = load_hparam(r'config_default/3D_FPM_experience_Bio_default.yaml')

    ''' config (change more often) '''
    hp_net_change = load_hparam(args.Net_config)
    hp_fpm_change = load_hparam(args.FPM_config)

    deep_dict_update(hp_net, hp_net_change)
    deep_dict_update(hp_fpm, hp_fpm_change)

    hp_net.model.device = hp_net.model.device.lower()
    hp_net.yaml_dir = args.Net_config
    hp_fpm.yaml_dir = args.FPM_config

    # random seed
    if hp_net.train.random_seed is None:
        hp_net.train.random_seed = random.randint(1, 10000)
    set_random_seed(hp_net.train.random_seed)

    # set GPUs used --> not suitable for remote development
    # os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, hp_net.train.dist.gpus))

    # set Visdom wins if used
    if hp_net.log.use_visdom:
        vis = visdom.Visdom(env = hp_net.log.name, port = 9000)
    else:
        vis = None

    if hp_net.train.dist.gpus[0] < 0:
        hp_net.train.dist.gpus = [torch.cuda.device_count()]

    if hp_net.model.device == "cpu" or hp_net.train.dist.gpus == [0]:
        train_loop(0, hp_fpm, hp_net, vis = vis)
    else:
        assert torch.cuda.is_available() and torch.cuda.device_count() >= len(hp_net.train.dist.gpus), \
            'cuda.is_available: {}, cuda.device_count: {}'.format(torch.cuda.is_available(), torch.cuda.device_count())
        distributed_run(train_loop, hp_fpm, hp_net, len(hp_net.train.dist.gpus), vis = vis)


if __name__ == "__main__":
    from util.util import load_hparam, set_random_seed, deep_dict_update
    set_random_seed(42)

    from model.model_arch import Net_arch
    from model.model_INR import Model
    from util.train_model import train_model
    from util.test_model import test_model
    from util.writer import Writer
    from util.logger import make_logger
    from loss.loss import loss_used
    from dataset.dataloader_INR import create_dataloader, DataloaderMode
    import visdom
    from tqdm import tqdm
    import warnings

    warnings.filterwarnings("ignore")
    main()
