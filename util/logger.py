import os
import logging
import shutil
from util.util import get_timestamp


def make_logger(hp_net, hp_fpm):
    # set log/checkpoint dir
    hp_net.log.chkpt_dir = os.path.join(hp_net.log.chkpt_dir, hp_net.log.name)
    hp_net.log.log_dir = os.path.join(hp_net.log.log_dir, hp_net.log.name)
    os.makedirs(hp_net.log.chkpt_dir, exist_ok=True)
    os.makedirs(hp_net.log.log_dir, exist_ok=True)

    shutil.copy(hp_net.yaml_dir, hp_net.log.log_dir)
    shutil.copy(hp_fpm.yaml_dir, hp_net.log.log_dir)

    hp_net.log.log_file_path = os.path.join(
        hp_net.log.log_dir, "%s-%s.log" % (hp_net.log.name, get_timestamp())
    )

    # set logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(hp_net.log.log_file_path), logging.StreamHandler(),],
    )
    logger = logging.getLogger()
    return logger
