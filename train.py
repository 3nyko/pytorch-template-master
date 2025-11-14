import argparse
import collections
import torch
import numpy as np
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from trainer import Trainer
from utils import prepare_device

import os
import sys
import subprocess
import threading
import webbrowser
from pathlib import Path

DEFAULT_CONFIG_PATH = r"C:\Users\fisar\Desktop\Diplomka\pytorch-template-master\configs\config_CICIoV_split.json"

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def main(config):
    logger = config.get_logger('train')

    start_tensorboard_for_latest()

    # setup data_loader instances
    data_loader = config.init_obj('data_loader', module_data)
    valid_data_loader = data_loader.split_validation()
    print_data_info(data_loader, valid_data_loader)

    # build model architecture, then print to console
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      device=device,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler)

    trainer.train()

def start_tensorboard():
    log_dir = os.path.join("saved", "log")
    subprocess.Popen(["tensorboard", "--logdir", log_dir, "--port", "6006"])
    threading.Timer(2.0, lambda: webbrowser.open("http://localhost:6006")).start()

def start_tensorboard_for_latest(base_log_dir="saved/log/CICIoV2024_split", port=6006):
    base = Path(base_log_dir)
    if not base.exists():
        print(f"[TensorBoard] Log dir not found: {base.resolve()}")
        return

    # Najdi nejnovější podadresář
    subdirs = [d for d in base.iterdir() if d.is_dir()]
    if not subdirs:
        print(f"[TensorBoard] No subdirectories in {base}")
        return

    latest_run = max(subdirs, key=lambda p: p.stat().st_mtime)
    print(f"[TensorBoard] Starting TensorBoard for: {latest_run}")

    # Spusť TensorBoard pouze pro aktuální běh
    cmd = [
        sys.executable, "-m", "tensorboard.main",
        "--logdir", str(latest_run),
        "--port", str(port),
        "--host", "127.0.0.1",
    ]
    subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    threading.Timer(2.0, lambda: webbrowser.open(f"http://localhost:{port}")).start()

def print_data_info(data_loader, valid_data_loader):
    """
    Get train and val sample count
    """
    num_train_samples = len(data_loader.dataset) if hasattr(data_loader, 'dataset') else len(data_loader.data_loader.dataset)
    num_val_samples = len(valid_data_loader.dataset) if valid_data_loader is not None else 0

    num_train_batches = len(data_loader)
    num_val_batches = len(valid_data_loader) if valid_data_loader is not None else 0

    print("\nDataset info:")
    print(f"\tTraining samples:   {num_train_samples} ({num_train_batches} batches)")
    print(f"\tValidation samples: {num_val_samples} ({num_val_batches} batches)")

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=DEFAULT_CONFIG_PATH, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
