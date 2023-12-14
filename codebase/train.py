import time
from collections import defaultdict
from datetime import timedelta
from typing import Tuple

import torch
from omegaconf import DictConfig
from tqdm import tqdm
import os

from codebase.utils import data_utils, utils


def train(opt, model, optimizer):
    """
    Model Train
    """
    train_start_time = time.time()

    train_loader = data_utils.get_data(opt, "train")

    step = 0
    while step < opt.training.steps:
        for images, labels in train_loader:
            flag_print_results = (opt.training.print_idx > 0 and step % opt.training.print_idx == 0)

            cur_iter_start_time = time.time()

            images = images.cuda(non_blocking=True)

            optimizer, lr = utils.update_learning_rate(optimizer, opt, step)
            optimizer.zero_grad()

            loss, metrics = model(images, labels, evaluate=flag_print_results)
            loss.backward()

            if opt.training.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), opt.training.gradient_clip)

            optimizer.step()

            # Print results.
            if flag_print_results:
                cur_iter_end_time = time.time()
                iteration_time = cur_iter_end_time - cur_iter_start_time
                log_name = os.path.splitext(opt.log.path)[0] + "_" + str(opt.model.rotation_dimensions) + \
                           os.path.splitext(opt.log.path)[1]
                utils.print_results("train", step, iteration_time, metrics, os.path.join(opt.cwd, log_name))

            # Validate.
            if opt.training.val_idx > 0 and step % opt.training.val_idx == 0:
                val_or_test(opt, step, model, "val")

            step += 1
            if step >= opt.training.steps:
                break

    train_end_time = time.time()
    total_train_time = train_end_time - train_start_time
    print(f"Total training time: {timedelta(seconds=total_train_time)}")
    return step, model


def val_or_test(opt, step, model, partition):
    """
    function of doing validation or testing.
    """
    test_start_time = time.time()
    test_results = defaultdict(float)

    data_loader = data_utils.get_data(opt, partition)

    model.eval()
    print(partition)
    with torch.no_grad():
        for images, labels in tqdm(data_loader):
            images = images.cuda(non_blocking=True)

            loss, metrics = model(images, labels, evaluate=True)

            test_results["Loss"] += loss.item() / len(data_loader)
            for key, value in metrics.items():
                test_results[key] += value / len(data_loader)

    test_end_time = time.time()
    total_test_time = test_end_time - test_start_time
    log_name = os.path.splitext(opt.log.path)[0] + "_" + str(opt.model.rotation_dimensions) + \
               os.path.splitext(opt.log.path)[1]
    utils.print_results(partition, step, total_test_time, test_results, os.path.join(opt.cwd, log_name))
    model.train()
