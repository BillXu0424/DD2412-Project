import time
from collections import defaultdict
from datetime import timedelta
from typing import Tuple

import hydra
import torch
from omegaconf import DictConfig
from tqdm import tqdm
import os

from codebase.utils import rotation_utils, data_utils, model_utils, eval_utils, utils
from skimage.transform import resize
from skimage.filters.rank import modal
from skimage.morphology import disk


def train(opt: DictConfig, model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> Tuple[int, torch.nn.Module]:
    """
    Train the model.

    Args:
        opt (DictConfig): Configuration options.
        model (torch.nn.Module): The model to be trained.
        optimizer (torch.optim.Optimizer): The optimizer used for training.

    Returns:
        torch.nn.Module: The trained model.
    """
    train_start_time = time.time()

    train_loader = data_utils.get_data(opt, "train")
    step = 0

    while step < opt.training.steps:
        for input_images, labels in train_loader:
            cur_iter_start_time = time.time()
            print_results = (opt.training.print_idx > 0 and step % opt.training.print_idx == 0)

            input_images = input_images.cuda(non_blocking=True)

            optimizer, lr = utils.update_learning_rate(optimizer, opt, step)
            optimizer.zero_grad()

            loss, metrics = model(input_images, labels, evaluate=print_results)
            loss.backward()

            if opt.training.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), opt.training.gradient_clip)

            optimizer.step()

            # Print results.
            if print_results:
                cur_iter_end_time = time.time()
                iteration_time = cur_iter_end_time - cur_iter_start_time
                log_name = os.path.splitext(opt.log.path)[0] + "_" + str(opt.model.rotation_dimensions) + \
                           os.path.splitext(opt.log.path)[1]
                utils.print_results("train", step, iteration_time, metrics, os.path.join(opt.cwd, log_name))

            # Validate.
            if opt.training.val_idx > 0 and step % opt.training.val_idx == 0:
                validate_or_test(opt, step, model, "val")

            step += 1
            if step >= opt.training.steps:
                break

    train_end_time = time.time()
    total_train_time = train_end_time - train_start_time
    print(f"Total training time: {timedelta(seconds=total_train_time)}")
    return step, model


def validate_or_test(opt: DictConfig, step: int, model: torch.nn.Module, partition: str) -> None:
    """
    Perform validation or testing of the model.

    Args:
        opt (DictConfig): Configuration options.
        step (int): Current training step.
        model (torch.nn.Module): The model to be evaluated.
        partition (str): Partition name ("val" or "test").
    """
    test_start_time = time.time()
    test_results = defaultdict(float)

    data_loader = data_utils.get_data(opt, partition)

    model.eval()
    print(partition)
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader):
            input_images = inputs.cuda(non_blocking=True)

            loss, metrics = model(input_images, labels, evaluate=True)

            test_results["Loss"] += loss.item() / len(data_loader)
            for key, value in metrics.items():
                test_results[key] += value / len(data_loader)

    test_end_time = time.time()
    total_test_time = test_end_time - test_start_time
    log_name = os.path.splitext(opt.log.path)[0] + "_" + str(opt.model.rotation_dimensions) + \
               os.path.splitext(opt.log.path)[1]
    utils.print_results(partition, step, total_test_time, test_results, os.path.join(opt.cwd, log_name))
    model.train()


@hydra.main(config_path="config", config_name="config")
def main(opt: DictConfig) -> None:
    opt = utils.parse_opt(opt)

    # Initialize model and optimizer.
    model, optimizer = model_utils.get_model_and_optimizer(opt)

    step, model = train(opt, model, optimizer)
    validate_or_test(opt, step, model, "test")

    save_name = os.path.splitext(opt.save.path)[0] + "_" + str(opt.model.rotation_dimensions) + \
                os.path.splitext(opt.save.path)[1]
    torch.save(model.state_dict(), os.path.join(opt.cwd, save_name))


if __name__ == "__main__":
    main()
