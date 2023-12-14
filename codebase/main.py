import hydra
import torch
from omegaconf import DictConfig
import os

from train import train, val_or_test
from codebase.utils import model_utils, utils


@hydra.main(config_path="config", config_name="config")
def main(opt):
    opt = utils.parse_opt(opt)

    # Initialize model and optimizer.
    model, optimizer = model_utils.get_model_and_optimizer(opt)

    step, model = train(opt, model, optimizer)
    val_or_test(opt, step, model, "test")

    save_name = os.path.splitext(opt.save.path)[0] + "_" + str(opt.model.rotation_dimensions) + \
                os.path.splitext(opt.save.path)[1]
    torch.save(model.state_dict(), os.path.join(opt.cwd, save_name))


if __name__ == "__main__":
    main()
