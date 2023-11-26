import torch
from omegaconf import DictConfig
import hydra
from ..utils.data_utils import get_data
from ..utils.utils import parse_opt
from ..utils.model_utils import get_model_and_optimizer
from ..utils.rotation_utils import norm_and_mask_rotating_output
from ..utils.eval_utils import apply_kmeans
import matplotlib.pyplot as plt


@hydra.main(config_path="../config", config_name="config")
def my_main(opt: DictConfig) -> None:
    opt = parse_opt(opt)
    index = 0
    model_path = "../saved_models/4Shapes_8.pth"
    compare_num_objects(opt, model_path, index)


def compare_reconstruction(opt: DictConfig, model_path: str, index: int) -> None:
    model, _ = get_model_and_optimizer(opt)
    model.load_state_dict(torch.load(model_path))
    data_loader = get_data(opt, "test")
    model.eval()
    with torch.no_grad():
        inputs, labels = next(iter(data_loader))
        input_images = inputs.cuda(non_blocking=True)

        images_to_process, _ = model.preprocess_input_images(input_images)
        z = model.encode(images_to_process)
        _, reconstruction = model.decode(z)

        original_image = input_images[index, 0].cpu().numpy()
        reconstruction_image = reconstruction[index, 0].cpu().numpy()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        img1 = ax1.imshow(original_image, cmap='gray')
        ax1.set_title('Original Image')
        plt.colorbar(img1, ax=ax1)

        img2 = ax2.imshow(reconstruction_image, cmap='gray')
        ax2.set_title(f'Reconstruction Image, n = {opt.model.rotation_dimensions}')
        plt.colorbar(img2, ax=ax2)

        plt.show()


def compare_num_objects(opt: DictConfig, model_path: str, index: int) -> None:
    model, _ = get_model_and_optimizer(opt)
    model.load_state_dict(torch.load(model_path))
    data_loader = get_data(opt, "test")
    model.eval()
    with torch.no_grad():
        inputs, labels = next(iter(data_loader))
        input_images = inputs.cuda(non_blocking=True)

        images_to_process, _ = model.preprocess_input_images(input_images)
        z = model.encode(images_to_process)
        rotation_output, _ = model.decode(z)

        norm_rotating_output = norm_and_mask_rotating_output(opt, rotation_output)
        pred_labels = apply_kmeans(opt, norm_rotating_output, labels["pixelwise_instance_labels"])

        original_label = labels["pixelwise_instance_labels"][index].cpu().numpy()
        pred_label = pred_labels[index]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        img1 = ax1.imshow(original_label, cmap='viridis')
        ax1.set_title('Original Label')
        plt.colorbar(img1, ax=ax1)

        img2 = ax2.imshow(pred_label, cmap='viridis')
        ax2.set_title(f'Predicted Label, n = {opt.model.rotation_dimensions}')
        plt.colorbar(img2, ax=ax2)

        plt.show()


if __name__ == '__main__':
    my_main()