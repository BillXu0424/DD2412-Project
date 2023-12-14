import numpy as np
import torch
from omegaconf import DictConfig
import hydra
from ..utils.data_utils import get_data
from ..utils.utils import parse_opt
from ..utils.model_utils import get_model_and_optimizer
from ..utils.rotation_utils import norm_and_mask_rotating_output
from ..utils.eval_utils import apply_kmeans, resize_pred_labels
import matplotlib.pyplot as plt


@hydra.main(config_path="../config", config_name="config")
def my_main(opt: DictConfig) -> None:
    opt = parse_opt(opt)
    index = 50
    # model_path = f"../saved_models/4Shapes_RGBD_{opt.model.rotation_dimensions}.pth"
    model_path = "../saved_models/Pascal.pth"
    compare_num_objects(opt, model_path, index)
    # compare_reconstruction(opt, model_path, index)


def compare_reconstruction(opt: DictConfig, model_path: str, index: int) -> None:
    model, _ = get_model_and_optimizer(opt)
    model.load_state_dict(torch.load(model_path))
    data_loader = get_data(opt, "eval")
    # data_loader = get_data(opt, "test")
    model.eval()
    with torch.no_grad():
        inputs, labels = next(iter(data_loader))
        input_images = inputs.cuda(non_blocking=True)

        images_to_process, _ = model.preprocess_input_images(input_images)
        z = model.encode(images_to_process)
        _, reconstruction = model.decode(z)

        original_image = input_images[index, 0].cpu().numpy()
        reconstruction_image = reconstruction[index, 0].cpu().numpy()

        # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        # img1 = ax1.imshow(original_image, cmap='gray')
        # img1 = ax1.imshow(original_image)
        # ax1.set_title('Original Image')
        # plt.colorbar(img1, ax=ax1)

        # img2 = ax2.imshow(reconstruction_image, cmap='gray')
        # img2 = ax2.imshow(reconstruction_image)
        # ax2.set_title(f'Reconstruction Image, n = {opt.model.rotation_dimensions}')
        # plt.colorbar(img2, ax=ax2)

        plt.imshow(reconstruction_image, cmap='gray')
        plt.axis('off')
        plt.savefig(f"../fig/4Shapes_RGBD_i{index}_n{opt.model.rotation_dimensions}.jpg", bbox_inches='tight')

        plt.show()


def compare_num_objects(opt: DictConfig, model_path: str, index: int) -> None:
    model, _ = get_model_and_optimizer(opt)
    model.load_state_dict(torch.load(model_path))
    data_loader = get_data(opt, "test")
    # data_loader = get_data(opt, "eval")
    model.eval()
    with torch.no_grad():
        inputs, labels = next(iter(data_loader))
        input_images = inputs.cuda(non_blocking=True)
        images_to_process, _ = model.preprocess_input_images(input_images)

        original_image = images_to_process[index].cpu()

        z = model.encode(images_to_process)
        rotation_output, _ = model.decode(z)

        norm_rotating_output = norm_and_mask_rotating_output(opt, rotation_output)
        pred_labels = apply_kmeans(opt, norm_rotating_output, labels["pixelwise_instance_labels"])  # pixelwise_instance_labels

        original_label = labels["pixelwise_instance_labels"][index].cpu().numpy()  # pixelwise_instance_labels

        # fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 6))
        #
        # img1 = ax1.imshow(original_label, cmap='viridis')
        # ax1.set_title('Original Label')
        # # plt.colorbar(img1, ax=ax1)
        #.
        # img2 = ax2.imshow(pred_label, cmap='viridis')
        # ax2.set_title(f'Predicted Label, n = {opt.model.rotation_dimensions}')
        # # plt.colorbar(img2, ax=ax2)
        #
        # # ax3.imshow(np.transpose(original_image, (1, 2, 0)))
        # ax3.imshow(inputs[index].squeeze(), cmap='gray')
        # ax3.set_title('Original Image')

        # plt.imshow(np.transpose(inputs[index, :3], (1, 2, 0)))
        # plt.imshow(original_label, cmap='viridis')
        # plt.axis('off')
        # plt.savefig(f"../fig/4Shapes_RGBD_i{index}_n{opt.model.rotation_dimensions}.jpg", bbox_inches='tight')
        # plt.savefig(f"../fig/Pascal_i{index}_gt.jpg", bbox_inches='tight')

        resized_pred_labels = resize_pred_labels(opt, pred_labels, labels["pixelwise_instance_labels"])
        plt.imshow(resized_pred_labels[index].squeeze(), cmap='viridis')
        plt.axis('off')
        plt.savefig(f"../fig/Pascal_i{index}_pred.jpg", bbox_inches="tight")
        plt.show()


if __name__ == '__main__':
    my_main()