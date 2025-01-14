from __future__ import annotations
import torch
import matplotlib.pyplot as plt  # only needed for plotting
from mpl_toolkits.axes_grid1 import ImageGrid  # only needed for plotting
from hydra.utils import get_original_cwd, to_absolute_path
import os
 

def corrupt_mnist() -> tuple[torch.utils.data.Dataset,torch.utils.data.Dataset]:
    """Return train and test dataloaders for corrupt MNIST."""
    train_images, train_target = [], []
    print(os.getcwd())
    DATA_PATH = "data/corruptedmnist"
    DATA_PATH = to_absolute_path(DATA_PATH)
    #MLOPS/data/corruptedmnist/test_images.pt
    for i in range(6):
        train_images.append(torch.load(f"{DATA_PATH}/train_images_{i}.pt",weights_only=True))
        train_target.append(torch.load(f"{DATA_PATH}/train_target_{i}.pt",weights_only=True))
    train_images = torch.cat(train_images)
    train_target = torch.cat(train_target)

    test_images: torch.Tensor = torch.load(f"{DATA_PATH}/test_images.pt",weights_only=True)
    test_target: torch.Tensor = torch.load(f"{DATA_PATH}/test_target.pt",weights_only=True)

    train_images = train_images.unsqueeze(1).float()
    test_images = test_images.unsqueeze(1).float()
    train_target = train_target.long()
    test_target = test_target.long()

    train_set = torch.utils.data.TensorDataset(train_images, train_target)
    test_set = torch.utils.data.TensorDataset(test_images, test_target)

    return train_set, test_set


def show_image_and_target(images: torch.Tensor, target: torch.Tensor) -> None:
    """Plot images and their labels in a grid."""
    row_col = int(len(images) ** 0.5)
    fig = plt.figure(figsize=(10.0, 10.0))
    grid = ImageGrid(fig, 111, nrows_ncols=(row_col, row_col), axes_pad=0.3)
    for ax, im, label in zip(grid, images, target):
        ax.imshow(im.squeeze(), cmap="gray")
        ax.set_title(f"Label: {label.item()}")
        ax.axis("off")
    plt.show()

if __name__ == "__main__":
    train_set, test_set = corrupt_mnist()
    print(f"Size of training set: {len(train_set)}")
    print(f"Size of test set: {len(test_set)}")
    print(f"Shape of a training point {(train_set[0][0].shape, train_set[0][1].shape)}")
    print(f"Shape of a test point {(test_set[0][0].shape, test_set[0][1].shape)}")
    show_image_and_target(train_set.tensors[0][:25], train_set.tensors[1][:25])