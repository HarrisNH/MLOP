import torch
from pytorch_lightning import LightningModule
from torch import nn

from data import corrupt_mnist


class MyAwesomeModel(LightningModule):
    """My awesome model."""

    def __init__(self, lr: float = 0.01, batch_size: int = 64) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128, 10)

        self.criterium = nn.CrossEntropyLoss()

    def forward(self, x):
        if x.dim() != 4:
            raise ValueError(f"Expected input of size (B, C, H, W), got {x.shape}")
        if x.shape[1:] != (1, 28, 28):
            raise ValueError(f"Expected input of size (B, 1, 28, 28), got {x.shape}")
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(
            x,
            2,
            2,
        )
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.relu(self.conv3(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc1(x)

    def training_step(self, batch, batch_idx):
        data, target = batch
        preds = self(data)
        loss = self.criterium(preds, target)
        acc = (target == preds.argmax(dim=-1)).float().mean()
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def testing_step(self, batch, batch_idx):
        data, target = batch
        preds = self(data)
        loss = self.criterium(preds, target)
        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), self.hparams["lr"])

    def train_dataloader(self):
        train_set, _ = corrupt_mnist()
        train_dataloader = torch.utils.data.DataLoader(
            train_set,
            batch_size=self.hparams["batch_size"],
            num_workers=7,
            persistent_workers=True,
        )
        return train_dataloader

    def test_dataloader(self):
        _, test_set = corrupt_mnist()
        test_dataloader = torch.utils.data.DataLoader(
            test_set,
            batch_size=self.hparams["batch_size"],
            num_workers=7,
            persistent_workers=True,
        )
        return test_dataloader

    def validation_step(self, batch) -> None:
        data, target = batch
        preds = self(data)
        loss = self.criterion(preds, target)
        acc = (target == preds.argmax(dim=-1)).float().mean()
        self.log("val_loss", loss, on_epoch=True)
        self.log("val_acc", acc, on_epoch=True)


if __name__ == "__main__":
    model = MyAwesomeModel()
    print(f"Model architecture: {model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    dummy_input = torch.randn(1, 1, 28, 28)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
