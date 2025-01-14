import hydra
from loguru import logger
from pytorch_lightning.loggers import WandbLogger  # Import WandbLogger
import torch
import wandb
import matplotlib.pyplot as plt
from data import corrupt_mnist
from hydra.utils import to_absolute_path
from model import MyAwesomeModel
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from omegaconf import OmegaConf
import os
from pytorch_lightning.cli import LightningCLI


# Disable W&B logging
os.environ["WANDB_SILENT"] = "true"

#wandb.login()
@hydra.main(config_path = "conf", config_name = "config.yaml",version_base="1.3.2")


def train(cfg) -> None:
    print(f"configuration: \n {OmegaConf.to_yaml(cfg)}")
    hparams = OmegaConf.to_container(cfg.experiment, resolve=True, structured_config_mode=False)

    # Initialize WandbLogger
    wandb_logger = WandbLogger(
        project="MNIST-CORRUPT",  # Your project name on W&B
        config=hparams           # Track hyperparameters and run metadata
    )
    """Train a model on MNIST."""
    model = MyAwesomeModel(lr=hparams["lr"],batch_size=hparams["batch_size"])
    early_stopping_callback = EarlyStopping(
        monitor="train_loss", patience=3, verbose=True, mode="min"
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{hydra.utils.get_original_cwd()}/MLOPS/MNIST_corrupt/models", monitor="train_loss", mode="min"
    )

    trainer= Trainer(callbacks=[early_stopping_callback, checkpoint_callback],max_epochs=hparams["epochs"],default_root_dir=hydra.utils.get_original_cwd(),logger=wandb_logger,limit_train_batches=0.2)
    trainer.fit(model)
        #statistics = {"train_loss": [], "train_accuracy": []}
    # criterion = torch.nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=hparams["lr"])
    # for epoch in range(hparams["epochs"]):
    #     for i, (img,target) in enumerate(train_dataloader):
    #         img, target = img.to(DEVICE), target.to(DEVICE)
    #         optimizer.zero_grad()
    #         y_pred = model(img)
    #         loss = criterion(y_pred,target)
    #         loss.backward()
    #         optimizer.step()
    #         statistics["train_loss"].append(loss.item())
    #         accuracy = (y_pred.argmax(dim=1)==target).float().mean().item()
    #         statistics["train_accuracy"].append(accuracy)
    #         wandb.log({"accuracy": accuracy, "loss": loss})
    #         if i % 100 == 0:
    #             print(f"Epoch {epoch}, iter {i}, loss: {loss.item()}")
    #             print(f"Accuracy: {accuracy}")
    # print("Training completed")
    # torch.save(model.state_dict(),to_absolute_path("MLOPS/MNIST_corrupt/models/model.pt"))
    # artifact = wandb.Artifact(name = "Model", type = "model")
    # artifact.add_file(to_absolute_path("MLOPS/MNIST_corrupt/models/model.pt"))
    # artifact.save()
    # run.log_artifact(artifact)
    # fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    
    # axs[0].plot(statistics["train_loss"])
    # axs[0].set_title("Train loss")
    # axs[1].plot(statistics["train_accuracy"])
    # axs[1].set_title("Train accuracy")
    # fig.savefig("training_statistics.png")
    # wandb.log({"Training perfomance": wandb.Image(fig)})
    #plt.show()

#@app.command()
def evaluate(model_checkpoint: str) -> None:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    """Evaluate a trained model."""
    print("Evaluating like my life depends on it")
    print(model_checkpoint)

    # TODO: Implement evaluation logic here
    model = MyAwesomeModel().to(DEVICE)
    model.load_state_dict(torch.load("MLOPS/MNIST_corrupt/models/model.pt"))

    _, test_set = corrupt_mnist()
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=32)

    model.eval()
    correct,total = 0,0
    for img,target in test_dataloader:
        img, target = img.to(DEVICE), target.to(DEVICE)
        y_pred = model(img)
        correct += (y_pred.argmax(dim=1) == target).float().sum().item()
        total += target.size(0)
    print(f"Test accuracy: {correct / total}")


    


if __name__ == "__main__":
    LightningCLI(MyAwesomeModel)
