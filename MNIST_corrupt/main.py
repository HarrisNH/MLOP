import torch
import matplotlib.pyplot as plt
from torch import nn
from torch import optim
from data import corrupt_mnist
from model import MyAwesomeModel
import typer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
app = typer.Typer()

@app.command()
def train(lr: float = 1e-3, batch_size: int = 32, epochs: int=5) -> None:
    """Train a model on MNIST."""
    print("Training day and night")
    print(lr)
    train_set, _ = corrupt_mnist()
    # TODO: Implement training loop here
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)



    statistics = {"train_loss": [], "train_accuracy": []}

    model = MyAwesomeModel().to(DEVICE)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        for i, (img,target) in enumerate(train_dataloader):
            img, target = img.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            y_pred = model(img)
            loss = criterion(y_pred,target)
            loss.backward()
            optimizer.step()
            statistics["train_loss"].append(loss.item())
            accuracy = (y_pred.argmax(dim=1)==target).float().mean().item()
            statistics["train_accuracy"].append(accuracy)

            if i % 100 == 0:
                print(f"Epoch {epoch}, iter {i}, loss: {loss.item()}")
    print("Training completed")
    torch.save(model.state_dict(),"MLOPS/MNIST_corrupt/models/model.pt")
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(statistics["train_loss"])
    axs[0].set_title("Train loss")
    axs[1].plot(statistics["train_accuracy"])
    axs[1].set_title("Train accuracy")
    fig.savefig("training_statistics.png")

    plt.show()
@app.command()
def evaluate(model_checkpoint: str) -> None:
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
    app()