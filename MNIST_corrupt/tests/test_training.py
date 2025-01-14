import pytest
from pytorch_lightning.cli import LightningCLI
from MNIST_corrupt.model import MyAwesomeModel
import tempfile
import os

@pytest.fixture
def dummy_args():
    """Fixture to provide dummy arguments for LightningCLI."""
    return [
        "--model.lr=0.001",
        "--model.batch_size=16",
        "--trainer.max_epochs=1",
        f"--trainer.default_root_dir={tempfile.gettempdir()}",
    ]

def test_training_cli(dummy_args, monkeypatch):
    """Test the training process using LightningCLI."""
    # Mock command-line arguments for LightningCLI
    monkeypatch.setattr("sys.argv", ["main.py"] + dummy_args)

    # Run the LightningCLI
    cli = LightningCLI(MyAwesomeModel, run=False)  # Initialize the CLI
    cli.trainer.fit(cli.model)  # Manually trigger training

    # Check if model checkpoint was created in the temporary directory
    checkpoint_dir = os.path.join(tempfile.gettempdir(), "lightning_logs")
    assert os.path.exists(checkpoint_dir), "Training did not produce expected outputs!"