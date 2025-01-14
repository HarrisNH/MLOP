import torch
import os.path
from MNIST_corrupt.data import corrupt_mnist
import pytest

@pytest.mark.skipif(not os.path.exists("/Users/harrishadzimahovic/Desktop/DTU/5/Machine Learning Operations/MLOPS/data/corruptedmnist/train_images_0.pt"), reason="Data files not found")
def test_data():
    train_data,test_data = corrupt_mnist()
    assert len(train_data) == 27000, "Expected 30000 samples in train data"
    assert len(test_data) == 5000
    for dataset in [train_data,test_data]:
        for x,y in dataset:
            assert x.shape == (1,28,28)
            assert y in range(10)
            
    train_targets = torch.unique(train_data.tensors[1])
    test_targets = torch.unique(test_data.tensors[1])
    assert (train_targets == torch.arange(0,10)).all()
    assert (test_targets == torch.arange(0,10)).all()
