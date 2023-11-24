import logging
import math
import os
import random
from pathlib import Path
from types import SimpleNamespace

import torch
import torchvision
import torchvision.transforms as T
import wandb
from torch import nn

# pylint: disable=logging-fstring-interpolation,not-callable,no-member

LOGGER = logging.getLogger("mnist_exp")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_HUB = str(os.getenv("TORCH_HUB"))
WANDB_DIR = Path(str(os.getenv("WANDB_DIR"))) / "mnist_exp"


def get_dataloader(is_train, batch_size, step=5):
    "Get train/validation dataloaders."
    full_dataset = torchvision.datasets.MNIST(
        root=TORCH_HUB, train=is_train, transform=T.ToTensor(), download=True
    )
    sub_dataset = torch.utils.data.Subset(
        full_dataset, indices=range(0, len(full_dataset), step)
    )
    loader = torch.utils.data.DataLoader(
        dataset=sub_dataset,
        batch_size=batch_size,
        shuffle=bool(is_train),
        pin_memory=True,
        num_workers=2,
    )
    return loader


def get_model(dropout):
    "A simple model"
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(256, 10),
    ).to(DEVICE)
    return model


def validate_model(model, valid_dl, loss_func, log_images=False, batch_idx=0):
    "Compute performance of the model on the validation dataset and log a wandb.Table"
    model.eval()
    val_loss = 0.0
    with torch.inference_mode():
        correct = 0
        for i, (images, labels) in enumerate(valid_dl):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            # Forward pass ➡
            outputs = model(images)
            val_loss += loss_func(outputs, labels) * labels.size(0)

            # Compute accuracy and accumulate
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()

            # Log one batch of images to the dashboard, always same batch_idx.
            if i == batch_idx and log_images:
                log_image_table(images, predicted, labels, outputs.softmax(dim=1))
    return val_loss / len(valid_dl.dataset), correct / len(valid_dl.dataset)


def log_image_table(images, predicted, labels, probs):
    "Log a wandb.Table with (img, pred, target, scores)"
    # Create a wandb Table to log images, labels and predictions to
    table = wandb.Table(
        columns=["image", "pred", "target"] + [f"score_{i}" for i in range(10)]
    )
    for img, pred, targ, prob in zip(
        images.to("cpu"), predicted.to("cpu"), labels.to("cpu"), probs.to("cpu")
    ):
        table.add_data(wandb.Image(img[0].numpy() * 255), pred, targ, *prob.numpy())
    wandb.log({"predictions_table": table}, commit=False)


def main():
    """Main function."""
    # pylint: disable=too-many-locals

    logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)

    # Launch 5 experiments, trying different dropout rates
    for _ in range(5):
        # Initialize wandb
        config = SimpleNamespace(
            epochs=5,
            batch_size=128,
            lr=1e-3,
            dropout=random.uniform(0.01, 0.80),
        )
        wandb.init(project="mnist-classification", config=config.__dict__)

        # Any logs should be after wandb initialization
        LOGGER.info(f"Torch version: {torch.__version__}")
        LOGGER.info(f"Computing device: {DEVICE}")
        LOGGER.info(f"Dataset location: {TORCH_HUB}")
        LOGGER.info(f"Generated file location by wandb: {WANDB_DIR}")
        LOGGER.info(f"wandb config: {config}")

        # Define dataloaders
        LOGGER.info("Define dataloaders...")
        train_dl = get_dataloader(is_train=True, batch_size=config.batch_size)
        valid_dl = get_dataloader(is_train=False, batch_size=2 * config.batch_size)
        n_steps_per_epoch = math.ceil(len(train_dl.dataset) / config.batch_size)
        LOGGER.info(f"  #train samples: {len(train_dl.dataset)}")
        LOGGER.info(f"  #valid samples: {len(valid_dl.dataset)}")
        LOGGER.info(f"  n_steps_per_epoch: {n_steps_per_epoch}")

        # A simple MLP model
        model = get_model(config.dropout)
        LOGGER.info(f"Model: {model}")

        # Make the loss and optimizer
        loss_func = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        LOGGER.info(f"Loss function: {loss_func}")
        LOGGER.info(f"Optimizer: {optimizer}")

        # Training
        example_ct = 0
        step_ct = 0
        for epoch in range(config.epochs):
            model.train()
            for step, (images, labels) in enumerate(train_dl):
                images, labels = images.to(DEVICE), labels.to(DEVICE)

                outputs = model(images)
                train_loss = loss_func(outputs, labels)
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                example_ct += len(images)
                metrics = {
                    "train/train_loss": train_loss,
                    "train/epoch": epoch,
                    "train/example_ct": example_ct,
                    "train/step_ct": step_ct,
                }

                # Log train metrics to wandb
                if step + 1 < n_steps_per_epoch:
                    wandb.log(metrics)

                step_ct += 1

            val_loss, accuracy = validate_model(
                model, valid_dl, loss_func, log_images=(epoch == (config.epochs - 1))
            )

            # Log train and validation metrics to wandb
            val_metrics = {
                "val/val_loss": val_loss,
                "val/val_accuracy": accuracy,
            }
            wandb.log({**metrics, **val_metrics})

            LOGGER.info(f"Epoch: {epoch}")
            LOGGER.info(f"  Train Loss: {train_loss:.3f}")
            LOGGER.info(f"  Valid Loss: {val_loss:.3f}")
            LOGGER.info(f"  Accuracy: {accuracy:.3f}")

        # If you had a test set, this is how you could log it as a summary metric
        wandb.summary["test_accuracy"] = 0.8

        # Close your wandb
        wandb.finish()


if __name__ == "__main__":
    main()
