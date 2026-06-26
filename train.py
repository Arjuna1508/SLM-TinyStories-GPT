import os
import math
import time

import numpy as np
import torch
import matplotlib.pyplot as plt

from model import GPT, GPTConfig


batch_size = 32
block_size = 128
max_iters = 16000
eval_interval = 500
eval_iters = 10
learning_rate = 3e-4
device = "cuda" if torch.cuda.is_available() else "cpu"

data_dir = "data"
checkpoint_path = "/content/drive/MyDrive/slm_best_model.pt"
loss_plot_path = "loss_curve.png"
train_data = np.memmap(os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r")
val_data = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r")


def get_batch(split):
    data = train_data if split == "train" else val_data

    ix = torch.randint(len(data) - block_size - 1, (batch_size,))

    x = torch.stack([
        torch.from_numpy((data[i:i + block_size]).astype(np.int64))
        for i in ix
    ])

    y = torch.stack([
        torch.from_numpy((data[i + 1:i + block_size + 1]).astype(np.int64))
        for i in ix
    ])

    x = x.to(device)
    y = y.to(device)

    return x, y


@torch.no_grad()
def estimate_loss(model):
    out = {}

    model.eval()

    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)

        for k in range(eval_iters):
            x, y = get_batch(split)
            logits, loss = model(x, y)
            losses[k] = loss.item()

        out[split] = losses.mean().item()

    model.train()
    return out


def main():
    print("Device:", device)

    config = GPTConfig(
        vocab_size=50257,
        block_size=block_size,
        n_layer=2,
        n_head=2,
        n_embd=64,
        dropout=0.1
    )

    model = GPT(config).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print("Total parameters:", total_params)

    x, y = get_batch("train")
    print("Batch x shape:", x.shape)
    print("Batch y shape:", y.shape)
    print("First x row:", x[0, :10].tolist())
    print("First y row:", y[0, :10].tolist())
    print("Checking shift:")
    print("x[0][1:10] =", x[0, 1:10].tolist())
    print("y[0][0:9]  =", y[0, 0:9].tolist())

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max_iters,
        eta_min=learning_rate / 10
    )

    print("Initial learning rate:", optimizer.param_groups[0]["lr"])

    dummy_x, dummy_y = get_batch("train")
    dummy_logits, dummy_loss = model(dummy_x, dummy_y)
    optimizer.zero_grad()
    dummy_loss.backward()
    optimizer.step()
    scheduler.step()

    print("Learning rate after one dummy step:", optimizer.param_groups[0]["lr"])

    train_losses = []
    val_losses = []
    steps = []

    best_val_loss = float("inf")
    start_time = time.time()

    for iteration in range(max_iters + 1):
        if iteration % eval_interval == 0:
            losses = estimate_loss(model)

            train_loss = losses["train"]
            val_loss = losses["val"]

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            steps.append(iteration)

            elapsed = time.time() - start_time

            print(
                f"step {iteration}: "
                f"train loss {train_loss:.4f}, "
                f"val loss {val_loss:.4f}, "
                f"time {elapsed:.1f}s"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss

                checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "best_val_loss": best_val_loss
                }

                torch.save(checkpoint, checkpoint_path)
                print("Saved best_model.pt")

        xb, yb = get_batch("train")

        logits, loss = model(xb, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    plt.figure()
    plt.plot(steps, train_losses, label="train loss")
    plt.plot(steps, val_losses, label="val loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.savefig(loss_plot_path)

    print("Training complete.")
    print("Best validation loss:", best_val_loss)
    print("Perplexity:", math.exp(best_val_loss))
    print("Saved loss curve to loss_curve.png")


if __name__ == "__main__":
    main()