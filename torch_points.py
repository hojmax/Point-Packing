import torch
import matplotlib.pyplot as plt
import os
import numpy as np
import math
import wandb
import matplotlib

# matplotlib.use("Agg")  # Switch to non-interactive backend

W = 800
H = 800
N = 1000
ALPHA = 250


def plot_points(p_x, p_y, loss=None, save=False, show=False, params=None):
    plt.figure(figsize=(8, 8))
    plt.scatter(p_x.cpu().detach(), p_y.cpu().detach(), alpha=1, s=7)
    plt.xlabel("X")
    plt.ylabel("Y")

    title = f"Point Packing (N={len(p_x)})"
    if loss is not None:
        title += f"\nLoss: {loss:.6f}"
    if params:
        param_text = f"\nIterations: {params['iterations']}, Min LR: {params['min_lr']:.2f}, Peak LR: {params['peak_lr']:.1f}"
        title += param_text

    plt.title(title)
    plt.xlim(0, W)
    plt.ylim(0, H)

    if save:
        os.makedirs("imgs", exist_ok=True)
        filename = f"sweep_optimized_points_{loss:.4f}.png"
        plt.savefig(os.path.join("imgs", filename))
        plt.close()

    if show:
        plt.show()


def get_loss(x, y, alpha, w, h):
    positions = torch.stack([x, y], dim=1)  # Shape (N, 2)
    distances = torch.pdist(positions)
    reciprocal_sum = torch.sum(1.0 / distances)
    border = torch.sum(1 / x + 1 / y + 1 / (w - x) + 1 / (h - y))
    loss = alpha * border + reciprocal_sum
    return loss


def plot_losses(losses, show_plateau=False):
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.title("Loss Over Time")
    if show_plateau:
        plt.ylim(
            min(losses) - 2, min(losses) + 2
        )  # Adjust y-axis to focus on the flat part
    plt.show()


def plot_lr_schedule(lrs: list[float]):
    plt.figure(figsize=(10, 5))
    plt.plot(lrs)
    plt.xlabel("Iteration")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Schedule")
    plt.show()


def get_lrs(iterations: int, peak_lr: float, min_lr: float):
    lrs = []
    for i in range(iterations):
        progress = i / (iterations - 1)
        lr = min_lr + (peak_lr - min_lr) * math.sin(math.pi * progress)
        lrs.append(lr)

    return lrs


def optimize_points(x, y, alpha, w, h, lrs: list[float], iterations: int):
    optimizer = torch.optim.SGD([x, y])
    losses = []

    for i in range(iterations):
        for param_group in optimizer.param_groups:
            param_group["lr"] = lrs[i]

        optimizer.zero_grad()
        loss = get_loss(x, y, alpha, w, h)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            x.clamp_(1, w - 1)
            y.clamp_(1, h - 1)

    return losses


def get_points(n, w, h, device):
    x = (
        1 + torch.rand(n, dtype=torch.float32, device=device) * (w - 1)
    ).requires_grad_()
    y = (
        1 + torch.rand(n, dtype=torch.float32, device=device) * (h - 1)
    ).requires_grad_()
    return x, y


def save_points(x, y, filename="optimized_points.npz"):
    os.makedirs("points", exist_ok=True)
    points = torch.stack([x.cpu().detach(), y.cpu().detach()], dim=1).numpy()
    np.savez(os.path.join("points", filename), points=points)


def main(
    iterations: int = 100,
    min_lr: float = 0.5,
    peak_lr: float = 900,
    device: torch.device = torch.device("cpu"),
):
    losses = []
    best_loss = float("inf")

    # Initialize with first run
    x, y = get_points(N, W, H, device)
    lrs = get_lrs(iterations, peak_lr, min_lr)
    run_losses = optimize_points(x, y, ALPHA, W, H, lrs, iterations)
    best_loss = run_losses[-1]
    best_x = x.clone()
    best_y = y.clone()
    losses.append(best_loss)

    # Run remaining iterations
    for _ in range(9):  # 9 more times for total of 10
        x, y = get_points(N, W, H, device)
        lrs = get_lrs(iterations, peak_lr, min_lr)
        run_losses = optimize_points(x, y, ALPHA, W, H, lrs, iterations)
        final_loss = run_losses[-1]
        losses.append(final_loss)

        if final_loss < best_loss:
            best_loss = final_loss
            best_x = x.clone()
            best_y = y.clone()

    avg_loss = sum(losses) / len(losses)

    # Save the best configuration
    if best_x is not None and best_y is not None:
        params = {"iterations": iterations, "min_lr": min_lr, "peak_lr": peak_lr}
        save_points(best_x, best_y, f"sweep_optimized_points_{best_loss:.4f}.npz")
        plot_points(best_x, best_y, best_loss, save=True, show=False, params=params)
    else:
        print("Warning: No valid points found to save")

    return avg_loss


def sweep():
    matplotlib.use("Agg")  # Set non-interactive backend only for sweeps
    sweep_config = {
        "method": "random",
        "metric": {"name": "avg_loss", "goal": "minimize"},
        "parameters": {
            "iterations": {"min": 50, "max": 500},
            "min_lr": {"min": 0.1, "max": 6.0},
            "peak_lr": {"min": 100, "max": 1200},
        },
    }

    sweep_id = wandb.sweep(sweep_config, project="point_packing")

    def sweep_train():
        wandb.init()

        avg_loss = main(
            iterations=int(wandb.config.iterations),
            min_lr=wandb.config.min_lr,
            peak_lr=wandb.config.peak_lr,
        )

        wandb.log({"avg_loss": avg_loss})

    wandb.agent(sweep_id, sweep_train, count=200)


def load_points(path: str):
    points = np.load(path)
    points_array = points["points"]
    p_x = torch.from_numpy(points_array[:, 0]).requires_grad_()
    p_y = torch.from_numpy(points_array[:, 1]).requires_grad_()
    return p_x, p_y


def train_further(path: str):
    # sweep()
    p_x, p_y = load_points(path)
    print("Initial loss: ", get_loss(p_x, p_y, ALPHA, W, H))

    # Use plot_points function instead of manual plotting
    plot_points(p_x, p_y, show=True)
    iterations = 100
    lr = 0.1
    lrs = [lr] * iterations
    losses = optimize_points(p_x, p_y, ALPHA, W, H, lrs, iterations)
    plot_losses(losses)
    plot_lr_schedule(lrs)
    print(f"Final loss: {losses[-1]}")


if __name__ == "__main__":
    train_further("points/optimized_points_5612.9111.npz")
