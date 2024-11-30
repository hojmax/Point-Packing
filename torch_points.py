import torch
import matplotlib.pyplot as plt
import os
import numpy as np


def plot_points(p_x, p_y):
    plt.figure(figsize=(8, 8))
    plt.scatter(p_x.cpu().detach(), p_y.cpu().detach(), alpha=1, s=7)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Random Points")
    plt.xlim(0, w)
    plt.ylim(0, h)
    plt.show()


def get_loss(x, y, alpha, w, h):
    positions = torch.stack([x, y], dim=1)  # Shape (N, 2)
    distances = torch.pdist(positions)
    reciprocal_sum = torch.sum(1.0 / distances)
    border = torch.sum(1 / x + 1 / y + 1 / (w - x) + 1 / (h - y))
    loss = alpha * border + reciprocal_sum
    return loss


def plot_loss(losses):
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.title("Loss Over Time")
    plt.show()


def optimize_points(x, y, alpha, w, h, iterations=1000):
    optimizer = torch.optim.SGD([x, y], lr=0.1)
    losses = []
    for _ in range(iterations):
        optimizer.zero_grad()
        loss = get_loss(x, y, alpha, w, h)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            x.clamp_(1, w - 1)
            y.clamp_(1, h - 1)

    plot_loss(losses)
    print(f"Initial loss: {losses[0]}")
    print(f"Final loss: {losses[-1]}")


def get_points(n, w, h, device):
    x = (
        1 + torch.rand(n, dtype=torch.float32, device=device) * (w - 1)
    ).requires_grad_()
    y = (
        1 + torch.rand(n, dtype=torch.float32, device=device) * (h - 1)
    ).requires_grad_()
    return x, y


def save_points(x, y, filename="optimized_points.npz"):
    points = torch.stack([x.cpu().detach(), y.cpu().detach()], dim=1).numpy()
    np.savez(filename, points=points)


def load_points(filename="optimized_points.npz", device=None):
    if os.path.exists(filename):
        data = np.load(filename)
        points = torch.from_numpy(data["points"]).to(device)
        return points[:, 0].requires_grad_(), points[:, 1].requires_grad_()
    return None


if __name__ == "__main__":
    w = 800
    h = 800
    n = 1000
    alpha = 250
    iterations = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    loaded_points = load_points(device=device)

    if loaded_points is not None:
        x, y = loaded_points
        print("Loaded optimized points from file")
    else:
        x, y = get_points(n, w, h, device)
        print("Generated new random points")

    plot_points(x, y)

    optimize_points(x, y, alpha, w, h, iterations)
    save_points(x, y)

    plot_points(x, y)
