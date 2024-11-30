import torch
import matplotlib.pyplot as plt


def plot_points(p_x, p_y):
    plt.figure(figsize=(8, 8))
    plt.scatter(p_x.cpu().detach(), p_y.cpu().detach(), alpha=0.5)
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
    plt.title("Loss Over Time")
    plt.show()


def optimize_points(x, y, alpha, w, h):
    optimizer = torch.optim.Adam([x, y], lr=1)
    losses = []
    for _ in range(10000):
        optimizer.zero_grad()
        loss = get_loss(x, y, alpha, w, h)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()

        epsilon = 1e-1
        with torch.no_grad():
            x.clamp_(epsilon, w - epsilon)
            y.clamp_(epsilon, h - epsilon)

    plot_loss(losses)


def get_points(n, w, h, device):
    x = (torch.rand(n, dtype=torch.float32, device=device) * w).requires_grad_()
    y = (torch.rand(n, dtype=torch.float32, device=device) * h).requires_grad_()
    return x, y


if __name__ == "__main__":
    w = 600
    h = 600
    n = 20
    alpha = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    x, y = get_points(n, w, h, device)
    plot_points(x, y)

    optimize_points(x, y, alpha, w, h)

    plot_points(x, y)
