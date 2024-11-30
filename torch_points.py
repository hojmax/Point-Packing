import torch
import matplotlib.pyplot as plt


def plot_points(p_x, p_y):
    plt.figure(figsize=(8, 8))
    plt.scatter(p_x.detach(), p_y.detach(), alpha=0.5)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Random Points")
    plt.xlim(0, w)
    plt.ylim(0, h)
    plt.show()


def get_squared_differences(x: torch.Tensor) -> torch.Tensor:
    differences = x.unsqueeze(0) - x.unsqueeze(1)
    upper_triangular = torch.triu(differences, diagonal=1)
    return upper_triangular


def get_loss(x, y, alpha, w, h):
    delta_x = get_squared_differences(x)
    delta_y = get_squared_differences(y)
    distances = torch.sqrt(torch.square(delta_x) + torch.square(delta_y))
    reciprocal_sum = torch.sum(1.0 / distances[distances > 0])
    border = torch.sum(1 / x + 1 / y + 1 / (w - x) + 1 / (h - y))
    loss = alpha * border + reciprocal_sum
    return loss


def optimize_points(x, y, alpha, w, h):
    optimizer = torch.optim.Adam([x, y], lr=1.0)
    for _ in range(2000):
        optimizer.zero_grad()
        loss = get_loss(x, y, alpha, w, h)
        loss.backward()
        optimizer.step()

        # Clamp values to stay within bounds
        with torch.no_grad():
            x.clamp_(1, w - 1)
            y.clamp_(1, h - 1)


def get_points(n, w, h):
    x = (torch.rand(n, dtype=torch.float32) * w).requires_grad_()
    y = (torch.rand(n, dtype=torch.float32) * h).requires_grad_()
    return x, y


if __name__ == "__main__":
    w = 600
    h = 600
    n = 20
    alpha = 1
    x, y = get_points(n, w, h)
    plot_points(x, y)

    optimize_points(x, y, alpha, w, h)

    plot_points(x, y)
