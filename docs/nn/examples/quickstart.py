"""sweep-nn demo: a SIREN fitted directly to a synthetic velocity model.

A toy fit with no wave equation: it shows the API and the optimization loop a
real implicit-FWI run uses (the two notebooks next to this file do the real
thing). Runs on a GPU when one is available, otherwise on the CPU in under
twenty seconds.
"""

import torch

from sweep_nn.siren import SIREN


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)                      # the same initial network on every run
    print("device:", device)

    nz, nx = 64, 128
    # synthetic target velocity model: linear gradient with a faster lens
    z, x = torch.meshgrid(
        torch.linspace(0, 1, nz, device=device), torch.linspace(0, 1, nx, device=device),
        indexing="ij",
    )
    vp_true = 1500 + 3000 * z + 500 * torch.exp(-((x - 0.5) ** 2 + (z - 0.5) ** 2) * 30)

    net = SIREN(out_shape=(nz, nx), hidden_features=128, hidden_layers=4,
                vp_min=1500.0, vp_max=5000.0).to(device)
    optim = torch.optim.Adam(net.parameters(), lr=1e-4)   # 1e-3 makes the loss jump

    for it in range(1000):
        vp = net()
        loss = (vp - vp_true).pow(2).mean()
        optim.zero_grad()
        loss.backward()
        optim.step()
        if it % 200 == 0:
            print(f"iter {it:4d}: loss={loss.item():.3e}")

    with torch.no_grad():
        final = net()
        rel = (final - vp_true).norm() / vp_true.norm()
    print(f"final relative L2 error: {rel.item():.3e}")


if __name__ == "__main__":
    main()
