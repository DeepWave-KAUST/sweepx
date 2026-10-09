"""sweep-nn demo: SIREN-reparameterized vp fitted to a synthetic target.

This is a toy fit (no wave equation) — it shows the API surface and the
typical optimization loop you'd use in a real FWI run.
"""

import torch

from sweep_nn.siren import SIREN


def main() -> None:
    nz, nx = 64, 128
    # synthetic target velocity model: linear gradient with a faster lens
    z, x = torch.meshgrid(
        torch.linspace(0, 1, nz), torch.linspace(0, 1, nx), indexing="ij"
    )
    vp_true = 1500 + 3000 * z + 500 * torch.exp(-((x - 0.5) ** 2 + (z - 0.5) ** 2) * 30)

    net = SIREN(out_shape=(nz, nx), hidden_features=128, hidden_layers=4,
                vp_min=1500.0, vp_max=5000.0)
    optim = torch.optim.Adam(net.parameters(), lr=1e-3)

    for it in range(200):
        vp = net()
        loss = (vp - vp_true).pow(2).mean()
        optim.zero_grad()
        loss.backward()
        optim.step()
        if it % 50 == 0:
            print(f"iter {it:4d}: loss={loss.item():.3e}")

    with torch.no_grad():
        final = net()
        rel = (final - vp_true).norm() / vp_true.norm()
    print(f"final relative L2 error: {rel.item():.3e}")


if __name__ == "__main__":
    main()
