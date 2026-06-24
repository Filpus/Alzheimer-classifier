import torch
import torch.nn as nn
import torch.nn.functional as F

class MambaEncoder(nn.Module):
    def __init__(self, num_channels, sequence_length, d_model=32, d_state=16):
        super().__init__()

        self.patch_size = 128
        self.num_channels = num_channels
        self.embed = _EEGPatchEmbed(num_channels, d_model, patch_size=self.patch_size)

        self.blocks = nn.ModuleList([
            _MambaBlock(d_model, d_state=d_state),
            _MambaBlock(d_model, d_state=d_state),
            #_MambaBlock(d_model, d_state=d_state),
        ])

        self.norm = _RMSNorm(d_model)

        with torch.no_grad():
            dummy = torch.zeros(1, num_channels, sequence_length)
            out = self.forward(dummy)
            self.encoded_size = out.view(1, -1).size(1)

    def forward(self, x):
        x = self.embed(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x  # (B, L, d_model) — zachowane

def _ssm_scan(deltaA, deltaBx):
    """
    Poprawna implementacja SSM (sekwencyjna rekurencja),
    zgodna z Mamba / S6 semantics.

    h_t = A_t ⊙ h_{t-1} + B_t
    """
    bsz, length, d_inner, d_state = deltaA.shape
    h = torch.zeros(
        bsz, d_inner, d_state,
        device=deltaA.device,
        dtype=deltaA.dtype
    )

    ys = []
    for t in range(length):
        h = deltaA[:, t] * h + deltaBx[:, t]
        ys.append(h)

    return torch.stack(ys, dim=1)

class _RMSNorm(nn.Module):
    """RMSNorm uzywany w oryginalnym backbone'ie Mamby (Gu & Dao, 2023, Sec. 3.4).

    Nie odejmuje sredniej (w odroznieniu od LayerNorm) — jeden uczony parametr
    skali, brak biasu. Czysty PyTorch, dziala identycznie na CPU, MPS, CUDA.
    """

    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.weight


class _EEGPatchEmbed(nn.Module):
    def __init__(self, n_channels, d_model, patch_size=32):
        super().__init__()
        self.patch_size = patch_size
        self.patch = nn.Conv1d(
            n_channels, d_model,
            kernel_size=patch_size,
            stride=patch_size
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # normalizacja per-patch przed konwolucją
        B, C, T = x.shape
        # (B, C, T) → (B, C, num_patches, patch_size)
        x = x.unfold(-1, self.patch_size, self.patch_size)
        mean = x.mean(dim=-1, keepdim=True)
        std  = x.std(dim=-1, keepdim=True) + 1e-6
        x = (x - mean) / std
        # (B, C, num_patches, patch_size) → (B, C, T)
        x = x.reshape(B, C, T)

        x = self.patch(x)
        x = x.transpose(1, 2)
        return self.norm(x)

class _SelectiveSSM(nn.Module):
    def __init__(self, d_model, d_state=16, expand=2, d_conv=4):
        super().__init__()

        self.d_inner = expand * d_model

        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)

        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=True
        )

        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + d_state, bias=False)
        self.dt_proj = nn.Linear(d_state, self.d_inner, bias=True)

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        self.d_state = d_state

        A = torch.arange(1, d_state + 1).float().repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))

        self.D = nn.Parameter(torch.ones(self.d_inner))

    def forward(self, x):
        B, L, _ = x.shape

        x = self.in_proj(x)
        x, res = x.chunk(2, dim=-1)

        x = self.conv1d(x.transpose(1, 2))[..., :L].transpose(1, 2)
        x = F.silu(x)

        x_proj = self.x_proj(x)
        dt, Bp, Cp = torch.split(x_proj, [self.d_state, self.d_state, self.d_state], dim=-1)

        dt = F.softplus(self.dt_proj(dt))

        A = -torch.exp(self.A_log)

        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device)

        ys = []
        for t in range(L):
            A_t = torch.exp(dt[:, t].unsqueeze(-1) * A)
            Bu = dt[:, t].unsqueeze(-1) * Bp[:, t].unsqueeze(1)
            h = A_t * h + Bu
            y = torch.einsum("bdn,bn->bd", h, Cp[:, t])
            ys.append(y)

        y = torch.stack(ys, dim=1)

        y = y + x * self.D.unsqueeze(0).unsqueeze(0)
        y = y * F.silu(res)  # ← węzeł × ze schematu
        return self.out_proj(y)

class _MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, expand=2):
        super().__init__()
        self.norm = _RMSNorm(d_model)
        self.ssm = _SelectiveSSM(d_model, d_state=d_state, expand=expand)

    def forward(self, x):
        return x + self.ssm(self.norm(x))


class MambaModel(nn.Module):
    def __init__(self, num_channels, sequence_length, d_model=64, d_state=16, k_steps=5, temperature=0.07):
        super().__init__()

        self.encoder = MambaEncoder(
            num_channels, sequence_length,
            d_model=d_model, d_state=d_state
        )
        self.encoded_size = self.encoder.encoded_size

        self.num_channels = num_channels
        self.patch_size = self.encoder.patch_size
        self.k_steps = k_steps
        self.temperature = temperature
        # Projekcja kontekstu → przestrzeń predykcji (oddzielna od enkodera)
        self.predictors = nn.ModuleList([
            nn.Linear(d_model, d_model, bias=False)
            for _ in range(k_steps)
        ])

    def encode_windows(self, windows):
        z = self.encoder(windows)          # (B, num_patches, d_model)
        return z.view(z.size(0), -1)

    def _pool(self, z):
        """(B, num_patches, d_model) → (B, d_model): mean pooling po czasie."""
        return z[:, -1, :]
        #return z.mean(dim=1)

    def forward(self, seq):
        N, C, T = seq.shape
        z = self._pool(self.encoder(seq))  # (N, d_model)
        z = F.normalize(z, dim=-1)

        total_loss, total_acc, valid = 0.0, 0.0, 0

        for i, k in enumerate(range(1, self.k_steps + 1)):
            if N <= k:
                continue
            ctx = z[:-k]
            tgt = z[k:]
            pred = F.normalize(self.predictors[i](ctx), dim=-1)

            sim = pred @ tgt.T / self.temperature
            labels = torch.arange(N - k, device=seq.device)
            total_loss += F.cross_entropy(sim, labels)

            with torch.no_grad():
                total_acc += (sim.argmax(dim=-1) == labels).float().mean().item()
            valid += 1

        if valid == 0:
            return None, None

        return total_loss / valid, total_acc / valid