import torch
from beast.beast_bspline_tokenizer import BEASTBsplineTokenizer
from beast.beast_bspline_bpe_tokenizer import BEASTBsplineBPETokenizer

torch.manual_seed(0)

batch_size = 32
seq_len = 10
num_dof = 32

base_time = torch.linspace(0, 1, seq_len)
wave = torch.stack([
    torch.sin(2 * torch.pi * (i + 1) * base_time) for i in range(5)
], dim=-1)
lin = torch.stack([(i + 1) * base_time for i in range(5)], dim=-1)
poly = torch.stack([base_time ** (i + 1) for i in range(5)], dim=-1)
noise = torch.randn(seq_len, 12)


def build_sample():
    components = [
        wave * torch.randn(5),
        lin * torch.randn(5),
        poly * torch.randn(5),
        noise * 0.1,
    ]
    traj = torch.cat(components, dim=-1)
    padding = torch.zeros(seq_len, 7)
    final_zero = torch.zeros(seq_len, 10)
    full = torch.cat([traj, padding, final_zero], dim=-1)
    full = full[:, :num_dof]
    full[:, -10:] = 0.0
    return full

trajs_train = [
    {'actions': torch.stack([build_sample() for _ in range(batch_size)], dim=0)}, 
    {'actions': torch.stack([build_sample() for _ in range(batch_size)], dim=0)},
    {'actions': torch.stack([build_sample() for _ in range(batch_size)], dim=0)},
    {'actions': torch.stack([build_sample() for _ in range(batch_size)], dim=0)},
]
trajs = torch.stack([build_sample() for _ in range(batch_size)], dim=0)
print('Trajs shape', trajs.shape)

tokenizer = BEASTBsplineTokenizer(
    num_dof=num_dof,
    num_basis=15,
    seq_len=seq_len,
    vocab_size=500,
    device='cpu',
)
tokenizer.fit_parameters(trajs_train)

tokens, params = tokenizer.encode(trajs)
recon = tokenizer.reconstruct_traj(tokens)

error = torch.mean((recon - trajs.to(recon.device)) ** 2, dim=(1,2))
print('Per-sample MSE:', error)
print('Mean MSE:', error.mean())

# ===================
# BEAST BPE tokenizer
# ===================
beast_tokenizer = BEASTBsplineBPETokenizer.from_beast(
    tokenizer, bpe_vocab_size=1024
)

try:
    beast_tokenizer.encode(trajs, update_bounds=False)
    raise AssertionError("Expected encode to fail before BPE is trained")
except RuntimeError as exc:
    assert "fit_from_trajectories" in str(exc)

fig_state = beast_tokenizer.fit_from_trajectories(trajs_train)

bpe_tokens, bpe_params = beast_tokenizer.encode(trajs, update_bounds=False)
recon_beast = beast_tokenizer.reconstruct_traj(bpe_tokens)
bpe_error = torch.mean((recon_beast - trajs.to(recon_beast.device)) ** 2, dim=(1, 2))
print('BEAST BPE Mean MSE:', bpe_error.mean())
