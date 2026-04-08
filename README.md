# ACE-Step 1.5 (Fork)

For the original project documentation, see [OLD_README.md](OLD_README.md).

## Noise Injection (Disabled by Default)
TODO: explain effects of noise with examples
Added an optional noise injection step after the output norm + projection in the DiT decoder. When enabled (`ADD_NOISE = True`), it adds std-scaled random noise to the hidden states. This is disabled by default.

Modified files:
- `acestep/models/base/modeling_acestep_v15_base.py` — PyTorch implementation
- `acestep/models/mlx/dit_model.py` — MLX (Apple Silicon) implementation
