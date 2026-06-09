"""Generate circular vs linear positional encoding comparison figure."""
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

GENOME_LENGTH = 16569
HIDDEN_SIZE = 64
N_POSITIONS = 200  # subsample for clarity

positions = np.linspace(0, GENOME_LENGTH - 1, N_POSITIONS, dtype=int)

# Linear PE
def linear_pe(pos, dim):
    pe = np.zeros(dim)
    for i in range(0, dim, 2):
        pe[i] = np.sin(pos / (10000 ** (i / dim)))
        if i + 1 < dim:
            pe[i + 1] = np.cos(pos / (10000 ** (i / dim)))
    return pe

# Circular PE
def circular_pe(pos, dim, genome_length):
    pe = np.zeros(dim)
    angle = 2 * math.pi * pos / genome_length
    for i in range(0, dim, 2):
        div = math.exp(-math.log(10000.0) * i / dim)
        pe[i] = np.sin(angle * div)
        if i + 1 < dim:
            pe[i + 1] = np.cos(angle * div)
    return pe

# Build PE matrices
linear_mat = np.array([linear_pe(p, HIDDEN_SIZE) for p in positions])
circular_mat = np.array([circular_pe(p, HIDDEN_SIZE, GENOME_LENGTH) for p in positions])

# Cosine similarity matrices
def cosine_sim_matrix(mat):
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    normed = mat / (norms + 1e-8)
    return normed @ normed.T

linear_sim = cosine_sim_matrix(linear_mat)
circular_sim = cosine_sim_matrix(circular_mat)

# Tick labels at key positions
key_positions = [0, 576, 4000, 8000, 12000, 16024, 16568]
tick_indices = [np.argmin(np.abs(positions - p)) for p in key_positions]
tick_labels = ['1', '576\n(D-loop\nstart)', '4k', '8k', '12k', '16024\n(D-loop\nend)', '16569']

fig = plt.figure(figsize=(14, 5.5))
gs = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[1, 1, 0.05], wspace=0.3)

ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])
cax = fig.add_subplot(gs[2])

vmin, vmax = -1, 1
im1 = ax1.imshow(linear_sim, aspect='auto', cmap='RdBu_r', vmin=vmin, vmax=vmax)
ax1.set_title('Linear PE\n(DNABERT2, HyenaDNA, NT)', fontsize=11, pad=10)
ax1.set_xticks(tick_indices)
ax1.set_xticklabels(tick_labels, fontsize=6.5)
ax1.set_yticks(tick_indices)
ax1.set_yticklabels(tick_labels, fontsize=6.5)
ax1.set_xlabel('Genome position', fontsize=9)
ax1.set_ylabel('Genome position', fontsize=9)

# Highlight the junction corner in linear (shows it as blue = dissimilar)
corner_idx = len(positions) - 1
ax1.add_patch(plt.Rectangle((0 - 0.5, corner_idx - 0.5), 6, 6, fill=False, edgecolor='red', linewidth=2, label='Junction (pos 1 vs 16,569)'))
ax1.add_patch(plt.Rectangle((corner_idx - 0.5, 0 - 0.5), 6, 6, fill=False, edgecolor='red', linewidth=2))

im2 = ax2.imshow(circular_sim, aspect='auto', cmap='RdBu_r', vmin=vmin, vmax=vmax)
ax2.set_title('Circular PE\n(mtDNA-FM)', fontsize=11, pad=10)
ax2.set_xticks(tick_indices)
ax2.set_xticklabels(tick_labels, fontsize=6.5)
ax2.set_yticks(tick_indices)
ax2.set_yticklabels(tick_labels, fontsize=6.5)
ax2.set_xlabel('Genome position', fontsize=9)

# Highlight junction corner in circular (shows it as red = similar)
ax2.add_patch(plt.Rectangle((0 - 0.5, corner_idx - 0.5), 6, 6, fill=False, edgecolor='red', linewidth=2))
ax2.add_patch(plt.Rectangle((corner_idx - 0.5, 0 - 0.5), 6, 6, fill=False, edgecolor='red', linewidth=2))

plt.colorbar(im2, cax=cax, label='Cosine similarity')

fig.suptitle(
    'Positional encoding similarity: position 1 and position 16,569 share a phosphodiester bond\n'
    'Linear PE (left): they appear maximally dissimilar (blue corner).   '
    'Circular PE (right): they appear similar (red corner).',
    fontsize=9, y=1.02
)

plt.savefig(
    '/home/user/Documents/Personal/ai_lab/mtdna_foundation_model/docs/figures/pe_comparison.png',
    dpi=300, bbox_inches='tight', facecolor='white'
)
print("Saved: docs/figures/pe_comparison.png")
