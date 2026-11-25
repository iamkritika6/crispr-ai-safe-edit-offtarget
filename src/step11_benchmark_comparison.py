"""
4-CHANNEL sgRNA ENCODING SCHEMA VISUALIZATION
==============================================

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import os

print("="*80)
print("CREATING 4-CHANNEL ENCODING SCHEMA FIGURE")
print("="*80)

# Ultra high-quality settings
plt.rcParams['figure.dpi'] = 400
plt.rcParams['savefig.dpi'] = 400
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['font.size'] = 12

# Create output directory
os.makedirs('figures/methods', exist_ok=True)

# Create figure
fig = plt.figure(figsize=(20, 14))
gs = GridSpec(4, 2, figure=fig, hspace=0.4, wspace=0.3, 
              height_ratios=[1, 1.5, 1.5, 0.8])

# Define vibrant colors for nucleotides
colors = {
    'A': '#E74C3C',  # Red
    'C': '#3498DB',  # Blue
    'G': '#2ECC71',  # Green
    'T': '#F39C12',  # Orange
    'N': '#95A5A6'   # Gray (for any nucleotide)
}

# Example sequence (23bp) - actual realistic sgRNA
example_sequence = "ATGCTAGCTAGCTAGCTAGTGG"  # 20bp + TGG PAM
positions = list(range(1, 24))

# ============================================================================
# PANEL A: EXAMPLE SEQUENCE
# ============================================================================
ax1 = fig.add_subplot(gs[0, :])
ax1.axis('off')
ax1.set_xlim([0, 23])
ax1.set_ylim([0, 2])

# Title
ax1.text(11.5, 1.7, 'A. Example 23bp sgRNA Sequence', 
        fontsize=18, fontweight='bold', ha='center')

# Draw sequence
for i, (pos, nuc) in enumerate(zip(positions, example_sequence)):
    # Box
    rect = mpatches.Rectangle((i, 0.5), 0.9, 0.8, 
                              facecolor=colors[nuc], 
                              edgecolor='black', linewidth=2)
    ax1.add_patch(rect)
    
    # Nucleotide letter
    ax1.text(i + 0.45, 0.9, nuc, 
            ha='center', va='center', 
            fontsize=16, fontweight='bold', color='white')
    
    # Position number
    ax1.text(i + 0.45, 0.2, str(pos), 
            ha='center', va='center', 
            fontsize=10, color='black')

# Region labels
ax1.text(6, 1.5, 'Seed Region (1-12)', 
        ha='center', fontsize=13, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='#6C5CE7', alpha=0.3))
ax1.text(16, 1.5, 'Distal (13-20)', 
        ha='center', fontsize=13, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='#95A5A6', alpha=0.3))
ax1.text(21.5, 1.5, 'PAM', 
        ha='center', fontsize=13, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='#00B894', alpha=0.3))

# ============================================================================
# PANEL B: ONE-HOT ENCODING PROCESS
# ============================================================================
ax2 = fig.add_subplot(gs[1, :])
ax2.set_xlim([0, 24])
ax2.set_ylim([0, 6])
ax2.axis('off')

ax2.text(12, 5.7, 'B. One-Hot Encoding Process', 
        fontsize=18, fontweight='bold', ha='center')

# Show first 8 positions for clarity
show_positions = 8

# Draw encoding for each channel
channel_names = ['A Channel', 'C Channel', 'G Channel', 'T Channel']
y_positions = [4.2, 3.0, 1.8, 0.6]

for ch_idx, (channel, y_pos, color_key) in enumerate(zip(channel_names, y_positions, ['A', 'C', 'G', 'T'])):
    # Channel label
    ax2.text(-0.5, y_pos + 0.4, channel, 
            fontsize=12, fontweight='bold', ha='right',
            color=colors[color_key])
    
    # Draw boxes for each position
    for i in range(show_positions):
        nuc = example_sequence[i]
        value = 1 if nuc == color_key else 0
        
        # Box color
        box_color = colors[color_key] if value == 1 else '#ECEFF1'
        box_alpha = 1.0 if value == 1 else 0.3
        
        # Draw box
        rect = mpatches.Rectangle((i * 2.8, y_pos), 2.5, 0.8,
                                  facecolor=box_color, alpha=box_alpha,
                                  edgecolor='black', linewidth=2)
        ax2.add_patch(rect)
        
        # Value
        text_color = 'white' if value == 1 else 'black'
        ax2.text(i * 2.8 + 1.25, y_pos + 0.4, str(value),
                ha='center', va='center', 
                fontsize=14, fontweight='bold', color=text_color)
        
        # Position label (only for first row)
        if ch_idx == 0:
            ax2.text(i * 2.8 + 1.25, 5.2, f'Pos {i+1}\n({example_sequence[i]})',
                    ha='center', va='bottom', fontsize=10)

# Continuation dots
ax2.text(show_positions * 2.8 + 1, 2.5, '... continues to position 23',
        fontsize=11, style='italic', ha='center')

# ============================================================================
# PANEL C: 4-CHANNEL TENSOR REPRESENTATION
# ============================================================================
ax3 = fig.add_subplot(gs[2, :])

ax3.text(0.5, 0.95, 'C. Resulting 4-Channel Tensor (23 × 4 matrix)', 
        transform=ax3.transAxes, fontsize=18, fontweight='bold', ha='center')

# Create the full encoding matrix
encoding_matrix = np.zeros((23, 4))
nucleotide_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

for pos_idx, nuc in enumerate(example_sequence):
    if nuc in nucleotide_map:
        encoding_matrix[pos_idx, nucleotide_map[nuc]] = 1

# Plot heatmap
im = ax3.imshow(encoding_matrix.T, cmap='RdYlGn', aspect='auto', 
               interpolation='nearest', vmin=0, vmax=1)

# Set ticks
ax3.set_yticks(range(4))
ax3.set_yticklabels(['A Channel', 'C Channel', 'G Channel', 'T Channel'], 
                    fontsize=12, fontweight='bold')
ax3.set_xticks(range(0, 23, 2))
ax3.set_xticklabels(range(1, 24, 2), fontsize=11)
ax3.set_xlabel('Position in Sequence', fontsize=13, fontweight='bold')

# Color the channel labels
for tick_label, color_key in zip(ax3.get_yticklabels(), ['A', 'C', 'G', 'T']):
    tick_label.set_color(colors[color_key])

# Add grid
ax3.set_xticks(np.arange(-0.5, 23, 1), minor=True)
ax3.set_yticks(np.arange(-0.5, 4, 1), minor=True)
ax3.grid(which='minor', color='black', linestyle='-', linewidth=1)

# Add values and highlight active cells
for i in range(23):
    for j in range(4):
        if encoding_matrix[i, j] == 1:
            # Highlight active cells
            ax3.add_patch(mpatches.Rectangle((i-0.45, j-0.45), 0.9, 0.9,
                                            fill=False, edgecolor='gold', 
                                            linewidth=3))
            ax3.text(i, j, '1', ha='center', va='center',
                    color='white', fontsize=10, fontweight='bold')

# Add region annotations
ax3.axvspan(-0.5, 11.5, alpha=0.1, color='purple', zorder=0)
ax3.axvspan(11.5, 19.5, alpha=0.1, color='gray', zorder=0)
ax3.axvspan(19.5, 22.5, alpha=0.15, color='green', zorder=0)

# Colorbar
cbar = plt.colorbar(im, ax=ax3, fraction=0.02, pad=0.02)
cbar.set_label('Encoding Value', fontsize=12, fontweight='bold')
cbar.set_ticks([0, 1])
cbar.set_ticklabels(['0 (Absent)', '1 (Present)'])

# ============================================================================
# PANEL D: MODEL INPUT SUMMARY
# ============================================================================
ax4 = fig.add_subplot(gs[3, :])
ax4.axis('off')
ax4.set_xlim([0, 10])
ax4.set_ylim([0, 3])

ax4.text(5, 2.5, 'D. Model Input Specification', 
        fontsize=18, fontweight='bold', ha='center')

# Input shape box
input_box = mpatches.FancyBboxPatch((1, 0.8), 8, 1.3,
                                    boxstyle="round,pad=0.1",
                                    facecolor='#E8F5E9', 
                                    edgecolor='#2ECC71', linewidth=3)
ax4.add_patch(input_box)

# Input details
ax4.text(5, 1.7, 'Input Tensor Shape: (Batch Size, 23, 4)', 
        ha='center', fontsize=14, fontweight='bold')
ax4.text(5, 1.3, '• 23 positions (sequence length)', 
        ha='center', fontsize=12)
ax4.text(5, 1.0, '• 4 channels (A, C, G, T nucleotides)', 
        ha='center', fontsize=12)

# Example tensor annotation
ax4.text(1.5, 0.3, 'Example: Single sequence tensor = [[0,0,0,1], [1,0,0,0], [0,0,0,1], ...]', 
        fontsize=11, style='italic', color='#555')

# ============================================================================
# LEGEND
# ============================================================================
legend_elements = [
    mpatches.Patch(facecolor=colors['A'], edgecolor='black', label='Adenine (A)'),
    mpatches.Patch(facecolor=colors['C'], edgecolor='black', label='Cytosine (C)'),
    mpatches.Patch(facecolor=colors['G'], edgecolor='black', label='Guanine (G)'),
    mpatches.Patch(facecolor=colors['T'], edgecolor='black', label='Thymine (T)'),
]
ax1.legend(handles=legend_elements, loc='upper left', fontsize=12, 
          framealpha=0.95, bbox_to_anchor=(0, 0.95))

# Main title
fig.suptitle('4-Channel One-Hot Encoding Schema for sgRNA Sequences', 
            fontsize=20, fontweight='bold', y=0.98)

# Save
plt.savefig('figures/methods/fig_encoding_schema.png', 
           dpi=400, bbox_inches='tight', facecolor='white')
print("\n✓ Saved: figures/methods/fig_encoding_schema.png")

plt.close()

print("\n" + "="*80)
print("ENCODING SCHEMA FIGURE COMPLETE")
print("="*80)
print("\nThis figure shows:")
print("  • Example 23bp sgRNA sequence with color-coded nucleotides")
print("  • Step-by-step one-hot encoding process")
print("  • Complete 23×4 tensor representation")
print("  • Model input specification")
print("\nPerfect for Methodology section of your thesis!")
print("="*80)