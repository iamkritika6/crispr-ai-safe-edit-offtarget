"""
STEP 9: PERFORMANCE + FINAL INTERPRETABILITY FIGURES
=====================================================

PERFORMANCE (3 figures):
1. Precision-Recall Curves
2. GA Evolution Plot
3. Confusion Matrices

FINAL INTERPRETABILITY (3 figures):
4. Mismatch Type Analysis
5. Layer-wise Neuron Importance
6. Activation Pattern Analysis


"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, confusion_matrix
from scipy.special import expit
from scipy.stats import gaussian_kde
import torch
import torch.nn as nn
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec
import os
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("STEP 9: PERFORMANCE + FINAL INTERPRETABILITY ANALYSIS")
print("6 Ultra High-Quality Figures (400 DPI)")
print("="*80)

# Ultra high-quality settings
plt.rcParams['figure.dpi'] = 400
plt.rcParams['savefig.dpi'] = 400
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['axes.titlesize'] = 15
plt.rcParams['legend.fontsize'] = 11

os.makedirs('figures/performance', exist_ok=True)
os.makedirs('figures/interpretability_final', exist_ok=True)
os.makedirs('results/interpretability', exist_ok=True)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {device}")

# Model architectures
class ConfigurableGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True,
                         dropout=dropout if num_layers > 1 else 0)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        gru_out, _ = self.gru(x)
        x = self.relu(self.fc1(gru_out[:, -1, :]))
        return self.fc2(self.dropout(x))

class ConfigurableBiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.bilstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                             bidirectional=True, dropout=dropout if num_layers > 1 else 0)
        self.fc1 = nn.Linear(hidden_size * 2, 64)
        self.fc2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        bilstm_out, _ = self.bilstm(x)
        x = self.relu(self.fc1(bilstm_out[:, -1, :]))
        return self.fc2(self.dropout(x))
    
    def forward_with_hidden(self, x):
        """Returns hidden layer activations"""
        bilstm_out, _ = self.bilstm(x)
        hidden1 = self.relu(self.fc1(bilstm_out[:, -1, :]))
        output = self.fc2(self.dropout(hidden1))
        return output, hidden1

class CNNBiLSTM4ch(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, num_filters, kernel_size):
        super().__init__()
        self.conv1 = nn.Conv1d(input_size, num_filters, kernel_size, padding=kernel_size//2)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)
        self.dropout_conv = nn.Dropout(dropout)
        self.lstm = nn.LSTM(num_filters, hidden_size, num_layers, batch_first=True,
                           bidirectional=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size * 2, 1)
    
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.dropout_conv(x)
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

# Load data
print("\nLoading data...")
X_test = np.load('data/X_test.npy')
y_test = np.load('data/y_test.npy')
print(f"Test samples: {len(X_test)}")
print(f"Positive: {sum(y_test)} ({sum(y_test)/len(y_test)*100:.2f}%)")
print(f"Negative: {len(y_test)-sum(y_test)} ({(len(y_test)-sum(y_test))/len(y_test)*100:.2f}%)")
print(f"Class imbalance: {(len(y_test)-sum(y_test))/sum(y_test):.1f}:1")

# Load models
print("\nLoading models...")
configs = {
    'GRU-4ch': {'hidden_size': 256, 'num_layers': 2, 'dropout': 0.25},
    'BiLSTM-4ch': {'hidden_size': 64, 'num_layers': 1, 'dropout': 0.2},
    'CNN-BiLSTM-4ch': {'hidden_size': 192, 'num_layers': 1, 'dropout': 0.18, 
                      'num_filters': 64, 'kernel_size': 3}
}

models = {}
predictions = {}

for model_name, config in configs.items():
    if model_name == 'GRU-4ch':
        model = ConfigurableGRU(4, config['hidden_size'], config['num_layers'], config['dropout'])
    elif model_name == 'BiLSTM-4ch':
        model = ConfigurableBiLSTM(4, config['hidden_size'], config['num_layers'], config['dropout'])
    else:
        model = CNNBiLSTM4ch(4, config['hidden_size'], config['num_layers'], 
                            config['dropout'], config['num_filters'], config['kernel_size'])
    
    model.load_state_dict(torch.load(f'models/{model_name.lower().replace("-", "_")}.pt', 
                                     map_location=device))
    model = model.to(device)
    model.eval()
    models[model_name] = model
    
    # Generate predictions
    with torch.no_grad():
        test_X = torch.FloatTensor(X_test).to(device)
        test_logits = model(test_X).cpu().numpy().flatten()
        test_probs = expit(test_logits)
    predictions[model_name] = test_probs
    
    print(f"  ✓ Loaded: {model_name}, Mean pred: {test_probs.mean():.4f}")

# Load test results
test_results_file = 'results/test_evaluation/final_test_results_optionB.csv'
results_df = pd.read_csv(test_results_file)

print("\n" + "="*80)
print("PART 1: PERFORMANCE FIGURES (3 FIGURES)")
print("="*80)

# =============================================================================
# FIGURE 1: PRECISION-RECALL CURVES
# =============================================================================
print("\n[1/6] Precision-Recall Curves...")

fig, ax = plt.subplots(figsize=(10, 8))

colors = ['#3498db', '#e74c3c', '#2ecc71']
model_names_short = ['GRU', 'BiLSTM', 'CNN-BiLSTM']

for idx, (model_name, probs) in enumerate(predictions.items()):
    precision, recall, _ = precision_recall_curve(y_test, probs)
    auprc = results_df[results_df['Model'] == model_name]['Test_AUPRC'].values[0]
    
    ax.plot(recall, precision, 
            label=f"{model_names_short[idx]} (AUPRC={auprc:.4f})",
            color=colors[idx], linewidth=2.5, alpha=0.85)

# Baselines
baseline = sum(y_test) / len(y_test)
ax.axhline(baseline, color='gray', linestyle='--', linewidth=2,
          label=f'Random (AUPRC={baseline:.4f})', alpha=0.7)

rahman_auprc = 0.7208
ax.axhline(rahman_auprc, color='red', linestyle=':', linewidth=2,
          label=f'Rahman 2023 (AUPRC={rahman_auprc:.4f})', alpha=0.7)

ax.set_xlabel('Recall', fontsize=13, fontweight='bold')
ax.set_ylabel('Precision', fontsize=13, fontweight='bold')
ax.set_title('Precision-Recall Curves - Test Set Performance\n(Imbalanced Data: 232:1)', 
             fontsize=14, fontweight='bold')
ax.legend(loc='best', fontsize=11, framealpha=0.9)
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])

plt.tight_layout()
plt.savefig('figures/performance/fig1_precision_recall_curves.png', dpi=400, bbox_inches='tight')
plt.close()
print("  ✓ Saved: fig1_precision_recall_curves.png")

# =============================================================================
# FIGURE 2: GA EVOLUTION
# =============================================================================
print("[2/6] GA Evolution Plot...")

ga_results_files = {
    'GRU-4ch': 'results/true_ga/GRU-4ch_progress.csv',
    'BiLSTM-4ch': 'results/true_ga/BiLSTM-4ch_progress.csv',
    'CNN-BiLSTM-4ch': 'results/true_ga/CNN-BiLSTM-4ch_progress.csv'
}

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for idx, (model_name, ga_file) in enumerate(ga_results_files.items()):
    ax = axes[idx]
    
    if os.path.exists(ga_file):
        try:
            ga_df = pd.read_csv(ga_file)
            
            # Auto-detect column names
            gen_col = None
            score_col = None
            
            for col in ga_df.columns:
                if 'gen' in col.lower():
                    gen_col = col
                if 'auprc' in col.lower() or 'score' in col.lower() or 'fitness' in col.lower():
                    score_col = col
            
            if gen_col is None or score_col is None:
                raise ValueError(f"Could not find generation or score columns. Columns: {ga_df.columns.tolist()}")
            
            if len(ga_df) > 0:
                gen_stats = ga_df.groupby(gen_col)[score_col].agg(['mean', 'max', 'std']).reset_index()
                gen_stats = gen_stats.rename(columns={gen_col: 'generation'})
                
                ax.plot(gen_stats['generation'], gen_stats['max'], 
                        marker='o', linewidth=2.5, markersize=8,
                        color=colors[idx], label='Best in Generation')
                ax.plot(gen_stats['generation'], gen_stats['mean'], 
                        marker='s', linewidth=2, markersize=6, alpha=0.7,
                        color=colors[idx], linestyle='--', label='Population Average')
                
                ax.fill_between(gen_stats['generation'], 
                                gen_stats['mean'] - gen_stats['std'],
                                gen_stats['mean'] + gen_stats['std'],
                                alpha=0.2, color=colors[idx])
                
                ax.set_xlabel('Generation', fontsize=11, fontweight='bold')
                ax.set_ylabel('Validation AUPRC', fontsize=11, fontweight='bold')
                ax.set_title(f'{model_name.replace("-4ch", "")}', fontsize=12, fontweight='bold')
                ax.legend(fontsize=9)
                ax.grid(True, alpha=0.3)
                ax.set_ylim([0, 1])
                
                improvement = gen_stats['max'].iloc[-1] - gen_stats['max'].iloc[0]
                ax.text(0.05, 0.95, f'Improvement: +{improvement:.4f}',
                        transform=ax.transAxes, fontsize=10,
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                print(f"  {model_name}: Gen 1 = {gen_stats['max'].iloc[0]:.4f}, Final = {gen_stats['max'].iloc[-1]:.4f}")
            else:
                ax.text(0.5, 0.5, f'{model_name}\nGA data empty', 
                        transform=ax.transAxes, ha='center', va='center', fontsize=12)
                ax.set_title(f'{model_name.replace("-4ch", "")}', fontsize=12, fontweight='bold')
        except Exception as e:
            ax.text(0.5, 0.5, f'{model_name}\nError: {str(e)[:30]}', 
                    transform=ax.transAxes, ha='center', va='center', fontsize=10)
            ax.set_title(f'{model_name.replace("-4ch", "")}', fontsize=12, fontweight='bold')
            print(f"  Warning: Error loading {model_name}: {e}")
    else:
        ax.text(0.5, 0.5, f'{model_name}\nFile not found:\n{ga_file}', 
                transform=ax.transAxes, ha='center', va='center', fontsize=10, color='red')
        ax.set_title(f'{model_name.replace("-4ch", "")}', fontsize=12, fontweight='bold')
        print(f"  Warning: File not found: {ga_file}")

plt.suptitle('Genetic Algorithm Evolution - Validation Performance Over Generations', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/performance/fig2_ga_evolution.png', dpi=400, bbox_inches='tight')
plt.close()
print("  ✓ Saved: fig2_ga_evolution.png")

# =============================================================================
# FIGURE 3: CONFUSION MATRICES
# =============================================================================
print("[3/6] Confusion Matrices...")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for idx, (model_name, probs) in enumerate(predictions.items()):
    ax = axes[idx]
    
    preds = (probs > 0.5).astype(int)
    cm = confusion_matrix(y_test, preds)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
               ax=ax, cbar_kws={'label': 'Percentage'},
               xticklabels=['Predicted Neg', 'Predicted Pos'],
               yticklabels=['Actual Neg', 'Actual Pos'])
    
    ax.set_title(f'{model_names_short[idx]}\n(n={len(y_test):,})', 
                 fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=11, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
    
    for i in range(2):
        for j in range(2):
            ax.text(j + 0.5, i + 0.7, f'n={cm[i, j]:,}',
                   ha="center", va="center", color="red", fontsize=9)
    
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    
    metrics_text = f'Sensitivity: {sensitivity:.2%}\nSpecificity: {specificity:.2%}'
    ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

plt.suptitle('Confusion Matrices - Classification Performance (Threshold=0.5)', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/performance/fig3_confusion_matrices.png', dpi=400, bbox_inches='tight')
plt.close()
print("  ✓ Saved: fig3_confusion_matrices.png")

print("\n" + "="*80)
print("PART 2: FINAL INTERPRETABILITY FIGURES (3 FIGURES)")
print("="*80)

# =============================================================================
# FIGURE 4: MISMATCH TYPE ANALYSIS
# =============================================================================
print("\n[4/6] Mismatch Type Analysis...")

nucleotides = ['A', 'C', 'G', 'T']
substitution_matrix = np.zeros((4, 4, 23))  # 4x4 for A/C/G/T, 23 positions

# Sample 100 sequences for analysis
np.random.seed(42)
sample_idx = np.random.choice(len(X_test), 100, replace=False)
X_samples = X_test[sample_idx]

print("\nAnalyzing substitution patterns...")
print("Testing all 16 substitution types at each position...")

model = models['CNN-BiLSTM-4ch']  # Use best model

with torch.no_grad():
    for seq_idx, seq in enumerate(X_samples):
        if (seq_idx + 1) % 20 == 0:
            print(f"  Processed {seq_idx + 1}/100 sequences...")
        
        # Get original prediction
        seq_tensor = torch.FloatTensor(seq).unsqueeze(0).to(device)
        orig_pred = torch.sigmoid(model(seq_tensor)).item()
        
        # Test each position
        for pos in range(23):
            # Find which nucleotide is present
            orig_nuc_idx = np.argmax(seq[pos, :4])
            
            # Try all 4 nucleotides
            for new_nuc_idx in range(4):
                if new_nuc_idx == orig_nuc_idx:
                    continue
                
                # Create mutated sequence
                mutated = seq.copy()
                mutated[pos, :4] = 0
                mutated[pos, new_nuc_idx] = 1
                
                # Get prediction
                mut_tensor = torch.FloatTensor(mutated).unsqueeze(0).to(device)
                mut_pred = torch.sigmoid(model(mut_tensor)).item()
                
                # Record ABSOLUTE impact (use abs to capture any change)
                impact = abs(orig_pred - mut_pred)
                substitution_matrix[orig_nuc_idx, new_nuc_idx, pos] += impact

# Average across positions
substitution_matrix /= 100  # Average over samples

print("\nGenerating mismatch type heatmap...")

fig = plt.figure(figsize=(20, 12))
gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.4)

# Main heatmap: Overall substitution impact
ax_main = fig.add_subplot(gs[:2, :2])

# Average across all positions
overall_impact = substitution_matrix.mean(axis=2)

# Create labels
labels = []
for i, from_nuc in enumerate(nucleotides):
    for j, to_nuc in enumerate(nucleotides):
        if i != j:
            labels.append(f'{from_nuc}→{to_nuc}')

# Flatten matrix (exclude diagonal)
impact_values = []
for i in range(4):
    for j in range(4):
        if i != j:
            impact_values.append(overall_impact[i, j])

# Beautiful heatmap
im = ax_main.imshow(overall_impact, cmap='RdYlGn_r', aspect='auto', 
                    interpolation='gaussian', vmin=-0.05, vmax=0.15)

ax_main.set_xticks(range(4))
ax_main.set_yticks(range(4))
ax_main.set_xticklabels(nucleotides, fontsize=14, fontweight='bold')
ax_main.set_yticklabels(nucleotides, fontsize=14, fontweight='bold')
ax_main.set_xlabel('To Nucleotide', fontsize=15, fontweight='bold')
ax_main.set_ylabel('From Nucleotide', fontsize=15, fontweight='bold')
ax_main.set_title('Average Substitution Impact Across All Positions', 
                 fontsize=16, fontweight='bold', pad=15)

# Add values
for i in range(4):
    for j in range(4):
        if i != j:
            color = 'white' if overall_impact[i, j] > 0.075 else 'black'
            text = ax_main.text(j, i, f'{overall_impact[i, j]:.3f}',
                              ha="center", va="center", color=color,
                              fontsize=13, fontweight='bold')

cbar = plt.colorbar(im, ax=ax_main, fraction=0.046, pad=0.04)
cbar.set_label('Prediction Drop', fontsize=13, fontweight='bold')

# Top substitutions bar chart
ax_bar = fig.add_subplot(gs[:2, 2])

sorted_idx = np.argsort(impact_values)[::-1][:10]
top_subs = [labels[i] for i in sorted_idx]
top_vals = [impact_values[i] for i in sorted_idx]

colors_gradient = plt.cm.RdYlGn_r(np.linspace(0.3, 0.9, 10))
bars = ax_bar.barh(range(10), top_vals, color=colors_gradient, 
                   edgecolor='black', linewidth=2, alpha=0.85)

ax_bar.set_yticks(range(10))
ax_bar.set_yticklabels(top_subs, fontsize=12, fontweight='bold')
ax_bar.set_xlabel('Average Impact', fontsize=13, fontweight='bold')
ax_bar.set_title('Top 10 Damaging\nSubstitutions', fontsize=14, fontweight='bold')
ax_bar.grid(axis='x', alpha=0.3, linestyle='--')
ax_bar.invert_yaxis()

for i, (bar, val) in enumerate(zip(bars, top_vals)):
    ax_bar.text(val + 0.003, i, f'{val:.3f}', va='center', 
               fontsize=11, fontweight='bold')

# Position-specific pattern
ax_pos = fig.add_subplot(gs[2, :])

# Show top 3 substitution types across positions
top3_subs = sorted_idx[:3]
positions = np.arange(1, 24)

for rank, sub_idx in enumerate(top3_subs):
    from_idx = sub_idx // 3
    to_idx = sub_idx % 3 + (1 if sub_idx % 3 >= from_idx else 0)
    
    pos_impacts = substitution_matrix[from_idx, to_idx, :]
    ax_pos.plot(positions, pos_impacts, marker='o', linewidth=3, 
               markersize=7, label=labels[sub_idx], alpha=0.8)

# Add region shading
ax_pos.axvspan(0.5, 12.5, alpha=0.15, color='#6C5CE7', label='Seed', zorder=1)
ax_pos.axvspan(12.5, 20.5, alpha=0.15, color='#95A5A6', label='Distal', zorder=1)
ax_pos.axvspan(20.5, 23.5, alpha=0.25, color='#00B894', label='PAM', zorder=1)

ax_pos.set_xlabel('Position', fontsize=14, fontweight='bold')
ax_pos.set_ylabel('Impact', fontsize=14, fontweight='bold')
ax_pos.set_title('Position-Specific Impact of Top 3 Substitutions', 
                fontsize=15, fontweight='bold')
ax_pos.legend(loc='upper left', fontsize=11, ncol=2, framealpha=0.95)
ax_pos.grid(True, alpha=0.3, linestyle='--')
ax_pos.set_xlim([0, 24])
ax_pos.set_facecolor('#F8F9FA')

plt.suptitle('Mismatch Type Analysis: Which Nucleotide Substitutions Matter Most?', 
            fontsize=18, fontweight='bold', y=0.98)

plt.savefig('figures/interpretability_final/fig4_mismatch_types.png', 
           dpi=400, bbox_inches='tight', facecolor='white')
plt.close()
print("  ✓ Saved: fig4_mismatch_types.png")

# Save data
mismatch_df = pd.DataFrame({
    'Substitution': labels,
    'Average_Impact': impact_values
})
mismatch_df = mismatch_df.sort_values('Average_Impact', ascending=False)
mismatch_df.to_csv('results/interpretability/mismatch_type_analysis.csv', index=False)
print("  ✓ Saved: mismatch_type_analysis.csv")

print(f"\nTop 5 damaging substitutions:")
for i in range(5):
    print(f"  {i+1}. {mismatch_df.iloc[i]['Substitution']}: {mismatch_df.iloc[i]['Average_Impact']:.4f}")

# Continue with Figures 11 and 12...

# =============================================================================
# [5/6] LAYER-WISE NEURON IMPORTANCE
# =============================================================================
print("\n" + "="*80)
print("[5/6] LAYER-WISE NEURON IMPORTANCE")
print("="*80)

print("\nAnalyzing BiLSTM hidden layer neurons...")

# Use BiLSTM model (has accessible hidden layers)
model_bilstm = models['BiLSTM-4ch']

# Get hidden layer activations for test samples
# Ensure we have enough positive samples
pos_indices = np.where(y_test == 1)[0]
neg_indices = np.where(y_test == 0)[0]

# Sample equal numbers
n_pos = min(100, len(pos_indices))
n_neg = min(100, len(neg_indices))

sample_pos = np.random.choice(pos_indices, n_pos, replace=False)
sample_neg = np.random.choice(neg_indices, n_neg, replace=False)
sample_idx = np.concatenate([sample_pos, sample_neg])

X_samples = torch.FloatTensor(X_test[sample_idx]).to(device)
y_samples = y_test[sample_idx]

# Separate positive and negative samples
pos_idx = y_samples == 1
neg_idx = y_samples == 0

hidden_activations_pos = []
hidden_activations_neg = []

print(f"Collecting activations from {len(sample_idx)} samples ({n_pos} pos, {n_neg} neg)...")

with torch.no_grad():
    for i, x in enumerate(X_samples):
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(sample_idx)}...")
        
        x_batch = x.unsqueeze(0)
        _, hidden = model_bilstm.forward_with_hidden(x_batch)
        hidden_np = hidden.cpu().numpy().flatten()
        
        if pos_idx[i]:
            hidden_activations_pos.append(hidden_np)
        else:
            hidden_activations_neg.append(hidden_np)

hidden_pos = np.array(hidden_activations_pos)
hidden_neg = np.array(hidden_activations_neg)

print(f"\nPositive samples: {len(hidden_pos)}")
print(f"Negative samples: {len(hidden_neg)}")
print(f"Hidden layer size: {hidden_pos.shape[1]}")

# Calculate neuron importance (mean activation difference)
neuron_importance = np.abs(hidden_pos.mean(axis=0) - hidden_neg.mean(axis=0))
top_neurons = np.argsort(neuron_importance)[::-1]

print("\nGenerating layer importance visualization...")

fig = plt.figure(figsize=(20, 12))
gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)

# Panel A: Neuron importance ranking
ax1 = fig.add_subplot(gs[0, :])

top_n = 20
top_importance = neuron_importance[top_neurons[:top_n]]
neuron_labels = [f'N{i}' for i in top_neurons[:top_n]]

colors_gradient = plt.cm.viridis(np.linspace(0.2, 0.9, top_n))
bars = ax1.bar(range(top_n), top_importance, color=colors_gradient, 
              edgecolor='black', linewidth=2, alpha=0.85)

ax1.set_xticks(range(top_n))
ax1.set_xticklabels(neuron_labels, fontsize=11, fontweight='bold')
ax1.set_ylabel('Importance Score\n(Pos-Neg Activation Difference)', 
              fontsize=14, fontweight='bold')
ax1.set_title('A. Top 20 Most Important Neurons in Hidden Layer', 
             fontsize=16, fontweight='bold', pad=15)
ax1.grid(axis='y', alpha=0.3, linestyle='--')
ax1.set_facecolor('#F8F9FA')

for i, (bar, val) in enumerate(zip(bars, top_importance)):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.3f}', ha='center', va='bottom',
            fontsize=9, fontweight='bold')

# Panel B: Activation heatmap - Positive samples
ax2 = fig.add_subplot(gs[1, 0])

# Show top 30 neurons, first 50 positive samples
heatmap_data_pos = hidden_pos[:50, top_neurons[:30]].T

im2 = ax2.imshow(heatmap_data_pos, cmap='YlOrRd', aspect='auto', 
                interpolation='gaussian')
ax2.set_ylabel('Neuron Index', fontsize=13, fontweight='bold')
ax2.set_xlabel('Sample Index', fontsize=13, fontweight='bold')
ax2.set_title('B. Activation Heatmap: Positive Samples (Off-Target)', 
             fontsize=14, fontweight='bold', pad=10)
ax2.set_yticks(range(0, 30, 5))
ax2.set_yticklabels([f'N{top_neurons[i]}' for i in range(0, 30, 5)], fontsize=10)

cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
cbar2.set_label('Activation', fontsize=12, fontweight='bold')

# Panel C: Activation heatmap - Negative samples
ax3 = fig.add_subplot(gs[1, 1])

heatmap_data_neg = hidden_neg[:50, top_neurons[:30]].T

im3 = ax3.imshow(heatmap_data_neg, cmap='YlGnBu', aspect='auto', 
                interpolation='gaussian')
ax3.set_ylabel('Neuron Index', fontsize=13, fontweight='bold')
ax3.set_xlabel('Sample Index', fontsize=13, fontweight='bold')
ax3.set_title('C. Activation Heatmap: Negative Samples (On-Target)', 
             fontsize=14, fontweight='bold', pad=10)
ax3.set_yticks(range(0, 30, 5))
ax3.set_yticklabels([f'N{top_neurons[i]}' for i in range(0, 30, 5)], fontsize=10)

cbar3 = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
cbar3.set_label('Activation', fontsize=12, fontweight='bold')

# Panel D: Activation distribution comparison
ax4 = fig.add_subplot(gs[2, :])

# Show distribution for top 5 neurons
for rank in range(5):
    neuron_idx = top_neurons[rank]
    
    # KDE plot
    from scipy.stats import gaussian_kde
    
    pos_activations = hidden_pos[:, neuron_idx]
    neg_activations = hidden_neg[:, neuron_idx]
    
    x_range = np.linspace(
        min(pos_activations.min(), neg_activations.min()),
        max(pos_activations.max(), neg_activations.max()),
        100
    )
    
    kde_pos = gaussian_kde(pos_activations)
    kde_neg = gaussian_kde(neg_activations)
    
    ax4.plot(x_range, kde_pos(x_range), linewidth=3, 
            label=f'N{neuron_idx} (Off-Target)', alpha=0.7,
            color=plt.cm.Reds(0.5 + rank*0.1))
    ax4.plot(x_range, kde_neg(x_range), linewidth=3, linestyle='--',
            label=f'N{neuron_idx} (On-Target)', alpha=0.7,
            color=plt.cm.Blues(0.5 + rank*0.1))

ax4.set_xlabel('Activation Value', fontsize=14, fontweight='bold')
ax4.set_ylabel('Density', fontsize=14, fontweight='bold')
ax4.set_title('D. Activation Distributions: Top 5 Neurons', 
             fontsize=15, fontweight='bold', pad=15)
ax4.legend(loc='upper right', fontsize=10, ncol=2, framealpha=0.95)
ax4.grid(True, alpha=0.3, linestyle='--')
ax4.set_facecolor('#F8F9FA')

plt.suptitle('Layer-wise Neuron Importance Analysis', 
            fontsize=18, fontweight='bold', y=0.98)

plt.savefig('figures/interpretability_final/fig5_layer_importance.png', 
           dpi=400, bbox_inches='tight', facecolor='white')
plt.close()
print("  ✓ Saved: fig5_layer_importance.png")

# Save neuron importance data
neuron_df = pd.DataFrame({
    'Neuron_Index': top_neurons[:30],
    'Importance_Score': neuron_importance[top_neurons[:30]],
    'Mean_Activation_Pos': hidden_pos[:, top_neurons[:30]].mean(axis=0),
    'Mean_Activation_Neg': hidden_neg[:, top_neurons[:30]].mean(axis=0)
})
neuron_df.to_csv('results/interpretability/neuron_importance.csv', index=False)
print("  ✓ Saved: neuron_importance.csv")

print(f"\nTop 5 most important neurons:")
for i in range(5):
    print(f"  {i+1}. Neuron {neuron_df.iloc[i]['Neuron_Index']}: {neuron_df.iloc[i]['Importance_Score']:.4f}")

# =============================================================================
# [6/6] ACTIVATION PATTERN ANALYSIS
# =============================================================================
print("\n" + "="*80)
print("[6/6] ACTIVATION PATTERN ANALYSIS")
print("="*80)

print("\nAnalyzing activation patterns across all neurons...")

fig = plt.figure(figsize=(20, 14))
gs = GridSpec(4, 2, figure=fig, hspace=0.4, wspace=0.3)

# Panel A: Full activation heatmap - All 64 neurons
ax1 = fig.add_subplot(gs[:2, 0])

# Average activation across all samples
avg_activation_pos = hidden_pos.mean(axis=0)
avg_activation_neg = hidden_neg.mean(axis=0)

# Create comparison matrix
activation_matrix = np.vstack([avg_activation_pos, avg_activation_neg])

im1 = ax1.imshow(activation_matrix, cmap='RdYlGn', aspect='auto', 
                interpolation='gaussian')
ax1.set_yticks([0, 1])
ax1.set_yticklabels(['Off-Target\n(Positive)', 'On-Target\n(Negative)'], 
                    fontsize=12, fontweight='bold')
ax1.set_xlabel('Neuron Index', fontsize=13, fontweight='bold')
ax1.set_title('A. Mean Activation Pattern Comparison\n(All 64 Neurons)', 
             fontsize=15, fontweight='bold', pad=15)

cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
cbar1.set_label('Mean Activation', fontsize=12, fontweight='bold')

# Panel B: Activation difference heatmap
ax2 = fig.add_subplot(gs[:2, 1])

activation_diff = avg_activation_pos - avg_activation_neg
diff_matrix = activation_diff.reshape(1, -1)

im2 = ax2.imshow(diff_matrix, cmap='coolwarm', aspect='auto', 
                interpolation='gaussian', vmin=-0.5, vmax=0.5)
ax2.set_yticks([])
ax2.set_xlabel('Neuron Index', fontsize=13, fontweight='bold')
ax2.set_title('B. Activation Difference\n(Off-Target - On-Target)', 
             fontsize=15, fontweight='bold', pad=15)

cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
cbar2.set_label('Activation Difference', fontsize=12, fontweight='bold')

# Highlight strongly differential neurons
strong_pos = np.where(activation_diff > 0.3)[0]
strong_neg = np.where(activation_diff < -0.3)[0]

for idx in strong_pos:
    rect = Rectangle((idx-0.5, -0.5), 1, 1, linewidth=2, 
                     edgecolor='red', facecolor='none')
    ax2.add_patch(rect)

for idx in strong_neg:
    rect = Rectangle((idx-0.5, -0.5), 1, 1, linewidth=2, 
                     edgecolor='blue', facecolor='none')
    ax2.add_patch(rect)

# Panel C: Activation variability
ax3 = fig.add_subplot(gs[2, :])

std_pos = hidden_pos.std(axis=0)
std_neg = hidden_neg.std(axis=0)

x_neurons = np.arange(len(std_pos))
width = 0.4

bars1 = ax3.bar(x_neurons - width/2, std_pos, width, 
               label='Off-Target', color='#E74C3C', alpha=0.7,
               edgecolor='black', linewidth=1)
bars2 = ax3.bar(x_neurons + width/2, std_neg, width,
               label='On-Target', color='#3498DB', alpha=0.7,
               edgecolor='black', linewidth=1)

ax3.set_xlabel('Neuron Index', fontsize=13, fontweight='bold')
ax3.set_ylabel('Activation Std Dev', fontsize=13, fontweight='bold')
ax3.set_title('C. Activation Variability Across Neurons', 
             fontsize=15, fontweight='bold', pad=15)
ax3.legend(fontsize=12, framealpha=0.95)
ax3.grid(axis='y', alpha=0.3, linestyle='--')
ax3.set_facecolor('#F8F9FA')

# Panel D: Activation difference score
ax4 = fig.add_subplot(gs[3, :])

# Calculate normalized difference score for each neuron
# Positive values = neuron activates more for off-target
# Negative values = neuron activates more for on-target
activation_diff_score = (avg_activation_pos - avg_activation_neg) / (avg_activation_pos + avg_activation_neg + 1e-8)

scatter = ax4.scatter(range(len(activation_diff_score)), activation_diff_score, 
                     s=100, c=activation_diff_score, cmap='RdYlGn_r',
                     edgecolors='black', linewidths=1.5, alpha=0.8,
                     vmin=-1, vmax=1)

ax4.axhline(0, color='gray', linestyle='--', linewidth=2, alpha=0.5)
ax4.set_xlabel('Neuron Index', fontsize=13, fontweight='bold')
ax4.set_ylabel('Activation Difference Score', fontsize=13, fontweight='bold')
ax4.set_title('D. Neuron Selectivity: Off-Target vs On-Target Activations', 
             fontsize=15, fontweight='bold', pad=15)
ax4.grid(True, alpha=0.3, linestyle='--')
ax4.set_ylim([-1, 1])
ax4.set_facecolor('#F8F9FA')

cbar4 = plt.colorbar(scatter, ax=ax4, fraction=0.046, pad=0.04)
cbar4.set_label('Selectivity Score', fontsize=12, fontweight='bold')

# Add interpretation box
textstr = (f'Positive score: Neuron prefers off-target\n'
          f'Negative score: Neuron prefers on-target\n'
          f'Near zero: Neuron responds equally to both')
props = dict(boxstyle='round', facecolor='wheat', alpha=0.4)
ax4.text(0.98, 0.02, textstr, transform=ax4.transAxes, fontsize=11,
        verticalalignment='bottom', horizontalalignment='right', bbox=props)

plt.suptitle('Activation Pattern Analysis: How Neurons Respond to Different Classes', 
            fontsize=18, fontweight='bold', y=0.98)

plt.savefig('figures/interpretability_final/fig6_activation_patterns.png', 
           dpi=400, bbox_inches='tight', facecolor='white')
plt.close()
print("  ✓ Saved: fig6_activation_patterns.png")

# Save activation pattern data
activation_df = pd.DataFrame({
    'Neuron_Index': range(len(avg_activation_pos)),
    'Mean_Activation_Pos': avg_activation_pos,
    'Mean_Activation_Neg': avg_activation_neg,
    'Activation_Difference': activation_diff,
    'Std_Pos': std_pos,
    'Std_Neg': std_neg,
    'Selectivity_Score': activation_diff_score
})
activation_df.to_csv('results/interpretability/activation_patterns.csv', index=False)
print("  ✓ Saved: activation_patterns.csv")

print("\n" + "="*80)
print("ALL 3 FINAL ANALYSES COMPLETE!")
print("="*80)
print("\nGenerated files (ultra high-quality, 400 DPI):")
print("  10. fig4_mismatch_types.png")
print("  11. fig5_layer_importance.png")
print("  12. fig6_activation_patterns.png")
print("\nData files:")
print("  - mismatch_type_analysis.csv")
print("  - neuron_importance.csv")
print("  - activation_patterns.csv")
print("\n" + "="*80)
print("INTERPRETABILITY ANALYSIS NOW COMPLETE")
print("="*80)

