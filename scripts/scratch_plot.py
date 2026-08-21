import re
import matplotlib.pyplot as plt
import seaborn as sns

log_file = '/Users/pranav/.gemini/antigravity-ide/brain/d09ca91a-ce82-4bfb-bdbd-706c130d25dc/.system_generated/tasks/task-30.log'
out_file = '/Users/pranav/.gemini/antigravity-ide/brain/d09ca91a-ce82-4bfb-bdbd-706c130d25dc/accuracy_plot.png'

train_acc = []
val_acc = []

with open(log_file, 'r') as f:
    for line in f:
        if 'val_accuracy:' in line:
            # Example line part: - accuracy: 0.8125 - loss: 0.4873 - val_accuracy: 0.8135 - val_loss: 0.4722
            m1 = re.search(r'- accuracy:\s+([0-9.]+)', line)
            m2 = re.search(r'- val_accuracy:\s+([0-9.]+)', line)
            if m1 and m2:
                train_acc.append(float(m1.group(1)) * 100)
                val_acc.append(float(m2.group(1)) * 100)

epochs = list(range(1, len(train_acc) + 1))

sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 6))

plt.plot(epochs, train_acc, label='Training Accuracy', color='steelblue', marker='o', markersize=4, linewidth=2)
plt.plot(epochs, val_acc, label='Validation Accuracy', color='darkorange', marker='s', markersize=4, linewidth=2)

# Highlight max validation accuracy
max_val = max(val_acc)
max_idx = val_acc.index(max_val)
plt.scatter([epochs[max_idx]], [max_val], color='red', s=100, zorder=5, label=f'Best Val Acc ({max_val:.2f}%)')

plt.title('Static Landmark MLP: Training vs Validation Accuracy', fontsize=14, pad=15)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.legend(fontsize=11, loc='lower right')
plt.grid(True, linestyle='--', alpha=0.7)

# Adjust axes limits for better viewing
plt.xlim(0, max(epochs) + 1)
plt.ylim(min(min(train_acc), min(val_acc)) - 2, max(max(train_acc), max(val_acc)) + 2)

plt.tight_layout()
plt.savefig(out_file, dpi=300)
print(f"Saved plot to {out_file}")
