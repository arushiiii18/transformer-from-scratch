import matplotlib.pyplot as plt

epochs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
train_loss = [7.11, 5.02, 4.29, 3.76, 3.40, 3.15, 2.97, 2.85, 2.76, 2.65, 2.54, 2.44, 2.35, 2.27, 2.19]
val_loss = [5.41, 4.46, 3.81, 3.44, 3.20, 3.05, 2.98, 2.94, 2.93, 2.88, 2.85, 2.84, 2.84, 2.88, 2.89]

plt.figure(figsize=(10, 5))
plt.plot(epochs, train_loss, 'b-o', label='Train Loss', markersize=5)
plt.plot(epochs, val_loss, 'r-o', label='Val Loss', markersize=5)
plt.axvline(x=12, color='gray', linestyle='--', alpha=0.7, label='Best model (epoch 12)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
print('Saved.')