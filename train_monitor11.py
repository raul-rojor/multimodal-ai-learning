"""
TRAINING MONITORING UTILITIES

Helpers to monitor training progress.
"""

import matplotlib.pyplot as plt
import json
import os

class TrainingMonitor:
    """Track and visualize training metrics"""
    
    def __init__(self, log_dir='./logs/'):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': []
        }
    
    def update(self, train_loss, val_loss, lr):
        """Update metrics"""
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        self.history['learning_rates'].append(lr)
    
    def save(self, filename='training_history.json'):
        """Save history to JSON"""
        path = os.path.join(self.log_dir, filename)
        with open(path, 'w') as f:
            json.dump(self.history, f)
        print(f"History saved to {path}")
    
    def load(self, filename='training_history.json'):
        """Load history from JSON"""
        path = os.path.join(self.log_dir, filename)
        if os.path.exists(path):
            with open(path, 'r') as f:
                self.history = json.load(f)
            print(f"History loaded from {path}")
        else:
            print(f"No history found at {path}")
    
    def plot(self, save=True):
        """Plot training curves"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss plot
        axes[0].plot(self.history['train_loss'], label='Train Loss')
        axes[0].plot(self.history['val_loss'], label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Learning rate plot
        axes[1].plot(self.history['learning_rates'])
        axes[1].set_xlabel('Step')
        axes[1].set_ylabel('Learning Rate')
        axes[1].set_title('Learning Rate Schedule')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(os.path.join(self.log_dir, 'training_curves.png'))
            print(f"Plot saved to {self.log_dir}/training_curves.png")
        
        plt.show()

# Example usage
if __name__ == "__main__":
    monitor = TrainingMonitor()
    
    # Simulate training
    for epoch in range(10):
        train_loss = 1.0 / (epoch + 1) + 0.1
        val_loss = 1.2 / (epoch + 1) + 0.1
        lr = 0.001 * (0.9 ** epoch)
        monitor.update(train_loss, val_loss, lr)
    
    monitor.save()
    monitor.plot()