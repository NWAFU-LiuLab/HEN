import matplotlib.pyplot as plt
import pandas as pd


def plot_trainloss(model_names):
    fig, ax1, = plt.subplots(1, 1, figsize=(8, 5))

    for model_name in model_names:
        data = pd.read_csv(f"results/{model_name}_metrics.txt", header=None,
                           names=['epoch', 'train_loss', 'train_iou', 'train_recall', 'train_f1', 'train_acc',
                                  'val_loss', 'val_iou', 'val_recall', 'val_f1', 'val_acc'])

        # Plotting Loss
        ax1.plot(data['epoch']*2, data['train_loss'], label=f'{model_name} ')
        # ax1.plot(data['epoch'], data['val_loss'], label=f'{model_name} Val Loss')

        # # Plotting IoU
        # ax2.plot(data['epoch'], data['train_iou'], label=f'{model_name} Train IoU')
        # ax2.plot(data['epoch'], data['val_iou'], label=f'{model_name} Val IoU')
        #
        # # Plotting Recall
        # ax3.plot(data['epoch'], data['train_recall'], label=f'{model_name} Train Recall')
        # ax3.plot(data['epoch'], data['val_recall'], label=f'{model_name} Val Recall')
        #
        # # Plotting F1-Score
        # ax4.plot(data['epoch'], data['train_f1'], label=f'{model_name} Train F1')
        # ax4.plot(data['epoch'], data['val_f1'], label=f'{model_name} Val F1')
        #
        # # Plotting Accuracy
        # ax5.plot(data['epoch'], data['train_acc'], label=f'{model_name} Train Accuracy')
        # ax5.plot(data['epoch'], data['val_acc'], label=f'{model_name} Val Accuracy')

    ax1.legend(loc=(0.79, 0.5))
    ax1.set_title('Train Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.tight_layout()

    plt.savefig(f"results/train_loss.png")
    plt.show()
def plot_trainacc(model_names):
    fig, ax, = plt.subplots(1, 1, figsize=(8, 5))

    for model_name in model_names:
        data = pd.read_csv(f"results/{model_name}_metrics.txt", header=None,
                           names=['epoch', 'train_loss', 'train_iou', 'train_recall', 'train_f1', 'train_acc',
                                  'val_loss', 'val_iou', 'val_recall', 'val_f1', 'val_acc'])

        # # Plotting Accuracy
        ax.plot(data['epoch']*2, data['train_acc'], label=f'{model_name}')

    ax.legend()
    ax.set_title('Train Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.tight_layout()

    plt.savefig(f"results/train_acc.png")
    plt.show()
def plot_valacc(model_names):
    fig, ax, = plt.subplots(1, 1, figsize=(8, 5))

    for model_name in model_names:
        data = pd.read_csv(f"results/{model_name}_metrics.txt", header=None,
                           names=['epoch', 'train_loss', 'train_iou', 'train_recall', 'train_f1', 'train_acc',
                                  'val_loss', 'val_iou', 'val_recall', 'val_f1', 'val_acc'])

        # # Plotting Accuracy
        ax.plot(data['epoch']*2, data['val_acc'], label=f'{model_name}')

    ax.legend()
    ax.set_title('Val Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.tight_layout()

    plt.savefig(f"results/val_acc.png")
    plt.show()

# Call the function for each model
model_names = ['DenseNet', 'ResNet', 'UNet++', 'DeepLabV3', 'HEN']
plot_trainloss(model_names)
plot_trainacc(model_names)
plot_valacc(model_names)
