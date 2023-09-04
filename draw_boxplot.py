import matplotlib.pyplot as plt
import numpy as np

fold_data = {
    'ResNet': {
        'ACC': [],
        'IoU': [],
        'Dice Score': []
    },
    'DenseNet': {
        'ACC': [],
        'IoU': [],
        'Dice Score': []
    },
    'UNet++': {
        'ACC': [],
        'IoU': [],
        'Dice Score': []
    },
    'DeepLabv3': {
        'ACC': [],
        'IoU': [],
        'Dice Score': []
    },
    'HEN': {
        'ACC': [],
        'IoU': [],
        'Dice Score': []
    }
}



metrics = ["ACC", "IoU", "Dice Score"]
for i, metric in enumerate(metrics):
    plt.figure(figsize=(10, 8))
    plt.boxplot([fold_data[model][metric] for model in fold_data], labels=fold_data.keys())
    plt.ylabel(metric)
    plt.title(f'{metric} for Each Model')
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.show()