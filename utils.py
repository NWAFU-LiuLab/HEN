import os
import torch
import numpy as np
import matplotlib.pyplot as plt

def save_model(model, save_path, epoch):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(model.state_dict(), os.path.join(save_path, f"model_epoch_{epoch}.pth"))

def visualize_prediction(image, mask, prediction):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(image)
    axs[0].set_title("Original Image")
    axs[1].imshow(mask, cmap="gray")
    axs[1].set_title("Ground Truth Mask")
    axs[2].imshow(prediction, cmap="gray")
    axs[2].set_title("Predicted Mask")
    plt.show()
