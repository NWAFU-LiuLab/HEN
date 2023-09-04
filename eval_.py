import os.path

import cv2
import torch
from config import config
from dataset import FloodDataset
from sklearn.metrics import accuracy_score, auc, roc_curve, recall_score, f1_score
import numpy as np
from model_unet import UNetPlusPlus
from torch.utils.data import DataLoader
from model_HEN import HEN
import matplotlib.pyplot as plt
from model_densenet import DenseNet
from model_resnet import ResNet
from model_deeplabv import DeepLav
def iou_metric(predicted, target):
    predicted = predicted > 0
    target = target > 0
    intersection = (predicted & target).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (predicted | target).float().sum((1, 2))         # Will be zero if both are 0
    iou = (intersection + 1e-6) / (union + 1e-6)            # We smooth our devision to avoid 0/0
    return iou.mean().item()

def dice_score(predicted, target):
    predicted = predicted > 0
    target = target > 0
    intersection = (predicted & target).float().sum()
    dice = (2. * intersection + 1e-6) / (predicted.float().sum() + target.float().sum() + 1e-6)
    return dice


def evaluate(model, dataloader, criterion, model_name, model_path):
    model.eval()
    total_loss = 0
    iou_sum = 0
    dice_sum = 0
    recall_sum = 0
    f1_sum = 0
    predictions = []
    targets = []

    id = 0

    with torch.no_grad():
        for images, masks in dataloader:
            images, masks = images.cuda(), masks.cuda()
            outputs = model(images)

            mask_out = outputs[0,:].detach().cpu().permute(1, 2, 0).numpy()
            _, mask_out = cv2.threshold(mask_out, 0.1, 1, cv2.THRESH_BINARY)


            save_path = r'results/contra/{}/resnet'.format(model_name)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            cv2.imwrite('{}/{}.jpg'.format(save_path, id), mask_out.astype(np.uint8)*255)
            id += 1

            outputs = outputs.squeeze(1)
            loss = criterion(outputs, masks)
            total_loss += loss.item()
            iou_sum += iou_metric(outputs, masks)
            dice_sum += dice_score(outputs, masks)

            predictions.extend(torch.sigmoid(outputs).cpu().numpy())
            targets.extend(masks.cpu().numpy())

    predictions = np.array(predictions)
    targets = np.array(targets)

    predictions_binary = predictions > 0.5
    targets_binary = targets > 0.5

    accuracy = accuracy_score(targets_binary.flatten(), predictions_binary.flatten())
    recall = recall_score(targets_binary.flatten(), predictions_binary.flatten())
    f1 = f1_score(targets_binary.flatten(), predictions_binary.flatten())

    fpr, tpr, _ = roc_curve(targets_binary.flatten(), predictions.flatten())
    roc_auc = auc(fpr, tpr)

    return total_loss / len(dataloader), iou_sum / len(dataloader), dice_sum / len(dataloader), accuracy, recall, f1, fpr, tpr, roc_auc


def main():
    val_dataset = FloodDataset(config.LABELED_DATA_PATH, None, split="test")
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    models = [ResNet(1), DenseNet(1), DeepLav(1), UNetPlusPlus(1), HEN(1)]
    # model_names = ['ResNet', 'DenseNet', 'DeepLav', 'UNetPlusPlus', 'HEN']
    model_names = ['ResNet']
    model_paths = [
        'results/contra/ResNet/best_ResNet.pth',
        'results/contra/DenseNet/best_DenseNet.pth',
        'results/contra/DeepLav/best_DeepLav.pth',
        'results/contra/UNetPlusPlus/best_UNetPlusPlus.pth',
        'results/ablation/HEN/best_HEN.pth',
    ]
    criterion = torch.nn.BCEWithLogitsLoss()

    # plt.figure()

    for model, model_name, model_path in zip(models, model_names, model_paths):
        model = model.cuda()
        model.load_state_dict(torch.load(model_path))

        eval_loss, eval_iou, eval_dice, eval_accuracy, eval_recall, eval_f1, fpr, tpr, roc_auc = evaluate(model, val_dataloader, criterion, model_name, model_path)

        # print(f"{model_name} Evaluation Loss: {eval_loss}")
        print(f"{model_name} Evaluation IoU: {eval_iou*100:<05.2f}")
        # print(f"{model_name} Evaluation Dice Score: {eval_dice}")
        print(f"{model_name} Evaluation Recall: {eval_recall*100:<05.2f}")
        print(f"{model_name} Evaluation F1 Score: {eval_f1*100:<05.2f}")
        print(f"{model_name} Evaluation Accuracy: {eval_accuracy*100:<05.2f}")
        # print(f"{model_name} ROC AUC: {roc_auc}")

    #     plt.plot(fpr, tpr, lw=2, label=f'{model_name} (area = {roc_auc:.4f})')
    #
    # plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver Operating Characteristic')
    # plt.legend(loc="lower right")
    # plt.show()


if __name__ == "__main__":
    main()

