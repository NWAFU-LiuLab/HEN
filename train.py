from torch.nn import BCEWithLogitsLoss
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from dataset import FloodDataset

from torch.utils.data import DataLoader
import os,time
from config import config
# from utils import save_model

from sklearn.metrics import accuracy_score, recall_score, f1_score
import numpy as np

from tqdm import tqdm
from ablation import HEN1, HEN2

from layer import HEN_L

from model_resnet import ResNet
from model_HEN import HEN
from model_densenet import DenseNet
from model_deeplabv import DeepLav
from model_unet import UNetPlusPlus

def self_training(model, unlabeled_dataloader, threshold=0.5):
    model.eval()
    for images in unlabeled_dataloader:
        images = images[0].cuda()
        with torch.no_grad():
            outputs = model(images)
        outputs = (torch.sigmoid(outputs) > threshold).float()
        yield images.cpu(), outputs.cpu()


def iou_metric(predicted, target):
    predicted = predicted > 0
    target = target > 0
    intersection = (predicted & target).float().sum((1, 2))  
    union = (predicted | target).float().sum((1, 2))         
    iou = (intersection + 1e-6) / (union + 1e-6)            
    return iou.mean().item()


def main(config, ModelClass):
    try:
        model = HEN_L(out_channels=1, layer=eval(ModelClass[-1])).cuda()
        Modelname = ModelClass
    except:
        model = ModelClass(1).cuda()
        Modelname = ModelClass.__name__

    if not os.path.exists(f"{config.MODEL_SAVE_PATH}/{Modelname}"):
        os.makedirs(f"{config.MODEL_SAVE_PATH}/{Modelname}")

    # 5-Fold Cross Validation
    num_folds = 5
    for fold_idx in range(2, 3):
        print(f"Training for fold {fold_idx + 1} out of {num_folds}")

        # Load datasets with the current fold index
        train_dataset = FloodDataset(config.LABELED_DATA_PATH, config.LABELED_DATA_PATH, split="train",fold_idx=fold_idx)
        val_dataset = FloodDataset(config.LABELED_DATA_PATH, config.LABELED_DATA_PATH, split="val", fold_idx=fold_idx)
        unlabeled_dataset = FloodDataset(config.LABELED_DATA_PATH, config.LABELED_DATA_PATH, split="unlabeled")

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
        # unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=config.BATCH_SIZE, shuffle=True)

        # Create model, loss function and optimizer
        criterion = BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
        scheduler = CosineAnnealingLR(optimizer, T_max=config.T_MAX, eta_min=config.ETA_MIN)

        best_val_iou = 0.0
        patience = 10
        no_improvement = 0
        for epoch in range(config.EPOCHS):
            start_time = time.time()
            print(f'Epoch: {epoch}, Training on labeled data...')

            # Training phase
            total_loss = 0
            iou_sum = 0
            predictions = []
            targets = []
            model.train()
            pbar = tqdm(total=len(train_loader))
            for i, (images, masks) in enumerate(train_loader):
                pbar.update(1)
                images = images.cuda()
                masks = masks.cuda()
                masks = masks.unsqueeze(1).cuda()  # add an extra dimension

                optimizer.zero_grad()

                outputs = model(images)
                loss = criterion(outputs, masks)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                iou_sum += iou_metric(outputs, masks)
                predictions.extend(torch.sigmoid(outputs).cpu().detach().numpy())
                targets.extend(masks.cpu().numpy())

            pbar.close()

            train_loss = total_loss/len(train_loader)
            train_iou = iou_sum/len(train_loader)
            predictions = np.array(predictions)
            targets = np.array(targets)

            predictions_binary = predictions > 0.5
            targets_binary = targets > 0.5
            scheduler.step()

            train_acc = accuracy_score(targets_binary.flatten(), predictions_binary.flatten())
            train_recall = recall_score(targets_binary.flatten(), predictions_binary.flatten())
            train_f1 = f1_score(targets_binary.flatten(), predictions_binary.flatten())

            # Validation phase

            model.eval()
            total_loss = 0
            iou_sum = 0
            val_predictions = []
            val_targets = []
            with torch.no_grad():
                for images, masks in val_loader:
                    images = images.cuda()
                    masks = masks.cuda()
                    masks = masks.unsqueeze(1).cuda()  # add an extra dimension

                    outputs = model(images)
                    loss = criterion(outputs, masks)

                    total_loss += loss.item()
                    iou_sum += iou_metric(outputs, masks)
                    val_predictions.extend(torch.sigmoid(outputs).cpu().detach().numpy())
                    val_targets.extend(masks.cpu().numpy())

            val_loss = total_loss / len(val_loader)
            val_iou = iou_sum / len(val_loader)
            val_predictions = np.array(val_predictions)
            val_targets = np.array(val_targets)

            val_predictions_binary = val_predictions > 0.5
            val_targets_binary = val_targets > 0.5

            val_acc = accuracy_score(val_targets_binary.flatten(), val_predictions_binary.flatten())
            val_recall = recall_score(val_targets_binary.flatten(), val_predictions_binary.flatten())
            val_f1 = f1_score(val_targets_binary.flatten(), val_predictions_binary.flatten())

            # Write metrics to file
            with open(f"{config.MODEL_SAVE_PATH}/{Modelname}/metrics.txt", 'a') as f:
                f.write(
                    f'{epoch},{train_loss},{train_iou},{train_recall},{train_f1},{train_acc},{val_loss},{val_iou},{val_recall},{val_f1},{val_acc}\n')
            end_time = time.time()
            print(f'Epoch: {epoch}, Train Loss: {train_loss:.3f}, Train IoU: {train_iou:.3f}, Val Loss: {val_loss:.3f}, '
                  f'Val IoU: {val_iou:.3f}, Time: {end_time - start_time:.3f}s')
            # Save model if it's the best so far
            if val_iou > best_val_iou:
                best_val_iou = val_iou
                # save_model(model, f"{config.MODEL_SAVE_PATH}/{ModelClass.__name__}/best_{ModelClass.__name__}")
                torch.save(model.state_dict(), f"{config.MODEL_SAVE_PATH}/{Modelname}/best_{Modelname}.pth")
                print('best model saved!')
                no_improvement = 0  # reset the count
            else:
                no_improvement += 1
            # if no_improvement >= patience:
            #     print("Early stopping")
            #     break
        # print('Training on unlabeled data...')
        # for images, pseudo_labels in self_training(model, unlabeled_loader):
        #     train_loader.dataset.extend(images, pseudo_labels)
        #

if __name__ == "__main__":
    Model_name = [ResNet]
    # Model_name = ['HENL3', 'HENL0']
    for ModelClass in Model_name:
        main(config, ModelClass)