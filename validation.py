import numpy as np
import utils.util
from torch import nn
import torch.nn.functional as F
import torch
import matplotlib as plt

def get_jaccard(y_true,y_pred,threshold=0.5):
    y_pred = torch.sigmoid(y_pred)
    y_pred = (y_pred > threshold).float()
    epsilon = 1e-6
    intersection = (y_pred * y_true).sum(dim=-2).sum(dim=-1)
    union =  (y_pred + y_true).sum(dim=-2).sum(dim=-1)
    return list((intersection / (union - intersection + epsilon)).data.cpu().numpy())

def get_dice(y_true,y_pred,threshold=0.5):
    y_pred = torch.sigmoid(y_pred)
    y_pred = (y_pred > threshold).float()
    epsilon = 1e-6
    intersection = (y_pred * y_true).sum(dim=-2).sum(dim=-1)
    union = (y_pred + y_true).sum(dim=-2).sum(dim=-1)
    return list(((2. * intersection) / (union + epsilon)).data.cpu().numpy())

def get_sen(y_true,y_pred,threshold=0.5):
    y_pred = torch.sigmoid(y_pred)
    y_pred = (y_pred > threshold).int()
    y_true = y_true.int()
    epsilon = 1e-6
    TP = (y_pred * y_true).sum(dim=-2).sum(dim=-1)
    FN = ((y_pred == 0) * y_true).sum(dim=-2).sum(dim=-1)
    return list((TP/ (TP + FN + epsilon)).data.cpu().numpy())

def get_spe(y_true,y_pred,threshold=0.5):
    y_pred = torch.sigmoid(y_pred)
    y_pred = (y_pred > threshold).int()
    y_true = y_true.int()
    epsilon = 1e-6
    TN = ((y_pred==0) * (y_true==0)).sum(dim=-2).sum(dim=-1)
    FP = (y_pred * (y_true==0)).sum(dim=-2).sum(dim=-1)
    return list((TN / (TN + FP + epsilon)).data.cpu().numpy())

def get_ppv(y_true,y_pred,threshold=0.5):
    y_pred = torch.sigmoid(y_pred)
    y_pred = (y_pred > threshold).int()
    y_true = y_true.int()
    epsilon = 1e-6
    TP = (y_pred * y_true).sum(dim=-2).sum(dim=-1)
    FP = (y_pred * (y_true==0)).sum(dim=-2).sum(dim=-1)
    return list((TP / (TP + FP + epsilon)).data.cpu().numpy())


def validation_binary(model,criterion,valid_dataset,num_classes=None):

    with torch.no_grad():
        model.eval()
        losses = []
        jaccard = []
        dice = []
        sen = []
        spe = []
        ppv = []

        for inputs,targets in valid_dataset:
            inputs = utils.util.cuda(inputs)
            targets = utils.util.cuda(targets)
            outputs = model(inputs)
            loss = criterion(outputs,targets)
            losses.append(loss.item())
            jaccard += get_jaccard(targets,outputs.float(),0.3)
            dice += get_dice(targets,outputs.float(),0.3)
            sen += get_sen(targets,outputs.float(),0.3)
            spe += get_spe(targets,outputs.float(),0.3)
            ppv += get_ppv(targets,outputs.float(),0.3)
        
        valid_loss = np.mean(losses)
        valid_jaccard = np.mean(jaccard).astype(np.float64)
        valid_dice = np.mean(dice).astype(np.float64)
        valid_se = np.mean(sen).astype(np.float64)
        valid_sp = np.mean(spe).astype(np.float64)
        valid_pp = np.mean(ppv).astype(np.float64)




        print('Valid loss: {:.5f} | Jaccard: {:.5f} | DICE: {:.5f} | SEN: {:.5f} | SPE： {:.5f} | PPV: {:.5f}'.
        format(valid_loss, valid_jaccard,valid_dice,valid_se,valid_sp,valid_pp)
        )

        metrics = { 'valid_loss': valid_loss, 
                    'jaccard_loss': valid_jaccard,
                    'dice':valid_dice,
                    'sen':valid_se,
                    'spe':valid_sp,
                    'ppv':valid_pp
        }

        return metrics


def train_binary(model,criterion,valid_dataset,num_classes=None):
    with torch.no_grad():
        model.eval()
        losses = []
        jaccard = []
        dice = []
        sen = []
        spe = []
        ppv = []
        
        for inputs,targets in valid_dataset:
            inputs = utils.util.cuda(inputs)
            targets = utils.util.cuda(targets)
            outputs = model(inputs)
            loss = criterion(outputs,targets)
            losses.append(loss.item())
            jaccard += get_jaccard(targets,outputs.float(),0.3)
            dice += get_dice(targets,outputs.float(),0.3)
            sen += get_sen(targets,outputs.float(),0.3)
            spe += get_spe(targets,outputs.float(),0.3)
            ppv += get_ppv(targets,outputs.float(),0.3)
        
        train_loss = np.mean(losses)
        train_jaccard = np.mean(jaccard).astype(np.float64)
        train_dice = np.mean(dice).astype(np.float64)
        train_se = np.mean(sen).astype(np.float64)
        train_sp = np.mean(spe).astype(np.float64)
        train_pp = np.mean(ppv).astype(np.float64)

        print('Train loss: {:.5f} | Jaccard: {:.5f} | DICE: {:.5f} | SEN: {:.5f} | SPE： {:.5f} | PPV: {:.5f}'.
        format(train_loss, train_jaccard,train_dice,train_se,train_sp,train_pp)
        )


        return train_loss, train_jaccard,train_dice,train_se,train_sp,train_pp
