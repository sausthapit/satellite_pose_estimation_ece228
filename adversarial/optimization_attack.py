import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
from utils import get_selected_element

def optimized_attack(target_model, target, x, device):
    y_pred =get_selected_element(target_model(x))# Only taking the x axis
    y_adv = y_pred
    # if y_pred > -0.1:
    #     y_target = y_pred - target
    # else:
    #     y_target = y_pred + target
    y_target = y_pred + target

    perturb = torch.zeros_like(x)
    perturb.requires_grad = True
    perturb = perturb.to(device)
    optimizer = optim.Adam(params=[perturb], lr=0.005)
    diff = 0

    # while abs(diff) < abs(target):
    for i in range(100):
        perturbed_image = x + perturb
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        y_adv = get_selected_element(target_model(perturbed_image))
        optimizer.zero_grad()
        loss_y = F.mse_loss(y_adv, y_target)
        loss_n = torch.mean(torch.pow(perturb, 2))
        loss_adv = loss_y + loss_n
        loss_adv.backward(retain_graph=True)
        optimizer.step()
        diff = y_adv.detach().cpu().numpy() - y_pred.detach().cpu().numpy()
        if abs(np.linalg.norm(diff,2)) >= abs(target):
            break
        # print(diff, target)


    
    return perturbed_image, perturb, y_pred, y_adv