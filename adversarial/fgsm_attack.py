import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np


def fgsm_attack_fun(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image

#def fgsm_attack(model, image, target, device, epsilon=0.01, image_size=(128, 128)):

def fgsm_attack(model, image, target, device, epsilon=0.01, image_size=(128, 128)):
    # image=torch.from_numpy(image)
    image =image.to(device)
    # image= image.type(torch.FloatTensor)
    output = model(image)
    r_x =output[1][0]
    perturbed_image = image.clone()
    # steer = steer.type(torch.FloatTensor)
    # if (steer.item() > -0.1):
    #     target_steer = steer + target
    # else:
    #     target_steer = steer - target
    target_r_x = r_x - target
    target_r_x = target_r_x.to(device)
    image.requires_grad = True
    output = model(image)[1]
    adv_output = output.clone()
    print(output)

    diff = 0
    # while abs(diff) < abs(target):
    for i in range(5):
        # print(i)
        loss = F.mse_loss(adv_output[0], target_r_x)
        model.zero_grad()
        loss.backward(retain_graph=True)
        image_grad = image.grad.data
        perturbed_image = fgsm_attack_fun(perturbed_image, epsilon, image_grad)
        adv_output = model(perturbed_image)[1]
        # print(adv_output)
        diff = abs(adv_output[0].detach().cpu().numpy() - output[0].detach().cpu().numpy())

    noise = torch.clamp(perturbed_image - image, 0, 1)

    return diff, perturbed_image, r_x, adv_output, noise
if __name__ == "__main__":
    pass


