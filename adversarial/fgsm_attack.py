import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import get_selected_element
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

    r_xy =get_selected_element(output)


    print('this is original ooutput')
    print(r_xy)
    #r_x = output[1][0]
    perturbed_image = image.clone()
    # steer = steer.type(torch.FloatTensor)
    # if (steer.item() > -0.1):
    #     target_steer = steer + target
    # else:
    #     target_steer = steer - target
    target_r_x = r_xy - target
    target_r_x = target_r_x.to(device)
    image.requires_grad = True
    output = model(image)
    adv_r_x=get_selected_element(output)
    #adv_output = output.clone()
    #print(output)
    # print(output[1])
    # print(adv_r_x)
    diff = 0
    loss_function = nn.GaussianNLLLoss()
    # while abs(diff) < abs(target):
    for i in range(5):
        print(i)
        #adv_r_x=get_selected_element(adv_output)
        # print("this is start")
        # print(adv_r_x)
        # print("this is mid")
        # print(target_r_x)
        # print("this is end")

        # print(len(adv_r_x))
        # print('target_r_x')
        # print(len(target_r_x))

        # loss = nn.GaussianNLLLoss(adv_r_x, target_r_x)
        loss = F.mse_loss(adv_r_x, target_r_x)
        # loss=loss_function(adv_r_x,r_x)
        # print(loss)
        model.zero_grad()
        loss.backward(retain_graph=True)
        image_grad = image.grad.data
        perturbed_image = fgsm_attack_fun(perturbed_image, epsilon, image_grad)
        adv_output = model(perturbed_image)
        # print(adv_output)
        adv_r_x = get_selected_element(adv_output)
        print (adv_r_x)
        diff = abs(adv_r_x.detach().cpu().numpy() - get_selected_element(output).detach().cpu().numpy())
        if(loss<0.01):
            break
    noise = torch.clamp(perturbed_image - image, 0, 1)

    return diff, perturbed_image, r_xy, adv_r_x, noise
if __name__ == "__main__":
    pass


