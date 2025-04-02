import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
from einops import rearrange
from scipy.ndimage import distance_transform_edt,binary_erosion, distance_transform_edt

def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0.):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule

# Dice 计算
def Dice(output, target, eps=1e-3):
    inter = torch.sum(output * target,dim=(1,2,-1)) + eps
    union = torch.sum(output,dim=(1,2,-1)) + torch.sum(target,dim=(1,2,-1)) + eps * 2
    x = 2 * inter / union
    dice = torch.mean(x)
    return dice
# HD95计算
# def HD95(output, target):
#     """
#     计算 HD95 (Hausdorff Distance 95%)
#     :param output: 模型输出的分割结果 (b, d, h, w)
#     :param target: 真实标签 (b, d, h, w)
#     :return: HD95 值
#     """
#     if output.shape != target.shape:
#         raise ValueError("Output and target must have the same shape.")

#     hd95_values = []
#     for i in range(output.shape[0]):  # 遍历 batch
#         pred = output[i].cpu().numpy()  # 转换为 numpy
#         gt = target[i].cpu().numpy()

#         # 计算预测结果的边界
#         pred_boundary = np.logical_and(pred, np.logical_not(binary_erosion(pred)))
#         # 计算真实标签的边界
#         gt_boundary = np.logical_and(gt, np.logical_not(binary_erosion(gt)))

#         # 如果预测或真实标签的边界为空，跳过计算
#         if np.sum(pred_boundary) == 0 or np.sum(gt_boundary) == 0:
#             hd95_values.append(0)
#             continue

#         # 计算距离变换
#         distance_pred = distance_transform_edt(np.logical_not(pred_boundary))
#         distance_gt = distance_transform_edt(np.logical_not(gt_boundary))

#         # 计算 HD95
#         hd1 = np.percentile(distance_pred[gt_boundary], 95)
#         hd2 = np.percentile(distance_gt[pred_boundary], 95)
#         hd95_values.append(max(hd1, hd2))

#     return np.mean(hd95_values)
def HD95(output, target, region):
    """
    计算 HD95 (Hausdorff Distance 95%) 针对不同区域
    :param output: 模型输出的分割结果 (b, d, h, w)
    :param target: 真实标签 (b, d, h, w)
    :param region: 区域类型 ('ET', 'TC', 'WT')
    :return: HD95 值
    """
    if output.shape != target.shape:
        raise ValueError("Output and target must have the same shape.")

    hd95_values = []
    for i in range(output.shape[0]):  # 遍历 batch
        pred = output[i].cpu().numpy()  # 转换为 numpy
        gt = target[i].cpu().numpy()

        # 根据区域类型生成掩码
        if region == 'ET':
            pred_mask = (pred == 3)
            gt_mask = (gt == 3)
        elif region == 'TC':
            pred_mask = (pred == 1) | (pred == 3)
            gt_mask = (gt == 1) | (gt == 3)
        elif region == 'WT':
            pred_mask = (pred != 0)
            gt_mask = (gt != 0)
        else:
            raise ValueError("Invalid region type. Must be 'ET', 'TC', or 'WT'.")

        # 计算预测结果的边界
        pred_boundary = np.logical_and(pred_mask, np.logical_not(binary_erosion(pred_mask)))
        # 计算真实标签的边界
        gt_boundary = np.logical_and(gt_mask, np.logical_not(binary_erosion(gt_mask)))

        # 如果预测或真实标签的边界为空，跳过计算
        if np.sum(pred_boundary) == 0 or np.sum(gt_boundary) == 0:
            hd95_values.append(0)
            continue

        # 计算距离变换
        distance_pred = distance_transform_edt(np.logical_not(pred_boundary))
        distance_gt = distance_transform_edt(np.logical_not(gt_boundary))

        # 计算 HD95
        hd1 = np.percentile(distance_pred[gt_boundary], 95)
        hd2 = np.percentile(distance_gt[pred_boundary], 95)
        hd95_values.append(max(hd1, hd2))

    return np.mean(hd95_values)
def cal_dice(output, target):
    '''
    output: (b, num_class, d, h, w)  target: (b, d, h, w)
    dice1(ET):label4
    dice2(TC):label1 + label4
    dice3(WT): label1 + label2 + label4
    注,这里的label4已经被替换为3
    '''
    output = torch.argmax(output,dim=1)
    dice1 = Dice((output == 3).float(), (target == 3).float())
    dice2 = Dice(((output == 1) | (output == 3)).float(), ((target == 1) | (target == 3)).float())
    dice3 = Dice((output != 0).float(), (target != 0).float())
    
    #HD95
#     hd95 = HD95(output.cpu().numpy(), target.cpu().numpy())
#     hd95 = HD95(output, target)
#     return dice1, dice2, dice3, hd95
    hd95_et = HD95(output, target, region='ET')
    hd95_tc = HD95(output, target, region='TC')
    hd95_wt = HD95(output, target, region='WT')

    return dice1, dice2, dice3, hd95_et, hd95_tc, hd95_wt
#     return dice1, dice2, dice3


class Loss(nn.Module):
    def __init__(self, n_classes, weight=None, alpha=0.5):
        "dice_loss_plus_cetr_weighted"
        super(Loss, self).__init__()
        self.n_classes = n_classes
        self.weight = weight.cuda()
        # self.weight = weight
        self.alpha = alpha

    def forward(self, input, target):
        # print(torch.unique(target))
        smooth = 0.01

        input1 = F.softmax(input, dim=1)
        target1 = F.one_hot(target,self.n_classes)
        input1 = rearrange(input1,'b n h w s -> b n (h w s)')
        target1 = rearrange(target1,'b h w s n -> b n (h w s)')

        input1 = input1[:, 1:, :]
        target1 = target1[:, 1:, :].float()

        # 以batch为单位计算loss和dice_loss，据说训练更稳定，那我试试
        inter = torch.sum(input1 * target1)
        union = torch.sum(input1) + torch.sum(target1) + smooth
        dice = 2.0 * inter / union

        loss = F.cross_entropy(input,target, weight=self.weight)

        total_loss = (1 - self.alpha) * loss + (1 - dice) * self.alpha

        return total_loss


if __name__ == '__main__':
    torch.manual_seed(3)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    losser = Loss(n_classes=4, weight=torch.tensor([0.2, 0.3, 0.25, 0.25])).to(device)
    x = torch.randn((2, 4, 16, 16, 16)).to(device)
    y = torch.randint(0, 4, (2, 16, 16, 16)).to(device)
    print(losser(x, y))
    print(cal_dice(x, y))
