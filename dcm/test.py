import os
import argparse

from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from BraTS import *
from networks.Unet import UNet
from utils import cal_dice
from scipy.ndimage import binary_erosion, distance_transform_edt


def test_loop(model, test_loader, device):
    """
    在测试集上评估模型
    :param model: 训练好的模型
    :param test_loader: 测试集 DataLoader
    :param device: 设备 (CPU 或 GPU)
    :return: 测试集上的评估结果
    """
    model.eval()
    dice1_test = 0
    dice2_test = 0
    dice3_test = 0
    hd95_test = 0
    pbar = tqdm(test_loader)
    with torch.no_grad():
        for images, masks in pbar:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            dice1, dice2, dice3, hd95 = cal_dice(outputs, masks)
            dice1_test += dice1.item()
            dice2_test += dice2.item()
            dice3_test += dice3.item()
            hd95_test += hd95
    dice1_avg = dice1_test / len(test_loader)
    dice2_avg = dice2_test / len(test_loader)
    dice3_avg = dice3_test / len(test_loader)
    hd95_avg = hd95_test / len(test_loader)
    return {'dice1': dice1_avg, 'dice2': dice2_avg, 'dice3': dice3_avg, 'hd95': hd95_avg}


def main(args):
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载测试数据集
    patch_size = (160, 160, 128)
    test_dataset = BraTS(args.data_path, args.test_txt, transform=transforms.Compose([
        CenterCrop(patch_size),
        ToTensor()
    ]))
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, num_workers=12, shuffle=False,
                             pin_memory=True)

    # 加载模型
    model = UNet(in_channels=4, num_classes=4).to(device)
    if os.path.exists(args.weights):
        weight_dict = torch.load(args.weights, map_location=device)
        model.load_state_dict(weight_dict['model'])
        print('Successfully loaded checkpoint.')
    else:
        raise FileNotFoundError(f"Model weights not found at {args.weights}")

    # 在测试集上评估模型
    print("Testing model on test set...")
    test_metrics = test_loop(model, test_loader, device)
    print(f"Test -- ET: {test_metrics['dice1']:.3f}, TC: {test_metrics['dice2']:.3f}, WT: {test_metrics['dice3']:.3f}, HD95: {test_metrics['hd95']:.3f}")


if __name__ == '__main__':
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data/BraTS2018_data/', help='Path to the dataset')
    parser.add_argument('--test_txt', type=str, default='./data/BraTS2018_data/val.txt', help='Path to the test list file')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for testing')
    parser.add_argument('--weights', type=str, default='results/UNet.pth', help='Path to the trained model weights')
    args = parser.parse_args()

    # 运行测试
    main(args)