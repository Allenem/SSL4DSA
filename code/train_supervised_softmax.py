import math, os, sys, argparse, logging, datetime, random
from re import I
from tqdm import tqdm
from tensorboardX import SummaryWriter
import numpy as np
import ml_collections

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid

from networks import vnet, unet, vnet4out, unet4out, unetpp, attunet, swinunet, isunetv1, isunetv2, isunetv3, isunetv4, isunetv5
from dataset import CoronaryArtery, RandomRotFlip, RandomCrop, CenterCrop, ToTensor, TwoStreamBatchSampler
from utils.losses import dice_loss, dice_loss_, DiceLoss


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../data/CA_DSA/LCA/', help='Name of dataset')
parser.add_argument('--exp', type=str, default='supervised/swinunet/LCA', help='Name of Experiment')
parser.add_argument('--gpus', type=str,  default='0, 1, 2', help='GPU to use')
parser.add_argument('--batch_size', type=int, default=4, help='batch_size per gpu')
parser.add_argument('--max_iterations', type=int,  default=6000, help='maximum epoch number to train')
# parser.add_argument('--base_lr', type=float,  default=1e-2, help='learn rate')
parser.add_argument('--base_lr', type=float,  default=1e-3, help='learn rate')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--num4train', type=int,  default=20, help='number of images for train')
args = parser.parse_args()


train_data_path = args.root_path
# snapshot_path = f'../model/{args.exp}_{str(args.num4train)}/'
snapshot_path = f'../model/{args.exp}_3layers_{str(args.num4train)}/'
netname = args.exp.split('/')[-2]
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
batch_size = args.batch_size * len(args.gpus.split(','))
max_iterations = args.max_iterations
base_lr = args.base_lr
if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
patch_size = (512, 512)
num_classes = 2
swin_config = ml_collections.ConfigDict({
    'IMG_SIZE': patch_size[0],
    'PATCH_SIZE': 4,
    'IN_CHANS': 3,
    'EMBED_DIM': 96,
    'DEPTHS': [2, 2, 6, 2],
    'NUM_HEADS': [3, 6, 12, 24],
    'WINDOW_SIZE': 8,
    'MLP_RATIO': 4.,
    'QKV_BIAS': True,
    'QK_SCALE': None,
    'DROP_RATE': 0.0,
    'DROP_PATH_RATE': 0.1,
    'APE': False,
    'PATCH_NORM': True,
    'USE_CHECKPOINT': False,
    # download from https://github.com/kamalkraj/Swin-Transformer-Serve
    'PRETRAIN_PATH': 'C:/Users/siat/Downloads/swin_tiny_patch4_window7_224.pth'
})
iswin_config = ml_collections.ConfigDict({
    'ALPHAS': [3/8, 4/8, 5/8, 6/8],
    'IMG_SIZE': 512,
    'PATCH_SIZE': 4,
    'IN_CHANS': 3,
    'NUM_CLASSES': 2,
    'EMBED_DIM': 96,
    'DEPTHS': [2, 2, 4, 2],
    'NUM_HEADS': [3, 6, 12, 24],
    'WINDOW_SIZE': 8,
    'MLP_RATIO': 4.,
    'QKV_BIAS': True,
    'QK_SCALE': None,
    'DROP_RATE': 0.0,
    'DROP_PATH_RATE': 0.1,
    'APE': False,
    'PATCH_NORM': True,
    'USE_CHECKPOINT': False
})
iswin_config_3layers = ml_collections.ConfigDict({
    'ALPHAS': [3/8, 4/8, 5/8],
    'IMG_SIZE': 512,
    'PATCH_SIZE': 4,
    'IN_CHANS': 3,
    'NUM_CLASSES': 2,
    'EMBED_DIM': 96,
    'DEPTHS': [2, 2, 4],
    'NUM_HEADS': [3, 6, 12],
    'WINDOW_SIZE': 8,
    'MLP_RATIO': 4.,
    'QKV_BIAS': True,
    'QK_SCALE': None,
    'DROP_RATE': 0.0,
    'DROP_PATH_RATE': 0.1,
    'APE': False,
    'PATCH_NORM': True,
    'USE_CHECKPOINT': False
})


def worker_init_fn(worker_id):
    random.seed(args.seed + worker_id)


def creat_model(netname, ema=False):

    if netname.lower() == 'swinunet':
        model = nn.DataParallel(swinunet(config=swin_config, img_size=patch_size[0], num_classes=num_classes)).cuda()
    elif netname.lower().__contains__('isunet'):
        # model = nn.DataParallel(eval(netname.lower())(config=iswin_config)).cuda()
        model = nn.DataParallel(eval(netname.lower())(config=iswin_config_3layers)).cuda()
    else:
        model = nn.DataParallel(eval(netname.lower())(1, num_classes)).cuda()
        
    if ema:
        for param in model.parameters():
            param.detach_()
    return model


if __name__ == '__main__':

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    # net = nn.DataParallel(SwinUnet(config=swin_config, img_size=patch_size[0], num_classes=num_classes)).cuda()
    # net = nn.DataParallel(InceptionSwinUnet(config=iswin_config)).cuda()
    # net = nn.DataParallel(InceptionSwinUnet(config=iswin_config_3layers)).cuda()
    # net = nn.DataParallel(InceptionSwinUnetV2(config=iswin_config_3layers)).cuda()
    # net = nn.DataParallel(InceptionSwinUnetV3(config=iswin_config_3layers)).cuda()
    net = creat_model(netname)
    train_dataset = CoronaryArtery(
        root=train_data_path,
        split='train',
        num=args.num4train,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=180),
            transforms.CenterCrop(size=patch_size[0]),
        ])
    )
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        worker_init_fn=worker_init_fn
    )

    lr_ = base_lr
    gamma = .99
    optimizer = optim.Adam(net.parameters(), lr=lr_, weight_decay=.0001)
    LossCE = nn.CrossEntropyLoss()
    LossDice = DiceLoss(num_classes)
    max_epoch = max_iterations // len(train_dataloader) + 1
    iter_num = 0

    # record loss, lr_, and so on
    writer = SummaryWriter(f'{snapshot_path}/log')
    # log
    logging.basicConfig(filename=f'{snapshot_path}/log.txt', level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    # logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(f'max_iterations: {max_iterations}\nmax_epoch: {max_epoch}\n{len(train_dataloader)} iterations per epoch')

    net.train()
    t0 = datetime.datetime.now()

    for epoch_num in tqdm(range(max_epoch), ncols=70):
        for i_batch, sample_batch in enumerate(train_dataloader):

            # 1. Get outputs of networks
            iter_num += 1
            img_batch, lb_batch = sample_batch['image'].cuda(), sample_batch['label'].cuda()
            outputs = net(img_batch)
            outputs_soft = F.softmax(outputs, dim=1)

            # # torch.float32 torch.float32 torch.float32 torch.float32
            # print(img_batch.dtype, lb_batch.dtype, outputs.dtype, outputs_soft.dtype)
            # # [0, 1], [0 or 1], [-, +], [0, 1]
            # print(img_batch[0], lb_batch[0], outputs[0], outputs_soft[0])
            # print(img_batch[0][0].max().max(), lb_batch[0][0].max().max(), outputs[0][0].max().max(), outputs_soft[0][0].max().max())
            # # [12, 1, 512, 512], [12, 1, 512, 512], [12, 2, 512, 512], [12, 2, 512, 512]
            # print(img_batch.shape, lb_batch.shape, outputs.shape, outputs_soft.shape)

            # 2. Calculate the loss and back-propagation
            # parameter1: [12, 2, 512, 512], torch.float32;  parameter2 must be: [12, 512, 512], torch.int64
            loss_ce = LossCE(outputs, lb_batch.squeeze(dim=1).long())
            # parameter1: [12, 2, 512, 512], torch.float32;  parameter2: [12, 1, 512, 512], torch.float32
            loss_dice = LossDice(outputs_soft, lb_batch)
            loss = .5 * (loss_ce + loss_dice)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 3. Log and record the loss_ce/loss_dice/loss, image/label/prediction
            logging.info(f'iteration {iter_num} : loss_ce = {loss_ce.item()} loss_dice = {loss_dice.item()} loss = {loss.item()}')
            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar('loss/loss_ce', loss_ce, iter_num)
            writer.add_scalar('loss/loss_dice', loss_dice, iter_num)
            writer.add_scalar('loss/loss', loss, iter_num)
            if iter_num % 100 == 0:
                # [batchsize, channel, h, w] -> torch.Size([4, 3, 512, 512])
                original_image = img_batch[:4, :, :, :].repeat(1, 3, 1, 1)
                grid_image = make_grid(original_image, 4, normalize=True)
                writer.add_image('train/Image', grid_image, iter_num)
                # [batchsize, channel, h, w] -> torch.Size([4, 3, 512, 512])
                label_image = lb_batch[:4, :, :, :].repeat(1, 3, 1, 1)
                grid_image = make_grid(label_image, 4, normalize=True)
                writer.add_image('train/Label', grid_image, iter_num)
                # [batchsize, keep one channel, h, w] -> torch.Size([4, 3, 512, 512]) # multi classes, multi values image
                predicted_image = torch.div(torch.argmax(outputs_soft[:4, :, :, :], 1, keepdim=True) * 255, num_classes - 1).repeat(1, 3, 1, 1)
                grid_image = make_grid(predicted_image, 4, normalize=True)
                writer.add_image('train/Prediction', grid_image, iter_num)
            
            # 4. Change the learning rate every 2000 iterations
            if iter_num % 2000 == 0:
                lr_ = base_lr * .1 ** (iter_num // 2000)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

            # 5. Save the model weights
            if iter_num % 1000 == 0:
                save_model_path = os.path.join(snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(net.state_dict(), save_model_path)
                logging.info(f'save model to {save_model_path}')

            if iter_num > max_iterations: break
        if iter_num > max_iterations: break

        # # 6. Change the learning rate every epoch
        # for param_group in optimizer.param_groups:
        #     lr_ = param_group['lr'] * gamma ** epoch_num
        #     param_group['lr'] = lr_

    save_model_path = os.path.join(snapshot_path, 'iter_' + str(max_iterations + 1) + '.pth')
    torch.save(net.state_dict(), save_model_path)
    logging.info(f'save model to {save_model_path}')

    t1 = datetime.datetime.now()
    logging.info(f'Training cost: {t1 - t0}')
    writer.close()