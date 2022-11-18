# train_semisupervised by Uncertainty Awareness and Mean Teaching in Networl

import math, os, sys, argparse, logging, random, ml_collections, cv2
from tqdm import tqdm
from datetime import datetime
from tensorboardX import SummaryWriter
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from torch.nn.modules.loss import CrossEntropyLoss

from networks import vnet, unet, vnet4out, unet4out, unetpp, attunet, swinunet, isunetv1, isunetv2, isunetv3, isunetv4, isunetv5
from dataset import CoronaryArtery, RandomRotFlip, RandomCrop, CenterCrop, ToTensor, TwoStreamBatchSampler
from utils import losses, ramps
from medpy import metric


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../data/CA_DSA/LCA/', help='Name of dataset')
parser.add_argument('--exp', type=str, default='semisupervised/UAMT/swinunet/LCA', help='Name of Experiment')
# parser.add_argument('--nets', type=str, default='vnet_swinunet', help='Name of networks')
parser.add_argument('--max_iterations', type=int,  default=6000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=4, help='batch_size per gpu')
parser.add_argument('--labeled_bs', type=int, default=2, help='labeled batch_size per gpu')
parser.add_argument('--gpus', type=str,  default='0, 1, 2', help='GPU to use')
parser.add_argument('--base_lr', type=float,  default=1e-3, help='learn rate')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency', type=float,  default=0.1, help='consistency')
parser.add_argument('--consistency_type', type=str,  default='mse', help='consistency_type')
# parser.add_argument('--consistency_rampup', type=float,  default=40.0, help='consistency_rampup')
parser.add_argument('--consistency_rampup', type=float,  default=200.0, help='consistency_rampup')
parser.add_argument('--num_classes', type=int,  default=2, help='output channel of network')
parser.add_argument('--patch_size', type=list,  default=[512, 512], help='patch size of network input')
args = parser.parse_args()


train_data_path = args.root_path
# snapshot_path = '../model/' + args.exp + '/'
# snapshot_path = '../model/' + args.exp + args.consistency_rampup + '_' + args.base_lr + '/'
snapshot_path = '../model/' + args.exp + '_3layers_' + str(args.consistency_rampup) + '_' + str(args.base_lr) + '/'
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
batch_size = args.batch_size * len(args.gpus.split(','))
max_iterations = args.max_iterations
base_lr = args.base_lr
# base_lr = args.base_lr * .1 # swinunet lr_ changes from .001
labeled_bs = args.labeled_bs
patch_size = args.patch_size
num_classes = args.num_classes
if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
swin_config = ml_collections.ConfigDict({
    'IMG_SIZE': args.patch_size[0],
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

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    # weight = .1 * exp(-5 * (1 - epoch / epochmax) ^ 2)
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    # data_ema = α data_ema + (1 - α) data
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        # ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)


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

    if not os.path.exists(snapshot_path): os.makedirs(snapshot_path)

    modelname = args.exp.split('/')[-2]

    model = creat_model(modelname)
    model_ema = creat_model(modelname, ema=True)

    train_dataset = CoronaryArtery(
        root=train_data_path,
        split='train',
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=180),
            transforms.CenterCrop(size=patch_size[0]),
        ])
    )
    labeled_idx, unlabeled_idx = list(range(20)), list(range(20, 100))
    batch_sampler = TwoStreamBatchSampler(labeled_idx, unlabeled_idx, batch_size, batch_size - labeled_bs)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_sampler=batch_sampler,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=worker_init_fn
    )
    
    model.train()
    model_ema.train()

    lr_ = base_lr
    optimizer = optim.Adam(model.parameters(), lr=lr_, weight_decay=1e-4)

    LossCE = CrossEntropyLoss()
    LossDice = losses.DiceLoss(num_classes)
    LossConsistency = losses.softmax_mse_loss if args.consistency_type == 'mse' else losses.softmax_kl_loss

    iter_num = 0
    max_epoch = max_iterations // len(train_dataloader) + 1

    writer = SummaryWriter(f'{snapshot_path}/log')
    logging.basicConfig(filename=f'{snapshot_path}/log.txt', level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    # logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(f'max_iterations: {max_iterations} \nmax_epoch: {max_epoch} \n{len(train_dataloader)} iterations per epoch')

    t0 = datetime.now()
    for epoch_num in tqdm(range(max_epoch), ncols=70):
        for i_batch, sample_batch in enumerate(train_dataloader):

            img_batch, lb_batch = sample_batch['image'].cuda(), sample_batch['label'].cuda() # [12, 1, 512, 512]*2, [0, 1]*2

            # 1.all kinds of outputs
            # outputs of model
            outputs = model(img_batch) # [12, 2, 512, 512] [-, +]
            outputs_soft = F.softmax(outputs, dim=1) # [12, 2, 512, 512] [0, 1]

            # outputs of model_ema
            ema_noise1 = torch.clamp(torch.randn_like(img_batch) * .1, -.2, .2) # [12, 1, 512, 512] [-.2, .2]
            ema_inputs1 = img_batch + ema_noise1
            with torch.no_grad():
                ema_outputs1 = model_ema(ema_inputs1) # [12, 2, 512, 512] [-, +]

            # repeat img_batch + noise and calculate the uncertainty mask to calculate the loss_consistency
            repeat_times = 8
            img_batch_r = img_batch.repeat(2, 1, 1, 1) # [24, 1, 512, 512]
            stride = img_batch.shape[0] # 12
            ema_outputs2 = torch.zeros([stride * repeat_times, num_classes, patch_size[0], patch_size[1]]).cuda() # [12 * 8, 2, 512, 512]
            for i in range(repeat_times // 2):
                ema_noise2 = torch.clamp(torch.rand_like(img_batch_r) * .1, -.2, .2)
                ema_inputs2 = img_batch_r + ema_noise2 # [24, 1, 512, 512]
                with torch.no_grad():
                    ema_outputs2[2 * stride * i:2 * stride * (i + 1)] = model_ema(ema_inputs2)
            preds = F.softmax(ema_outputs2, dim=1) # [12 * 8, 2, 512, 512]
            preds = preds.reshape(repeat_times, stride, num_classes, patch_size[0], patch_size[1]) # [8, 12, 2, 512, 512]
            preds = torch.mean(preds, dim=0) # [12, 2, 512, 512]
            uncertainty = -1.0 * torch.sum(preds * torch.log(preds + 1e-6), dim=1, keepdim=True) # [12, 1, 512, 512]
            thresh = (.75 + .25 * ramps.sigmoid_rampup(iter_num, max_iterations)) * np.log(2)
            mask = (uncertainty < thresh).float()

            # 2.loss
            loss_ce = LossCE(outputs[:labeled_bs], lb_batch[:labeled_bs].squeeze(dim=1).long()) # [2, 2, 512, 512], [2, 512, 512]
            loss_dice = LossDice(outputs[:labeled_bs], lb_batch[:labeled_bs]) # [2, 2, 512, 512], [2, 1, 512, 512]
            
            consistency_weight = get_current_consistency_weight(epoch_num)
            consistency_dist = LossConsistency(outputs, ema_outputs1) # [12, 2, 512, 512]*2
            consistency_dist = torch.sum(mask * consistency_dist) / (2 * torch.sum(mask) + 1e-16)
            consistency_loss = consistency_weight * consistency_dist

            loss = .5 * (loss_ce + loss_dice) + consistency_loss

            # 3.BP
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update_ema_variables(model, model_ema, args.ema_decay, iter_num)
            iter_num += 1

            # 4.log, writer
            logging.info(f'iteration {iter_num} : loss: {loss}, loss_ce: {loss_ce}, loss_dice: {loss_dice}, consistency_loss: {consistency_loss}, consistency_weight: {consistency_weight}, consistency_dist: {consistency_dist}')

            writer.add_scalar('train/lr', lr_, iter_num)

            writer.add_scalar('uncertainty/mean', uncertainty[0,0].mean(), iter_num)
            writer.add_scalar('uncertainty/max', uncertainty[0,0].max(), iter_num)
            writer.add_scalar('uncertainty/min', uncertainty[0,0].min(), iter_num)
            writer.add_scalar('uncertainty/thresh', thresh, iter_num)
            writer.add_scalar('uncertainty/mask_per', torch.sum(mask)/mask.numel(), iter_num)

            writer.add_scalar('loss/loss', loss, iter_num)
            writer.add_scalar('loss/loss_ce', loss_ce, iter_num)
            writer.add_scalar('loss/loss_dice', loss_dice, iter_num)
            writer.add_scalar('loss/consistency_loss', consistency_loss, iter_num)
            
            writer.add_scalar('consistency/consistency_weight', consistency_weight, iter_num)
            writer.add_scalar('consistency/consistency_dist', consistency_dist, iter_num)
            writer.add_scalar('consistency/consistency_loss', consistency_loss, iter_num)
            

            if iter_num % 100 == 0:
            # if iter_num % 1 == 0:

                # [bs, ch, h, w] = [12, 1, 512, 512] -> [4, 3, 512, 512]
                original_img = img_batch[:4].repeat(1, 3, 1, 1)
                grid_img = make_grid(original_img, 4, normalize=True)
                writer.add_image('train/Image', grid_img, iter_num)

                # [bs, ch, h, w] = [12, 1, 512, 512] -> [4, 3, 512, 512]
                label_img = lb_batch[:4].repeat(1, 3, 1, 1)
                grid_image = make_grid(label_img, 4, normalize=True)
                writer.add_image('train/Label', grid_image, iter_num)

                # [4, 1, 512, 512] -> [4, 3, 512, 512]
                pred_img =  torch.div( torch.argmax(outputs_soft[:4].detach(), dim=1, keepdim=True) * 255, num_classes - 1).repeat(1, 3, 1, 1)
                grid_img = make_grid(pred_img, 4, normalize=True)
                writer.add_image('train/Model1_Prediction', grid_img, iter_num)

                # [4, 1, 512, 512] -> [4, 3, 512, 512]
                uncertainty_img = uncertainty[:4].repeat(1, 3, 1, 1)
                grid_img = make_grid(uncertainty_img, 4, normalize=True)
                writer.add_image('train/Uncertainty', grid_img, iter_num)

                # [4, 1, 512, 512] -> [4, 3, 512, 512]
                mask2 = (uncertainty > thresh).float()
                mask_img = mask2[:4].repeat(1, 3, 1, 1)
                grid_img = make_grid(mask_img, 4, normalize=True)
                writer.add_image('train/ZMask', grid_img, iter_num)

                # unlabeled Image/Label/Prediction
                # [4, 3, 512, 512]
                unlabeled_img = img_batch[-4:].repeat(1, 3, 1, 1)
                grid_img = make_grid(unlabeled_img, 4, normalize=True)
                writer.add_image('unlabeled/Image', grid_img, iter_num)

                # [4, 3, 512, 512]
                label_img = lb_batch[-4:].repeat(1, 3, 1, 1)
                grid_img = make_grid(label_img, 4, normalize=True)
                writer.add_image('unlabeled/Label', grid_img, iter_num)

                # [4, 3, 512, 512]
                pred_img = torch.div(torch.argmax(outputs_soft[-4:].detach(), dim=1, keepdim=True) * 255, num_classes - 1).repeat(1, 3, 1, 1)
                grid_img = make_grid(pred_img, 4, normalize=True)
                writer.add_image('unlabeled/Prediction', grid_img, iter_num)
            
            # 5.change lr_
            if iter_num % 2000 == 0:
                lr_ = base_lr * .1 ** (iter_num // 2000)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

            # 6.save model weights
            if iter_num % 1000 == 0:
            # if iter_num % 1 == 0:
                save_model_path = f'{snapshot_path}iter_{str(iter_num)}.pth'
                torch.save(model.state_dict(), save_model_path)
                logging.info(f'save model to {save_model_path}')

            if iter_num > max_iterations: break
        if iter_num > max_iterations: break

    save_model_path = f'{snapshot_path}iter_{str(max_iterations + 1)}.pth'
    torch.save(model.state_dict(), save_model_path)
    logging.info(f'save model to {save_model_path}')

    t1 = datetime.now()
    logging.info(f'Train & validate data cost {t1 - t0}')
    writer.close()
