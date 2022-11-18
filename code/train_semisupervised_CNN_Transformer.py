# train_semisupervised by Cross Teaching between CNN and Transformer

import math, os, sys, argparse, logging, random, ml_collections, cv2
# from matplotlib.pyplot import grid
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

from networks import Unet, Unetpp, AttUnet, SwinUnet, Vnet, InceptionSwinUnet, InceptionSwinUnetV2, InceptionSwinUnetV3
from dataset import CoronaryArtery, RandomRotFlip, RandomCrop, CenterCrop, ToTensor, TwoStreamBatchSampler
from utils import losses, ramps
from medpy import metric


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../data/CA_DSA/LCA/', help='Name of dataset')
parser.add_argument('--exp', type=str, default='semisupervised/CNN_Transformer/vnet_swinunet/LCA', help='Name of Experiment')
parser.add_argument('--nets', type=str, default='vnet_swinunet', help='Name of networks')
parser.add_argument('--max_iterations', type=int,  default=6000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=4, help='batch_size per gpu')
parser.add_argument('--labeled_bs', type=int, default=2, help='labeled batch_size per gpu')
parser.add_argument('--gpus', type=str,  default='0, 1, 2', help='GPU to use')
parser.add_argument('--base_lr', type=float,  default=1e-2, help='learn rate')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency', type=float,  default=0.1, help='consistency')
# parser.add_argument('--consistency_rampup', type=float,  default=40.0, help='consistency_rampup')
parser.add_argument('--consistency_rampup', type=float,  default=200.0, help='consistency_rampup')
parser.add_argument('--num_classes', type=int,  default=2, help='output channel of network')
parser.add_argument('--patch_size', type=list,  default=[512, 512], help='patch size of network input')
args = parser.parse_args()


train_data_path = args.root_path
snapshot_path = f'../model/{args.exp}/'
net1name = args.nets.split('_')[0]
net2name = args.nets.split('_')[1]
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
batch_size = args.batch_size * len(args.gpus.split(','))
max_iterations = args.max_iterations
# base_lr = args.base_lr
base_lr1 = args.base_lr
base_lr2 = args.base_lr * .1
labeled_bs = args.labeled_bs
num_classes = args.num_classes
patch_size = args.patch_size
networks = [Unet, Unetpp, AttUnet, Vnet]
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
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    # data_ema = α data_ema + (1 - α) data
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, 1 - alpha)


def worker_init_fn(worker_id):
    random.seed(args.seed + worker_id)


def creat_model(netname, ema=False):
    if netname.lower() == 'swinunet':
        model = nn.DataParallel(SwinUnet(config=swin_config, img_size=patch_size[0], num_classes=num_classes)).cuda()
    elif netname.lower() == 'inceptionswinunet':
        # model = nn.DataParallel(InceptionSwinUnet(config=iswin_config)).cuda()
        model = nn.DataParallel(InceptionSwinUnet(config=iswin_config_3layers)).cuda()
    elif netname.lower() == 'inceptionswinunetv2':
        # model = nn.DataParallel(InceptionSwinUnetV2(config=iswin_config)).cuda()
        model = nn.DataParallel(InceptionSwinUnetV2(config=iswin_config_3layers)).cuda()
    elif netname.lower() == 'inceptionswinunetv3':
        # model = nn.DataParallel(InceptionSwinUnetV3(config=iswin_config)).cuda()
        model = nn.DataParallel(InceptionSwinUnetV3(config=iswin_config_3layers)).cuda()
    else:
        for network in networks:
            if netname.lower() == str(network).split('.')[-1][:-2].lower():
                model = nn.DataParallel(network(1, num_classes)).cuda()
                break    
    if ema:
        for param in model.parameters():
            param.detach_()
    return model


def calculate_metrics(pred, gt):
    dice = metric.binary.dc(pred, gt)
    # jc = metric.binary.jc(pred, gt)
    hd = metric.binary.hd95(pred, gt)
    # asd = metric.binary.asd(pred, gt)

    return dice, hd


def calculate_all_cases(net, testdataloader, numclasses, save_result=False, test_save_path=None):

    metric_sum = np.zeros((numclasses - 1, 2)) # [numclasses-1, 2metrics]
    metric_total = [] # save all metrics of each case, [cases, numclasses-1, 2metrics]
    num_test = 0 # number of test cases

    # for sample_batch in tqdm(testdataloader, ncols=70):
    for sample_batch in testdataloader:

        img_batch, lb_batch = sample_batch['image'].cuda(), sample_batch['label'].cuda() # [1, 1, 512, 512]*2, [0, 1]*2
        outputs = net(img_batch) # [1, 2, 512, 512], [-, +]
        outputs_soft = F.softmax(outputs, dim=1) # [1, 2, 512, 512], [0, 1]

        img = img_batch.cpu().data.numpy()[0, 0].astype(np.float32) # [1, 1, 512, 512] -> [512, 512]
        lb = lb_batch.cpu().data.numpy()[0, 0].astype(np.float32) # [1, 1, 512, 512] -> [512, 512]
        pred = np.argmax(outputs_soft.cpu().data.numpy()[0], axis=0)  # [1, 2, 512, 512] -> [2, 512, 512] -> [512, 512]
        pred_norm = pred / (numclasses - 1)

        if save_result:
            cv2.imwrite(f'{test_save_path}{str(num_test+100)}_img.bmp', img * 255)
            cv2.imwrite(f'{test_save_path}{str(num_test+100)}_label.bmp', lb * 255)
            cv2.imwrite(f'{test_save_path}{str(num_test+100)}_pred.bmp', pred_norm * 255)

        single_metric = [] # [numclasses-1, 2metrics]
        if np.sum(pred) == 0:
            single_metric = [0, 0] * (numclasses - 1)
        else:
            # ignore class 0
            for i in range(1, num_classes):
                single_metric.append(calculate_metrics(pred == i, np.around(lb * (numclasses - 1)) == i))
        
        metric_sum += np.asarray(single_metric) # [numclasses-1, 2metrics]
        metric_total.append(np.asarray(single_metric)) # [cases, numclasses-1, 2metrics]
        num_test += img_batch.shape[0]

    metric_avg = metric_sum / num_test # [numclasses-1, 2metrics]
    metric_total = np.asarray(metric_total, dtype=object) # [cases, numclasses-1, 2metrics]
    # print(metric_avg.shape, metric_total.shape) # (1, 2) (50, 1, 2)

    return metric_avg, metric_total


if __name__ == '__main__':

    if not os.path.exists(snapshot_path): os.makedirs(snapshot_path)

    model1 = creat_model(net1name)
    model2 = creat_model(net2name)

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

    valid_dataset = CoronaryArtery(
        root=train_data_path,
        split='test',
        transform=transforms.ToTensor()
    )
    valid_dataloader = DataLoader(
        dataset=valid_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
    )

    model1.train()
    model2.train()

    # lr_ = base_lr
    # optimizer1 = optim.Adam(model1.parameters(), lr=lr_, weight_decay=0.0001)
    # optimizer2 = optim.Adam(model2.parameters(), lr=lr_, weight_decay=0.0001)
    lr_1 = base_lr1
    lr_2 = base_lr2
    optimizer1 = optim.Adam(model1.parameters(), lr=lr_1, weight_decay=0.0001)
    optimizer2 = optim.Adam(model2.parameters(), lr=lr_2, weight_decay=0.0001)

    LossCE = CrossEntropyLoss()
    LossDice = losses.DiceLoss(num_classes)

    iter_num = 0
    max_epoch = max_iterations // len(train_dataloader) + 1
    best_performance1 = 0.
    best_performance2 = 0.

    # record loss, lr_, and so on
    writer = SummaryWriter(f'{snapshot_path}/log')
    # log
    logging.basicConfig(filename=f'{snapshot_path}/log.txt', level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    # logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(f'max_iterations: {max_iterations} \nmax_epoch: {max_epoch} \n{len(train_dataloader)} iterations per epoch')

    t0 = datetime.now()
    for epoch_num in tqdm(range(max_epoch), ncols=70):
        for i_batch, sample_batch in enumerate(train_dataloader):

            # 1.input & output
            iter_num += 1
            img_batch, lb_batch = sample_batch['image'].cuda(), sample_batch['label'].cuda() # [12, 1, 512, 512]*2, [0, 1]*2

            outputs1 = model1(img_batch) # [12, 2, 512, 512], [-, +]
            outputs_soft1 = torch.softmax(outputs1, dim=1) # [12, 2, 512, 512], [0, 1]
            outputs_pseodu1 = torch.argmax(outputs_soft1.detach(), dim=1, keepdim=True) # [12, 1, 512, 512], [0, 1]

            outputs2 = model2(img_batch) # [12, 2, 512, 512], [-, +]
            outputs_soft2 = torch.softmax(outputs2, dim=1) # [12, 2, 512, 512], [0, 1]
            outputs_pseodu2 = torch.argmax(outputs_soft2.detach(), dim=1, keepdim=True) # [12, 1, 512, 512], [0, 1]

            # 2.loss
            loss1 = .5 * (LossCE(outputs1[:labeled_bs], lb_batch[:labeled_bs].squeeze(dim=1).long()) # [2, 2, 512, 512], [2, 512, 512]
                        + LossDice(outputs1[:labeled_bs], lb_batch[:labeled_bs])) # [2, 2, 512, 512], [2, 1, 512, 512]
            loss2 = .5 * (LossCE(outputs2[:labeled_bs], lb_batch[:labeled_bs].squeeze(dim=1).long()) # [2, 2, 512, 512], [2, 512, 512]
                        + LossDice(outputs2[:labeled_bs], lb_batch[:labeled_bs])) # [2, 2, 512, 512], [2, 1, 512, 512]

            pseudo_supervision1 = LossDice(outputs_soft1[labeled_bs:], outputs_pseodu2[labeled_bs:]) # [10, 2, 512, 512], [10, 1, 512, 512]
            pseudo_supervision2 = LossDice(outputs_soft2[labeled_bs:], outputs_pseodu1[labeled_bs:]) # [10, 2, 512, 512], [10, 1, 512, 512]

            consistency_weight = get_current_consistency_weight(iter_num // 150)
            loss_model1 = loss1 + pseudo_supervision1 * consistency_weight
            loss_model2 = loss2 + pseudo_supervision2 * consistency_weight
            loss = loss_model1 + loss_model2

            # 3.backpropagation
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            loss.backward()
            optimizer1.step()
            optimizer2.step()

            # 4.writer & loging
            logging.info(f'iteration {iter_num} : loss_model1: {loss_model1.item()}, loss_model2: {loss_model2.item()}, consistency_weight: {consistency_weight}')

            # writer.add_scalar('train/lr', lr_, iter_num)
            writer.add_scalar('train/lr1', lr_1, iter_num)
            writer.add_scalar('train/lr2', lr_2, iter_num)
            writer.add_scalar('train/consistency_weight', consistency_weight, iter_num)

            writer.add_scalar('loss1/loss_model1', loss_model1, iter_num)
            writer.add_scalar('loss1/loss_seg', loss1, iter_num)
            writer.add_scalar('loss1/loss_pseudo_supervision1', pseudo_supervision1, iter_num)

            writer.add_scalar('loss2/loss_model2', loss_model2, iter_num)
            writer.add_scalar('loss2/loss_seg', loss2, iter_num)
            writer.add_scalar('loss2/loss_pseudo_supervision2', pseudo_supervision2, iter_num)

            if iter_num % 100 == 0:
                # [bs, ch, h, w] = [12, 1, 512, 512] -> [4, 3, 512, 512]
                original_img = img_batch[:4].repeat(1, 3, 1, 1)
                grid_img = make_grid(original_img, 4, normalize=True)
                writer.add_image('train/Image', grid_img, iter_num)

                # [bs, ch, h, w] = [12, 1, 512, 512] -> [4, 3, 512, 512]
                label_img = lb_batch[:4].repeat(1, 3, 1, 1)
                grid_image = make_grid(label_img, 4, normalize=True)
                writer.add_image('train/Label', grid_image, iter_num)

                # [4, 1, 512, 512] -> [4, 3, 512, 512]
                pred_img =  torch.div(outputs_pseodu1[:4] * 255, num_classes - 1).repeat(1, 3, 1, 1)
                grid_img = make_grid(pred_img, 4, normalize=True)
                writer.add_image('train/Model1_Prediction', grid_img, iter_num)

                # [4, 1, 512, 512] -> [4, 3, 512, 512]
                pred_img =  torch.div(outputs_pseodu2[:4] * 255, num_classes - 1).repeat(1, 3, 1, 1)
                grid_img = make_grid(pred_img, 4, normalize=True)
                writer.add_image('train/Model2_Prediction', grid_img, iter_num)

            # 5.validation
            if iter_num > 0 and iter_num % 200 == 0:
            # if iter_num > 0 and iter_num % 1 == 0:

                model1.eval()

                # avg_metric: [(nclass-1), nmetrics]; total_metric: [ncase, (nclass-1), nmetrics]
                metric_avg, metric_total = calculate_all_cases(model1, valid_dataloader, num_classes, save_result=False, test_save_path=None)
                if num_classes > 2: # >2 add more scalars
                    for class_i in range(num_classes - 1):
                        writer.add_scalar(f'info/model1_val_class{class_i+1}_dice', metric_avg[class_i, 0], iter_num)
                        writer.add_scalar(f'info/model1_val_class{class_i+1}_hd95', metric_avg[class_i, 1], iter_num)
                class_mean_dice1 = np.mean(metric_avg, axis=0)[0]
                class_mean_hd951 = np.mean(metric_avg, axis=0)[1]
                writer.add_scalar('info/model1_val_class_mean_dice', class_mean_dice1, iter_num)
                writer.add_scalar('info/model1_val_class_mean_hd95', class_mean_hd951, iter_num)

                if class_mean_dice1 > best_performance1:
                    best_performance1 = class_mean_dice1
                    save_mode_path = f'{snapshot_path}model1_iter_{iter_num}_dice_{round(class_mean_dice1, 4)}_hd95_{round(class_mean_hd951, 2)}.pth'
                    save_best_path = f'{snapshot_path}{args.nets}_best_model1.pth'
                    torch.save(model1.state_dict(), save_mode_path)
                    torch.save(model1.state_dict(), save_best_path)

                logging.info(f'iteration {iter_num} : model1_mean_dice : {class_mean_dice1} model1_mean_hd95 : {class_mean_hd951}')

                model1.train()

                model2.eval()

                # avg_metric: [(nclass-1), nmetrics]; total_metric: [ncase, (nclass-1), nmetrics]
                metric_avg, metric_total = calculate_all_cases(model2, valid_dataloader, num_classes, save_result=False, test_save_path=None)
                if num_classes > 2: # >2 add more scalars
                    for class_i in range(num_classes - 1):
                        writer.add_scalar(f'info/model2_val_class{class_i+1}_dice', metric_avg[class_i, 0], iter_num)
                        writer.add_scalar(f'info/model2_val_class{class_i+1}_hd95', metric_avg[class_i, 1], iter_num)
                class_mean_dice2 = np.mean(metric_avg, axis=0)[0]
                class_mean_hd952 = np.mean(metric_avg, axis=0)[1]
                writer.add_scalar('info/model2_val_class_mean_dice', class_mean_dice2, iter_num)
                writer.add_scalar('info/model2_val_class_mean_hd95', class_mean_hd952, iter_num)

                if class_mean_dice2 > best_performance2:
                    best_performance2 = class_mean_dice2
                    save_mode_path = f'{snapshot_path}model2_iter_{iter_num}_dice_{round(class_mean_dice2, 4)}_hd95_{round(class_mean_hd952, 2)}.pth'
                    save_best_path = f'{snapshot_path}{args.nets}_best_model2.pth'
                    torch.save(model2.state_dict(), save_mode_path)
                    torch.save(model2.state_dict(), save_best_path)

                logging.info(f'iteration {iter_num} : model2_mean_dice : {class_mean_dice2} model2_mean_hd95 : {class_mean_hd952}')

                model2.train()

            # 6.lr
            # lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            # for param_group in optimizer1.param_groups:
            #     param_group['lr'] = lr_
            # for param_group in optimizer2.param_groups:
            #     param_group['lr'] = lr_

            if iter_num % 2000 == 0:
                lr_1 = base_lr1 * .1 ** (iter_num // 2000)
                for param_group in optimizer1.param_groups:
                    param_group['lr'] = lr_1
                lr_2 = base_lr2 * .1 ** (iter_num // 2000)
                for param_group in optimizer2.param_groups:
                    param_group['lr'] = lr_2

                
            # 7.save weights
            if iter_num % 1000 == 0:
                save_model_path = f'{snapshot_path}model1_iter_{str(iter_num)}.pth'
                torch.save(model1.state_dict(), save_model_path)
                logging.info(f'save model1 to {save_model_path}')
                
                save_model_path = f'{snapshot_path}model2_iter_{str(iter_num)}.pth'
                torch.save(model2.state_dict(), save_model_path)
                logging.info(f'save model2 to {save_model_path}')

            if iter_num > max_iterations: break
        if iter_num > max_iterations: break

    t1 = datetime.now()
    logging.info(f'Train & validate data cost {t1 - t0}')
    writer.close()
