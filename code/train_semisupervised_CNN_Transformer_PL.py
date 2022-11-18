# train_semisupervised by Cross Teaching, Confident Learning between CNN and Transformer

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

from networks import vnet, unet, vnet4out, unet4out, unetpp, attunet, swinunet, isunetv1, isunetv2, isunetv3, isunetv4, isunetv5
from dataset import CoronaryArtery, RandomRotFlip, RandomCrop, CenterCrop, ToTensor, TwoStreamBatchSampler
from utils import losses, ramps
from medpy import metric
from sklearn import metrics


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../data/CA_DSA/LCA/', help='Name of dataset')
parser.add_argument('--exp', type=str, default='semisupervised/CNN_Transformer_PL/vnet_swinunet/LCA', help='Name of Experiment')
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


def create_model(netname, ema=False):

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

def calculate_metrics(pred, gt):
    dice = metric.binary.dc(pred, gt)
    # jc = metric.binary.jc(pred, gt)
    # hd = metric.binary.hd95(pred, gt)
    # asd = metric.binary.asd(pred, gt)
    se = metric.binary.sensitivity(pred, gt)
    sp = metric.binary.specificity(pred, gt)
    acc = metrics.accuracy_score(pred.flatten(), gt.flatten())

    all_metrics = [dice, acc, se, sp]

    return all_metrics
    # return dice, jc, hd, asd


def calculate_all_cases(net, testdataloader, numclasses, save_result=True, test_save_path=None):

    nmetrics = 4
    metric_sum = np.zeros((numclasses - 1, nmetrics)) # [numclasses-1, 2metrics]
    metric_total = [] # save all metrics of each case, [cases, numclasses-1, 2metrics]
    num_test = 0 # number of test cases

    for sample_batch in testdataloader:

        img_batch, lb_batch = sample_batch['image'].cuda(), sample_batch['label'].cuda() # [1, 1, 512, 512]*2, [0, 1]*2
        outputs = net(img_batch)[-1] # [1, 2, 512, 512], [-, +]
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
            single_metric = [[0] * nmetrics for _ in range(numclasses - 1)] # 4metrics
        else:
            # ignore class 0
            for i in range(1, num_classes):
                single_metric.append(calculate_metrics(pred == i, np.around(lb * (numclasses - 1)) == i))
        
        metric_sum += np.asarray(single_metric) # [numclasses-1, 2metrics]
        metric_total.append(np.asarray(single_metric)) # [cases, numclasses-1, 2metrics]
        num_test += img_batch.shape[0]

    metric_avg = metric_sum / num_test # [numclasses-1, 2metrics]
    metric_total = np.asarray(metric_total) # [cases, numclasses-1, 2metrics]
    # print(metric_avg.shape, metric_total.shape) # (1, 2) (50, 1, 2)

    return metric_avg, metric_total


if __name__ == '__main__':

    if not os.path.exists(snapshot_path): os.makedirs(snapshot_path)

    model1 = create_model(net1name)
    model2 = create_model(net2name)

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
    kl_distance = nn.KLDivLoss(reduction='none')

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

            out11, out12, out13, out14 = model1(img_batch) # [12, 2, 512, 512], [-, +]
            out_soft11 = torch.softmax(out11, dim=1)  # [12, 2, 512, 512], [0, 1]
            out_soft12 = torch.softmax(out12, dim=1)
            out_soft13 = torch.softmax(out13, dim=1)
            out_soft14 = torch.softmax(out14, dim=1)
            out_pseodu11 = torch.argmax(out_soft11.detach(), dim=1, keepdim=True) # [12, 1, 512, 512], [0, 1]
            out_pseodu12 = torch.argmax(out_soft12.detach(), dim=1, keepdim=True)
            out_pseodu13 = torch.argmax(out_soft13.detach(), dim=1, keepdim=True)
            out_pseodu14 = torch.argmax(out_soft14.detach(), dim=1, keepdim=True)

            out21, out22, out23, out24 = model2(img_batch) # [12, 2, 512, 512], [-, +]
            out_soft21 = torch.softmax(out21, dim=1) # [12, 2, 512, 512], [0, 1]
            out_soft22 = torch.softmax(out22, dim=1)
            out_soft23 = torch.softmax(out23, dim=1)
            out_soft24 = torch.softmax(out24, dim=1)
            out_pseodu21 = torch.argmax(out_soft21.detach(), dim=1, keepdim=True) # [12, 1, 512, 512], [0, 1]
            out_pseodu22 = torch.argmax(out_soft22.detach(), dim=1, keepdim=True)
            out_pseodu23 = torch.argmax(out_soft23.detach(), dim=1, keepdim=True)
            out_pseodu24 = torch.argmax(out_soft24.detach(), dim=1, keepdim=True)

            # 2.loss
            # 2.1.supervise loss
            supervise_loss_CE1 = (.15 * LossCE(out11[:labeled_bs], lb_batch[:labeled_bs].squeeze(dim=1).long()) # [2, 2, 512, 512], [2, 512, 512] 
                                + .20 * LossCE(out12[:labeled_bs], lb_batch[:labeled_bs].squeeze(dim=1).long())
                                + .30 * LossCE(out13[:labeled_bs], lb_batch[:labeled_bs].squeeze(dim=1).long())
                                + .35 * LossCE(out14[:labeled_bs], lb_batch[:labeled_bs].squeeze(dim=1).long()))
            supervise_loss_Dice1 = (.15 * LossDice(out11[:labeled_bs], lb_batch[:labeled_bs]) # [2, 2, 512, 512], [2, 1, 512, 512]
                                + .20 * LossDice(out12[:labeled_bs], lb_batch[:labeled_bs])
                                + .30 * LossDice(out13[:labeled_bs], lb_batch[:labeled_bs])
                                + .35 * LossDice(out14[:labeled_bs], lb_batch[:labeled_bs]))
            supervise_loss_CE2 = (.15 * LossCE(out21[:labeled_bs], lb_batch[:labeled_bs].squeeze(dim=1).long()) # [2, 2, 512, 512], [2, 512, 512] 
                                + .20 * LossCE(out22[:labeled_bs], lb_batch[:labeled_bs].squeeze(dim=1).long())
                                + .30 * LossCE(out23[:labeled_bs], lb_batch[:labeled_bs].squeeze(dim=1).long())
                                + .35 * LossCE(out24[:labeled_bs], lb_batch[:labeled_bs].squeeze(dim=1).long()))
            supervise_loss_Dice2 = (.15 * LossDice(out21[:labeled_bs], lb_batch[:labeled_bs]) # [2, 2, 512, 512], [2, 1, 512, 512]
                                + .20 * LossDice(out22[:labeled_bs], lb_batch[:labeled_bs])
                                + .30 * LossDice(out23[:labeled_bs], lb_batch[:labeled_bs])
                                + .35 * LossDice(out24[:labeled_bs], lb_batch[:labeled_bs]))
            supervise_loss1 = (supervise_loss_CE1 + supervise_loss_Dice1) / 2
            supervise_loss2 = (supervise_loss_CE2 + supervise_loss_Dice2) / 2

            # 2.2.pseudo loss
            pseudo_supervision1 = LossDice(out_soft14[labeled_bs:], out_pseodu24[labeled_bs:]) # [10, 2, 512, 512], [10, 1, 512, 512]
            pseudo_supervision2 = LossDice(out_soft24[labeled_bs:], out_pseodu14[labeled_bs:]) # [10, 2, 512, 512], [10, 1, 512, 512]

            # 2.3.pyramid consistency loss
            if iter_num < 400:
                consistency_loss1 = 0
                consistency_loss2 = 0
            else:
                avg1 = .15 * out_soft11 + .20 * out_soft12 + .30 * out_soft13 + .35 * out_soft14
                avg2 = .15 * out_soft21 + .20 * out_soft22 + .30 * out_soft23 + .35 * out_soft24

                variance_aux11 = torch.sum(kl_distance(torch.log(out_soft11[labeled_bs:]), avg2[labeled_bs:]), dim=1, keepdim=True)
                exp_variance_aux11 = torch.exp(-variance_aux11)
                consistency_dist_aux11 = (out_soft11[labeled_bs:] - avg2[labeled_bs:]) ** 2
                consistency_loss_aux11 = torch.mean(consistency_dist_aux11 * exp_variance_aux11) / (torch.mean(exp_variance_aux11) + 1e-8) + torch.mean(variance_aux11)

                variance_aux12 = torch.sum(kl_distance(torch.log(out_soft12[labeled_bs:]), avg2[labeled_bs:]), dim=1, keepdim=True)
                exp_variance_aux12 = torch.exp(-variance_aux12)
                consistency_dist_aux12 = (out_soft12[labeled_bs:] - avg2[labeled_bs:]) ** 2
                consistency_loss_aux12 = torch.mean(consistency_dist_aux12 * exp_variance_aux12) / (torch.mean(exp_variance_aux12) + 1e-8) + torch.mean(variance_aux12)

                variance_aux13 = torch.sum(kl_distance(torch.log(out_soft13[labeled_bs:]), avg2[labeled_bs:]), dim=1, keepdim=True)
                exp_variance_aux13 = torch.exp(-variance_aux13)
                consistency_dist_aux13 = (out_soft13[labeled_bs:] - avg2[labeled_bs:]) ** 2
                consistency_loss_aux13 = torch.mean(consistency_dist_aux13 * exp_variance_aux13) / (torch.mean(exp_variance_aux13) + 1e-8) + torch.mean(variance_aux13)

                variance_aux14 = torch.sum(kl_distance(torch.log(out_soft14[labeled_bs:]), avg2[labeled_bs:]), dim=1, keepdim=True)
                exp_variance_aux14 = torch.exp(-variance_aux14)
                consistency_dist_aux14 = (out_soft14[labeled_bs:] - avg2[labeled_bs:]) ** 2
                consistency_loss_aux14 = torch.mean(consistency_dist_aux14 * exp_variance_aux14) / (torch.mean(exp_variance_aux14) + 1e-8) + torch.mean(variance_aux14)

                consistency_loss1 = (consistency_loss_aux11 + consistency_loss_aux12 + consistency_loss_aux13 + consistency_loss_aux14) / 4

                variance_aux21 = torch.sum(kl_distance(torch.log(out_soft21[labeled_bs:]), avg1[labeled_bs:]), dim=1, keepdim=True)
                exp_variance_aux21 = torch.exp(-variance_aux21)
                consistency_dist_aux21 = (out_soft21[labeled_bs:] - avg1[labeled_bs:]) ** 2
                consistency_loss_aux21 = torch.mean(consistency_dist_aux21 * exp_variance_aux21) / (torch.mean(exp_variance_aux21) + 1e-8) + torch.mean(variance_aux21)

                variance_aux22 = torch.sum(kl_distance(torch.log(out_soft22[labeled_bs:]), avg1[labeled_bs:]), dim=1, keepdim=True)
                exp_variance_aux22 = torch.exp(-variance_aux22)
                consistency_dist_aux22 = (out_soft22[labeled_bs:] - avg1[labeled_bs:]) ** 2
                consistency_loss_aux22 = torch.mean(consistency_dist_aux22 * exp_variance_aux22) / (torch.mean(exp_variance_aux22) + 1e-8) + torch.mean(variance_aux22)

                variance_aux23 = torch.sum(kl_distance(torch.log(out_soft23[labeled_bs:]), avg1[labeled_bs:]), dim=1, keepdim=True)
                exp_variance_aux23 = torch.exp(-variance_aux23)
                consistency_dist_aux23 = (out_soft23[labeled_bs:] - avg1[labeled_bs:]) ** 2
                consistency_loss_aux23 = torch.mean(consistency_dist_aux23 * exp_variance_aux23) / (torch.mean(exp_variance_aux23) + 1e-8) + torch.mean(variance_aux23)

                variance_aux24 = torch.sum(kl_distance(torch.log(out_soft24[labeled_bs:]), avg1[labeled_bs:]), dim=1, keepdim=True)
                exp_variance_aux24 = torch.exp(-variance_aux24)
                consistency_dist_aux24 = (out_soft24[labeled_bs:] - avg1[labeled_bs:]) ** 2
                consistency_loss_aux24 = torch.mean(consistency_dist_aux24 * exp_variance_aux24) / (torch.mean(exp_variance_aux24) + 1e-8) + torch.mean(variance_aux24)

                consistency_loss2 = (consistency_loss_aux21 + consistency_loss_aux22 + consistency_loss_aux23 + consistency_loss_aux24) / 4
            
            # 2.4.total loss = supervised loss + pseudo loss + pyramid consistency loss
            consistency_weight = get_current_consistency_weight(iter_num // 150)
            loss_model1 = supervise_loss1 + consistency_weight * (pseudo_supervision1 + consistency_loss1)
            loss_model2 = supervise_loss2 + consistency_weight * (pseudo_supervision2 + consistency_loss2)
            loss = (supervise_loss1 + supervise_loss2) + consistency_weight * (pseudo_supervision1 + pseudo_supervision2 + consistency_loss1 + consistency_loss2)

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
            writer.add_scalar('train/loss', loss, iter_num)

            writer.add_scalar('loss1/loss_model1', loss_model1, iter_num)
            writer.add_scalar('loss1/loss_seg', supervise_loss1, iter_num)
            writer.add_scalar('loss1/loss_pseudo_supervision1', pseudo_supervision1, iter_num)
            writer.add_scalar('loss1/consistency_loss1', consistency_loss1, iter_num)

            writer.add_scalar('loss2/loss_model2', loss_model2, iter_num)
            writer.add_scalar('loss2/loss_seg', supervise_loss2, iter_num)
            writer.add_scalar('loss2/loss_pseudo_supervision2', pseudo_supervision2, iter_num)
            writer.add_scalar('loss2/consistency_loss2', consistency_loss2, iter_num)

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
                pred_img =  torch.div(out_pseodu14[:4] * 255, num_classes - 1).repeat(1, 3, 1, 1)
                grid_img = make_grid(pred_img, 4, normalize=True)
                writer.add_image('train/Model1_Prediction', grid_img, iter_num)

                # [4, 1, 512, 512] -> [4, 3, 512, 512]
                pred_img =  torch.div(out_pseodu24[:4] * 255, num_classes - 1).repeat(1, 3, 1, 1)
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
                        writer.add_scalar(f'info/model1_val_class{class_i+1}_acc', metric_avg[class_i, 1], iter_num)
                class_mean_dice1 = np.mean(metric_avg, axis=0)[0]
                class_mean_acc1 = np.mean(metric_avg, axis=0)[1]
                writer.add_scalar('info/model1_val_class_mean_dice', class_mean_dice1, iter_num)
                writer.add_scalar('info/model1_val_class_mean_acc', class_mean_acc1, iter_num)

                if class_mean_dice1 > best_performance1:
                    best_performance1 = class_mean_dice1
                    save_mode_path = f'{snapshot_path}model1_iter_{iter_num}_dice_{round(class_mean_dice1, 4)}_acc_{round(class_mean_acc1, 2)}.pth'
                    save_best_path = f'{snapshot_path}{args.nets}_best_model1.pth'
                    torch.save(model1.state_dict(), save_mode_path)
                    torch.save(model1.state_dict(), save_best_path)

                logging.info(f'iteration {iter_num} : model1_mean_dice : {class_mean_dice1} model1_mean_acc : {class_mean_acc1}')

                model1.train()

                model2.eval()

                # avg_metric: [(nclass-1), nmetrics]; total_metric: [ncase, (nclass-1), nmetrics]
                metric_avg, metric_total = calculate_all_cases(model2, valid_dataloader, num_classes, save_result=False, test_save_path=None)
                if num_classes > 2: # >2 add more scalars
                    for class_i in range(num_classes - 1):
                        writer.add_scalar(f'info/model2_val_class{class_i+1}_dice', metric_avg[class_i, 0], iter_num)
                        writer.add_scalar(f'info/model2_val_class{class_i+1}_acc', metric_avg[class_i, 1], iter_num)
                class_mean_dice2 = np.mean(metric_avg, axis=0)[0]
                class_mean_acc2 = np.mean(metric_avg, axis=0)[1]
                writer.add_scalar('info/model2_val_class_mean_dice', class_mean_dice2, iter_num)
                writer.add_scalar('info/model2_val_class_mean_acc', class_mean_acc2, iter_num)

                if class_mean_dice2 > best_performance2:
                    best_performance2 = class_mean_dice2
                    save_mode_path = f'{snapshot_path}model2_iter_{iter_num}_dice_{round(class_mean_dice2, 4)}_acc_{round(class_mean_acc2, 2)}.pth'
                    save_best_path = f'{snapshot_path}{args.nets}_best_model2.pth'
                    torch.save(model2.state_dict(), save_mode_path)
                    torch.save(model2.state_dict(), save_best_path)

                logging.info(f'iteration {iter_num} : model2_mean_dice : {class_mean_dice2} model2_mean_acc : {class_mean_acc2}')

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
