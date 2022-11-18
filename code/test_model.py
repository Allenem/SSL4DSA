import os, sys, argparse, torch, random, cv2, logging, ml_collections
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from networks import vnet, unet, vnet4out, unet4out, unetpp, attunet, swinunet, isunetv1, isunetv2, isunetv3, isunetv4, isunetv5
from dataset import CoronaryArtery, RandomRotFlip, RandomCrop, CenterCrop, ToTensor, TwoStreamBatchSampler
from tqdm import tqdm
from medpy import metric
from sklearn import metrics
from datetime import datetime


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../data/CA_DSA/LCA/', help='Name of dataset')
parser.add_argument('--model', type=str, default='supervised/swinunet/LCA_20', help='Name of model')
parser.add_argument('--whichmodel', type=str, default=None, help='model1 or model2')
parser.add_argument('--gpus', type=str,  default='2', help='GPU to use')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
args = parser.parse_args()


os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
snapshot_path = f'../model/{args.model}/'
whichmodel = args.whichmodel
test_save_path = f'../model/predictions/{args.model}{whichmodel}_post/' if whichmodel else f'../model/predictions/{args.model}_post/'
if not os.path.exists(test_save_path):
    os.makedirs(test_save_path)
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

    for sample_batch in tqdm(testdataloader, ncols=70):

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


def test_main(net, saved_iter_num, testdataloader, numclasses, test_save_path):

    # saved_mode_path = f'{snapshot_path}vnet_swinunet_best_{whichmodel}.pth' if whichmodel else f'{snapshot_path}iter_{str(saved_iter_num)}.pth'
    # saved_mode_path = f'{snapshot_path}vnet_best_{whichmodel}.pth' if whichmodel else f'{snapshot_path}iter_{str(saved_iter_num)}.pth'
    saved_mode_path = f'{snapshot_path}{args.model.split("/")[-2]}_best_{whichmodel}.pth' if whichmodel else f'{snapshot_path}iter_{str(saved_iter_num)}.pth'
    net.load_state_dict(torch.load(saved_mode_path))
    print(f'Init weights from {saved_mode_path}')
    net.eval()
    metric_avg, metric_total = calculate_all_cases(net, testdataloader, 
        numclasses=numclasses, save_result=True, test_save_path=test_save_path)
    return metric_avg, metric_total


if __name__ == '__main__':

    t0 = datetime.now()
    test_dataset = CoronaryArtery(
        root=args.root_path,
        split='test',
        transform = transforms.ToTensor()
    )
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=worker_init_fn
    )
    
    if whichmodel:
        if whichmodel == 'model1':
            modelname = args.model.split('/')[-2].split('_')[0]
        elif whichmodel == 'model2':
            modelname = args.model.split('/')[-2].split('_')[1]
    else:
        modelname = args.model.split('/')[-2]
    net = creat_model(modelname)

    metric_avg, metric_total = test_main(
        net, saved_iter_num=6000, testdataloader=test_dataloader, numclasses=num_classes, test_save_path=test_save_path)
    t1 = datetime.now()

    logging.basicConfig(filename=test_save_path+'/log.txt', level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    # logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(f'\nSingle metrics are : \n{metric_total}') # [ncases, nclass-1, nmetric] = [50, 1, 2]
    logging.info(f'\nAverage metrics are : \n{metric_avg}')  # [nclass-1, nmetric] = [1, 2]
    logging.info(f'\nTest data cost {t1 - t0}')
    print(f'\nAverage metrics are : \n{metric_avg*100}')  # [nclass-1, nmetric] = [1, 2]
    print(f'\nTest data cost {t1 - t0}')
