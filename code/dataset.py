import os, cv2
import PIL.Image as Image
import numpy as np
import torch
from torch.utils.data import Dataset
import itertools
from torch.utils.data.sampler import Sampler
from torchvision import transforms


class CoronaryArtery(Dataset):
    '''Load image & label as sample'''

    def __init__(self, root=None, split='train', num=None, transform=None):
        self.root = root
        self.transform = transform
        self.sample = {}
        if split == 'train':
            with open(self.root + 'train.list', 'r') as f:
            # with open('./data/train.list', 'r') as f:
                self.image_list = f.readlines()
        elif split == 'test':
            with open(self.root + 'test.list', 'r') as f:
            # with open('./data/test.list', 'r') as f:
                self.image_list = f.readlines()
        if num: self.image_list = self.image_list[:num]
        print(f'total {len(self.image_list)} samples for {split}')
    
    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = str(self.image_list[idx]).replace('\n', '').replace('\r', '')
        # path_x, path_y = f'{self.root}data/{img_name}.bmp', f'{self.root}label/{img_name}.bmp'
        path_x, path_y = f'{self.root}enhance/{img_name}.bmp', f'{self.root}label/{img_name}.bmp'
        img_x, img_y = cv2.imread(path_x, 0), cv2.imread(path_y, 0)
        # print(img_x.shape, img_y.shape) # (512, 512) (512, 512)
        # img_x, img_y = Image.open(path_x), Image.open(path_y)
        # print(img_x.shape, img_y.shape) # (512, 512) (512, 512)
        self.sample = {'image': img_x, 'label': img_y}
        if self.transform:
            # self.sample = self.transform(self.sample)
            seed = np.random.randint(2147483647)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            img_x = self.transform(img_x)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            img_y = self.transform(img_y)
            
        self.sample = {'image': img_x, 'label': img_y}
        # print(img_x.shape, img_y.shape) # [1, 512, 512] [1, 512, 512]
        # print(img_x.dtype, img_y.dtype) # torch.float32 torch.float32
        
        return self.sample


class RandomRotFlip(object):
    '''Random rotate 90 0~3 times, flip updown or leftright'''
    
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        k1, axis= np.random.randint(0, 4), np.random.randint(0, 2)
        image = np.rot90(image, k1)
        label = np.rot90(label, k1)
        image = np.flip(image, axis=axis).copy()
        label = np.flip(label, axis=axis).copy()
        return {'image': image, 'label': label}


class CenterCrop(object):
    '''Crop sample as desired size from sample center'''

    def __init__(self, out_size):
        self.out_size = out_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        # pad the sample if necessary
        if label.shape[0] <= self.out_size[0] or label.shape[1] <= self.out_size[1]:
            pw = max((self.out_size[0] - label.shape[0]) // 2, 0)
            ph = max((self.out_size[1] - label.shape[1]) // 2, 0)
            image = np.pad(image, [(pw, pw), (ph, ph)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph)], mode='constant', constant_values=0)
        
        (w, h) = image.shape

        w_start = int(round((w - self.out_size[0]) / 2.))
        h_start = int(round((h - self.out_size[1]) / 2.))

        image = image[w_start:w_start + self.out_size[0], h_start:h_start + self.out_size[1]]
        label = label[w_start:w_start + self.out_size[0], h_start:h_start + self.out_size[1]]

        return {'image': image, 'label': label}


class RandomCrop(object):
    '''Crop sample as desired size from sample random position'''

    def __init__(self, out_size):
        self.out_size = out_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        # pad the sample if necessary
        if label.shape[0] <= self.out_size[0] or label.shape[1] <= self.out_size[1]:
            pw = max((self.out_size[0] - label.shape[0]) // 2, 0)
            ph = max((self.out_size[1] - label.shape[1]) // 2, 0)
            image = np.pad(image, [(pw, pw), (ph, ph)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph)], mode='constant', constant_values=0)
        
        (w, h) = image.shape

        w_start = 0 if w == self.out_size[0] else np.random.randint(0, w - self.out_size[0])
        h_start = 0 if h == self.out_size[1] else np.random.randint(0, h - self.out_size[1])

        image = image[w_start:w_start + self.out_size[0], h_start:h_start + self.out_size[1]]
        label = label[w_start:w_start + self.out_size[0], h_start:h_start + self.out_size[1]]

        return {'image': image, 'label': label}


class CreateOnehotLabel(object):
    '''Creat onhot_label according label'''

    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        onehot_label = np.zeros((self.num_classes, label.shape[0], label.shape[1], label.shape[2]), dtype=np.float32)
        for i in range(self.num_classes):
            onehot_label[i, :, :] = (label == i).astype(np.float32)
        return {'image': image, 'label': label,'onehot_label':onehot_label}


class ToTensor(object):
    '''Convert ndarrays in sample to Tensors.'''

    def __call__(self, sample):
        image = sample['image']
        image = image.reshape(1, image.shape[0], image.shape[1]).astype(np.float32)
        if 'onehot_label' in sample:
            return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long(),
                    'onehot_label': torch.from_numpy(sample['onehot_label']).long()}
        else:
            return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long()}


def iterate_once(iterable):
    '''Randomly permute a sequence, or return a permuted range.'''
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    '''Create an iterator, it can return all elements in the all iterations.'''
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    '''Collect data into fixed-length chunks or blocks'''
    # grouper('ABCDEFG', 3) --> ABC DEF
    args = [iter(iterable)] * n
    # print(args)
    return zip(*args)


class TwoStreamBatchSampler(Sampler):
    '''Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated throughas many times as needed.
    '''
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.primary_batch_size = batch_size - secondary_batch_size
        self.secondary_batch_size = secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                    grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size
