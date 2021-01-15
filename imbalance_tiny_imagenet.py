import torch
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import numpy as np

class TinyImageFolder(dset.ImageFolder):
    def __init__(self, **kwds):
        super().__init__(**kwds)

class IMBALANCE_TINY_IMAGENET(dset.ImageFolder):

    def __init__(self,
                 root,
                 imb_type='exp',
                 imb_factor=0.01,
                 rand_number=0,
                 train=True,
                 transform=None,
                 target_transform=None,
                 download=False):

        super(IMBALANCE_TINY_IMAGENET, self).__init__(root, transform)
        np.random.seed(rand_number)
        # [5000, 2997, .... 50]
        num_samples = len(self.targets)

        self.data = []
        self.targets = []
        for i in range(num_samples):
            self.data.append(self.__getitem__(i)[0])
            self.targets.append(self.__getitem__(i)[1])

        self.data = torch.stack(self.data)

        if train:
            self.cls_num = 200
            img_num_list = self.get_img_num_per_cls(self.cls_num,
                                                    imb_type,
                                                    imb_factor)
            self.gen_imbalanced_data(img_num_list)
        else:
            None

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        # self.data [50000, 32, 32, 3]
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list

if __name__ == '__main__':

    # tiny-imagenet for training
    is_train = True
    if is_train :
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                         std=[0.25, 0.25, 0.25])
        side = 64; padding = 8

        train_transform = transforms.Compose(
            [transforms.RandomCrop(side, padding=padding),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(), normalize])

        trainset = IMBALANCE_TINY_IMAGENET(root='./data/tiny-imagenet-200/train',
                                           transform=train_transform)
        trainloader = iter(trainset)
        data, label = next(trainloader)

    else:
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25])
        test_transform = transforms.Compose([transforms.ToTensor(), normalize])
        test_set = TinyImageFolder(root="./data/tiny-imagenet-200/val",
                                   transform=test_transform)

