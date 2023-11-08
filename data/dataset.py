import sys
import torch.utils.data as torchdata
from os import listdir
import os
import pandas as pd
import torchvision.transforms as transforms
import torch
import numpy as np
import multiprocessing

from utils.tools import default_loader, is_image_file, normalize

storeOnRAM = True

class Dataset(torchdata.Dataset):
    def __init__(self, data_path, image_shape, with_subfolder=False, random_crop=True, return_name=False):
        super(Dataset, self).__init__()
        if with_subfolder:
            self.samples = self._find_samples_in_subfolders(data_path)
        else:
            self.samples = [x for x in listdir(data_path) if is_image_file(x)]
        self.data_path = data_path
        self.image_shape = image_shape[:-1]
        self.random_crop = random_crop
        self.return_name = return_name

        if storeOnRAM:
            print(f'Getting {len(self.samples)} samples...')
            paths = [os.path.join(self.data_path, sample) for sample in self.samples]
            with multiprocessing.Pool(8) as pool:
                self.samples_img = pool.map(self.load_item, paths)
            print('Got samples')

    def __getitem__(self, index):
        if not storeOnRAM:
            path = os.path.join(self.data_path, self.samples[index])
            img = self.load_item(path)
        else:
            img = self.samples_img[index]

        img = transforms.RandomCrop(self.image_shape)(img)

        """from matplotlib import pyplot as plt
        plt.imshow(img, 'gray')
        plt.show()"""

        img = transforms.ToTensor()(img)  # turn the image to a tensor
        img = normalize(img)

        if self.return_name:
            return self.samples[index], img
        else:
            return img

    def load_item(self, path):
        img = default_loader(path)
        if self.random_crop:
            imgw, imgh = img.size
            if imgh < self.image_shape[0] or imgw < self.image_shape[1]:
                img = transforms.Resize(min(self.image_shape))(img)
        else:
            img = transforms.Resize(self.image_shape)(img)
        return img

    def _find_samples_in_subfolders(self, dir):
        """
        Finds the class folders in a dataset.
        Args:
            dir (string): Root directory path.
        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.
        Ensures:
            No class is a subdirectory of another.
        """
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        samples = []
        for target in sorted(class_to_idx.keys()):
            d = os.path.join(dir, target)
            if not os.path.isdir(d):
                continue
            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    if is_image_file(fname):
                        path = os.path.join(root, fname)
                        # item = (path, class_to_idx[target])
                        # samples.append(item)
                        samples.append(path)
        return samples

    def __len__(self):
        return len(self.samples)

class Dataset_BboxCsv(Dataset):
    def __init__(self, data_path, image_shape, bbox_csvfname, with_subfolder=False, random_crop=True,
                 return_name=False, min_mask_shape=None):
        super(Dataset_BboxCsv, self).__init__(data_path, image_shape, with_subfolder=with_subfolder,
                                              random_crop=random_crop, return_name=return_name)
        self.bbox_csvfname = bbox_csvfname
        df = pd.read_csv(self.bbox_csvfname)
        self.samples = [fname[0] for fname in df[['filename']].to_numpy()]
        self.samples_bbox = df[['bbox_t', 'bbox_l', 'bbox_h', 'bbox_w']].to_numpy()
        self.samples_cl = df[['class_id']].to_numpy()
        self.samples_bbox = [bbox for bbox, s in zip(self.samples_bbox, self.samples) if s in os.listdir(data_path)]
        self.samples_cl = [cl for cl, s in zip(self.samples_cl, self.samples) if s in os.listdir(data_path)]
        self.samples = [s for s in self.samples if s in os.listdir(data_path)]

        self.data_path = data_path
        self.image_shape = image_shape[:-1]
        self.random_crop = random_crop
        self.return_name = return_name
        self.min_mask_shape = min_mask_shape

    def __getitem__(self, index):
        if self.return_name:
            s, img = super(Dataset_BboxCsv, self).__getitem__(index)
        else:
            img = super(Dataset_BboxCsv, self).__getitem__(index)

        bbox = self.samples_bbox[index]
        bbox = torch.Tensor([int(bbox[0] * img.shape[1]), int(bbox[1] * img.shape[2]),
                             int(bbox[2] * img.shape[1]), int(bbox[3] * img.shape[2])])

        if self.min_mask_shape is not None:
            if bbox[2] < self.min_mask_shape[0]:
                dh = (self.min_mask_shape[0] - bbox[2]) / 2
                if dh % 2 == 1:
                    dh +=1
                bbox[0] -=dh
                bbox[2] += dh * 2

            if bbox[3] < self.min_mask_shape[1]:
                dh = (self.min_mask_shape[1] - bbox[3]) / 2
                if dh % 2 == 1:
                    dh +=1
                bbox[1] -=dh
                bbox[3] += dh * 2

        if self.return_name:
            return s, img, bbox, self.samples_cl[index]
        else:
            return img, bbox, self.samples_cl[index]

class Dataset_RefCXR(Dataset):
    def __init__(self, data_path, image_shape, with_subfolder=False, random_crop=True, return_name=False, ref_func=None):
        super(Dataset_RefCXR, self).__init__(data_path, image_shape, with_subfolder=with_subfolder,
                                             random_crop=random_crop, return_name=return_name)

        if ref_func is None:
            self.ref_func = 'random_func'
        elif ref_func[-4:] == '.csv':
            self.ref_func = 'csv_func'
            df = pd.read_csv(f'.\\media\\{ref_func}')
            self.fnames_self = df['0'].to_numpy()
            self.fnames_nbor = df['1'].to_numpy()
            if storeOnRAM:
                print(f'Getting {len(self.samples)} reference samples...')
                paths = [self.fnames_nbor[np.where(self.fnames_self == sample)[0][0]] for sample in self.samples]
                with multiprocessing.Pool(8) as pool:
                    self.samples_imgnbor = pool.map(self.load_item, paths)
                print('Got samples')
        else:
            print('Reference CXR retrieval function unknown!!!')
            xxx

    def __getitem__(self, index):
        if not storeOnRAM:
            path = os.path.join(self.data_path, self.samples[index])
            img = self.load_item(path)
        else:
            img = self.samples_img[index]
        img1 = getattr(self, self.ref_func)(index)

        img = transforms.RandomCrop(self.image_shape)(img)
        img1 = transforms.RandomCrop(self.image_shape)(img1)

        """from matplotlib import pyplot as plt
        plt.imshow(img, 'gray')
        plt.show()"""

        img = transforms.ToTensor()(img)  # turn the image to a tensor
        img = normalize(img)
        img1 = transforms.ToTensor()(img1)  # turn the image to a tensor
        img1 = normalize(img1)

        if self.return_name:
            return self.samples[index], img, img1
        else:
            return img, img1

    def random_func(self, index):
        index1 = np.random.randint(len(self.samples))
        if not storeOnRAM:
            path = os.path.join(self.data_path, self.samples[index1])
            return self.load_item(path)
        else:
            return self.samples_img[index1]

    def csv_func(self, index):
        if not storeOnRAM:
            try:
                fname_index = np.where(self.fnames_self == self.samples[index])[0][0]
            except:
                print(f'Could not find {self.samples[index]} on csv!!!')
                xxx
            return self.load_item(self.fnames_nbor[fname_index])
        else:
            return self.samples_imgnbor[index]

class Dataset_BboxCsv_RefCXR(Dataset_RefCXR):
    def __init__(self, data_path, image_shape, bbox_csvfname, with_subfolder=False, random_crop=True,
                 return_name=False, ref_func=None, min_mask_shape=None):
        super(Dataset_BboxCsv_RefCXR, self).__init__(data_path, image_shape, with_subfolder=with_subfolder,
                                              random_crop=random_crop, return_name=return_name, ref_func=ref_func)
        self.bbox_csvfname = bbox_csvfname
        df = pd.read_csv(self.bbox_csvfname)
        self.samples = [fname[0] for fname in df[['filename']].to_numpy()]
        self.samples_bbox = df[['bbox_t', 'bbox_l', 'bbox_h', 'bbox_w']].to_numpy()
        self.samples_cl = df[['class_id']].to_numpy()
        self.samples_bbox = [bbox for bbox, s in zip(self.samples_bbox, self.samples) if s in os.listdir(data_path)]
        self.samples_cl = [cl for cl, s in zip(self.samples_cl, self.samples) if s in os.listdir(data_path)]
        self.samples = [s for s in self.samples if s in os.listdir(data_path)]

        self.data_path = data_path
        self.image_shape = image_shape[:-1]
        self.random_crop = random_crop
        self.return_name = return_name
        self.min_mask_shape = min_mask_shape

    def __getitem__(self, index):
        if self.return_name:
            s, img, img1 = super(Dataset_BboxCsv_RefCXR, self).__getitem__(index)
        else:
            img, img1 = super(Dataset_BboxCsv_RefCXR, self).__getitem__(index)

        bbox = self.samples_bbox[index]
        bbox = torch.Tensor([int(bbox[0] * img.shape[1]), int(bbox[1] * img.shape[2]),
                             int(bbox[2] * img.shape[1]), int(bbox[3] * img.shape[2])])

        if self.min_mask_shape is not None:
            if bbox[2] < self.min_mask_shape[0]:
                dh = (self.min_mask_shape[0] - bbox[2]) / 2
                if dh % 2 == 1:
                    dh +=1
                bbox[0] -=dh
                bbox[2] += dh * 2

            if bbox[3] < self.min_mask_shape[1]:
                dh = (self.min_mask_shape[1] - bbox[3]) / 2
                if dh % 2 == 1:
                    dh +=1
                bbox[1] -=dh
                bbox[3] += dh * 2

        if self.return_name:
            return s, img, img1, bbox, self.samples_cl[index]
        else:
            return img, img1, bbox, self.samples_cl[index]

class Dataset_MaskCXR(Dataset):
    def __init__(self, data_path, image_shape, with_subfolder=False, random_crop=True, return_name=False, addmask_suff=None):
        super(Dataset_MaskCXR, self).__init__(data_path, image_shape, with_subfolder=with_subfolder,
                                             random_crop=random_crop, return_name=return_name)

        self.addmask_suff = addmask_suff

        if storeOnRAM:
            print(f'Getting {len(self.samples)} mask samples...')
            paths = [os.path.join(f'{self.data_path}_{self.addmask_suff}', sample) for sample in self.samples]
            with multiprocessing.Pool(8) as pool:
                self.samples_mask = pool.map(self.load_item, paths)
            print('Got samples')

    def __getitem__(self, index):
        if not storeOnRAM:
            path = os.path.join(self.data_path, self.samples[index])
            img = self.load_item(path)
            path_mask = os.path.join(f'{self.data_path}_{self.addmask_suff}', self.samples[index])
            mask = self.load_item(path_mask)
        else:
            img = self.samples_img[index]
            mask = self.samples_mask[index]

        img = transforms.RandomCrop(self.image_shape)(img)
        mask = transforms.RandomCrop(self.image_shape)(mask)

        """from matplotlib import pyplot as plt
        import matplotlib
        matplotlib.use('TkAgg')
        plt.figure()
        plt.imshow(img, 'gray')
        plt.figure()
        plt.imshow(mask, 'gray')
        plt.show()"""

        img = transforms.ToTensor()(img)  # turn the image to a tensor
        img = normalize(img)
        mask = transforms.ToTensor()(mask)  # turn the image to a tensor
        mask = normalize(mask)

        if self.return_name:
            return self.samples[index], img, mask
        else:
            return img, mask

class Dataset_BboxCsv_MaskCXR(Dataset_MaskCXR):
    def __init__(self, data_path, image_shape, bbox_csvfname, with_subfolder=False, random_crop=True,
                 return_name=False, addmask_suff=None, min_mask_shape=None):
        super(Dataset_BboxCsv_MaskCXR, self).__init__(data_path, image_shape, with_subfolder=with_subfolder,
                                                     random_crop=random_crop, return_name=return_name,
                                                     addmask_suff=addmask_suff)
        self.bbox_csvfname = bbox_csvfname
        df = pd.read_csv(self.bbox_csvfname)
        self.samples = [fname[0] for fname in df[['filename']].to_numpy()]
        self.samples_bbox = df[['bbox_t', 'bbox_l', 'bbox_h', 'bbox_w']].to_numpy()
        self.samples_cl = df[['class_id']].to_numpy()
        self.samples_bbox = [bbox for bbox, s in zip(self.samples_bbox, self.samples) if
                             s in os.listdir(data_path)]
        self.samples_cl = [cl for cl, s in zip(self.samples_cl, self.samples) if s in os.listdir(data_path)]
        self.samples = [s for s in self.samples if s in os.listdir(data_path)]

        self.data_path = data_path
        self.image_shape = image_shape[:-1]
        self.random_crop = random_crop
        self.return_name = return_name
        self.min_mask_shape = min_mask_shape

    def __getitem__(self, index):
        if self.return_name:
            s, img, img1 = super(Dataset_BboxCsv_MaskCXR, self).__getitem__(index)
        else:
            img, img1 = super(Dataset_BboxCsv_MaskCXR, self).__getitem__(index)

        bbox = self.samples_bbox[index]
        bbox = torch.Tensor([int(bbox[0] * img.shape[1]), int(bbox[1] * img.shape[2]),
                             int(bbox[2] * img.shape[1]), int(bbox[3] * img.shape[2])])

        if self.min_mask_shape is not None:
            if bbox[2] < self.min_mask_shape[0]:
                dh = (self.min_mask_shape[0] - bbox[2]) / 2
                if dh % 2 == 1:
                    dh += 1
                bbox[0] -= dh
                bbox[2] += dh * 2

            if bbox[3] < self.min_mask_shape[1]:
                dh = (self.min_mask_shape[1] - bbox[3]) / 2
                if dh % 2 == 1:
                    dh += 1
                bbox[1] -= dh
                bbox[3] += dh * 2

        if self.return_name:
            return s, img, img1, bbox, self.samples_cl[index]
        else:
            return img, img1, bbox, self.samples_cl[index]

