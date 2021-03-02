import tarfile
import urllib
import PIL
import os
import numpy
import torch
import albumentations as A

class VOCdataset:
    def __init__(self, split='train', root='datasets',
                 aug=A.Compose([
                    A.PadIfNeeded(min_height=256, min_width=256),
                    A.RandomCrop(height=256, width=256),
                    A.HorizontalFlip(p=0.5),
                    A.RandomBrightnessContrast(p=0.2),
                    A.ShiftScaleRotate(p=0.5),
                    A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
                    A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
                    A.Blur(blur_limit=3)
                 ])):
        self.root = root
        self.aug = aug
        tar_path = os.path.join(root, 'VOC/VOC.zip')
        dir_path = os.path.join(root, 'VOC')
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            url = "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"
            with open(tar_path, 'wb+') as fout:
                fout.write(urllib.request.urlopen(url).read())
                fout.close()

        if os.path.exists(tar_path):
            with tarfile.open(tar_path) as tar_ref:
                tar_ref.extractall(dir_path)

            os.remove(tar_path)

        self.images = []
        for line in open(os.path.join(dir_path, 'VOCdevkit', 'VOC2012',
                                      'ImageSets', 'Segmentation',
                                      split + '.txt')):
            self.images.append(line.strip())

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_name = self.images[index]
        img_path = os.path.join(self.root, 'VOC', 'VOCdevkit', 'VOC2012',
                                'JPEGImages', img_name + '.jpg')
        seg_path = os.path.join(self.root, 'VOC', 'VOCdevkit', 'VOC2012',
                                'SegmentationClass', img_name + '.png')
        img = numpy.array(PIL.Image.open(img_path))
        seg = numpy.array(PIL.Image.open(seg_path))

        res = self.aug(image=img, mask=seg)

        img = torch.tensor(res['image']).permute(2, 0, 1)
        seg = torch.tensor(res['mask'])
        seg[seg == 255] = 0

        return {'image': img.float() / 255.0, 'seg': seg.long()}
