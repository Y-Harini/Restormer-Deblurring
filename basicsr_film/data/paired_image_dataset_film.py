from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.data_util import (paired_paths_from_folder,
                                    paired_DP_paths_from_folder,
                                    paired_paths_from_lmdb,
                                    paired_paths_from_meta_info_file)
from basicsr.data.transforms import augment, paired_random_crop, paired_random_crop_DP, random_augmentation
from basicsr.utils import FileClient, imfrombytes, img2tensor, padding, padding_DP, imfrombytesDP

import random
import numpy as np
import torch
import cv2


def random_affine_pair(
    img_gt,
    img_lq,
    degrees=35,
    translate=0.30,
    scale_x_range=(0.60, 1.40),
    scale_y_range=(0.60, 1.40)
):
    """
    Apply the same random affine transformation to GT and LQ images.

    Returns:
        gt_aug: Transformed ground-truth image.
        lq_aug: Transformed low-quality image.
        affine_params: Dictionary containing the actual sampled parameters.
    """

    h, w = img_gt.shape[:2]

    # Sample transformation parameters
    angle = random.uniform(-degrees, degrees)
    scale_x = random.uniform(*scale_x_range)
    scale_y = random.uniform(*scale_y_range)

    tx = random.uniform(-translate, translate) * w
    ty = random.uniform(-translate, translate) * h

    # Construct affine transformation matrix
    theta = np.deg2rad(angle)

    # M = np.array([
    #     [scale_x * np.cos(theta), -scale_y * np.sin(theta), tx],
    #     [scale_x * np.sin(theta),  scale_y * np.cos(theta), ty]
    # ], dtype=np.float32)
    
    
    # Construct affine transformation around image center
    theta = np.deg2rad(angle)

    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    cx, cy = w / 2, h / 2

    matrix = np.array([
        [
            scale_x * cos_t,
            scale_x * sin_t,
            (1 - scale_x * cos_t) * cx
            - scale_x * sin_t * cy
        ],
        [
            -scale_y * sin_t,
            scale_y * cos_t,
            scale_y * sin_t * cx
            + (1 - scale_y * cos_t) * cy
        ]
    ], dtype=np.float32)

    matrix[0, 2] += tx
    matrix[1, 2] += ty

    M = matrix

    # Apply exactly the same transformation to GT and LQ
    gt_aug = cv2.warpAffine(
        img_gt,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101
    )

    lq_aug = cv2.warpAffine(
        img_lq,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101
    )

    # Store the actual sampled parameters
    affine_params = {
        'angle': angle,
        'tx_fraction': tx / w,
        'ty_fraction': ty / h,
        'scale_x': scale_x,
        'scale_y': scale_y
    }

    return gt_aug, lq_aug, affine_params


def random_gaussian_blur(img_lq, prob=1.0):
    """
    Apply Gaussian blur to the LQ image.

    Returns:
        lq_aug: Blurred or unchanged LQ image.
        blur_params: Dictionary containing the actual sampled parameters.
    """

    # Initialize neutral parameters
    blur_params = {
        'kernel_size': 0,
        'sigma': 0.0
    }

    # Skip blur according to probability
    if random.random() >= prob:
        return img_lq, blur_params

    # Sample blur parameters
    kernel_size = random.choice([13, 15, 17])
    sigma = random.uniform(3.0, 6.0)

    # Apply Gaussian blur
    lq_aug = cv2.GaussianBlur(
        img_lq,
        (kernel_size, kernel_size),
        sigmaX=sigma
    )

    blur_params = {
        'kernel_size': kernel_size,
        'sigma': sigma
    }

    return lq_aug, blur_params


def random_gaussian_noise(img_lq, prob=1.0, sigma_range=(20, 30)):
    """
    Add Gaussian noise to the LQ image.

    Returns:
        lq_aug: Noisy or unchanged LQ image.
        noise_params: Dictionary containing the actual sampled noise level.
    """

    # Initialize neutral parameters
    noise_params = {
        'sigma': 0.0
    }

    # Skip noise according to probability
    if random.random() >= prob:
        return img_lq, noise_params

    # Sample noise standard deviation on the 0-255 scale
    sigma = random.uniform(*sigma_range)

    # Convert sigma to the normalized 0-1 image scale
    noise = np.random.normal(
        0,
        sigma / 255.0,
        img_lq.shape
    ).astype(np.float32)

    # Add noise and clip values
    noisy_img = img_lq + noise
    noisy_img = np.clip(noisy_img, 0.0, 1.0)

    noise_params = {
        'sigma': sigma
    }

    return noisy_img, noise_params


class Dataset_PairedImageFiLM(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
    GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the
                template excludes the file extension. Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            geometric_augs (bool): Use geometric augmentations.

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(Dataset_PairedImageFiLM, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        
        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'])
        elif 'meta_info_file' in self.opt and self.opt[
                'meta_info_file'] is not None:
            self.paths = paired_paths_from_meta_info_file(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'],
                self.opt['meta_info_file'], self.filename_tmpl)
        else:
            self.paths = paired_paths_from_folder(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'],
                self.filename_tmpl)

        # if self.opt['phase'] == 'train':
        #     self.geometric_augs = opt['geometric_augs']
        
        if self.opt['phase'] == 'train':
            self.geometric_augs = opt['geometric_augs']
            self.affine_augs = opt.get('affine_augs', False)
            self.affine_degrees = opt.get('affine_degrees', 35)
            self.affine_translate = opt.get('affine_translate', 0.18)
            self.affine_scale_x_min = opt.get('affine_scale_x_min', 0.70)
            self.affine_scale_x_max = opt.get('affine_scale_x_max', 1.30)
            self.affine_scale_y_min = opt.get('affine_scale_y_min', 0.80)
            self.affine_scale_y_max = opt.get('affine_scale_y_max', 1.20)
            self.gaussian_blur_aug = opt.get('gaussian_blur_aug', False)
            self.gaussian_blur_prob = opt.get('gaussian_blur_prob', 0.5)
            self.gaussian_noise_aug = opt.get('gaussian_noise_aug', False)
            self.gaussian_noise_prob = opt.get('gaussian_noise_prob', 0.5)
            self.gaussian_noise_sigma_min = opt.get('gaussian_noise_sigma_min', 20)
            self.gaussian_noise_sigma_max = opt.get('gaussian_noise_sigma_max', 30)
    
    
    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt
            )

        scale = self.opt['scale']
        index = index % len(self.paths)

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')

        try:
            img_gt = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("gt path {} not working".format(gt_path))

        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')

        try:
            img_lq = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("lq path {} not working".format(lq_path))

        # --------------------------------------------------
        # Initialize neutral augmentation parameters
        # --------------------------------------------------
        affine_params = {
            'angle': 0.0,
            'tx_fraction': 0.0,
            'ty_fraction': 0.0,
            'scale_x': 1.0,
            'scale_y': 1.0
        }

        blur_params = {
            'kernel_size': 0,
            'sigma': 0.0
        }

        noise_params = {
            'sigma': 0.0
        }

        # --------------------------------------------------
        # Augmentations for training
        # --------------------------------------------------
        if self.opt['phase'] == 'train':

            gt_size = self.opt['gt_size']

            # Padding
            img_gt, img_lq = padding(img_gt, img_lq, gt_size)

            # Random crop
            img_gt, img_lq = paired_random_crop(
                img_gt, img_lq, gt_size, scale, gt_path
            )

            # Flip and rotation augmentations
            if self.geometric_augs:
                img_gt, img_lq = random_augmentation(img_gt, img_lq)

            # --------------------------------------------------
            # 1. Affine transformation
            # --------------------------------------------------
            if self.affine_augs:
                img_gt, img_lq, affine_params = random_affine_pair(
                    img_gt,
                    img_lq,
                    degrees=self.affine_degrees,
                    translate=self.affine_translate,
                    scale_x_range=(
                        self.affine_scale_x_min,
                        self.affine_scale_x_max
                    ),
                    scale_y_range=(
                        self.affine_scale_y_min,
                        self.affine_scale_y_max
                    )
                )

            # --------------------------------------------------
            # 2. Gaussian blur (LQ only)
            # --------------------------------------------------
            if self.gaussian_blur_aug:
                img_lq, blur_params = random_gaussian_blur(
                    img_lq,
                    prob=self.gaussian_blur_prob
                )

            # --------------------------------------------------
            # 3. Gaussian noise (LQ only)
            # --------------------------------------------------
            if self.gaussian_noise_aug:
                img_lq, noise_params = random_gaussian_noise(
                    img_lq,
                    prob=self.gaussian_noise_prob,
                    sigma_range=(
                        self.gaussian_noise_sigma_min,
                        self.gaussian_noise_sigma_max
                    )
                )

        # --------------------------------------------------
        # Construct 8-dimensional FiLM conditioning vector
        # --------------------------------------------------
        aug_params = torch.tensor([
            affine_params['angle'] / 35.0,
            affine_params['tx_fraction'] / 0.30,
            affine_params['ty_fraction'] / 0.30,
            (affine_params['scale_x'] - 1.0) / 0.40,
            (affine_params['scale_y'] - 1.0) / 0.40,
            blur_params['sigma'] / 6.0,
            blur_params['kernel_size'] / 17.0,
            noise_params['sigma'] / 30.0
        ], dtype=torch.float32)

        # --------------------------------------------------
        # BGR to RGB, HWC to CHW, numpy to tensor
        # --------------------------------------------------
        img_gt, img_lq = img2tensor(
            [img_gt, img_lq],
            bgr2rgb=True,
            float32=True
        )

        # Normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        # Return images and FiLM conditioning vector
        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': lq_path,
            'gt_path': gt_path,
            'aug_params': aug_params
        }
        
    def __len__(self):
        return len(self.paths)
   

class Dataset_GaussianDenoising(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
    GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            gt_size (int): Cropped patched size for gt patches.
            use_flip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(Dataset_GaussianDenoising, self).__init__()
        self.opt = opt

        if self.opt['phase'] == 'train':
            self.sigma_type  = opt['sigma_type']
            self.sigma_range = opt['sigma_range']
            assert self.sigma_type in ['constant', 'random', 'choice']
        else:
            self.sigma_test = opt['sigma_test']
        self.in_ch = opt['in_ch']

        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None        

        self.gt_folder = opt['dataroot_gt']

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.gt_folder]
            self.io_backend_opt['client_keys'] = ['gt']
            self.paths = paths_from_lmdb(self.gt_folder)
        elif 'meta_info_file' in self.opt:
            with open(self.opt['meta_info_file'], 'r') as fin:
                self.paths = [
                    osp.join(self.gt_folder,
                             line.split(' ')[0]) for line in fin
                ]
        else:
            self.paths = sorted(list(scandir(self.gt_folder, full_path=True)))

        if self.opt['phase'] == 'train':
            self.geometric_augs = self.opt['geometric_augs']

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        index = index % len(self.paths)
        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')

        if self.in_ch == 3:
            try:
                img_gt = imfrombytes(img_bytes, float32=True)
            except:
                raise Exception("gt path {} not working".format(gt_path))

            img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)
        else:
            try:
                img_gt = imfrombytes(img_bytes, flag='grayscale', float32=True)
            except:
                raise Exception("gt path {} not working".format(gt_path))

            img_gt = np.expand_dims(img_gt, axis=2)
        img_lq = img_gt.copy()


        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # padding
            img_gt, img_lq = padding(img_gt, img_lq, gt_size)

            # random crop
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale,
                                                gt_path)
            # flip, rotation
            if self.geometric_augs:
                img_gt, img_lq = random_augmentation(img_gt, img_lq)

            img_gt, img_lq = img2tensor([img_gt, img_lq],
                                        bgr2rgb=False,
                                        float32=True)


            if self.sigma_type == 'constant':
                sigma_value = self.sigma_range
            elif self.sigma_type == 'random':
                sigma_value = random.uniform(self.sigma_range[0], self.sigma_range[1])
            elif self.sigma_type == 'choice':
                sigma_value = random.choice(self.sigma_range)

            noise_level = torch.FloatTensor([sigma_value])/255.0
            # noise_level_map = torch.ones((1, img_lq.size(1), img_lq.size(2))).mul_(noise_level).float()
            noise = torch.randn(img_lq.size()).mul_(noise_level).float()
            img_lq.add_(noise)

        else:            
            np.random.seed(seed=0)
            img_lq += np.random.normal(0, self.sigma_test/255.0, img_lq.shape)
            # noise_level_map = torch.ones((1, img_lq.shape[0], img_lq.shape[1])).mul_(self.sigma_test/255.0).float()

            img_gt, img_lq = img2tensor([img_gt, img_lq],
                            bgr2rgb=False,
                            float32=True)

        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': gt_path,
            'gt_path': gt_path
        }

    def __len__(self):
        return len(self.paths)

class Dataset_DefocusDeblur_DualPixel_16bit(data.Dataset):
    def __init__(self, opt):
        super(Dataset_DefocusDeblur_DualPixel_16bit, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        
        self.gt_folder, self.lqL_folder, self.lqR_folder = opt['dataroot_gt'], opt['dataroot_lqL'], opt['dataroot_lqR']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        self.paths = paired_DP_paths_from_folder(
            [self.lqL_folder, self.lqR_folder, self.gt_folder], ['lqL', 'lqR', 'gt'],
            self.filename_tmpl)

        if self.opt['phase'] == 'train':
            self.geometric_augs = self.opt['geometric_augs']
            
    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        index = index % len(self.paths)
        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        try:
            img_gt = imfrombytesDP(img_bytes, float32=True)
        except:
            raise Exception("gt path {} not working".format(gt_path))

        lqL_path = self.paths[index]['lqL_path']
        img_bytes = self.file_client.get(lqL_path, 'lqL')
        try:
            img_lqL = imfrombytesDP(img_bytes, float32=True)
        except:
            raise Exception("lqL path {} not working".format(lqL_path))

        lqR_path = self.paths[index]['lqR_path']
        img_bytes = self.file_client.get(lqR_path, 'lqR')
        try:
            img_lqR = imfrombytesDP(img_bytes, float32=True)
        except:
            raise Exception("lqR path {} not working".format(lqR_path))


        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # padding
            img_lqL, img_lqR, img_gt = padding_DP(img_lqL, img_lqR, img_gt, gt_size)

            # random crop
            img_lqL, img_lqR, img_gt = paired_random_crop_DP(img_lqL, img_lqR, img_gt, gt_size, scale, gt_path)
            
            # flip, rotation            
            if self.geometric_augs:
                img_lqL, img_lqR, img_gt = random_augmentation(img_lqL, img_lqR, img_gt)
        # TODO: color space transform
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_lqL, img_lqR, img_gt = img2tensor([img_lqL, img_lqR, img_gt],
                                    bgr2rgb=True,
                                    float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lqL, self.mean, self.std, inplace=True)
            normalize(img_lqR, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        img_lq = torch.cat([img_lqL, img_lqR], 0)
        
        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': lqL_path,
            'gt_path': gt_path
        }

    def __len__(self):
        return len(self.paths)
