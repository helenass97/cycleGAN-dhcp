import torch
from torch.utils.data import Dataset, Subset
from torch.utils.data.sampler import SubsetRandomSampler
import os
import numpy as np
import torchvision
import scipy.misc
import imageio
import pickle
import imgaug as ia
import imgaug.augmenters as iaa
import matplotlib.pyplot as plt
import pandas as pd
from skimage.transform import rescale, resize, downscale_local_mean
import SimpleITK as sitk
from dhcp_dataloader import *
import torchvision.transforms as transforms
import random

def line_best_fit(X, Y):

    xbar = sum(X)/len(X)
    ybar = sum(Y)/len(Y)
    n = len(X) # or len(Y)

    numer = sum([xi*yi for xi,yi in zip(X, Y)]) - n * xbar * ybar
    denum = sum([xi**2 for xi in X]) - n * xbar**2

    b = numer / denum
    a = ybar - b * xbar

    return a, b


def whitening(image):
    """Whitening. Normalises image to zero mean and unit variance."""

    image = image.astype(np.float32)

    mean = np.mean(image)
    std = np.std(image)

    if std > 0:
        ret = (image - mean) / std
    else:
        ret = image * 0.
    return ret

def normalise_zero_one_torch(image):
    """Image normalisation. Normalises image to fit [0, 1] range."""

    minimum = torch.min(image)
    maximum = torch.max(image)

    if maximum > minimum:
        ret = (image - minimum) / (maximum - minimum)
    else:
        ret = image * 0.
    return ret


def normalise_zero_one(image):
    """Image normalisation. Normalises image to fit [0, 1] range."""

    image = image.astype(np.float32)

    minimum = np.min(image)
    maximum = np.max(image)

    if maximum > minimum:
        ret = (image - minimum) / (maximum - minimum)
    else:
        ret = image * 0.
    return ret

def normalise_negative_one(image):
    """Image normalisation. Normalises image to fit [0, 1] range."""

    image = image.astype(np.float32)

    minimum = np.min(image)
    maximum = np.max(image)

    if maximum > minimum:
        ret = (2*(image - minimum) / (maximum-minimum)) - 1
    else:
        ret = image * 0.
    return ret

class NormMinMax(object):
    """Normalize image between 0 and 1.
    """

    def __init__(self):
        self.int = 1

    def __call__(self, data):
        image = data.astype(np.float32)

        minimum = np.min(image)
        maximum = np.max(image)

        if maximum > minimum:
            ret = (image - minimum) / (maximum - minimum)
        else:
            ret = image * 0.
        return ret


class ImgAugTransform:
    def __init__(self, gauss_noise, flip_lr, elastic):
        self.aug = iaa.Sequential([
            iaa.AdditiveGaussianNoise(loc=gauss_noise[0], scale=gauss_noise[1]),
            iaa.Fliplr(flip_lr),
            iaa.ElasticTransformation(elastic[0], elastic[1])])

    def __call__(self, image):
        image = np.array(image)
        return self.aug.augment_image(image)


class NormMeanSTD(object):
    """Normalize image with standardization technique.
    Args:
        data_mean (float): The dataset mean.
        data_std (float): The dataset std.
    """

    def __init__(self, data_mean=None, data_std=None):
        self.data_mean = data_mean
        self.data_std = data_std

    def __call__(self, data):
        if self.data_mean:
            data_norm = (data - self.data_mean) / self.data_std
        else:
            data_norm = data
        return data_norm

class ResizeImage(object):
    """
    Rescale image- default image size - [128, 160, 128]
    Args:
        new image size
    """
    def __init__(self, image_size=(128, 160, 128)):
        self.image_size = image_size

    def __call__(self, data):
        if len(self.image_size) == 2:
            image_resized = resize(data, (self.image_size[0], self.image_size[1]),
                                   anti_aliasing=True)
        else:
            image_resized = resize(data, (self.image_size[0], self.image_size[1], self.image_size[2]),
                                   anti_aliasing=True)

        return image_resized


class RicianNoiseNegOne(object):
    """Fourier transformed Gaussian Noise is Rician Noise.
    Args:
        noise_level (int): The amount of noise to add.
    """
    def __init__(self, noise_level):
        self.noise_level = noise_level
    def add_complex_noise(self, inverse_image, noise_level):
        # Convert the noise from decibels to a linear scale: See: http://www.mogami.com/e/cad/db.html
        noise_level_linear = 10 ** (noise_level / 10)
        # Real component of the noise: The noise "map" should span the entire image, hence the multiplication
        real_noise = np.sqrt(noise_level_linear / 2) * np.random.randn(inverse_image.shape[0],
                                                                       inverse_image.shape[1], inverse_image.shape[2])
        # Imaginary component of the noise: Note the 1j term
        imaginary_noise = np.sqrt(noise_level_linear / 2) * 1j * np.random.randn(inverse_image.shape[0],
                                                                                 inverse_image.shape[1], inverse_image.shape[2])
        noisy_inverse_image = inverse_image + real_noise + imaginary_noise
        return noisy_inverse_image
    def __call__(self, image):
        prob = random.uniform(0, 1)
        if prob > 0.5:
            if len(self.noise_level) == 2:
                noise_level = np.random.randint(self.noise_level[0], self.noise_level[1])
                noise_level = noise_level
            else:
                noise_level = self.noise_level[0]
            # normalize image betwwen 0-1
            norm_image = (image - image.min())/(image.max()-image.min())
            # Fourier transform the input image
            inverse_image = np.fft.fftn(norm_image)
            # Add complex noise to the image in k-space
            inverse_image_noisy = self.add_complex_noise(inverse_image, noise_level)
            # Reverse Fourier transform the image back into real space
            complex_image_noisy = np.fft.ifftn(inverse_image_noisy)
            # Calculate the magnitude of the image to get something entirely real
            magnitude_image_noisy = np.sqrt(np.real(complex_image_noisy) ** 2 + np.imag(complex_image_noisy) ** 2)
            # revert to original image range
            image_noisy = (image.max()-image.min())*magnitude_image_noisy+image.min()
        else:
            image_noisy = image
        return image_noisy


class RicianNoise(object):
    """Fourier transformed Gaussian Noise is Rician Noise.
    Args:
        noise_level (int): The amount of noise to add.
    """
    def __init__(self, noise_level):
        self.noise_level = noise_level

    def add_complex_noise(self, inverse_image, noise_level):
        # Convert the noise from decibels to a linear scale: See: http://www.mogami.com/e/cad/db.html
        noise_level_linear = 10 ** (noise_level / 10)
        # Real component of the noise: The noise "map" should span the entire image, hence the multiplication
        real_noise = np.sqrt(noise_level_linear / 2) * np.random.randn(inverse_image.shape[0],
                                                                       inverse_image.shape[1], inverse_image.shape[2])
        # Imaginary component of the noise: Note the 1j term
        imaginary_noise = np.sqrt(noise_level_linear / 2) * 1j * np.random.randn(inverse_image.shape[0],
                                                                                 inverse_image.shape[1], inverse_image.shape[2])
        noisy_inverse_image = inverse_image + real_noise + imaginary_noise
        return noisy_inverse_image

    def __call__(self, image):
        prob = random.uniform(0, 1)
        if prob > 0.5:
            if len(self.noise_level) == 2:
                noise_level = np.random.randint(self.noise_level[0], self.noise_level[1])
                noise_level = noise_level
            else:
                noise_level = self.noise_level[0]
            # Fourier transform the input image
            inverse_image = np.fft.fftn(image)
            # Add complex noise to the image in k-space
            inverse_image_noisy = self.add_complex_noise(inverse_image, noise_level)
            # Reverse Fourier transform the image back into real space
            complex_image_noisy = np.fft.ifftn(inverse_image_noisy)
            # Calculate the magnitude of the image to get something entirely real
            magnitude_image_noisy = np.sqrt(np.real(complex_image_noisy) ** 2 + np.imag(complex_image_noisy) ** 2)
        else:
            magnitude_image_noisy = image
        return magnitude_image_noisy

class ElasticDeformationsBspline(object):
    """Elastic deformation with a b-spline.
    Args:
        num_controlpoints (int):
        sigma (float): STD
    """
    def __init__(self, num_controlpoints=5, sigma=1):
        self.num_controlpoints = num_controlpoints
        self.sigma = sigma

    def create_elastic_deformation(self, image, num_controlpoints, sigma):
        """
        We need to parameterise our b-spline transform
        The transform will depend on such variables as image size and sigma
        Sigma modulates the strength of the transformation
        The number of control points controls the granularity of our transform
        """
        # Create an instance of a SimpleITK image of the same size as our image
        itkimg = sitk.GetImageFromArray(np.zeros(image.shape))
        # This parameter is just a list with the number of control points per image dimensions
        trans_from_domain_mesh_size = [num_controlpoints] * itkimg.GetDimension()
        # We initialise the transform here: Passing the image size and the control point specifications
        bspline_transformation = sitk.BSplineTransformInitializer(itkimg, trans_from_domain_mesh_size)
        # Isolate the transform parameters: They will be all zero at this stage
        params = np.asarray(bspline_transformation.GetParameters(), dtype=float)
        # Let's initialise the transform by randomly initialising each parameter according to sigma
        params = params + np.random.randn(params.shape[0]) * sigma
        bspline_transformation.SetParameters(tuple(params))
        return bspline_transformation

    def __call__(self, image):
        prob = random.uniform(0, 1)
        if prob > 0.5:
            if len(self.num_controlpoints) == 2:
                num_controlpoints = np.random.randint(self.num_controlpoints[0], self.num_controlpoints[1])
                num_controlpoints = num_controlpoints
            else:
                num_controlpoints = self.num_controlpoints[0]
            if len(self.sigma) == 2:
                sigma = np.random.uniform(self.sigma[0], self.sigma[1])
                sigma = sigma
            else:
                sigma = self.sigma[0]
            # We need to choose an interpolation method for our transformed image, let's just go with b-spline
            resampler = sitk.ResampleImageFilter()
            resampler.SetInterpolator(sitk.sitkBSpline)
            # Let's convert our image to an sitk image
            sitk_image = sitk.GetImageFromArray(image)
            # sitk_grid = self.create_grid(image)
            # Specify the image to be transformed: This is the reference image
            resampler.SetReferenceImage(sitk_image)
            resampler.SetDefaultPixelValue(0)
            # Initialise the transform
            bspline_transform = self.create_elastic_deformation(image, num_controlpoints, sigma)
            # Set the transform in the initialiser
            resampler.SetTransform(bspline_transform)
            # Carry out the resampling according to the transform and the resampling method
            out_img_sitk = resampler.Execute(sitk_image)
            # out_grid_sitk = resampler.Execute(sitk_grid)
            # Convert the image back into a python array
            out_img = sitk.GetArrayFromImage(out_img_sitk)
            out_img = out_img.reshape(image.shape)
        else:
            out_img = image
        return out_img


class GenHelper(Dataset):
    def __init__(self, mother, length, mapping):
        # here is a mapping from this index to the mother ds index
        self.mapping = mapping
        self.length = length
        self.mother = mother

    def __getitem__(self, index):
        return self.mother[self.mapping[index]]

    def __len__(self):
        return self.length

def train_valid_split(ds, split_fold=0.1, random_seed=None):
    '''
    This is a pytorch generic function that takes a data.Dataset object and splits it to validation and training
    efficiently.
    :return: train, val datasets
    '''
    if random_seed != None:
        np.random.seed(random_seed)

    dslen = len(ds)
    indices = list(range(dslen))
    valid_size = int(dslen * split_fold)
    np.random.shuffle(indices)
    train_mapping = indices[valid_size:]
    valid_mapping = indices[:valid_size]
    train = GenHelper(ds, dslen - valid_size, train_mapping)
    valid = GenHelper(ds, valid_size, valid_mapping)

    return train, valid

def train_val_test_split(ds, val_split=0.1, test_split=0.1, random_seed=None):
    '''
    This is a pytorch generic function that takes a data.Dataset object and splits it to validation and training
    efficiently.
    :return: train, val datasets
    '''
    if random_seed != None:
        np.random.seed(random_seed)

    dslen = len(ds)
    indices = list(range(dslen))
    val_size = int(dslen * val_split)
    test_size = int(dslen * test_split)
    train_size = int(dslen-val_size-test_size)
    np.random.shuffle(indices)
    train_mapping = indices[:train_size]
    val_mapping = indices[train_size:train_size+val_size]
    test_mapping = indices[train_size+val_size:train_size+val_size+test_size]
    train = GenHelper(ds, train_size, train_mapping)
    val = GenHelper(ds, val_size, val_mapping)
    test = GenHelper(ds, test_size, test_mapping)

    return train, val, test


def train_valid_split_ind(ds, ind_path):
    '''
    This is a pytorch generic function that takes a data.Dataset object and splits it to validation and training
    efficiently.
    :return: train, val, test datasets
    '''
    df = pd.read_csv(ind_path)
    train_ind = np.array(df[df['split'] == 0].index)
    val_ind = np.array(df[df['split'] == 1].index)
    test_ind = np.array(df[df['split'] == 2].index)

    train = GenHelper(ds, train_ind.shape[0], train_ind)
    val = GenHelper(ds, val_ind.shape[0], val_ind)
    test = GenHelper(ds, test_ind.shape[0], test_ind)

    return train, val, test


# -------------------- dataloaders  ---------------------------------------------------------

def init_dhcp_dataloader(args, shuffle_test=False):
    '''
    Initialize both datasets and dataloaders
    image_size = [128, 160, 112]
    '''
    if (not args.aug_rician_noise == None) or (not args.aug_bspline_deformation == None) or (not args.resize_image == None):
        transforms = []
    else:
        transforms = None

    if args.resize_image:
        transforms.append(ResizeImage(image_size=args.resize_size))

    if args.aug_rician_noise:
        transforms.append(RicianNoise(noise_level=args.aug_rician_noise))

    if args.aug_bspline_deformation:
        transforms.append(ElasticDeformationsBspline(num_controlpoints=args.aug_bspline_deformation[0], sigma=args.aug_bspline_deformation[1]))

    if args.aug_rician_noise or args.aug_bspline_deformation or args.resize_image:
        transforms = torchvision.transforms.Compose(transforms)

    healthy_train = DHCP_2D(image_path= args.dataset_dir,
                         label_path= args.labels_path, # was label_path before
                         num_classes=2,
                         task='classification',
                         class_label=0,
                         transform=transforms)

    anomaly_train = DHCP_2D(image_path= args.dataset_dir,
                         label_path= args.labels_path, # was label_path before
                         num_classes=2,
                         task='classification',
                         class_label=1,
                         transform=transforms)

    healthy_dataloader_train, healthy_dataloader_val, healthy_dataloader_test = train_val_test_split(healthy_train, val_split=0.1, test_split=0.1,
                                                                         random_seed=8)
    anomaly_dataloader_train, anomaly_dataloader_val, anomaly_dataloader_test = train_val_test_split(anomaly_train, val_split=0.1, test_split=0.1,
                                                                         random_seed=8)


    print('Train data length: ', len(healthy_dataloader_train), 'Val data length: ',len(healthy_dataloader_val), 'Test data length: ', len(healthy_dataloader_test))
    print('Train data length: ', len(anomaly_dataloader_train), 'Val data length: ',len(anomaly_dataloader_val), 'Test data length: ', len(anomaly_dataloader_test))

    healthy_dataloader_train = torch.utils.data.DataLoader(healthy_dataloader_train, batch_size=args.batch_size//2,
                                                           shuffle=True)
    anomaly_dataloader_train = torch.utils.data.DataLoader(anomaly_dataloader_train, batch_size=args.batch_size//2,
                                                           shuffle=True)

    healthy_dataloader_val = torch.utils.data.DataLoader(healthy_dataloader_val, batch_size=args.batch_size//2,
                                                         shuffle=True)
    anomaly_dataloader_val = torch.utils.data.DataLoader(anomaly_dataloader_val, batch_size=args.batch_size//2,
                                                         shuffle=True)
    healthy_dataloader_test = torch.utils.data.DataLoader(healthy_dataloader_test, batch_size=args.batch_size//2,
                                                         shuffle=shuffle_test)
    anomaly_dataloader_test = torch.utils.data.DataLoader(anomaly_dataloader_test, batch_size=args.batch_size//2,
                                                         shuffle=shuffle_test)

    return healthy_dataloader_train, healthy_dataloader_val, healthy_dataloader_test, anomaly_dataloader_train, anomaly_dataloader_val, anomaly_dataloader_test
