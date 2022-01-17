from pykuwahara import kuwahara
from skimage.util import random_noise

import cv2
import numpy as np
import SimpleITK as sitk


def apply_bilateral(image, domain_sigma=15, range_sigma=0.08):
    sitk_image = sitk.GetImageFromArray(image.astype(np.float32))

    bilateral_filter = sitk.BilateralImageFilter()
    bilateral_filter.SetDomainSigma(domain_sigma)
    bilateral_filter.SetRangeSigma(range_sigma)

    bilateral_image = bilateral_filter.Execute(sitk_image)
    bilateral_image = sitk.GetArrayFromImage(bilateral_image)

    return bilateral_image


def apply_diffusion(image, conductance=5, iterations=50):
    sitk_image = sitk.GetImageFromArray(image.astype(np.float32))

    diffusion_filter = sitk.CurvatureAnisotropicDiffusionImageFilter()
    diffusion_filter.SetConductanceParameter(conductance)
    diffusion_filter.SetNumberOfIterations(iterations)

    diffused_image = diffusion_filter.Execute(sitk_image)
    diffused_image = sitk.GetArrayFromImage(diffused_image)

    return diffused_image


class Kuwahara(object):

    def __call__(self, image):
        img = kuwahara(image.astype(np.float32), method='gaussian', radius=np.random.randint(2,4)).astype(np.float32)

        return img


class Bilateral(object):

    def __call__(self, image):
        img = apply_bilateral(image, range_sigma=np.random.uniform(0.03,0.1)).astype(np.float32)

        return img


class Diffusion(object):

    def __call__(self, image):
        img = apply_diffusion(image, conductance=np.random.randint(3,11))

        return img


class GaussianNoise(object):
    
    def __call__(self, image):
        img = random_noise(image, seed=2, mode='gaussian', var=np.random.uniform(0.03,0.1))

        return img

class ToRGB(object):
    
    def __call__(self, image):
        img = np.repeat(image, 3, axis=0)
        return img


class RandomNoiseDenoise(object):

    def __call__(self, image):

        random_number = np.random.randint(0,4)
        
        if random_number == 0:
            img = kuwahara(image, method='gaussian', radius=np.random.randint(2,4)).astype(np.float32)
        elif random_number == 1:
            img = apply_bilateral(image, range_sigma=np.random.uniform(0.03,0.1)).astype(np.float32)
        elif random_number == 2:
            img = apply_diffusion(image, conductance=np.random.randint(3,11)).astype(np.float32)
        elif random_number == 3:
            img = random_noise(image, seed=2, mode='gaussian', var=np.random.uniform(0.03,0.1)).astype(np.float32)
        else:
            img = image

        return img