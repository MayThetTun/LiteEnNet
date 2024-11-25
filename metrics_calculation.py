import os
import cv2
import numpy as np
import piq
import sewar
import skimage.metrics
from sewar.full_ref import mse, rmse, psnr, uqi, ssim, ergas, scc, rase, sam, msssim, vifp
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from uiqm_utils import getUIQM,getUCIQE
from skimage.measure import compare_niqe

def calculate_metrics(generated_image_path, ground_truth_image_path, resize_size=(256, 256)):
    generated_image_list = os.listdir(generated_image_path)
    # Initialize lists to store SSIM and PSNR errors
    error_list_ssim, error_list_msssim, error_list_uqi, error_list_vif, error_list_psnr = [], [], [], [], []
    # Iterate through each generated image
    for img in generated_image_list:
        label_img = img
        generated_image = os.path.join(generated_image_path, img)
        ground_truth_image = os.path.join(ground_truth_image_path, label_img)
        # Read and resize the generated image
        generated_image = cv2.imread(generated_image)
        generated_image = cv2.resize(generated_image, resize_size)
        # Read and resize the ground truth image
        ground_truth_image = cv2.imread(ground_truth_image)
        ground_truth_image = cv2.resize(ground_truth_image, resize_size)
        # calculate SSIM
        # error_ssim = sewar.full_ref.ssim(ground_truth_image, generated_image)
        error_ssim, diff_ssim = skimage.metrics.structural_similarity(ground_truth_image, generated_image,full=True,channel_axis=-1)
        error_list_ssim.append(error_ssim)
        # calculate MS_SSIM
        error_msssim = sewar.full_ref.msssim(ground_truth_image, generated_image)
        error_list_msssim.append(np.real(error_msssim))
        # calculate UQI
        error_uqi = sewar.full_ref.uqi(ground_truth_image, generated_image)
        error_list_uqi.append(error_uqi)
        # calculate VIF
        error_vif = sewar.full_ref.vifp(ground_truth_image, generated_image)
        error_list_vif.append(error_vif)
        # Convert images to grayscale
        generated_image = cv2.cvtColor(generated_image, cv2.COLOR_BGR2GRAY)
        ground_truth_image = cv2.cvtColor(ground_truth_image, cv2.COLOR_BGR2GRAY)
        # calculate PSNR
        error_psnr = sewar.full_ref.psnr(ground_truth_image, generated_image)
        error_list_psnr.append(error_psnr)

    # Convert error lists to numpy arrays and return
    return np.array(error_list_ssim), np.array(error_list_msssim), np.array(
        error_list_uqi), np.array(error_list_vif), np.array(error_list_psnr)


def calculate_UIQM(image_path, resize_size=(256, 256)):
    image_list = os.listdir(image_path)
    uiqms = []

    for img in image_list:
        image = os.path.join(image_path, img)
        image = cv2.imread(image)
        image = cv2.resize(image, resize_size)
        # calculate UIQM
        uiqms.append(getUIQM(image))
    return np.array(uiqms)


def calculate_UCIQE(image_path, resize_size=(256, 256)):
    image_list = os.listdir(image_path)
    uciqes = []

    for img in image_list:
        image = os.path.join(image_path, img)
        image = cv2.imread(image)
        image = cv2.resize(image, resize_size)
        # calculate UCIQE
        uciqes.append(getUCIQE(image))
    return np.array(uciqes)
