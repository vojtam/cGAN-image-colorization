
import sys
import os
from skimage import io, color
from sklearn.metrics import f1_score
from sklearn.metrics import average_precision_score
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import pandas as pd
import numpy as np



def dssim(img_ref, img_pred):
	ssim_none = ssim(img_ref, img_pred, channel_axis=2, data_range=255)
	dssim_score = 1 - ssim_none
	return dssim_score


def psnr(img_ref, img_pred):
	psnr_none = psnr(img_ref, img_pred)
	return psnr_none


def compute_simple_score_images(project_code, folder_ref, folder_pre):
	# reading the images
	ref_files = sorted([f for f in os.listdir(folder_ref)])
	pre_files = sorted([f for f in os.listdir(folder_pre)])

	scores = []
	for ref_file, pre_file in zip(ref_files, pre_files):
		# Open images
		ref_img = io.imread(os.path.join(folder_ref, ref_file))
		pre_img = io.imread(os.path.join(folder_pre, pre_file))

		if project_code == 'COL':
			score_pair = dssim(ref_img, pre_img)
		elif project_code == 'SUP':
			score_pair = psnr(ref_img, pre_img)
		# print(f'Img {ref_file}: score pair = {score_pair}')
		scores.append(score_pair)
	return sum(scores) / len(scores)


