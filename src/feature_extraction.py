import os
from multiprocessing import Pool

import numpy as np
import pandas as pd

from PIL import Image

from collections import Counter

import pywt
from scipy import stats
import cv2

import logging
logger = logging.getLogger(__name__)

def img_to_segments(img, crop_size, num_segments,
                    random_state, masked_region=None) -> list:
    segments = []
    # Create random bounding boxes
    num_extracted_segments = 0
    i = 0
    while num_extracted_segments < num_segments:
        xtl = random_state.randint(0, img.size[0] - crop_size)
        ytl = random_state.randint(0, img.size[1] - crop_size)

        xbr = xtl + crop_size
        ybr = ytl + crop_size

        i += 1

        if masked_region is not None:
            # Check if the bounding box is within the mask
            if xtl <= masked_region[2] \
               and xbr >= masked_region[0] \
               and ytl <= masked_region[3] \
               and ybr >= masked_region[1]:
                # Overlapping
                continue

        num_extracted_segments += 1
        segments.append((xtl, ytl, xbr, ybr))

    return segments

def scan_to_features(img, crop_size, num_segments, random_state, feature_fn, use_multiprocessing=True, resample_patience=5, masked_region=None):
    segments = img_to_segments(
                        img, crop_size=crop_size,
                        num_segments=num_segments * resample_patience, random_state=random_state, masked_region=masked_region)

    crops = []
    for segment in segments:
        crop = img.crop(box=segment)

        crops.append(np.array(crop))

    if use_multiprocessing:
        pool = Pool(processes=os.cpu_count())
        X = pool.map(feature_fn, crops)
        pool.close()
    else:
        X = []
        n_valid = 0
        for crop in crops:
            x = feature_fn(crop)
            X.append(x)
            if x is not None:
                n_valid += 1

                if n_valid == num_segments:
                    # We have enough samples
                    break
        # Fill x with Nones so it matches the expected length
        for _ in range(num_segments * resample_patience - len(X)):
            X.append(None)

    return X, segments, crops

def get_features(df, img_data_prefix, num_segments, crop_size,
                 random_state, feature_fn, save_crops,
                 printer_models=None, use_multiprocessing=True, resample_patience=5,
                 masked_region=None):
    result = []
    descriptors = None

    for i, row in df.iterrows():
        row_dict = row.to_dict()

        filename = row['filename']
        manufacturer = row['printer_manufacturer']
        model = row['printer_model']

        img = Image.open(os.path.join(img_data_prefix, filename))

        X, segments, crops = scan_to_features(img=img, crop_size=crop_size, num_segments=num_segments,
                         random_state=random_state, feature_fn=feature_fn,
                                              use_multiprocessing=use_multiprocessing, resample_patience=resample_patience, masked_region=masked_region)

        if printer_models is not None:
            printer_model_str = f'{manufacturer}//{model}'
            printer_model_index = np.where(printer_models == printer_model_str)[0][0]
            row_dict.update({'y': printer_model_index})

        num_valid = 0
        for x, segment, crop in zip(X, segments, crops):
            # Take the first num_segments samples from X that are not None
            if x is None:
                continue
            x, descriptors = x
            d = row_dict.copy()
            d['x'] = x
            d['crop_area'] = segment
            if save_crops:
                d['crop'] = crop
            result.append(d)
            num_valid += 1
            if num_valid == num_segments:
                # We are done
                break

        if num_valid < num_segments:
            print(f"Warning: {filename} only has {num_valid}/{num_segments} valid crops")
        else:
            print(f"Successfully extracted features from {filename}")

    assert(len(result) > 0)
    assert(descriptors is not None)

    return pd.DataFrame(result), descriptors

def process_segment(crop):
    crop = Image.fromarray(crop)

    try:
        ft = extract_features(crop)
    except Exception as e:
        ft = None

    return ft

class FeatureExtractionException(Exception):
    pass

def extract_features(
        img,
        ):
    # Preprocess crop
    crop_rgb = np.array(img)

    # Remove droplets as noise
    denoised_rgb_inv = cv2.medianBlur(crop_rgb, 5)
    denoised_rgb = 255 - denoised_rgb_inv

    # Diff image, resulting in the droplets being left
    diff = crop_rgb.copy()
    cv2.absdiff(crop_rgb, denoised_rgb, diff)
    diff_inv = crop_rgb.copy()
    cv2.absdiff(crop_rgb, denoised_rgb_inv, diff_inv)

    # Do the same for cmy channels
    crop_cmyk = Image.fromarray(diff).convert('CMYK')
    c_im, m_im, y_im, _ = crop_cmyk.split()
    crop_c = np.array(c_im)
    crop_m = np.array(m_im)
    crop_y = np.array(y_im)

    dst_c = cv2.medianBlur(crop_c, 5)
    dst_m = cv2.medianBlur(crop_m, 5)
    dst_y = cv2.medianBlur(crop_y, 5)

    diff_c = crop_c.copy()
    diff_m = crop_m.copy()
    diff_y = crop_y.copy()

    cv2.absdiff(crop_c, dst_c, diff_c)
    cv2.absdiff(crop_m, dst_m, diff_m)
    cv2.absdiff(crop_y, dst_y, diff_y)

    # Thresholding to retrieve highlighted droplets
    _, c = cv2.threshold(diff_c,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    _, m = cv2.threshold(diff_m,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    _, y = cv2.threshold(diff_y,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    diff_inv = cv2.cvtColor(diff_inv, cv2.COLOR_RGB2GRAY)
    _, greyscale_img = cv2.threshold(diff_inv,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    c = Image.fromarray(c)
    m = Image.fromarray(m)
    y = Image.fromarray(y)
    greyscale_img = Image.fromarray(greyscale_img)

    colorbands = [greyscale_img, c, m, y]
    colorband_names = ['greyscale', 'c', 'm', 'y']

    # Feature Extraction
    features = []
    descriptors = []

    ft, desc = _get_image_features(img)
    features += ft
    descriptors += [f'image_{d}' for d in desc]

    for channel, name in zip(colorbands, colorband_names):
        ft, desc = _get_droplet_features(np.array(channel))
        desc = [f'{name}_droplet_{d}' for d in desc]

        features += ft
        descriptors += desc

    # reset the colorbands as we do not want to preprocess for wavelet features
    c, m, y, _ = img.convert('CMYK').split()
    greyscale_img = img.convert('L')

    colorbands = [greyscale_img, c, m, y]

    for colorband, colorband_name in zip(colorbands, colorband_names):
        ft, desc = _get_wavelet_features(colorband)
        desc = [f'wavelet_{colorband_name}_{d}' for d in desc]
        features += ft
        descriptors += desc


    assert(features)

    result = np.array(features).flatten()

    if np.isnan(result).any():
        # For instance if no dots were recognized. We just label this as invalid
        raise FeatureExtractionException()

    return result, descriptors

def _get_image_features(img):
    bgr_opencv_img = cv2.cvtColor(np.array(img.convert('RGB')), cv2.COLOR_RGB2BGR)

    color_ft, color_desc = feat_color(im=img)
    contrast_ft, contrast_desc = feat_contrast(img=bgr_opencv_img)

    features = color_ft + contrast_ft
    descriptors = [f'color_{d}' for d in color_desc] + [f'contrast_{d}' for d in contrast_desc]

    return features, descriptors

def _get_droplet_features(c):
    contours, _ = cv2.findContours(c, 1, 2)

    dot_size_ft, dot_size_desc = features_dot_size(contours=contours)
    perimeter_ft, perimeter_desc = features_perimeter(contours=contours)

    features = dot_size_ft + perimeter_ft
    descriptors = [f'size_{d}' for d in dot_size_desc] + [
        f'perimeter_{d}' for d in perimeter_desc]

    return features, descriptors

def _get_wavelet_features(colorband, wavelets_2d: bool=False):
    if wavelets_2d:
        coeffs = pywt.wavedec2(colorband, 'db5', level=3)
    else:
        coeffs = pywt.wavedec(colorband, 'db5', level=3)

    ft = []
    descriptors = []
    for j, coeff in enumerate(coeffs):
        flattened_coeff = np.array(coeff).flatten()

        entropy, entropy_desc = _calculate_entropy(flattened_coeff)
        crossings, crossings_desc = _calculate_crossings(flattened_coeff)
        statistics, statistics_desc = _calculate_statistics(flattened_coeff)

        coeff_features = [entropy] + crossings + statistics

        desc = entropy_desc + crossings_desc + statistics_desc
        desc = [f'{d}_l{j+1}' for d in desc]

        ft += coeff_features
        descriptors += desc

    return ft, descriptors

def _calculate_entropy(list_values):
    counter_values = Counter(list_values).most_common()
    probabilities = [elem[1]/len(list_values) for elem in counter_values]
    entropy=stats.entropy(probabilities)

    return entropy, ['entropy']

def _calculate_statistics(list_values):
    n5 = np.percentile(list_values, 5)
    n25 = np.percentile(list_values, 25)
    n75 = np.percentile(list_values, 75)
    n95 = np.percentile(list_values, 95)
    median = np.percentile(list_values, 50)
    mean = np.mean(list_values)
    std = np.std(list_values)
    var = np.var(list_values)
    rms = np.mean(np.sqrt(list_values**2))

    return [n5, n25, n75, n95, median, mean, std, var, rms], ['n5','n25','n75','n95','median','mean','std','var','rms']

def _calculate_crossings(list_values):
    zero_crossing_indices = np.nonzero(np.diff(np.array(list_values) > 0))[0]
    no_zero_crossings = len(zero_crossing_indices)
    mean_crossing_indices = np.nonzero(np.diff(np.array(list_values) > np.mean(list_values)))[0]
    no_mean_crossings = len(mean_crossing_indices)

    return [no_zero_crossings, no_mean_crossings], ['no_zero_crossings', 'no_mean_crossings']

def feat_color(im):
    im = im.convert('CMYK')
    c_im, m_im, y_im, _ = im.split()
    # convert to numpy array for opencv
    C = np.array(c_im)
    M = np.array(m_im)
    Y = np.array(y_im)

    ret_c, thresh_c = cv2.threshold(C, 0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    ret_m, thresh_m = cv2.threshold(M, 0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    ret_y, thresh_y = cv2.threshold(Y, 0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    if np.sum(thresh_c == 255) > 1:
        C_mean = np.mean(C[thresh_c == 255])
        C_std = np.std(C[np.where(thresh_c==255)])
    else:
        C_mean = 0
        C_std = 0

    if np.sum(thresh_m == 255) > 1:
        M_mean = np.mean(M[thresh_m == 255])
        M_std = np.std(M[np.where(thresh_m==255)])
    else:
        M_mean = 0
        M_std = 0

    if np.sum(thresh_y == 255) > 1:
        Y_mean = np.mean(Y[thresh_y == 255])
        Y_std = np.std(Y[np.where(thresh_y==255)])
    else:
        Y_mean = 0
        Y_std = 0

    return [C_mean, C_std, M_mean, M_std, Y_mean, Y_std], ['c_mean', 'c_std', 'm_mean', 'm_std', 'y_mean', 'y_std']

def feat_contrast(img):
    Y = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)[:,:,0]
    min_val = int(np.min(Y))
    max_val = int(np.max(Y))
    # compute contrast
    denominator = (max_val + min_val)
    if denominator == 0:
        contrast = 0
    else:
        contrast = (max_val - min_val) / denominator

    return [min_val, max_val, contrast], ['min', 'max', 'value']

def features_dot_size(contours):
    non_empty_contours = [cv2.contourArea(c) for c in contours if cv2.contourArea(c) > 0]
    if len(non_empty_contours) > 1:
        areas = np.array(non_empty_contours)
        feat_conts = [
            areas.std(),
            areas.mean(),
            np.median(areas),
            np.quantile(areas, 0.25),
            np.quantile(areas, 0.75)]
    else:
        raise FeatureExtractionException("No droplets found")

    return feat_conts, ['std', 'mean', 'median', 'q25', 'q75']

def features_perimeter(contours):
    non_empty_contours = [cv2.arcLength(cntr, True) for cntr in contours if cv2.arcLength(cntr, True) > 0]
    if len(non_empty_contours) <= 1:
        raise FeatureExtractionException("No droplets found")

    perimeter = np.array(non_empty_contours)
    feat_perimeter = [
        perimeter.std(),
        perimeter.mean(),
        np.median(perimeter),
        np.quantile(perimeter, 0.25),
        np.quantile(perimeter, 0.75)]

    return feat_perimeter, ['std', 'mean', 'median', 'q25', 'q75']
