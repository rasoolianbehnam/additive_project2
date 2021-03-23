from typing import Tuple, Dict, Callable, List

import joblib
import numpy as np
import cv2
from dask import delayed, compute
import additive.feature_functions as ff
import scipy.stats as st

METRIC_MAP = {
    'r_a': '$R_a$',
    'r_q': '$R_q$',
    'r_p': '$R_p$',
    'r_v': '$R_v$',
    'r_sk': '$R_{rsk}$',
    'r_ku': '$R_{ku}$',

    's_a': '$S_a$',
    's_q': '$S_q$',
    's_p': '$S_p$',
    's_v': '$S_v$',
    's_sk': '$S_{rsk}$',
    's_ku': '$S_{ku}$',

    'rq': '$R_q$',
    'rp': '$R_p$',
    'rv': '$R_v$',
    'rsk': '$R_{rsk}$',
    'rku': '$R_{ku}$',
    'sa': '$S_a$',
    'sq': '$S_q$',
    'sp': '$S_p$',
    'sv': '$S_v$',
    'ssk': '$S_{rsk}$',
    'sku': '$S_{ku}$',
    'ra_2d': '$S_a$',
    'rq_2d': '$S_q$',
    'rp_2d': '$S_p$',
    'rv_2d': '$S_v$',
    'rsk_2d': '$S_{rsk}$',
    'rku_2d': '$S_{ku}$',
    'mode_2d': '$S_{mode}$',
    'ra_1d': '$R_a$',
    'rq_1d': '$R_q$',
    'rp_1d': '$R_p$',
    'rv_1d': '$R_v$',
    'rsk_1d': '$R_{rsk}$',
    'rku_1d': '$R_{ku}$',
    'mode_1d': '$R_{mode}$',
    'median': 'M',
    'mean': r'$\mu$',
    'r10_iso': r'$\rho_{iso}^{10}$',
}


def erode(x: np.ndarray, thresh: int):
    thresh_img = np.uint8(x < thresh)
    eroded = cv2.erode(thresh_img, np.ones((70, 70)), 4)
    _, contours, _ = cv2.findContours(eroded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    good_contours = sorted(contours, key=lambda x: -len(x))
    imt = np.zeros_like(eroded)
    e = cv2.drawContours(imt, good_contours[:2], -1, 1, -1)
    return e


def tilt_image(img, line: Tuple[float, float], axis=0):
    m, b = line
    w, h = img.shape
    n = w
    if axis:
        n = h
    x = np.arange(n).reshape(-1, 1)
    if axis:
        x = x.reshape(1, -1)
    return img + (m * x + b)


def get_all_stats(file_name: str,
                  transform_functions: Dict[str, Callable[[np.ndarray], float]]) -> List[Tuple[str, str, str, float]]:
    print(file_name)
    img = joblib.load(file_name)
    if not isinstance(img, np.ndarray):
        img = np.array(img['value'].x)
    img = img[1000:-1000, 1000:-1000]
    imgs = {k: delayed(fun)(img) for k, fun in transform_functions.items()}
    out, *_ = compute(
        [(file_name, metric, variation, delayed(feature_fun)(transformed_img))
         for variation, transformed_img in imgs.items()
         for metric, feature_fun in ff.feature_functions.items()]
    )
    return out


def pairwise_ttest(df, c1, c2):
    return st.ttest_1samp(df[c1]-df[c2], 0)