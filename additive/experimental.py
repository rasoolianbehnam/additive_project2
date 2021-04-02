import collections
import os
from collections import namedtuple

import cv2
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Ellipse
from matplotlib.pyplot import *
from scipy import optimize
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

Arg = namedtuple('Arg', 'meta value')
Params = namedtuple('Params', 'name, variation, fun, extra')

MATCH_METHOD = cv2.TM_CCORR_NORMED

info = """v05_T2_R_3d,7302.47,228.58,7500
v06_T1_L_3d,7500,274.21,7500
v04_T1_L_3d,7420.48,308.71,7500
v02_T2_L_3d,7311.48,306.65,7500
v04_T1_R_3d,7326.07,319.37,7500
v01_T2_L_3d,7500,253.19,7500
v02_T1_R_3d,7411.04,281.72,7500
v03_T1_L_3d,7201.62,306.67,7500
v04_T2_R_3d,7424.77,214.81,7500
v05_T1_L_3d,7500,324.97,7500
v06_T1_R_3d,7476.69,397.99,7500
v03_T1_R_3d,7199.05,267.50,7500
v05_T2_L_3d,7403.31,405.86,7500
v02_T2_R_3d,7408.89,287.19,7500
v01_T2_R_3d,7500,340.53,7500
v04_T2_L_3d,7425.20,274.28,7500
v02_T1_L_3d,7212.35,253.20,7500
v01_T1_L_3d,7408.89,359.44,7500
v05_T1_R_3d,7500,317.93,7500
v06_T2_L_3d,7376.71,269.32,7500
v03_T2_L_3d,7257.84,243.52,7500
v03_T2_R_3d,7312.77,268.11,7500
v01_T1_R_3d,7206.34,305.86,7500"""

Circle = collections.namedtuple('Circle', 'beg, end, index, h, cx, cy, r, x_mean,  x_std, y_mean, y_std')

ImageInfo = collections.namedtuple('ImageInfo', 'max_height, max_peak, max_width')

# image_info is a dictionary with image file name as key and ImageInfo instances as values
image_info = {name: ImageInfo(*(float(x) for x in rest))
              for name, *rest in [x.split(',') for x in info.split('\n')]}


def rescale_image(img, img_info, scale=1):
    scales = np.array((img_info.max_height, img_info.max_width)) / np.array(img.shape)
    print("scales = %f, %f" % tuple(scales))
    resized_img = cv2.resize(img, (int(img_info.max_width * scale), int(img_info.max_height * scale))) * scale
    return resized_img


def adjust_tilt(img, n_jobs):
    median = np.median(img)
    q1, q3 = np.percentile(img, [25, 75])
    iqr = q3 - q1
    poinst = np.where((img > median - 2 * iqr) & (img < median + 2 * iqr))
    y = img[poinst]
    X = np.array(poinst).T
    model = LinearRegression(n_jobs=n_jobs, normalize=True)
    model.fit(X, y)
    poinst = np.where(img == img)
    y = img[poinst]
    X = np.array(poinst).T
    y_pred = model.predict(X)
    print(model.coef_)
    adjusted = (y + y.mean() - y_pred).reshape(img.shape)
    negative_indices = np.where(img < 0)
    adjusted[negative_indices] = img[negative_indices]
    return adjusted


def remove_outliers(x):
    x = np.array(x)
    median = np.median(x)
    Q1 = np.percentile(x, 25)
    Q3 = np.percentile(x, 75)
    iqr = Q3 - Q1
    indices = (x >= median - 3 * iqr) & (x <= median + 3 * iqr)
    return np.where(indices)[0], x[indices], median, iqr


def findTopAndBottom(x):
    *_, median, iqr = remove_outliers(x)
    # X = X.reshape(-1, 1)
    X = np.where((x < median - 3 * iqr))[0].reshape(-1, 1)
    if len(X) == 0:
        # print("len(X) is 0")
        return 0, len(x)
    try:
        kmeans = KMeans(n_clusters=2).fit(X)
        min_index = np.argmin(kmeans.cluster_centers_)
        return np.max((X[np.where(kmeans.predict(X) == min_index)])), np.min(
            (X[np.where(kmeans.predict(X) != min_index)]))
    except:
        return 0, len(x)


def align_image(resized_img):
    print('[*] Starting to align ...')
    limits = [findTopAndBottom(resized_img[:, w]) for w in range(resized_img.shape[1])]
    a1, b1 = np.polyfit(*(remove_outliers([x[0] for x in limits])[:2]), 1)
    a2, b2 = np.polyfit(*(remove_outliers([x[1] for x in limits])[:2]), 1)
    a = min(a1, a2)
    # print('[*] a = %r'%(a))
    h, w = resized_img.shape
    M = cv2.getRotationMatrix2D((h // 2, w // 2), a * 180 / np.pi, 1)
    rotated = cv2.warpAffine(resized_img, M, (w, h))
    return rotated  # , a, M, limits


# def align_image(img, resize_factor=10):
#    print('[*] Starting to align ...')
#    W, H = img.shape
#    resized_img = cv2.resize(img, (H//resize_factor, W//resize_factor))
#    limits = [findTopAndBottom(resized_img[:, w]) for w in range(resized_img.shape[1])]
#    a1, b1 = np.polyfit(*(remove_outliers([x[0] for x in limits])[:2]), 1)
#    a2, b2 = np.polyfit(*(remove_outliers([x[1] for x in limits])[:2]), 1)
#    a = min(a1, a2)
#    #print('[*] a = %r'%(a))
#    h, w = resized_img.shape
#    M = cv2.getRotationMatrix2D((W//2, H//2), a * 180/np.pi, 1)
#    rotated = cv2.warpAffine(img, M, (H, W)) 
#    return rotated#, a, M, limits

def get_local_minima_3d(img, kernel_size_param, thresh=None):
    if isinstance(kernel_size_param, tuple):
        kernel = np.ones(kernel_size_param)
    else:
        kernel = np.ones((kernel_size_param, kernel_size_param))
    eroded = cv2.erode(img, kernel)
    diff = img - eroded
    if thresh is None:
        thresh = img.mean() - img.std()
    return (diff == 0) & (img >= 0) & (img < thresh)


def draw_local_minima_3d(img, local_minima):
    drawing_img = np.zeros_like(img)  # ((h, w))
    for j, i in zip(*local_minima):
        cv2.circle(drawing_img, (i, j), 2, (255))
    plt.imshow(drawing_img)


def clean_y(y, s):
    y = cv2.GaussianBlur(y.reshape(1, -1), (1, 2 * int(s) + 1), 1).reshape(-1)
    return getFirstAndSecondDerivatives(y, s)


def getFirstAndSecondDerivatives(x, y, s=1):
    n = len(y)
    der1 = []
    der2 = []
    y_new = []
    for i in range(s, n - s):
        a, b, c = np.polyfit(x=x[i - s:i + s], y=y[i - s:i + s], deg=2)
        y_new.append(a * i ** 2 + b * i + c)
        der1.append(2 * a * i + b)
        der2.append(a)
    return x[s:n - s], np.array(y_new), np.array(der1), np.array(der2)


def get_local_minima_2d(y, kernel_size_param, t=-1.1, mn=float('-infinity')):
    kernel = np.ones((1, kernel_size_param))
    eroded = cv2.erode(y.reshape(1, -1), kernel).reshape(-1)
    # return (y>mn)&(eroded==y)&(y<y.mean()+t*y.std())
    return (y > mn) & (eroded == y) & (y < np.percentile(y, 25))


# def get_neighborhood(i, der2, kernel_size_param, alpha=0):
#    k1 = k2 = 0
#    while i-k1 > 0 and der2[i-k1] > alpha:
#        k1 += 1
#    while i+k2 < len(der2) and der2[i+k2] > alpha:
#        k2 += 1   
#    if k1 == 0 or k2 == 0:
#        print("invalid situation")
#        return
#    #NOTE: you might or might not want to do this:
#    k = k1 if k1 < k2 else k2
#    #if k < kernel_size_param:
#    #    print('[!!!] skipped')
#    #    return
#    return i-k, i+k+1
def get_neighborhood(i, der2, kernel_size_param, alpha=0, verbose=True):
    pos_concav_areas = der2 > 0
    # kernel = np.ones((1,2))
    # pos_concav_areas = cv2.dilate((pos_concav_areas*1).astype('uint8').reshape(1, -1), kernel).reshape(-1)
    # plt.plot(pos_concav_areas)
    # plt.show()
    i1 = i2 = i
    while i1 >= 0 and pos_concav_areas[i1] == 1:
        i1 -= 1
    i1 += 1
    while i2 < len(pos_concav_areas) and pos_concav_areas[i2] == 1:
        i2 += 1
    if i2 <= i1 + 3:
        if verbose:
            print("i2 <= i1+3")
        return
    l = np.min([i - i1, i2 - i])
    if l <= 2:
        if verbose:
            print("l <= 2")
        return
    # print(i-l, ":", i+l)
    return i - l, i + l + 1


def normalize(x):
    # _, x, *_ = remove_outliers(x)
    x_mean = np.mean(x)
    x_std = np.std(x)
    return (x - x_mean) / x.std(), x_mean, x_std


def get_cut_points(profile, thresh):
    """
    Input: 
    Profile: A 1D numpy array
    thresh: scalar: $y = thresh$ is the line that crosses the profile
    TODO: visualization"""
    cut_points = np.where(profile < thresh)[0]
    clusters = []
    cluster = [0]
    for i in range(1, len(cut_points)):
        if cut_points[i] == cut_points[i - 1] + 1:
            cluster.append(i)
        else:
            clusters.append(cluster)
            cluster = [i]
    # indices = np.where([cut_points[i]-cut_points[i-1] > 1 for i in range(1, len(cut_points))])[0].tolist()
    # indices.insert(0, 0)
    # print(clusters)
    out = []
    for cluster in clusters:
        if len(cluster):
            out.append(len(cluster))
    # out = [len(cluster) for cluster in clusters]
    if len(out) == 0:
        return [0]
    return out


class Fit_circle():
    def __init__(self, x_t, y_t):
        self.x_t = x_t
        self.y_t = y_t

    def calc_R(self, xc, yc):
        """ calculate the distance of each 2D points from the center (xc, yc) """
        return np.sqrt((self.x_t - xc) ** 2 + (self.y_t - yc) ** 2)

    def f_2(self, c):
        """ calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc) """
        cx, cy, r = c
        Ri = self.calc_R(cx, cy)
        return ((Ri - r) ** 2 \
                # + 50 * (Ri*((r-Ri)>0))**2 \
                + 1000 * ((r - Ri) > 0) \
                # + .2 * r**2
                )

    def fit_circle(self):
        center_estimate = self.x_t.mean(), self.y_t.mean(), self.calc_R(self.x_t.mean(), self.y_t.mean()).mean()
        # print("center estimate", center_estimate)
        center_2, ier = optimize.leastsq(self.f_2, center_estimate)
        cx, cy, r = center_2
        return cx, cy, r


def extract_circles(x_orig, y_orig, kernel_size_param, same_scale=False, verbose=True):
    # if ax is None:
    #    fig, ax = subplots()
    y = cv2.GaussianBlur(y_orig.reshape(1, -1), (1, 2 * int(kernel_size_param) + 1), 1).reshape(-1)
    # print(len(x_orig), len(y), s)
    n = len(y)
    x, y, der1, der2 = getFirstAndSecondDerivatives(x_orig, y, max(5, kernel_size_param // 8))
    if verbose:
        print(len(x), len(y))
    local_minima_2d = np.where(get_local_minima_2d(y, kernel_size_param) & (y > 0))[0]
    # plot(x, y)
    # scatter(x[local_minima_2d], y[local_minima_2d], marker='*')
    circles = []
    for i, index in enumerate(local_minima_2d):
        # print(i)
        tmp = get_neighborhood(index, der2, kernel_size_param, verbose=verbose)
        if tmp is None:
            if verbose:
                print("get_neighborhood at index %r returned None" % (index))
            continue
        beg, end = tmp

        x_n = x[beg:end]
        y_n = y[beg:end]
        if same_scale:
            x_n, x_mean, x_std = x_n, 0, 1
            y_n, y_mean, y_std = y_n, 0, 1
        else:
            x_n, x_mean, x_std = normalize(x_n)
            y_n, y_mean, y_std = normalize(y_n)
        # print(x_n.shape, y_n.shape)
        if x_n.shape != y_n.shape:
            if verbose:
                print("Skipped circle at index=%r since x_n.shape(%r) != y_n.shape(%r)" % (index, x_n.shape, y_n.shape))
            continue

        cx, cy, r = Fit_circle(x_n, y_n).fit_circle()
        # print(index, x_mean, x_std, y_mean, y_std)
        if cy > y_n.min():
            # circles.append({"center":index, "cx":cx, "cy":cy, "r":r, "x_mean":x_mean,
            #                "x_std":x_std, "y_mean":y_mean, "y_std":y_std})
            circles.append(Circle(beg, end, index, y_orig[index], cx, cy, r, x_mean,
                                  x_std, y_mean, y_std))
    return x, y, der1, der2, local_minima_2d, circles


def draw_circles(x_new, y_new, circles, ax=None, drop_large_radius=True):
    radii = np.array([circle.r for circle in circles])
    radii_x = np.array([circle.r * circle.x_std for circle in circles])
    radii_y = np.array([circle.r * circle.y_std for circle in circles])
    std_x = np.array([circle.x_std for circle in circles])
    std_y = np.array([circle.y_std for circle in circles])
    radii_of_curvature = radii * (std_x ** 2) / std_y
    if ax is None:
        fig, ax = subplots()
    for beg, end, index, h, cx, cy, r, x_mean, x_std, y_mean, y_std in circles:
        rx = r * x_std
        ry = r * y_std
        radius_of_curvature = r * x_std ** 2 / y_std
        if drop_large_radius and radius_of_curvature > radii_of_curvature.mean() + radii_of_curvature.std():
            print('Skipped drawing circle at %r due to radius of curvature being too high' % (index))
            continue
        # print("radius_of_curvature", radius_of_curvature, rx, ry)
        # plt.scatter(np.array([index])*x_scale, y_new[index])
        plot(x_new[range(beg, end)], y_new[beg:end], color='cyan')
        ellipse1 = Ellipse((x_mean + cx * x_std, y_mean + cy * y_std - ry + radius_of_curvature),
                           2 * radius_of_curvature, 2 * radius_of_curvature, alpha=1, color='r')
        ax.add_artist(ellipse1)

    ax.plot(x_new, y_new, '-o', marker='', linewidth=1.5)
    return ax


def dfe(p):
    d = os.path.dirname(p) + "/"
    b = os.path.basename(p)
    return (d, *os.path.splitext(b))


def get_mask(file):
    if isinstance(file, str):
        x = joblib.load(file)
    else:
        x = file

    model = KMeans(2)
    pred1 = model.fit_predict(x.mean(axis=1).reshape(-1, 1))
    if pred1.sum() / len(pred1) > .2:
        pred1 = 1 - pred1
    msk1 = (np.zeros(x.shape) + pred1.reshape(-1, 1)).astype('bool')

    q1, q3 = np.percentile(x, [25, 75], axis=1)
    iqr = q3 - q1
    msk2 = x < (q1 - 4 * iqr).reshape(-1, 1)

    mask = msk1 | (x <= 0) | msk2
    return mask


def get_file_info(res):
    a = res.str.lower().str.contains('polished').rename('ispolished').to_frame()
    b = pd.DataFrame(res.map(lambda x: re.findall(f"(v\d+)_(t\d+)_(l|r)", x, re.IGNORECASE)[0]).tolist(),
                     columns=['specimen', 'T', 'RL'], index=a.index)
    return a.join(b)  # .join(res.str.rsplit('/', n=1, expand=True)[1])


import scipy.stats as st


def gkern2d(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel."""
    x = np.linspace(-nsig, nsig, kernlen + 1)
    kern1d = np.diff(st.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)
    return kern2d / kern2d.max()


def gkern1d(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel."""
    x = np.linspace(-nsig, nsig, kernlen + 1)
    kern1d = np.diff(st.norm.cdf(x))
    return kern1d / kern1d.max()


def plot_3d_surface(image, step=30, ax=None, xshift=0, yshift=0, **kwargs):
    sub_image = image[500:-500:step, 500:-500:step]
    xx, yy = np.mgrid[0:sub_image.shape[0], 0:sub_image.shape[1]]

    # create the figure
    if ax is None:
        fig, axes = plt.subplots(figsize=(8, 8))
        ax = fig.gca(projection='3d')
    out = ax.plot_surface(xx + xshift, yy + yshift, sub_image, rstride=1, cstride=1, linewidth=0, **kwargs)

    ax.axis('off')
    return out
    # show it
    # plt.show()


def get_top_left_best_template_match(src, template, method=MATCH_METHOD):
    res = cv2.matchTemplate(src, template, method)
    min_val1, max_val1, min_loc1, max_loc1 = cv2.minMaxLoc(res)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    return top_left[::-1]


def get_best_template_match(src, template, method=MATCH_METHOD):
    w, h = template.shape  # [::-1]
    x, y = get_top_left_best_template_match(src, template, method)
    return src[x:x + w, y:y + h]


def equalize_hist(img):
    mn, mx = img.min(), img.max()
    rng = mx - mn
    scaled_img = np.uint8((img - mn) / rng * 255)
    return cv2.equalizeHist(scaled_img)


def get_ratio_under_thresh(profile, thresh):
    cut_points = np.array(get_cut_points(profile, thresh))
    return (cut_points.max() - cut_points.min()) / len(cut_points)


def random_sub_image(img, size):
    W, H = img.shape
    w, h = size
    if w <= 1: w = int(w * W)
    if h <= 1: h = int(h * H)
    x = np.random.randint(0, W - w + 1)
    y = np.random.randint(0, H - h + 1)
    return img[x:x + w, y:y + h], (x, y)


def match_asbuilt_unpolished(asbuilt, polished, template_size_ratio, method=MATCH_METHOD):
    template, l1 = random_sub_image(asbuilt, [template_size_ratio] * 2)
    l2 = get_top_left_best_template_match(polished, template)
    mn = np.minimum(l1, l2)
    return np.array(l1) - mn, np.array(l2) - mn
    # return (x,y), l2


# def get_image_from_top_left(img, top_left, wh, dxy=(0, 0)):
#     x, y = top_left
#     w, h = wh
#     dx, dy = dxy
#     return img[x+dx:x+dx+w, y+dy:y+dy+h]

def get_image_from_top_left(img, top_left, dxy=(0, 0)):
    x, y = top_left
    dx, dy = dxy
    return img[x + dx:, y + dy:]


def image_correlation(x, y):
    return np.mean((x - x.mean()) * (y - y.mean())) / x.std() / y.std()
