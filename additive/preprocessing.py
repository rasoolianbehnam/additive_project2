import numpy as np
import pandas as pd
import scipy.stats as st
import joblib

from scipy.stats import linregress
from scipy.ndimage import rotate
from sklearn.linear_model import LinearRegression


def get_image_alignment_slope(image, thresh=None):
    # l: length to crop from both sides
    if thresh is None:
        thresh = image.mean()-image.std()
    pts = np.where(image>thresh)
    df = pd.DataFrame(np.array(pts).T, columns=["x", "y"])
    dfmn = df.groupby('y').x.agg(['min', 'max'])
    s1 = linregress(dfmn.index, dfmn['min']).slope
    s2 = linregress(dfmn.index, dfmn['max']).slope
    return s1, s2


def correct_aligment(image, l):
    # l: length to crop from both sides
    thresh = image.mean() - image.std()
    s1, s2 = get_image_alignment_slope(image[:, l:-l])
    return rotate(image, (s1+s2)/2*180/np.pi)


def adjust_tilt(img, degree=2, n_jobs=1):
    median = np.median(img)
    q1, q3 = np.percentile(img, [25, 75])
    iqr = q3 - q1
    points = np.where((img > median - 2 * iqr)) #& (img < median + 2 * iqr))
    y = img[points]
    X = np.array(points).T
    if degree == 2:
        X = np.concatenate([X, X**2], axis=1)
    model = LinearRegression(n_jobs=n_jobs, normalize=True)
    model.fit(X, y)
    points = np.where(img > -10)
    y = img.reshape(-1)
    X = np.array(points).T   
    if degree == 2:
        X = np.concatenate([X, X**2], axis=1)
    y_pred = model.predict(X)
    print(model.coef_)
    adjusted = (y + y.mean() - y_pred).reshape(img.shape)
    #negative_indices = np.where(img < 0)
    #adjusted[negative_indices] = img[negative_indices]
    return adjusted

def image_from_info_file(file):
    data = joblib.load(file)
    try:
        image = np.array(data['value'].x)
    except IndexError:
        image = data
    return image
        
        
def gkern2d(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel."""
    x = np.linspace(-nsig, nsig, kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)
    return kern2d/kern2d.max()

def process_image(image, tilt_crop_size=None, l=300, **tilt_kwargs):
    """
    :param f: path to the dict data or the data itself.
        f['value'] is a Feature object whose x field is a masked
        array
    :param transform_fun: a custom function to transform the loaded data
        before further processing.
    :param tilt_crop_size: a tuple (s1, s2) where s1 is the distance
        to cut from top and bottomn and s2 is the distance
        to cut from left and right :param l: length to crop from both sides in alignment
    """
    s1, s2 = tilt_crop_size or (1, 1)
    rotated_image = correct_aligment(image, l)
    tilted_rotated_image = adjust_tilt(rotated_image[s1:-s1, s2:-s2], **tilt_kwargs)
    return tilted_rotated_image