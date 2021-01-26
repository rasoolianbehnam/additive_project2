import os
from datetime import datetime

import matplotlib
from cycler import cycler
import pandas as pd
import re
import dask
from dask import delayed, compute
from multiprocessing.pool import ThreadPool
import glob
import joblib
from numpy import genfromtxt
from additive.preprocessing import gkern2d
from scipy.ndimage import zoom, convolve
import numpy as np
import shutil
from dask import bag
import matplotlib.pyplot as plt
import sh
dwget = delayed(sh.wget)

def dfe(p):
    d = os.path.dirname(p)+"/"
    b = os.path.basename(p)
    return (d, *os.path.splitext(b))

def get_file_info(res, col=None):
    if col:
        t = res[col]
    else:
        t = res
    t.name = t.name or 'files'
    a = t.str.lower().str.contains('polished').rename('ispolished').to_frame()
    b = pd.DataFrame(t.map(lambda x: re.findall(f"(v\d+)_(t\d+)_(l|r)", x, re.IGNORECASE)[0]).tolist(),
                columns=['specimen', 'T', 'RL'], index=a.index)
    out = a.join(b).join(res)
    return out

def pick_cols(df, pattern, reverse=False):
    c1 = re.compile(pattern)
    if not reverse:
        cols = [x for x in df.columns if c1.search(x)]
    else:
        cols = [x for x in df.columns if not c1.search(x)]
    return df[cols]


def download_from_dict(links_and_names, output_dir):
    """
    @param: links_and_names: dictionary: key: file name value: link
    @param: output_dir: obvious"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    def ddwget(k, v):
        try:
            return dwget(*f"{v} -O {output_dir}/{k}".split())
        except:
            return (k, v)
    out = [ddwget for k, v in links_and_names.items()]
    with dask.config.set(pool=ThreadPool(4)):
        print(compute(out))

def extract_array_from_csv(file):
    d, f, e = dfe(file)
    #return joblib.dump(x.values, d+f+'.pd')
    to_write = d+f+'.pd'
    print(file, to_write)
    if not os.path.exists(to_write):
        x = pd.read_csv(file, header=None, delimiter=',').values.astype('float32')
        joblib.dump(x, to_write) 
    else:
        print(f"file {to_write} exists")

def image_rescale(files):
    M = 31
    k = gkern2d(M, 5)
    k3 = k / k.sum()
    s = np.array([2.330435, 2.33016])
    original_images = bag.from_sequence(files).map(joblib.load)
    resized_images   = original_images.map(zoom, zoom=1/s)
    smoothed_images = resized_images.map(convolve, weights=k3)
    return smoothed_images

def save_list_to_files(lst, files, root, extension, save_fun=joblib.dump, overwrite=None):
    if not os.path.exists(root):
        os.mkdir(root)
    for file, item in zip(files, lst):
        d, f, e = dfe(file)
        out_path = f"{root}/{f}{extension}"
        if os.path.exists(out_path):
            if overwrite is None:
                print("File exists. Not overwriting")
                continue
            elif overwrite == 'backup':
                old_out_path = out_path+".old"
                print(f"File {out_path} exists Moving it to {old_out_path}")
                shutil.move(out_path, old_out_path)
            else:
                raise ValueError(f"overwrite parameter {overwrite} invalid")
        save_fun(item, out_path)


def subplots(figsize=(10, 10), nrows=1, ncols=1, **kwargs):
    return plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, **kwargs)


def custom_matplotlib_style():
    s = {
        "lines.linewidth": 2.0,
        "axes.edgecolor": "#bcbcbc",
        "patch.linewidth": 0.5,
        "legend.fancybox": True,
        "axes.prop_cycle": cycler('color', [
            "#348ABD",
            "#A60628",
            "#7A68A6",
            "#467821",
            "#CF4457",
            "#188487",
            "#E24A33"
        ]),
        "axes.facecolor": "#eeeeee",
        "axes.labelsize": "large",
        "axes.grid": True,
        "grid.linestyle": 'dashed',
        "grid.color": 'black',
        "grid.alpha": .2,
        "patch.edgecolor": "#eeeeee",
        "axes.titlesize": "x-large",
        "svg.fonttype": "path",
    }

    matplotlib.rcParams.update(s)


def timestamp_file(file, format="%Y%m%d-%H%M-"):
    d, f = os.path.dirname(file), os.path.basename(file)
    s = datetime.strftime(datetime.now(), format)
    return os.path.join(d, s+f)


def savefig(path, dpi=300):
    plt.savefig(timestamp_file(path), bbox_inches="tight", dpi=dpi)