# -*- coding: utf-8 -*-
"""A set of utility functions to support outlier detection.
"""
# Author: Yue Zhao <zhaoy@cmu.edu>
# License: BSD 2 clause

import os
import torch
import shutil
import numbers
import requests
import warnings
import numpy as np
from importlib import import_module

from ..metric import *

MAX_INT = np.iinfo(np.int32).max
MIN_INT = np.iinfo(np.int32).min


def validate_device(gpu_id):
    """Validate the input GPU ID is valid on the given environment.
    If no GPU is presented, return 'cpu'.

    Parameters
    ----------
    gpu_id : int
        GPU ID to check.

    Returns
    -------
    device : str
        Valid device, e.g., 'cuda:0' or 'cpu'.
    """

    # cast to int for checking
    gpu_id = int(gpu_id)

    # if it is cpu
    if gpu_id == -1:
        return 'cpu'

    # if gpu is available
    if torch.cuda.is_available():
        # check if gpu id is between 0 and the total number of GPUs
        check_parameter(gpu_id, 0, torch.cuda.device_count(),
                        param_name='gpu id', include_left=True,
                        include_right=False)
        device = 'cuda:{}'.format(gpu_id)
    else:
        if gpu_id != 'cpu':
            warnings.warn('The cuda is not available. Set to cpu.')
        device = 'cpu'

    return device


def check_parameter(param, low=MIN_INT, high=MAX_INT, param_name='',
                    include_left=False, include_right=False):
    """Check if an input is within the defined range.
    Parameters
    ----------
    param : int, float
        The input parameter to check.
    low : int, float
        The lower bound of the range.
    high : int, float
        The higher bound of the range.
    param_name : str, optional (default='')
        The name of the parameter.
    include_left : bool, optional (default=False)
        Whether includes the lower bound (lower bound <=).
    include_right : bool, optional (default=False)
        Whether includes the higher bound (<= higher bound).
    Returns
    -------
    within_range : bool or raise errors
        Whether the parameter is within the range of (low, high)
    """

    # param, low and high should all be numerical
    if not isinstance(param, (numbers.Integral, int, float)):
        raise TypeError('{param_name} is set to {param} Not numerical'.format(
            param=param, param_name=param_name))

    if not isinstance(low, (numbers.Integral, int, float)):
        raise TypeError('low is set to {low}. Not numerical'.format(low=low))

    if not isinstance(high, (numbers.Integral, int, float)):
        raise TypeError('high is set to {high}. Not numerical'.format(
            high=high))

    # at least one of the bounds should be specified
    if low is MIN_INT and high is MAX_INT:
        raise ValueError('Neither low nor high bounds is undefined')

    # if wrong bound values are used
    if low > high:
        raise ValueError(
            'Lower bound > Higher bound')

    # value check under different bound conditions
    if (include_left and include_right) and (param < low or param > high):
        raise ValueError(
            '{param_name} is set to {param}. '
            'Not in the range of [{low}, {high}].'.format(
                param=param, low=low, high=high, param_name=param_name))

    elif (include_left and not include_right) and (
            param < low or param >= high):
        raise ValueError(
            '{param_name} is set to {param}. '
            'Not in the range of [{low}, {high}).'.format(
                param=param, low=low, high=high, param_name=param_name))

    elif (not include_left and include_right) and (
            param <= low or param > high):
        raise ValueError(
            '{param_name} is set to {param}. '
            'Not in the range of ({low}, {high}].'.format(
                param=param, low=low, high=high, param_name=param_name))

    elif (not include_left and not include_right) and (
            param <= low or param >= high):
        raise ValueError(
            '{param_name} is set to {param}. '
            'Not in the range of ({low}, {high}).'.format(
                param=param, low=low, high=high, param_name=param_name))
    else:
        return True


def load_data(name, cache_dir=None):
    """
    Data loading function. See `data repository
    <https://github.com/pygod-team/data>`_ for supported datasets.
    For injected/generated datasets, the labels meanings are as follows.

    - 0: inlier
    - 1: contextual outlier only
    - 2: structural outlier only
    - 3: both contextual outlier and structural outlier

    Parameters
    ----------
    name : str
        The name of the dataset.
    cache_dir : str, optional
        The directory for dataset caching.
        Default: ``None``.

    Returns
    -------
    data : torch_geometric.data.Data
        The outlier dataset.

    Examples
    --------
    >>> from pygod.utils import load_data
    >>> data = load_data(name='weibo') # in PyG format
    >>> y = data.y.bool()    # binary labels (inlier/outlier)
    >>> yc = data.y >> 0 & 1 # contextual outliers
    >>> ys = data.y >> 1 & 1 # structural outliers
    """

    if cache_dir is None:
        cache_dir = os.path.join(os.path.expanduser('~'), '.pygod/data')
    file_path = os.path.join(cache_dir, name+'.pt')
    zip_path = os.path.join(cache_dir, name+'.pt.zip')

    if os.path.exists(file_path):
        data = torch.load(file_path)
    else:
        url = "https://github.com/pygod-team/data/raw/main/" + name + ".pt.zip"
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        r = requests.get(url, stream=True)
        if r.status_code != 200:
            raise RuntimeError("Failed downloading url %s" % url)
        with open(zip_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
        shutil.unpack_archive(zip_path, cache_dir)
        data = torch.load(file_path)
    return data


def logger(epoch=0,
           loss=0,
           score=None,
           target=None,
           time=None,
           verbose=0,
           train=True,
           deep=True):
    """
    Logger for detector.

    Parameters
    ----------
    epoch : int, optional
        The current epoch.
    loss : float, optional
        The current epoch loss value.
    score : torch.Tensor, optional
        The current outlier scores.
    target : torch.Tensor, optional
        The ground truth labels.
    time : float, optional
        The current epoch time.
    verbose : int, optional
        Verbosity mode. Range in [0, 3]. Larger value for printing out
        more log information. Default: ``0``.
    train : bool, optional
        Whether the logger is used for training.
    deep : bool, optional
        Whether the logger is used for deep detector.
    """
    if verbose > 0:
        if deep:
            if train:
                print("Epoch {:04d}: ".format(epoch), end='')
            else:
                print("Test: ", end='')

            if isinstance(loss, tuple):
                print("Loss G {:.4f} | Loss D {:.4f} | "
                      .format(loss[0], loss[1]), end='')
            else:
                print("Loss {:.4f} | ".format(loss), end='')

        if verbose > 1:
            if target is not None:
                auc = eval_roc_auc(target, score)
                print("AUC {:.4f}".format(auc), end='')

            if verbose > 2:
                if target is not None:
                    pos_size = target.nonzero().size(0)
                    rec = eval_recall_at_k(target, score, pos_size)
                    pre = eval_precision_at_k(target, score, pos_size)
                    ap = eval_average_precision(target, score)

                    contamination = sum(target) / len(target)
                    threshold = np.percentile(score,
                                              100 * (1 - contamination))
                    pred = (score > threshold).long()
                    f1 = eval_f1(target, pred)

                    print(" | Recall {:.4f} | Precision {:.4f} "
                          "| AP {:.4f} | F1 {:.4f}"
                          .format(rec, pre, ap, f1), end='')

            if time is not None:
                print(" | Time {:.2f}".format(time), end='')

        print()


def init_detector(name, **kwargs):
    """
    Detector initialization function.
    """
    module = import_module('pygod.detector')
    assert name in module.__all__, "Detector {} not found".format(name)
    return getattr(module, name)(**kwargs)


def init_nn(name, **kwargs):
    """
    Neural network initialization function.
    """
    module = import_module('pygod.nn')
    assert name in module.__all__, "Neural network {} not found".format(name)
    return getattr(module, name)(**kwargs)


def pprint(params, offset=0, printer=repr):
    """Pretty print the dictionary 'params'

    Parameters
    ----------
    params : dict
        The dictionary to pretty print
    offset : int, optional
        The offset at the beginning of each line.
    printer : callable, optional
        The function to convert entries to strings, typically
        the builtin str or repr.
    """

    params_list = list()
    this_line_length = offset
    line_sep = ',\n' + (1 + offset) * ' '
    for i, (k, v) in enumerate(sorted(params.items())):
        if type(v) is float:
            # use str for representing floating point numbers
            # this way we get consistent representation across
            # architectures and versions.
            this_repr = '%s=%s' % (k, str(v))
        else:
            # use repr of the rest
            this_repr = '%s=%s' % (k, printer(v))
        if len(this_repr) > 500:
            this_repr = this_repr[:300] + '...' + this_repr[-100:]
        if i > 0:
            if this_line_length + len(this_repr) >= 75 or '\n' in this_repr:
                params_list.append(line_sep)
                this_line_length = len(line_sep)
            else:
                params_list.append(', ')
                this_line_length += 2
        params_list.append(this_repr)
        this_line_length += len(this_repr)

    lines = ''.join(params_list)
    # Strip trailing space to avoid nightmare in doctests
    lines = '\n'.join(l.rstrip(' ') for l in lines.split('\n'))
    return lines


def is_fitted(detector, attributes=None):
    """
    Check if the detector is fitted.

    Parameters
    ----------
    detector : pygod.detector.Detector
        The detector to check.
    attributes : list, optional
        The attributes to check.
        Default: ``None``.

    Returns
    -------
    is_fitted : bool
        Whether the detector is fitted.
    """
    if attributes is None:
        attributes = ['model']
    assert all(hasattr(detector, attr) and
               eval('detector.%s' % attr) is not None
               for attr in attributes), \
        "The detector is not fitted yet"
