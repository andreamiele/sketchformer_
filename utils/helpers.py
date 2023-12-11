# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 11:05:57 2016
some help functions to perform basic tasks
@author: tb00083
"""
import os
import sys
import csv
import numpy as np
import json
import pickle  # python3.x
import time
from datetime import timedelta, datetime
import subprocess
import struct
import errno
from pprint import pprint
import glob
import tensorflow as tf
import torch


def get_time_id_str():
    """
    returns a string with DDHHM format, where M is the minutes cut to the tenths
    """
    now = datetime.now()
    time_str = "{:02d}{:02d}{:02d}".format(now.day, now.hour, now.minute)
    time_str = time_str[:-1]
    return time_str


def setup_gpu(gpu_id):
    """
    Set the specified GPU to be used in PyTorch
    :param gpu_id: (int) GPU ID
    """
    # Check if CUDA (GPU support) is available
    if not torch.cuda.is_available():
        print("CUDA (GPU support) is not available. Using CPU...")
        return torch.device('cpu')

    # Ensure the GPU ID is valid
    num_gpus = torch.cuda.device_count()
    if gpu_id >= num_gpus:
        print(f"Invalid GPU ID: {gpu_id}. Using CPU instead.")
        return torch.device('cpu')

    # Set the device to the specified GPU
    device = torch.device(f'cuda:{gpu_id}')
    print(f"Using GPU: {torch.cuda.get_device_name(device)}")
    return device


def read_json_into_hparams(hparams, filepath):
    with open(filepath, 'r') as f:
        json_str = f.read()
        data = json.loads(json_str)
    for key, val in data.items():
        try:
            hparams.add_hparam(key, val)
        except Exception as e:
            pass


def match_matrix_sizes(a, b):
    if len(a) != len(b):
        max_len = max([len(a), len(b)])
        for i in range(max_len):
            if i >= len(a):
                a.append(b[i])
            if i >= len(b):
                b.append(a[i])
    return a, b


def interpolate_a_b(a, b, n, kind='linear'):
    a, b = np.array(a), np.array(b)
    interps = []
    prop = 1. / (n + 1)
    if kind == 'slerp':
        max_, min_ = np.max(np.concatenate((a, b))), np.min(np.concatenate((a, b)))
        rng = max_ - min_
        a_norm = 1. - (((1. - (-1.)) * (max_ - a)) / rng)
        b_norm = 1. - (((1. - (-1.)) * (max_ - b)) / rng)
        omega = np.arccos(a_norm * b_norm)
    for i in range(1, n + 1):
        if kind == 'slerp':
            t = i * prop
            interp = ((np.sin((1 - t) * omega)) / (np.sin(omega))) * a + ((np.sin((t) * omega)) / (np.sin(omega))) * b
        elif kind == 'linear':
            interp = a * (1 - i * prop) + b * i * prop
        interps.append(interp)
    return interps


def incorporate_flags_into_hparams(hparams, flags):
    for key, val in flags.items():
        try:
            hparams.add_hparam(key, val)
        except Exception as e:
            pass


def time_format(t):
    m, s = divmod(t, 60)
    h, m = divmod(m, 60)
    m, h, s = int(m), int(h), int(s)

    if m == 0 and h == 0:
        return "{}s".format(s)
    elif h == 0:
        return "{}m{}s".format(m, s)
    else:
        return "{}h{}m{}s".format(h, m, s)


def restore(model, save_file, raise_if_not_found=False, strict=True):
    if not os.path.exists(save_file) and raise_if_not_found:
        raise Exception('File {} not found'.format(save_file))

    print('Restoring model from {}'.format(save_file))
    model.load_state_dict(torch.load(save_file), strict=strict)
    print('Model restored from {}'.format(save_file))

    try:
        start_iter = int(save_file.split('-')[-1])
    except ValueError:
        print('Could not parse start iter, assuming 0')
        start_iter = 0
    return start_iter


def restore_from_dir(model, folder_path, raise_if_not_found=False):
    start_iter = 0
    latest_checkpoint = None
    highest_iter = -1

    for file in os.listdir(folder_path):
        if file.endswith(".pth") or file.endswith(".pt"):
            try:
                iter = int(file.split('-')[-1].split('.')[0])
                if iter > highest_iter:
                    highest_iter = iter
                    latest_checkpoint = file
            except ValueError:
                pass

    checkpoint_path = os.path.join(folder_path, latest_checkpoint) if latest_checkpoint else None

    if checkpoint_path:
        print('Restoring from latest checkpoint')
        start_iter = restore(model, checkpoint_path)
    else:
        if raise_if_not_found:
            raise Exception('No checkpoint to restore in {}'.format(folder_path))
        else:
            print('No checkpoint to restore in {}'.format(folder_path))

    return start_iter



def get_all_files(dir_path, trim=0, extension=''):
    """
    Recursively get list of all files in the given directory
    trim = 1 : trim the dir_path from results, 0 otherwise
    extension: get files with specific format
    """
    file_paths = []  # List which will store all of the full filepaths.

    # Walk the tree.
    for root, directories, files in os.walk(dir_path):
        for filename in files:
            # Join the two strings in order to form the full filepath.
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)  # Add it to the list.

    if trim == 1:  # trim dir_path from results
        if dir_path[-1] != os.sep:
            dir_path += os.sep
        trim_len = len(dir_path)
        file_paths = [x[trim_len:] for x in file_paths]

    if extension:  # select only file with specific extension
        extension = extension.lower()
        tlen = len(extension)
        file_paths = [x for x in file_paths if x[-tlen:] == extension]

    return file_paths  # Self-explanatory.


def get_all_dirs(dir_path, trim=0):
    """
    Recursively get list of all directories in the given directory
    excluding the '.' and '..' directories
    trim = 1 : trim the dir_path from results, 0 otherwise
    """
    out = []
    # Walk the tree.
    for root, directories, files in os.walk(dir_path):
        for dirname in directories:
            # Join the two strings in order to form the full filepath.
            dir_full = os.path.join(root, dirname)
            out.append(dir_full)  # Add it to the list.

    if trim == 1:  # trim dir_path from results
        if dir_path[-1] != os.sep:
            dir_path += os.sep
        trim_len = len(dir_path)
        out = [x[trim_len:] for x in out]

    return out


def read_list(file_path, delimeter=' ', keep_original=True):
    """
    read list column wise
    deprecated, should use pandas instead
    """
    out = []
    with open(file_path, 'r') as f:
        reader = csv.reader(f, delimiter=delimeter)
        for row in reader:
            out.append(row)
    out = zip(*out)

    if not keep_original:
        for col in range(len(out)):
            if out[col][0].isdigit():  # attempt to convert to numerical array
                out[col] = np.array(out[col]).astype(np.int64)

    return out


def save_pickle2(file_path, **kwargs):
    """
    save variables to file (using pickle)
    """
    # check if any variable is a dict
    var_count = 0
    for key in kwargs:
        var_count += 1
        if isinstance(kwargs[key], dict):
            sys.stderr.write('Opps! Cannot write a dictionary into pickle')
            sys.exit(1)
    with open(file_path, 'wb') as f:
        pickler = pickle.Pickler(f, -1)
        pickler.dump(var_count)
        for key in kwargs:
            pickler.dump(key)
            pickler.dump(kwargs[key])


def load_pickle2(file_path, varnum=0):
    """
    load variables that previously saved using self.save()
    varnum : number of variables u want to load (0 mean it will load all)
    Note: if you are loading class instance(s), you must have it defined in advance
    """
    with open(file_path, 'rb') as f:
        pickler = pickle.Unpickler(f)
        var_count = pickler.load()
        if varnum:
            var_count = min([var_count, varnum])
        out = {}
        for i in range(var_count):
            key = pickler.load()
            out[key] = pickler.load()

    return out


def save_pickle(path, obj):
    """
    simple method to save a picklable object
    :param path: path to save
    :param obj: a picklable object
    :return: None
    """
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_pickle(path):
    """
    load a pickled object
    :param path: .pkl path
    :return: the pickled object
    """
    with open(path, 'rb') as f:
        return pickle.load(f)


def make_new_dir(dir_path, remove_existing=False, mode=511):
    """note: default mode in ubuntu is 511"""
    if not os.path.exists(dir_path):
        try:
            if mode == 777:
                oldmask = os.umask(000)
                os.makedirs(dir_path, 0o777)
                os.umask(oldmask)
            else:
                os.makedirs(dir_path, mode)
        except OSError as exc:  # Python >2.5
            if exc.errno == errno.EEXIST and os.path.isdir(dir_path):
                pass
            else:
                raise
    if remove_existing:
        for file_obj in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file_obj)
            if os.path.isfile(file_path):
                os.unlink(file_path)


class Locker(object):
    """place a lock file in specified location
    useful for distributed computing"""

    def __init__(self, name='lock.txt', mode=511):
        """INPUT: name default file name to be created as a lock
                  mode if a directory has to be created, set its permission to mode"""
        self.name = name
        self.mode = mode

    def lock(self, path):
        make_new_dir(path, False, self.mode)
        with open(os.path.join(path, self.name), 'w') as f:
            f.write('progress')

    def finish(self, path):
        make_new_dir(path, False, self.mode)
        with open(os.path.join(path, self.name), 'w') as f:
            f.write('finish')

    def customise(self, path, text):
        make_new_dir(path, False, self.mode)
        with open(os.path.join(path, self.name), 'w') as f:
            f.write(text)

    def is_locked(self, path):
        out = False
        check_path = os.path.join(path, self.name)
        if os.path.exists(check_path):
            text = open(check_path, 'r').readline().strip()
            out = True if text == 'progress' else False
        return out

    def is_finished(self, path):
        out = False
        check_path = os.path.join(path, self.name)
        if os.path.exists(check_path):
            text = open(check_path, 'r').readline().strip()
            out = True if text == 'finish' else False
        return out

    def clean(self, path):
        check_path = os.path.join(path, self.name)
        if os.path.exists(check_path):
            try:
                os.remove(check_path)
            except Exception as e:
                print('Unable to remove %s: %s.' % (check_path, e))


class ProgressBar(object):
    """show progress"""

    def __init__(self, total, increment=5):
        self.total = total
        self.point = self.total / 100.0
        self.increment = increment
        self.interval = self.total * self.increment / 100
        self.milestones = range(0, total, self.interval) + [self.total, ]
        self.id = 0

    def show_progress(self, i):
        if i >= self.milestones[self.id]:
            while i >= self.milestones[self.id]:
                self.id += 1
            sys.stdout.write("\r[" + "=" * (i / self.interval) +
                             " " * ((self.total - i) / self.interval) + "]" + str(int((i + 1) / self.point)) + "%")
            sys.stdout.flush()


class Timer(object):

    def __init__(self):
        self.start_t = time.time()
        self.last_t = self.start_t

    def time(self, lap=False):
        end_t = time.time()
        if lap:
            out = timedelta(seconds=int(end_t - self.last_t))  # count from last stop point
        else:
            out = timedelta(seconds=int(end_t - self.start_t))  # count from beginning
        self.last_t = end_t
        return out


def get_gpu_free_mem():
    """return a list of free GPU memory"""
    sp = subprocess.Popen(['nvidia-smi', '-q'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out_str = sp.communicate()
    out_list = out_str[0].split('\n')

    out = []
    for i in range(len(out_list)):
        item = out_list[i]
        if item.strip() == 'FB Memory Usage':
            free_mem = int(out_list[i + 3].split(':')[1].strip().split(' ')[0])
            out.append(free_mem)
    return out


def float2hex(x):
    """
    x: a vector
    return: x in hex
    """
    f = np.float32(x)
    out = ''
    if f.size == 1:  # just a single number
        f = [f, ]
    for e in f:
        h = hex(struct.unpack('<I', struct.pack('<f', e))[0])
        out += h[2:].zfill(8)
    return out


def hex2float(x):
    """
    x: a string with len divided by 8
    return x as array of float32
    """
    assert len(x) % 8 == 0, 'Error! string len = {} not divided by 8'.format(len(x))
    l = len(x) / 8
    out = np.empty(l, dtype=np.float32)
    x = [x[i:i + 8] for i in range(0, len(x), 8)]
    for i, e in enumerate(x):
        out[i] = struct.unpack('!f', e.decode('hex'))[0]
    return out


def nice_print(inputs, stream=sys.stdout):
    """print a list of string to file stream"""
    if type(inputs) is not list:
        tstrings = inputs.split('\n')
        pprint(tstrings, stream=stream)
    else:
        for string in inputs:
            nice_print(string, stream=stream)
    stream.flush()


def remove_latest_similar_file_if_it_exists(path):
    list_of_files = glob.glob(path)  # * means all if need specific format then *.csv
    if list_of_files:
        latest_file = max(list_of_files, key=os.path.getctime)
        os.remove(latest_file)


def string_to_int_tuple(s):
    return tuple(int(i) for i in s.split(','))
