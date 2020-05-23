# !/usr/bin/env python
# -*- coding: utf-8 -*-
# File: useful functions


import os
import datetime
import numpy as np
from scipy import signal
from scipy.io import wavfile


class Logger:
    def __init__(self, logf, add=True):
        if not add and os.path.isfile(logf):
            os.remove(logf)
        self.out = open(logf, 'a')
        self.out.write("\n{}\n".format(datetime.datetime.now()))

    def __del__(self):
        if self.out is not None:
            self.close()

    def __call__(self, msg):
        print(msg)
        self.out.write("{}\n".format(msg))
        self.out.flush()

    def close(self):
        self.out.close()
        self.out = None


def set_log(path, add=True):
    if path.endswith('.txt'):
        folder = os.path.dirname(path)
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)
    else:
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, "log.txt")
    logprint = Logger(path, add)

    return path, logprint


def dat_load_trunc(files, path, seg_len, max_size):
    i = 0
    X_train = np.array([])
    for f in files:
        file_path = path + f
        X = np.load(file_path)

        if i == 0:
            X_train = X[1:, :].T
            frame_num_list = np.array([X.shape[1]])
        else:
            X_train = np.append(X_train, X[1:, :].T, axis=0)
            frame_num_list = np.append(frame_num_list, X.shape[1])
        i += 1

    n_frame, n_freq = X_train.shape
    if n_frame > seg_len:
        n_frame = int(n_frame / seg_len) * seg_len
        X_train = X_train[:n_frame, :]
        X_train = X_train.reshape(-1, seg_len, n_freq)
    else:
        X_tmp = np.zeros((seg_len, n_freq), dtype=X_train.dtype)
        X_tmp[:n_frame, :] = X_train
        X_train = X_tmp.reshape(-1, seg_len, n_freq)
    n_seg = X_train.shape[0]

    Y = X_train.real[np.newaxis]
    Y = np.append(Y, X_train.imag[np.newaxis], axis=0)

    Y = np.transpose(Y, (1, 0, 3, 2))

    if n_seg > max_size:
        Y = Y[:maxbsize]
        frame_num_list = frame_num_list[:maxbsize]

    return Y, frame_num_list


def prenorm(stat_path, X):
    # X must be a 4D array with size (N, n_ch, n_freq, n_frame)
    # stat_path is a path for a txt file containing mean and standard deviation of X
    # The txt file is assumed to contain a 1D array with size 2 where
    # the first and second elements are the mean and standard deviation of X.
    if stat_path is None or not os.path.exists(stat_path):
        X_abs = np.linalg.norm(X, axis=1, keepdims=True)
        gv = np.mean(np.power(X_abs, 2), axis=(0, 1, 2, 3), keepdims=True)
        gs = np.sqrt(gv)
        X = X / gs
    else:
        gs = np.load(stat_path)[1]
        X = X / gs

    return X, gs


def back_projection(Y, X):
    I, J, M = Y.shape

    if X.shape[2] == 1:
        A = np.zeros((1, M, I), dtype=np.complex)
        Z = np.zeros((I, J, M),dtype=np.complex)
        for i in range(I):
            Yi = np.squeeze(Y[i, :, :]).T  # channels x frames (M x J)
            Yic = np.conjugate(Yi.T)
            A[0, :, i] = X[i, :, 0] @ Yic @ np.linalg.inv(Yi @ Yic)

        A[np.isnan(A)] = 0
        A[np.isinf(A)] = 0
        for m in range(M):
            for i in range(I):
                Z[i, :, m] = A[0, m, i] * Y[i, :, m]

    elif X.shape[2] == M:
        A = np.zeros(M, M, I)
        Z = np.zeros(I, J, M, M)
        for i in range(I):
            for m in range(M):
                Yi = np.squeeze(Y[i, :, :]).T
                Yic = np.conjugate(Yi.T)
                A[0, :, i] = X[i, :, m] @ Yic @ np.linalg.inv(Yi @ Yic)
        A[np.isnan(A)] = 0
        A[np.isinf(A)] = 0
        for n in range(M):
            for m in range(M):
                for i in range(I):
                    Z[i, :, n, m] = A[m, n, i] * Y[i, :, n]

    else:
        print('The number of channels in X must be 1 or equal to that in Y.')

    return Z


def load_wav(wpath, fs_resample):
    wdir = sorted(os.listdir(wpath))

    # define max length of data
    max_len, n_ch = 0, 0
    for f in wdir:
        path_fname = os.path.join(wpath, f)
        fs, data = wavfile.read(path_fname)
        ddim = data.ndim
        if len(data) > max_len:
            max_len = len(data)
        if ddim == 1:
            data = data.reshape(len(data), -1)
        n_ch += data.shape[1]

    # load data
    sig = np.asarray([]).reshape(max_len, 0)
    for f in wdir:
        path_fname = os.path.join(wpath, f)
        fs, data = wavfile.read(path_fname)
        ddim = data.ndim
        if ddim == 1:
            data = data.reshape(len(data), -1)
        sig_ = np.zeros((max_len, data.shape[1]))
        sig_[:len(data), :] = data

        sig = np.append(sig, sig_, axis=1)

    # resample data
    J = int(np.ceil(sig.shape[0] * fs_resample / fs))
    y = signal.resample(sig, J)
    nsamples = len(y)

    return y
