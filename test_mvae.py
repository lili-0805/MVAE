# !/usr/bin/env python
# -*- coding: utf-8 -*-
# File: MVAE test file


import os
import argparse
import utils
import numpy as np
from scipy.io import wavfile
from scipy import signal
from MVAE import MVAE


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test script for MVAE")
    parser.add_argument('--gpu', '-g', type=int, help="GPU ID (negative value indicates CPU)", default=-1)
    parser.add_argument('--input_root', '-i', type=str, help="path for input test data", default="./data/test_input/")
    parser.add_argument('--output_root', '-o', type=str, help="path for output data", default="./output/")
    parser.add_argument('--n_src', '-n', type=int, help="number of sources", default=4)

    parser.add_argument('--fs', '-r', type=int, help="Resampling frequency", default=16000)
    parser.add_argument('--fft_size', '-l', type=int, help="Frame length of STFT in sample points", default=2048)
    parser.add_argument('--shift_size', '-s', type=int, help="Frame shift of STFT in samplie points", default=1024)

    parser.add_argument('--nb', '-nb', type=int, help="Number of basis for initialization", default=1)
    parser.add_argument('--n_itr0', '-n0', type=int, help="Number of iterations for initialization using ILRMA", default=30)
    parser.add_argument('--n_itr1', '-n1', type=int, help="Number of iterations for MVAE", default=30)

    parser.add_argument('--model_path', '-m', type=str, help="path for a trained encoder")

    args = parser.parse_args()

    # ============== parameter and path settings ================
    STFTPara = {'fs': args.fs, 'window_size': args.fft_size, 'window_shift': args.shift_size, 'type': "hamming"}
    AlgPara = {'it0': args.n_itr0, 'it1': args.n_itr1,
               'nb': args.nb, 'whitening': False, 'norm': True, 'RefMic': 0}
    NNPara = {'n_src': args.n_src, 'model_path': args.model_path}

    # path
    input_root = args.input_root
    output_root = args.output_root
    os.makedirs(output_root, exist_ok=True)
    input_dir_names = sorted(os.listdir(input_root))

    gpu = args.gpu

    # ================ separation ================
    for idx, f in enumerate(input_dir_names):
        input_dir = os.path.join(input_root, f)
        print("Processing {}...".format(input_dir), end="")
        save_dir = os.path.join(output_root, f)
        os.makedirs(save_dir, exist_ok=True)

        # Input data and resample
        mix = utils.load_wav(input_dir, STFTPara['fs'])
        ns = mix.shape[1]

        # STFT
        frames_ = np.floor((mix.shape[0] + 2*STFTPara['window_shift']) / STFTPara['window_shift'])  # to meet NOLA
        frames = int(np.ceil(frames_ / 8) * 8)

        X = np.zeros((int(STFTPara['window_size'] / 2 + 1), int(frames), mix.shape[1]), dtype=np.complex)
        for n in range(mix.shape[1]):
            f, t, X[:, :int(frames_), n] = signal.stft(mix[:, n], nperseg=STFTPara['window_size'],
                        window=STFTPara['type'], noverlap=STFTPara['window_size'] - STFTPara['window_shift'])

        # source separation
        Y = MVAE(X, AlgPara, NNPara, gpu)

        # projection back
        XbP = np.zeros((X.shape[0], X.shape[1], 1), dtype=np.complex)
        XbP[:, :, 0] = X[:, :, AlgPara['RefMic']]
        Z = utils.back_projection(Y, XbP)

        # iSTFT and save
        for n in range(ns):
            sep = signal.istft(Z[:, :, n], window=STFTPara['type'])[1]
            sep = sep / max(abs(sep)) * 30000
            wavfile.write(os.path.join(save_dir, 'estimated_signal{}.wav'.format(n)), STFTPara['fs'], sep.astype(np.int16))

        print("Done!")
