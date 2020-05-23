# !/usr/bin/env python
# -*- coding: utf-8 -*-
# File: Training script


import os
import math
import argparse
import torch
import utils
import net
import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training CVAE source model")

    parser.add_argument('--gpu', '-g', type=int, help="GPU ID (negative value indicates CPU)", default=-1)
    parser.add_argument('--dataset', '-i', type=str, help="training dataset", choices=["vcc"], default="vcc")
    parser.add_argument('--save_root', '-o', type=str, help="path for saving model", default="./model/")

    parser.add_argument('--epoch', '-e', type=int, help="# of epochs for training", default=1000)
    parser.add_argument('--snapshot', '-s', type=int, help="interval of snapshot", default=100)
    parser.add_argument('--iteration', '-it', type=int, help="number of iterations", default=9)
    parser.add_argument('--lrate', '-lr', type=float, help="learning rate", default=0.0001)
    parser.add_argument('--model_path', '-m', type=str, help="pretrained model", default=None)

    config = parser.parse_args()

    # Constant values
    N_EPOCH = config.epoch
    N_ITER = config.iteration
    SEGLEN = 128


    # =============== Directories and data ===============
    # Make directories and create log file
    save_path = os.path.join(config.save_root, "vcc")
    logprint = utils.set_log(save_path, add=False)[1]

    # Set input directories and data paths
    if config.dataset == "vcc":
        data_root = "./data/vcc/"
    src_folders = sorted(os.listdir(data_root))
    data_paths = ["{}{}/cspec/".format(data_root, f) for f in src_folders]
    stat_paths = ["{}{}/train_cspecstat.npy".format(data_root, f) for f in src_folders]
    label_paths = ["{}{}/label.npy".format(data_root, f) for f in src_folders]
    n_src = len(src_folders)

    src_data = [sorted(os.listdir(p)) for p in data_paths]
    n_src_data = [len(d) for d in src_data]
    src_batch_size = [math.floor(n) // N_ITER for n in n_src_data]
    labels = [np.load(p) for p in label_paths]


    # =============== Set model ==============
    # Set up model and optimizer
    x_tmp = np.load(data_paths[0] + src_data[0][0])
    n_freq = x_tmp.shape[0] - 1
    del x_tmp

    encoder = net.Encoder(n_freq, n_src)
    decoder = net.Decoder(n_freq, n_src)
    cvae = net.CVAE(encoder, decoder)

    n_para_enc = sum(p.numel() for p in encoder.parameters())
    n_para_dec = sum(p.numel() for p in decoder.parameters())

    if config.gpu >= 0:
        device = torch.device("cuda:{}".format(config.gpu))
        cvae.cuda(device)
    else:
        device = torch.device("cpu")

    optimizer = torch.optim.Adam(cvae.parameters(), lr=config.lrate)

    # load pretrained model
    if config.model_path is not None:
        checkpoint = torch.load(config.model_path)
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    start_epoch = 1 if config.model_path is None else checkpoint['epoch'] + 1

    # set cudnn
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


    # =============== Define functions ===============
    def print_msg(epoch, iter, src, batch_size, kl_loss, rec_loss):
        logprint("epoch {}, iter {}, src {}, batch size={}: KLD={}, rec_loss={}".format(
            epoch, iter, src, batch_size, float(kl_loss.data), float(rec_loss.data)))

    def snapshot(epoch):
        print('save the model at {} epoch'.format(epoch))
        torch.save({'epoch': epoch, 'encoder_state_dict': encoder.state_dict(),
                    'decoder_state_dict': decoder.state_dict(), 'optimizer_state_dict': optimizer.state_dict()},
                   os.path.join(save_path, '{}.model'.format(epoch)))

    # =============== Write log file ===============
    logprint("Save folder: {}".format(save_path))
    logprint("\nTraining data:")
    logprint("\tDataset: {}".format(config.dataset))
    logprint("\tNumber of sources: {}".format(n_src))
    logprint("\nNetwork:")
    logprint("\tEncoder architecture:\n\t\t{}".format(encoder))
    logprint("\tDecoder architecture:\n\t\t{}".format(decoder))
    logprint("\tParameter # of encoder: {}".format(n_para_enc))
    logprint("\tParameter # of decoder: {}".format(n_para_dec))
    logprint("\tParameter # of the whole network: {}".format(n_para_enc + n_para_dec))
    if config.model_path is not None:
        logprint("\tPretrained model: {}".format(config.model_path))
    logprint("\nTraining conditions:")
    print("\tGPU #: {}".format(config.gpu))
    logprint("\tEpoch #: {}".format(N_EPOCH))
    logprint("\tIteration #: {}".format(N_ITER))
    logprint("\tFile # used in a batch: {}".format(src_batch_size))
    logprint("\tOptimizer: Adam")
    logprint("\tLearning rate: {}".format(config.lrate))
    logprint("\tSnapshot: every {} iteration(s)".format(config.snapshot))

    # =============== Train model ===============
    try:
        for epoch in range(start_epoch, N_EPOCH+1):
            perms = [np.random.permutation(n) for n in n_src_data]
            perms_data = []

            for i in range(n_src):
                perms_data.append([src_data[i][j] for j in perms[i]])

            for i in range(N_ITER):
                # data pre-processing
                for j in range(n_src):
                    x = utils.dat_load_trunc(perms_data[j][i*src_batch_size[j] : (i+1)*src_batch_size[j]],
                                            data_paths[j], SEGLEN, 16)[0]
                    # Normalization
                    x = utils.prenorm(stat_paths[j], x)[0]
                    # Turn X into magnitude spectrograms
                    mag_x = np.linalg.norm(x, axis=1, keepdims=True)
                    # to GPU
                    x = torch.from_numpy(np.asarray(mag_x, dtype="float32")).to(device)
                    l = torch.from_numpy(np.asarray(labels[j], dtype="float32")).to(device)

                    # update trainable parameters
                    optimizer.zero_grad()
                    x_logvar = cvae(x, l)
                    loss, kl_loss, rec_loss = cvae.loss(x)
                    loss.backward()
                    optimizer.step()

                    print_msg(epoch, i+1, j+1, x.size(0), kl_loss, rec_loss)

            if epoch % config.snapshot == 0:
                snapshot(epoch)

    except KeyboardInterrupt:
        logprint("\nKeyboard interrupt, exit.")

    else:
        logprint("Training done!")

    finally:
        print("Output: {}".format(save_path))
        logprint.close()
