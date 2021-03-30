# !/usr/bin/env python
# -*- coding: utf-8 -*-
# File: MVAE algorithm (PyTorch ver.)


import sys
import torch
import net
import numpy as np
import numpy.linalg as LA

epsi = sys.float_info.epsilon


def MVAE(X, AlgPara, NNPara, gpu=-1):
    # check errors and set default values
    I, J, M = X.shape
    N = M
    if N > I:
        sys.stderr.write('The input spectrogram might be wrong. The size of it must be (freq x frame x ch).\n')

    W = np.zeros((I, M, N), dtype=np.complex)
    for i in range(I):
        W[i, :, :] = np.eye(N)

    # Parameter for ILRMA
    if AlgPara['nb'] is None:
        AlgPara['nb'] = np.ceil(J / 10)
    L = AlgPara['nb']
    T = np.maximum(np.random.rand(I, L, N), epsi)
    V = np.maximum(np.random.rand(L, J, N), epsi)

    R = np.zeros((I, J, N))  # variance matrix
    Y = np.zeros((I, J, N), dtype=np.complex)
    for i in range(0, I):
        Y[i, :, :] = (W[i, :, :] @ X[i, :, :].T).T
    P = np.maximum(np.abs(Y) ** 2, epsi)  # power spectrogram

    # ILRMA
    Y, W, R, P = ilrma(X, W, R, P, T, V, AlgPara['it0'], AlgPara['norm'])

    ####  CVAE ####
    # load trained networks
    n_freq = I - 1
    encoder = net.Encoder(n_freq, NNPara['n_src'])
    decoder = net.Decoder(n_freq, NNPara['n_src'])

    checkpoint = torch.load(NNPara['model_path'])
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])

    if gpu >= 0:
        device = torch.device("cuda:{}".format(gpu))
        encoder.cuda(device)
        decoder.cuda(device)
    else:
        device = torch.device("cpu")

    Q = np.zeros((N, I, J))  # estimated variance matrix
    P = P.transpose(2, 0, 1)
    R = R.transpose(2, 0, 1)

    # initial z and l
    Y_abs = abs(Y).astype(np.float32).transpose(2, 0, 1)
    gv = np.mean(np.power(Y_abs[:, 1:, :], 2), axis=(1, 2), keepdims=True)
    Y_abs_norm = Y_abs / np.sqrt(gv)
    eps = np.ones(Y_abs_norm.shape) * epsi
    Y_abs_array_norm = np.maximum(Y_abs_norm, eps)[:, None]

    zs, ls, models, optims = [], [], [], []
    for n in range(N):
        y_abs = torch.from_numpy(np.asarray(Y_abs_array_norm[n, None, :, 1:, :], dtype="float32")).to(device)
        label = torch.from_numpy(np.ones((1, NNPara['n_src']), dtype="float32") / NNPara['n_src']).to(device)
        z = encoder(y_abs, label)[0]
        zs.append(z)
        ls.append(label)
        Q[n, 1:, :] = np.squeeze(np.exp(decoder(z, label).detach().to("cpu").numpy()), axis=1)

    Q = np.maximum(Q, epsi)
    gv = np.mean(np.divide(P[:, 1:, :], Q[:, 1:, :]), axis=(1, 2), keepdims=True)
    Rhat = np.multiply(Q, gv)
    Rhat[:, 0, :] = R[:, 0, :]
    R = Rhat

    # Model construction
    for para in decoder.parameters():
        para.requires_grad = False

    for n in range(N):
        z_para = torch.nn.Parameter(zs[n].type(torch.float), requires_grad=True)
        l_para = torch.nn.Parameter(ls[n].type(torch.float), requires_grad=True)
        src_model = net.SourceModel(decoder, z_para, l_para)
        if gpu >= 0:
            src_model.cuda(device)
        optimizer = torch.optim.Adam(src_model.parameters(), lr=0.01)
        models.append(src_model)
        optims.append(optimizer)

    # initialize z, l by running BP 100 iterations
    Q = np.zeros((N, I, J))
    for n in range(N):
        y_abs = torch.from_numpy(np.asarray(Y_abs_array_norm[n, None, :, 1:, :], dtype="float32")).to(device)
        for iz in range(100):
            optims[n].zero_grad()
            loss = models[n].loss(y_abs)
            loss.backward()
            optims[n].step()
        Q[n, 1:I, :] = models[n].get_power_spec(cpu=True)

    Q = np.maximum(Q, epsi)
    gv = np.mean(np.divide(P[:, 1:I, :], Q[:, 1:I, :]), axis=(1, 2), keepdims=True)
    Rhat = np.multiply(Q, gv)
    Rhat[:, 0, :] = R[:, 0, :]
    R = Rhat

    # Algorithm for MVAE
    # Iterative update
    for it in range(AlgPara['it1']):
        Y_abs_array_norm = Y_abs / np.sqrt(gv)
        for n in range(N):
            y_abs = torch.from_numpy(np.asarray(Y_abs_array_norm[n, None, None, 1:, :], dtype="float32")).to(device)
            for iz in range(100):
                optims[n].zero_grad()
                loss = models[n].loss(y_abs)
                loss.backward()
                optims[n].step()
            Q[n, 1:, :] = models[n].get_power_spec(cpu=True)
        Q = np.maximum(Q, epsi)
        gv = np.mean(np.divide(P[:, 1:, :], Q[:, 1:, :]), axis=(1, 2), keepdims=True)
        Rhat = np.multiply(Q, gv)
        Rhat[:, 0, :] = R[:, 0, :]
        R = Rhat.transpose(1, 2, 0)

        # update W
        W = update_w(X, R, W)
        Y = X @ W.conj()
        Y_abs = np.abs(Y)
        Y_pow = np.power(Y_abs, 2)
        P = np.maximum(Y_pow, epsi)

        if AlgPara['norm']:
            W, R, P = local_normalize(W, R, P, I, J)

        Y_abs = Y_abs.transpose(2, 0, 1)
        P = P.transpose(2, 0, 1)
        R = R.transpose(2, 0, 1)

    return Y


#### Local functions ####
def ilrma(X, W, R, P, T, V, iteration, normalise):
    I, J, N = X.shape
    for n in range(N):
        R[:, :, n] = T[:, :, n] @ V[:, :, n]  # low-rank source model

    # Iterative update
    for it in range(iteration):
        for n in range(N):
            # Update T
            T[:, :, n] = T[:, :, n] * np.sqrt(
                (P[:, :, n] * (R[:, :, n] ** -2)) @ V[:, :, n].T / (R[:, :, n] ** -1 @ V[:, :, n].T))
            T[:, :, n] = np.maximum(T[:, :, n], epsi)
            R[:, :, n] = T[:, :, n] @ V[:, :, n]
            # Update V
            V[:, :, n] = V[:, :, n] * np.sqrt(
                T[:, :, n].T @ (P[:, :, n] * R[:, :, n] ** -2) / (T[:, :, n].T @ (R[:, :, n] ** -1)))
            V[:, :, n] = np.maximum(V[:, :, n], epsi)
            R[:, :, n] = T[:, :, n] @ V[:, :, n]
        # Update W
        W = update_w(X, R, W)

        Y = X @ W.conj()
        Y_abs = np.abs(Y)
        Y_pow = np.power(Y_abs, 2)
        P = np.maximum(Y_pow, epsi)
        if normalise:
            W, R, P, T = local_normalize(W, R, P, I, J, T)

    if iteration == 0:
        Y = X @ W.conj()
        Y_abs = np.abs(Y)
        Y_pow = np.power(Y_abs, 2)
        P = np.maximum(Y_pow, epsi)
        R = P
        if normalise:
            W, R, P, T = local_normalize(W, R, P, I, J, T)

    return Y, W, R, P


def local_normalize(W, R, P, I, J, *args):
    lamb = np.sqrt(np.sum(np.sum(P, axis=0), axis=0) / (I * J))  # 1 x 1 x N

    W = W / np.squeeze(lamb)
    lambPow = lamb ** 2
    P = P / lambPow
    R = R / lambPow
    if len(args) == 1:
        T = args[0]
        T = T / lambPow
        return W, R, P, T
    elif len(args) == 0:
        return W, R, P


def update_w(s, r, w):
    L = w.shape[-1]
    _, N, M = s.shape
    sigma = np.einsum('fnp,fnl,fnq->flpq', s, 1 / r, s.conj())
    sigma /= N
    for l in range(L):
        w[..., l] = LA.solve(
            w.swapaxes(-2, -1).conj() @ sigma[:, l, ...],
            np.eye(L)[None, :, l])
        den = np.einsum(
            'fp,fpq,fq->f',
            w[..., l].conj(), sigma[:, l, ...], w[..., l])
        w[..., l] /= np.maximum(np.sqrt(np.abs(den))[:, None], 1.e-8)
    w += epsi * np.eye(M)
    return w
