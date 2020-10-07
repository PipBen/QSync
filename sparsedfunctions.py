#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 09:29:29 2016
@author: stefansiwiak-jaszek
"""
import numpy as np
import scipy.integrate as integrate
import scipy.sparse as sp
from scipy.special import factorial


def dissgeneral(o, r):  # generate Loui-D for typical lindblad form. Any operator o, rate r.
    D = r * (sp.kron(o, o.conj()).tocsr() \
             - 0.5 * sp.kron(o.getH() * o, sp.eye(o.shape[1])).tocsr() \
             - 0.5 * sp.kron(sp.eye(o.shape[1]), o.getH() * o).tocsr())
    return D


def dissgeneral2(o, r):  # generate Loui-D for typical lindblad form. Any operator o, rate r.
    D = r * (sp.kron(o, o.conj()).tocsr() \
             - 0.5 * sp.kron(o.getH() * o, sp.eye(o.shape[1])).tocsr() \
             - 0.5 * sp.kron(sp.eye(o.shape[1]), (o.getH() * o).transpose()).tocsr())
    return D


def louiham(H):
    D = sp.kron(H, sp.eye(H.shape[1])).tocsr() - sp.kron(sp.eye(H.shape[1]), H).tocsr()
    return D


def louiham2(H):
    D = sp.kron(H, sp.eye(H.shape[1])).tocsr() - sp.kron(sp.eye(H.shape[1]), H.transpose()).tocsr()
    return D


def initthermal(kBT, N, w):  # thermal dist, in cm-1 please!
    p = np.matrix(np.zeros((N, N)))
    for i in np.arange(N):
        p[i, i] = (1 - np.exp(-w / kBT)) * np.exp(-w * i / kBT)
    return p


def initcoherent(N, x):  # thermal dist, in cm-1 please!
    v = np.zeros(N)
    for i in np.arange(N):
        v[i] = np.exp(-x ** 2 / 2) * (1 / np.sqrt(factorial(i))) * x ** i
    v = np.matrix(v)
    p = np.kron(v, v.getH())
    return p


def initthermal2(kBT, N, w):  # thermal dist, in cm-1 please!
    p = np.matrix(np.zeros((N, N)))
    for i in np.arange(N):
        p[i, i] = np.exp(-w * (2 * i + 1) / kBT)
    norm = p * (1 / np.trace(p))
    return norm


def destroy(N):
    b = np.matrix(np.zeros((N, N)))
    for i in np.arange(N - 1):
        b[i, i + 1] = np.sqrt(i + 1)
    return b


def corrfunc(f1, f2, delta):
    f1bar = np.zeros(np.size(f1) - delta)
    f2bar = np.zeros(np.size(f1) - delta)
    df1df2bar = np.zeros(np.size(f1) - 2 * delta)
    df1sqbar = np.zeros(np.size(f1) - 2 * delta)
    df2sqbar = np.zeros(np.size(f1) - 2 * delta)
    for i in np.arange(np.size(f1) - delta):
        f1bar[i] = (1 / delta) * integrate.trapz(f1[i:i + delta + 1])
        f2bar[i] = (1 / delta) * integrate.trapz(f2[i:i + delta + 1])
    df1 = f1[0:(np.size(f1) - delta)] - f1bar
    df2 = f2[0:(np.size(f1) - delta)] - f2bar
    df1df2 = df1 * df2
    df1sq = df1 ** 2
    df2sq = df2 ** 2
    for i in np.arange(np.size(f1) - 2 * delta):
        df1df2bar[i] = integrate.trapz(df1df2[i:i + delta + 1])
        df1sqbar[i] = integrate.trapz(df1sq[i:i + delta + 1])
        df2sqbar[i] = integrate.trapz(df2sq[i:i + delta + 1])
    C = df1df2bar / np.sqrt(df1sqbar * df2sqbar)
    return C


def saveallnames(omg, ve, rtherm, rel):
    np.save('x1_dw' + omg + '_' + ve + 'V' + rtherm + rel, x1)
    np.save('x1sq_dw' + omg + '_' + ve + 'V' + rtherm + rel, x1sq)
    np.save('x2_dw' + omg + '_' + ve + 'V' + rtherm + rel, x2)
    np.save('x2sq_dw' + omg + '_' + ve + 'V' + rtherm + rel, x2sq)
    np.save('p1_dw' + omg + '_' + ve + 'V' + rtherm + rel, p1)
    np.save('p1sq_dw' + omg + '_' + ve + 'V' + rtherm + rel, p1sq)
    np.save('p2_dw' + omg + '_' + ve + 'V' + rtherm + rel, p2)
    np.save('p2sq_dw' + omg + '_' + ve + 'V' + rtherm + rel, p2sq)
    np.save('m1_dw' + omg + '_' + ve + 'V' + rtherm + rel, m1)
    np.save('m1e0_dw' + omg + '_' + ve + 'V' + rtherm + rel, m1e0)
    np.save('m1e1_dw' + omg + '_' + ve + 'V' + rtherm + rel, m1e1)
    np.save('m1e2_dw' + omg + '_' + ve + 'V' + rtherm + rel, m1e2)
    np.save('m2e2_dw' + omg + '_' + ve + 'V' + rtherm + rel, m1e3)
    np.save('m2e2_dw' + omg + '_' + ve + 'V' + rtherm + rel, m1e4)
    np.save('m2_dw' + omg + '_' + ve + 'V' + rtherm + rel, m2)
    np.save('m2e0_dw' + omg + '_' + ve + 'V' + rtherm + rel, m2e0)
    np.save('m2e1_dw' + omg + '_' + ve + 'V' + rtherm + rel, m2e1)
    np.save('m2e2_dw' + omg + '_' + ve + 'V' + rtherm + rel, m2e2)
    np.save('m2e3_dw' + omg + '_' + ve + 'V' + rtherm + rel, m2e3)
    np.save('m2e4_dw' + omg + '_' + ve + 'V' + rtherm + rel, m2e4)
    np.save('c_x12_dw' + omg + '_' + ve + 'V' + rtherm + rel, c_X12)
    np.save('c_x12sq_dw' + omg + '_' + ve + 'V' + rtherm + rel, c_X12sq)
    #    np.save('c_xpm_dw'+omg+'_'+ve+'V'+rtherm+rel,c_Xpm)
    np.save('c_p12_dw' + omg + '_' + ve + 'V' + rtherm + rel, c_p12)
    np.save('c_p12sq_dw' + omg + '_' + ve + 'V' + rtherm + rel, c_p12sq)
    #    np.save('c_ppm_dw'+omg+'_'+ve+'V'+rtherm+rel,c_ppm)
    np.save('e1_dw' + omg + '_' + ve + 'V' + rtherm + rel, el_1)
    np.save('e2_dw' + omg + '_' + ve + 'V' + rtherm + rel, el_2)
