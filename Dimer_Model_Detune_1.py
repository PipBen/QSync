#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""

Created on Mon Feb 27 11:02:44 2017

Code for two coupled dimer chromophores, strongly coupled to modes. 

Tensor order is excitonic, mode 1, mode 2.

@author: stefansiwiak-jaszek



If you want a zoomable navigable figure use matplotlib qt5. Default is inline



"""

import numpy as np

import scipy.constants as constant

import scipy.integrate as integrate

import sparsedfunctions as sj

import time

import scipy.sparse as sp


import QCcorrelations as QC

# %% Define Parameters all in cm-1


omega = 1111

huang = 0.0578

detuning = 1.00

w1 = omega

w2 = detuning * omega

e1 = 0 + w1 * huang

e2 = 1042 + w2 * huang  # 946.412

de = e2 - e1

V = 92  # 236.603 # cm-1

dE = np.sqrt(de ** 2 + 4 * V ** 2)  # energy between excitonic states

theta = 0.5 * np.arctan(2 * V / de)

a = 1

g1 = w1 * np.sqrt(huang) * a

g2 = w2 * np.sqrt(huang) * a

##ALTERNATIVE PARAMETER REGIME 

# heta = 0.5

# theta = 0.5*np.arctan(heta)

# theta = 0.5*np.arctan(2*0.0001*92/1042) #original

# dE = np.sqrt(1042**2+4*0.0001*92**2)

# de = np.sqrt(dE**2/(1+np.tan(2*theta)**2))

# de = 1042

# V = 1*92 # cm-1

# V = np.tan(2*theta)*de*0.5

# omega = 1111

# omega = np.sqrt(de**2+4*V**2)

# domega = 0*omega # 10 cm-1

# w1 = omega

# w2 = omega + domega

# dE = np.sqrt(de**2+4*V**2) # energy between excitonic states

# theta = 0.5*np.arctan(2*V/de)

# g = 267.1 #cm-1

# g = 0

## Maximum amplitude

# coh = 1/(1+((dE-omega)/(2*g*np.sin(2*theta)))**2)


kBT = (constant.k * (298)) / (constant.h * constant.c * 100)  # cm-1

thermal_dissipation = 33.3564  # 70

electronic_dephasing = 333.564

# taudiss = 1 / (1e-12 * thermal_dissipation * 100 * constant.c)

# taudeph = 1 / (1e-12 * electronic_dephasing * 100 * constant.c)

taudeph = 1 / (1e-12 * thermal_dissipation * 100 * constant.c)

taudiss = 1 / (1e-12 * electronic_dephasing * 100 * constant.c) 

N = 5  # 7

# %% scaling effects from setting 2pi x c = 1 and hbar = 1

r_el = electronic_dephasing / (2 * constant.pi)

r_th = thermal_dissipation / (2 * constant.pi)

r_v1 = r_th  # 6 # cm-1

r_v2 = r_th  # 6 # cm-1

# %% Define Pure State Vectors and Operators in Subspaces


# Exciton Vectors

E1 = sp.lil_matrix(np.matrix([[1.], [0.]])).tocsr()

E2 = sp.lil_matrix(np.matrix([[0.], [1.]])).tocsr()

# Exciton Operators

oE1 = sp.kron(E1, E1.getH()).tocsr()

oE2 = sp.kron(E2, E2.getH()).tocsr()

oE2E1 = sp.kron(E2, E1.getH()).tocsr()  # sigma plus

oE1E2 = sp.kron(E1, E2.getH()).tocsr()  # sigma minus

sigmaz = oE2 - oE1

sigmax = oE2E1 + oE1E2

Ie = sp.eye(2, 2).tocsr()

elmix = np.cos(2 * theta) * sigmaz - np.sin(2 * theta) * sigmax

# Unitary rotation from site to exciton

U = sp.lil_matrix(np.matrix([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])).tocsr()

# U = sp.lil_matrix(np.matrix([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])).tocsr()

eldis1 = U * oE1 * U.getH()

eldis2 = U * oE2 * U.getH()

# Mode Vectors

mode0 = np.matrix(np.zeros([N, 1]))

mode1 = np.matrix(np.zeros([N, 1]))

mode2 = np.matrix(np.zeros([N, 1]))

mode3 = np.matrix(np.zeros([N, 1]))

mode4 = np.matrix(np.zeros([N, 1]))

mode0[0, 0] = 1

mode1[1, 0] = 1

mode2[2, 0] = 1

# mode3[3,0] = 1

# mode4[4,0] = 1


# Mode Operators

om0 = sp.lil_matrix(np.kron(mode0, mode0.getH())).tocsr()

om0m1 = sp.lil_matrix(np.kron(mode0, mode1.getH())).tocsr()

om1m0 = sp.lil_matrix(np.kron(mode1, mode0.getH())).tocsr()

om1 = sp.lil_matrix(np.kron(mode1, mode1.getH())).tocsr()

om2 = sp.lil_matrix(np.kron(mode2, mode2.getH())).tocsr()

# om3 = sp.lil_matrix(np.kron(mode3,mode3.getH())).tocsr()

# om4 = sp.lil_matrix(np.kron(mode4,mode4.getH())).tocsr()

b = sp.lil_matrix(sj.destroy(N)).tocsr()

Iv = sp.eye(N, N).tocsr()

M1thermal = sp.lil_matrix(sj.initthermal(kBT, N, w1)).tocsr()

M2thermal = sp.lil_matrix(sj.initthermal(kBT, N, w2)).tocsr()

# %% Define Exciton-Vibration State Vectors in Full Space


E1m0m0 = sp.lil_matrix(sp.kron(E1, sp.kron(mode0, mode0))).tocsr()

E1m1m0 = sp.lil_matrix(sp.kron(E1, sp.kron(mode1, mode0))).tocsr()

E1m0m1 = sp.lil_matrix(sp.kron(E1, sp.kron(mode0, mode1))).tocsr()

E1m1m1 = sp.lil_matrix(sp.kron(E1, sp.kron(mode1, mode1))).tocsr()

E1m2m0 = sp.lil_matrix(sp.kron(E1, sp.kron(mode2, mode0))).tocsr()

E1m0m2 = sp.lil_matrix(sp.kron(E1, sp.kron(mode0, mode2))).tocsr()

E2m0m0 = sp.lil_matrix(sp.kron(E2, sp.kron(mode0, mode0))).tocsr()

E2m0m1 = sp.lil_matrix(sp.kron(E2, sp.kron(mode0, mode1))).tocsr()

E2m1m0 = sp.lil_matrix(sp.kron(E2, sp.kron(mode1, mode0))).tocsr()

E2m1m1 = sp.lil_matrix(sp.kron(E2, sp.kron(mode1, mode1))).tocsr()

# %% Full Space Projectors


# pure states

oE2m0m0 = sp.kron(E2m0m0, E2m0m0.getH()).tocsr()

oE2m0m1 = sp.kron(E2m0m1, E2m0m1.getH()).tocsr()

oE2m1m0 = sp.kron(E2m1m0, E2m1m0.getH()).tocsr()

oE2m1m1 = sp.kron(E2m1m1, E2m1m1.getH()).tocsr()

oE1m0m0 = sp.kron(E1m0m0, E1m0m0.getH()).tocsr()

oE1m1m0 = sp.kron(E1m1m0, E1m1m0.getH()).tocsr()

oE1m0m1 = sp.kron(E1m0m1, E1m0m1.getH()).tocsr()

oE1m1m1 = sp.kron(E1m1m1, E1m1m1.getH()).tocsr()

oE1m2m0 = sp.kron(E1m2m0, E1m2m0.getH()).tocsr()

oE1m0m2 = sp.kron(E1m0m2, E1m0m2.getH()).tocsr()

oE2E1m0m0 = sp.kron(oE2E1, sp.kron(om0, om0)).tocsr()

oE1E2m0m0 = sp.kron(oE1E2, sp.kron(om0, om0)).tocsr()

oE100E200 = sp.kron(E1m0m0, E2m0m0.getH()).tocsr()

# local operators

oB1 = sp.kron(Ie, sp.kron(b, Iv)).tocsr()

oB2 = sp.kron(Ie, sp.kron(Iv, b)).tocsr()

oE1mImI = sp.kron(oE1, sp.kron(Iv, Iv)).tocsr()

oE2mImI = sp.kron(oE2, sp.kron(Iv, Iv)).tocsr()

oE1E2mImI = sp.kron(oE1E2, sp.kron(Iv, Iv)).tocsr()

oE2E1mImI = sp.kron(oE2E1, sp.kron(Iv, Iv)).tocsr()

oEIm0mI = sp.kron(Ie, sp.kron(om0, Iv)).tocsr()

oEIm1mI = sp.kron(Ie, sp.kron(om1, Iv)).tocsr()

oEIm2mI = sp.kron(Ie, sp.kron(om2, Iv)).tocsr()

oEImIm0 = sp.kron(Ie, sp.kron(Iv, om0)).tocsr()

oEImIm1 = sp.kron(Ie, sp.kron(Iv, om1)).tocsr()

oEImIm2 = sp.kron(Ie, sp.kron(Iv, om2)).tocsr()

oEIm0m0 = sp.kron(Ie, sp.kron(om0, om0)).tocsr()

oEIm1m1 = sp.kron(Ie, sp.kron(om0, om0)).tocsr()

oEIm2m2 = sp.kron(Ie, sp.kron(om2, om2)).tocsr()

oEIm0m1 = sp.kron(Ie, sp.kron(om0, om1)).tocsr()

oEIm1m0 = sp.kron(Ie, sp.kron(om1, om0)).tocsr()

oEIm0m1mI = sp.kron(Ie, sp.kron(om0m1, Iv)).tocsr()

oEIm1m0mI = sp.kron(Ie, sp.kron(om1m0, Iv)).tocsr()

oEImIm1m0 = sp.kron(Ie, sp.kron(Iv, om1m0)).tocsr()

oEImIm0m1 = sp.kron(Ie, sp.kron(Iv, om0m1)).tocsr()

oEIm1m0m1m0 = sp.kron(Ie, sp.kron(om1m0, om1m0)).tocsr()

oX1 = oB1 + oB1.getH()

oX1sq = oX1.dot(oX1)

oX2 = oB2 + oB2.getH()

oX2sq = oX2.dot(oX2)

oM1 = oB1.getH() * oB1

oM1sq = oM1.dot(oM1)

oM2 = oB2.getH() * oB2

oM2sq = oM2.dot(oM2)

oP1 = -1j * (oB1 - oB1.getH())

oP1sq = oP1.dot(oP1)

oP2 = -1j * (oB2 - oB2.getH())

oP2sq = oP2.dot(oP2)

sigmax_I_I = sp.kron(sigmax, sp.kron(Iv, Iv)).tocsr()

sigmaz_I_I = sp.kron(sigmaz, sp.kron(Iv, Iv)).tocsr()

# Generate Hamiltonian in full space.

H = sp.kron((dE / 2) * sigmaz, sp.kron(Iv, Iv)) \
    + sp.kron(Ie, sp.kron(w1 * b.getH() * b, Iv)) \
    + sp.kron(Ie, sp.kron(Iv, w2 * b.getH() * b)) \
    + g1 * sp.kron(eldis1, sp.kron((b + b.getH()), Iv)) \
    + g2 * sp.kron(eldis2, sp.kron(Iv, (b + b.getH())))

# Stationary Hamiltonian Solutions

enE2m0m0 = np.trace((H * oE2m0m0).todense())

enE2m0m1 = np.trace((H * oE2m0m1).todense())

enE2m1m0 = np.trace((H * oE2m1m0).todense())

enE2m1m1 = np.trace((H * oE2m1m1).todense())

enE1m0m0 = np.trace((H * oE1m0m0).todense())

enE1m1m0 = np.trace((H * oE1m1m0).todense())

enE1m0m1 = np.trace((H * oE1m0m1).todense())

enE1m1m1 = np.trace((H * oE1m1m1).todense())

enE1m2m0 = np.trace((H * oE1m2m0).todense())

enE1m0m2 = np.trace((H * oE1m0m2).todense())

vals, eigs = np.linalg.eigh(H.todense())

# %% Initial Conditions


P0 = sp.kron(oE2, sp.kron(M1thermal, M2thermal)).todense()

# P0 = oE2m0m0.todense()

# P0 = sp.kron(orderedeig5[:,1],orderedeig5[:,1].getH()).todense()

# P0 = eqstate_0dw_1V.reshape(50,50)

# P0 = opsi02

# P0 = rho_4ps_PE545

# P0 = rho8ps


# %%


P0eig = eigs.getH() * P0 * eigs

##

# opsi00 = np.kron(orderedeig5[:,0],orderedeig5[:,0].getH()) # eigenvectors in exciton mode mode basis

# psi00eig = orderedeig5.getH()*psi00*orderedeig5 # eigenvector in eigenvector basis. Each state is a 1,0,0,0,0,0... vector that makes the diagonal of H.

# opsi11 = np.kron(eigs[:,1],eigs[:,1].getH())

# opsi22 = np.kron(orderedeig5[:,2],orderedeig5[:,2].getH())

# opsi33 = np.kron(orderedeig5[:,3],orderedeig5[:,3].getH())

# opsi44 = np.kron(orderedeig5[:,4],orderedeig5[:,4].getH())

# opsi55 = np.kron(orderedeig5[:,5],orderedeig5[:,5].getH())

# opsi66 = np.kron(orderedeig5[:,6],orderedeig5[:,6].getH())

# opsi77 = np.kron(orderedeig5[:,7],orderedeig5[:,7].getH())

# opsi88 = np.kron(orderedeig5[:,8],orderedeig5[:,8].getH())

# opsi99 = np.kron(orderedeig5[:,9],orderedeig5[:,9].getH())


opsi01 = np.kron(eigs[:, 0], eigs[:, 1].getH())

opsi10 = np.kron(eigs[:, 1], eigs[:, 0].getH())

opsi03 = np.kron(eigs[:, 0], eigs[:, 3].getH())

opsi14 = np.kron(eigs[:, 1], eigs[:, 4].getH())

opsi13 = np.kron(eigs[:, 1], eigs[:, 3].getH())

opsi16 = np.kron(eigs[:, 1], eigs[:, 6].getH())

opsi25 = np.kron(eigs[:, 2], eigs[:, 5].getH())

opsi36 = np.kron(eigs[:, 3], eigs[:, 6].getH())

opsi27 = np.kron(eigs[:, 2], eigs[:, 7].getH())

opsi28 = np.kron(eigs[:, 2], eigs[:, 8].getH())

opsi38 = np.kron(eigs[:, 3], eigs[:, 8].getH())

opsi02 = np.kron(eigs[:, 0], eigs[:, 2].getH())

opsi15 = np.kron(eigs[:, 1], eigs[:, 5].getH())

opsi26 = np.kron(eigs[:, 2], eigs[:, 6].getH())

opsi37 = np.kron(eigs[:, 3], eigs[:, 7].getH())

#

##alphax1_01 = orderedeig5[:,0].getH()*oX1.todense()*orderedeig5[:,1]

# alphax1_01test = np.trace(psi01*oX1)

#

oX1eig = eigs.getH() * oX1 * eigs

oX2eig = eigs.getH() * oX2 * eigs

diffX12chop = np.tril(oX1eig - oX2eig, k=-1)

##

coefx1 = np.multiply(oX1eig, P0eig)

coefx2 = np.multiply(oX2eig, P0eig)

scalexpsi01 = oX1eig[0, 1]

##

coefx1chop = np.tril(coefx1,k=-1)

coefx2chop = np.tril(coefx2,k=-1)

##

omegaarray = np.repeat(vals, 2 * N ** 2).reshape(2 * N ** 2, 2 * N ** 2) - np.repeat(vals, 2 * N ** 2).reshape(
    2 * N ** 2, 2 * N ** 2).transpose()

omegachop = np.tril(omegaarray,k=-1)

# anaX1array = np.zeros([2*N**2,2*N**2,np.size(t_cm)])

# anaX2array = np.zeros([2*N**2,2*N**2,np.size(t_cm)])

# anaX1 = np.zeros(np.size(t_cm))

# anaX2 = np.zeros(np.size(t_cm))

#

# for a in np.arange(np.size(t_cm)):

#    anaX1array[:,:,a] = np.multiply((2*np.cos(omegachop*2*constant.pi*t_cm[a])),coefx1chop)

#    anaX1[a] = np.sum(anaX1array[:,:,a]) + np.trace(coefx1)

#    anaX2array[:,:,a] = np.multiply((2*np.cos(omegachop*2*constant.pi*t_cm[a])),coefx2chop)

#    anaX2[a] = np.sum(anaX2array[:,:,a]) + np.trace(coefx2)


f13 = np.round(np.abs(omegaarray[1, 3]), decimals=2)

f01 = np.round(np.abs(omegaarray[0, 1]), decimals=2)

f02 = np.round(np.abs(omegaarray[0, 2]), decimals=2)

f03 = np.round(np.abs(omegaarray[0, 3]), decimals=2)

f14 = np.round(np.abs(omegaarray[1, 4]), decimals=2)

f15 = np.round(np.abs(omegaarray[1, 5]), decimals=2)

f37 = np.round(np.abs(omegaarray[3, 7]), decimals=2)

f38 = np.round(np.abs(omegaarray[3, 8]), decimals=2)

f13 = np.round(np.abs(omegaarray[1, 3]), decimals=2)

# ph13 = np.round(np.log(np.abs(oX1eig[1,3]/oX2eig[1,3])),decimals=3)

# ph01 = np.round(np.log(np.abs(oX1eig[0,1]/oX2eig[0,1])),decimals=3)

# ph02 = np.round(np.log(np.abs(oX1eig[0,2]/oX2eig[0,2])),decimals=3)

# ph03 = np.round(np.log(np.abs(oX1eig[0,3]/oX2eig[0,3])),decimals=3)

# ph14 = np.round(np.log(np.abs(oX1eig[1,4]/oX2eig[1,4])),decimals=3)

# ph15 = np.round(np.log(np.abs(oX1eig[1,5]/oX2eig[1,5])),decimals=3)

# ph37 = np.round(np.log(np.abs(oX1eig[3,7]/oX2eig[3,7])),decimals=3)

# ph38 = np.round(np.log(np.abs(oX1eig[3,8]/oX2eig[3,8])),decimals=3)

#

#

# phases = np.zeros([oX1eig.shape[0],oX1eig.shape[0]])

# for i in np.arange(oX1eig.shape[0]):

#    for k in np.arange(oX1eig.shape[1]):        

#        phases[i,k] = np.round(np.log(np.abs(oX2eig[i,k]/oX1eig[i,k])),decimals=3)

#        if oX2eig[i,k] < 0:

#            phases[i,k] += -np.round(np.pi,decimals=3)


# ph01 = oX1eig[0,1]/oX2eig[0,1]


# %% Hamiltonian Evolution


# LH = -1j*sj.louiham(H)

##LH = -1j*sj.louiham(Heff)

# RowP0 = P0.reshape(1,(P0.shape[1])**2)

#    

# def ge(t,y): # need to define a function for the ODE solver. output must be a row vector.

#    pt = LH.dot(y.transpose()).transpose() # the lindblad master equation. Row*matrixTranspose

#    return pt

#

# evo = integrate.complex_ode(ge) # call my ode solver evo.

# t0 = 0 # start time

# tmax_ps = 5 #5

# tmax = tmax_ps*100*constant.c*2*constant.pi*1e-12 #3 end time

# dt = ((2*constant.pi)/omega)/100 # 0.0001 # time steps at which we want to record the data. The solver will

#           # automatically choose the best time step for calculation.

# steps = np.int((tmax-t0)/dt) # total number of steps. Must be int.

# evo.set_initial_value(RowP0,t0) # initial conditions

#

## Next we define the empty arrays we are going to store to.

# t = np.zeros((steps))

# t[0] = t0

# rhoT = np.zeros((steps,P0.shape[1]**2),dtype=complex) # time is 3rd dim.

# rhoT[0,:] = RowP0

#

#

## now do the iteration.

# k=1

# while evo.successful() and k < steps:

#    evo.integrate(evo.t + dt) # time to integrate at at each loop

#    t[k] = evo.t # save current loop time

#    rhoT[k,:] = evo.y # save current loop data

#    k += 1 # keep index increasing with time.


# %% Dissipator (Lindblad) Operators


ELdis1 = sp.kron(eldis1, sp.kron(Iv, Iv)).tocsr()

ELdis2 = sp.kron(eldis2, sp.kron(Iv, Iv)).tocsr()

# %% Lindblad Master Equation Evolution


nw1 = 1 / (np.exp(w1 / kBT) - 1)  # thermal distribution N - see notes.

nw2 = 1 / (np.exp(w2 / kBT) - 1)

L = -1j * sj.louiham2(H) + sj.dissgeneral2(ELdis1, r_el) + sj.dissgeneral2(ELdis2, r_el) \
    + sj.dissgeneral2(oB1, r_v1 * (nw1 + 1)) + sj.dissgeneral2(oB1.getH(), r_v1 * nw1) \
    + sj.dissgeneral2(oB2, r_v2 * (nw2 + 1)) + sj.dissgeneral2(oB2.getH(), r_v2 * nw2)

RowP0 = P0.reshape(1, (P0.shape[1]) ** 2)


def f(t, y):  # need to define a function for the ODE solver. output must be a row vector.

    pt = L.dot(y.transpose()).transpose()  # the lindblad master equation. Row*matrixTranspose

    return pt


count1 = time.time()

evo = integrate.complex_ode(f)  # call my ode solver evo.

t0 = 0  # start time

tmax_ps = 4

tmax = tmax_ps * 100 * constant.c * 2 * constant.pi * 1e-12  # 3 end time

dt = ((2 * constant.pi) / omega) / 100  # 0.0001 # time steps at which we want to record the data. The solver will

# automatically choose the best time step for calculation.

steps = np.int((tmax - t0) / dt)  # total number of steps. Must be int.

evo.set_initial_value(RowP0, t0)  # initial conditions

# Next we define the empty arrays we are going to store to.

t = np.zeros(steps)

t[0] = t0

rhoT = np.zeros((steps, P0.shape[1] ** 2), dtype=complex)  # time is 3rd dim.

rhoT[0, :] = RowP0

# now do the iteration.

k = 1

while evo.successful() and k < steps:
    evo.integrate(evo.t + dt)  # time to integrate at at each loop

    t[k] = evo.t  # save current loop time

    rhoT[k, :] = evo.y  # save current loop data

    k += 1  # keep index increasing with time.

count2 = time.time()

print('Integration =', count2 - count1)

# %%

#

# count2 = time.time()

#

# nw1 = 1/(np.exp(w1/kBT)-1) # thermal distribution N - see notes.

# nw2 = 1/(np.exp(w2/kBT)-1)

#

# Q = sj.dissgeneral(ELdis1,r_el) + sj.dissgeneral(ELdis2,r_el) \

#    + sj.dissgeneral(oB1,r_v1*(nw1+1)) + sj.dissgeneral(oB1.getH(),r_v1*nw1)     \

#    + sj.dissgeneral(oB2,r_v2*(nw2+1)) + sj.dissgeneral(oB2.getH(),r_v2*nw2)   

#

# eigL = np.linalg.eig(Q.todense())

##inds = np.abs(np.imag(eigL[0])).argsort()

# inds = eigL[0].argsort()

# inds = inds[::-1]

# orderedeigL = eigL[1][:,inds]

# orderedenL = eigL[0][inds]

#

# count3 = time.time()

# print('Diagonalise Louivillian =',count3-count2)


# %%

# oL0 = orderedeigL[:,0].reshape(50,50)

# oL0eig = orderedeig5.getH()*oL0*orderedeig5

# oL1 = orderedeigL[:,1].reshape(50,50)

# oL1eig = orderedeig5.getH()*oL1*orderedeig5

# oL2 = orderedeigL[:,2].reshape(50,50)

# oL2eig = orderedeig5.getH()*oL2*orderedeig5

# oL3 = orderedeigL[:,3].reshape(50,50)

# oL3eig = orderedeig5.getH()*oL3*orderedeig5

# oL4 = orderedeigL[:,4].reshape(50,50)

# oL4eig = orderedeig5.getH()*oL4*orderedeig5

# oL5 = orderedeigL[:,5].reshape(50,50)

# oL5eig = orderedeig5.getH()*oL5*orderedeig5

# oL6 = orderedeigL[:,6].reshape(50,50)

# oL6eig = orderedeig5.getH()*oL6*orderedeig5

# oL7 = orderedeigL[:,7].reshape(50,50)

# oL7eig = orderedeig5.getH()*oL7*orderedeig5

# oL8 = orderedeigL[:,8].reshape(50,50)

# oL8eig = orderedeig5.getH()*oL8*orderedeig5

# oL9 = orderedeigL[:,9].reshape(50,50)

# oL9eig = orderedeig5.getH()*oL9*orderedeig5

#

# oLsum2to9 = oL2+oL3+oL4+oL5+oL6+oL7+oL8+oL9


# %%

# oL111 = orderedeigL[:,23].reshape(50,50)

# oL111eig = orderedeig5.getH()*oL111*orderedeig5


# %%


count2 = time.time()

x1 = np.zeros((steps))

for i in np.arange(steps):
    x1[i] = np.real(np.trace(oX1.dot(rhoT[i, :].reshape(np.shape(P0)[0], np.shape(P0)[1]))))

# x1sq = np.zeros((steps))

# for i in np.arange(steps):

#    x1sq[i] = np.real(np.trace(oX1sq.dot(rhoT[i,:].reshape(np.shape(P0)[0],np.shape(P0)[1]))))


x2 = np.zeros((steps))

for i in np.arange(steps):
    x2[i] = np.real(np.trace(oX2.dot(rhoT[i, :].reshape(np.shape(P0)[0], np.shape(P0)[1]))))

# x2sq = np.zeros((steps))

# for i in np.arange(steps):

#    x2sq[i] = np.real(np.trace(oX2sq.dot(rhoT[i,:].reshape(np.shape(P0)[0],np.shape(P0)[1]))))


bb1 = np.zeros((steps))

for i in np.arange(steps):
    bb1[i] = np.real(np.trace(oM1.dot(rhoT[i, :].reshape(np.shape(P0)[0], np.shape(P0)[1]))))

# bb1sq = np.zeros((steps))

# for i in np.arange(steps):

#    bb1sq[i] = np.real(np.trace(oM1sq.dot(rhoT[i,:].reshape(np.shape(P0)[0],np.shape(P0)[1]))))


bb2 = np.zeros((steps))

for i in np.arange(steps):
    bb2[i] = np.real(np.trace(oM2.dot(rhoT[i, :].reshape(np.shape(P0)[0], np.shape(P0)[1]))))

# bb2sq = np.zeros((steps))

# for i in np.arange(steps):

#    bb2sq[i] = np.real(np.trace(oM2sq.dot(rhoT[i,:].reshape(np.shape(P0)[0],np.shape(P0)[1]))))


ex1 = np.zeros((steps))

for i in np.arange(steps):
    ex1[i] = np.real(np.trace(oE1mImI.dot(rhoT[i, :].reshape(np.shape(P0)[0], np.shape(P0)[1]))))

ex2 = np.zeros((steps))

for i in np.arange(steps):
    ex2[i] = np.real(np.trace(oE2mImI.dot(rhoT[i, :].reshape(np.shape(P0)[0], np.shape(P0)[1]))))

ex12 = np.zeros((steps))

for i in np.arange(steps):
    ex12[i] = np.abs(np.trace(oE1E2mImI.dot(rhoT[i, :].reshape(np.shape(P0)[0], np.shape(P0)[1]))))

psi01 = np.zeros((steps), dtype=complex)

for i in np.arange(steps):
    psi01[i] = np.trace(opsi01.dot(rhoT[i, :].reshape(np.shape(P0)[0], np.shape(P0)[1])))

psi02 = np.zeros((steps), dtype=complex)

for i in np.arange(steps):
    psi02[i] = np.trace(opsi02.dot(rhoT[i, :].reshape(np.shape(P0)[0], np.shape(P0)[1])))

psi03 = np.zeros((steps), dtype=complex)

for i in np.arange(steps):
    psi03[i] = np.trace(opsi03.dot(rhoT[i, :].reshape(np.shape(P0)[0], np.shape(P0)[1])))

psi14 = np.zeros((steps), dtype=complex)

for i in np.arange(steps):
    psi14[i] = np.trace(opsi14.dot(rhoT[i, :].reshape(np.shape(P0)[0], np.shape(P0)[1])))

psi15 = np.zeros((steps), dtype=complex)

for i in np.arange(steps):
    psi15[i] = np.trace(opsi15.dot(rhoT[i, :].reshape(np.shape(P0)[0], np.shape(P0)[1])))

psi37 = np.zeros((steps), dtype=complex)

for i in np.arange(steps):
    psi37[i] = np.trace(opsi37.dot(rhoT[i, :].reshape(np.shape(P0)[0], np.shape(P0)[1])))

psi38 = np.zeros((steps), dtype=complex)

for i in np.arange(steps):
    psi38[i] = np.trace(opsi38.dot(rhoT[i, :].reshape(np.shape(P0)[0], np.shape(P0)[1])))

psi13 = np.zeros((steps), dtype=complex)

for i in np.arange(steps):
    psi13[i] = np.trace(opsi13.dot(rhoT[i, :].reshape(np.shape(P0)[0], np.shape(P0)[1])))

# psi46 = np.zeros((steps),dtype=complex)

# for i in np.arange(steps):

#    psi46[i] = np.trace(opsi46.dot(rhoT[i,:].reshape(np.shape(P0)[0],np.shape(P0)[1])))


count3 = time.time()

print('Measurements =', count3 - count2)

# %%


# pr = np.zeros((steps))

# for i in np.arange(steps):

#    pr[i] = np.real(np.trace(rhoT[i,:].reshape(np.shape(P0)[0],np.shape(P0)[1]).dot(rhoT[i,:].reshape(np.shape(P0)[0],np.shape(P0)[1]).transpose().conj())))

#

# bbratio = bb2/bb1


# %%


# q_n1 = (bb1sq-bb1**2)/bb1 - 1

# q_n2 = (bb2sq-bb2**2)/bb2 - 1


# q_x1 = (x1sq-x1**2)/x1 - 1

# q_n1 = (bb1sq-bb1**2)/bb1 - 1


# v_n1 = bb1sq-bb1**2

# v_n2 = bb2sq-bb2**2

# v_x1 = x1sq-x1**2

# v_x2 = x2sq-x2**2


# %% Time correlation function

elta = np.int(np.round(((2 * constant.pi) / omega) / dt))  # 57

c_X12 = sj.corrfunc(x1, x2, elta)

# c_Xsq12 = sj.corrfunc(x1sq,x2sq,elta)

# c_p12 = sj.corrfunc(p1,p2,elta)


# c_n12 = sj.corrfunc(bb1,bb2,elta)

# c_nsq12 = sj.corrfunc(bb1sq,bb2sq,elta)

# c_vx12 = sj.corrfunc(v_x1,v_x2,elta)

# c_vn12 = sj.corrfunc(v_n1,v_n2,elta)


# %%

import matplotlib.pyplot as plt

t_cm = t / (2 * constant.pi)

t_ps = (t_cm * 1e12) / (100 * constant.c)

t_w = t_ps / (1e12 / (omega * 100 * constant.c))

dtperps = (100 * constant.c * 2 * constant.pi * 1e-12) / dt

itvl = 5

# %% Quantum Correlations


counta = time.time()



q_mutual = []

c_info = []

q_discord = []

corr_times = []

maxstep = np.int(np.round(8*dtperps))



for i in np.arange(0,maxstep,100):

   test_matrix = rhoT[i,:].reshape(np.shape(P0)[0],np.shape(P0)[1])

   quantum_mutual_info, classical_info, quantum_discord = QC.correlations(test_matrix, 2, N, N, 1, 2)

   q_mutual.append(quantum_mutual_info)

   c_info.append(classical_info)

   q_discord.append(quantum_discord)

   corr_times.append(t_ps[i])

   print(i)



q_mutual = np.array(q_mutual)

c_info = np.array(c_info)

q_discord = np.array(q_discord)

corr_times = np.array(corr_times)



countb = time.time()

print('Quantum Correlation Measures =',countb-counta)


# %%

# np.save('m_inf_dw002_100dt',q_mutual)

# np.save('q_dis_dw002_100dt',q_discord)

# np.save('c_inf_dw002_100dt',c_info)

##np.save('cx12_dw002_100dt',c_X12)

# np.save('corr_times_100dt',corr_times)


# %%


# dtperps = (100*constant.c*2*constant.pi*1e-12)/dt

##

# P0eig = orderedeig5.getH()*P0*orderedeig5

##

# P02eig = np.real(orderedeig5.getH()*np.abs(rhoT[int(0.2*dtperps),:].reshape(50,50))*orderedeig5)

# P05eig = np.real(orderedeig5.getH()*np.abs(rhoT[int(0.5*dtperps),:].reshape(50,50))*orderedeig5)

# P1eig = np.real(orderedeig5.getH()*np.abs(rhoT[int(1*dtperps),:].reshape(50,50))*orderedeig5)

# P2eig = np.real(orderedeig5.getH()*np.abs(rhoT[int(2*dtperps),:].reshape(50,50))*orderedeig5)

# P3eig = np.real(orderedeig5.getH()*np.abs(rhoT[int(3*dtperps),:].reshape(50,50))*orderedeig5)

# P4eig = np.real(orderedeig5.getH()*np.abs(rhoT[int(4*dtperps),:].reshape(50,50))*orderedeig5)

##P6eig = np.real(orderedeig5.getH()*np.abs(rhoT[int(6*dtperps),:].reshape(50,50))*orderedeig5)

##P8eig = np.real(orderedeig5.getH()*np.abs(rhoT[int(8*dtperps),:].reshape(50,50))*orderedeig5)

##P10eig = np.real(orderedeig5.getH()*np.abs(rhoT[int(10*dtperps)-1,:].reshape(50,50))*orderedeig5)

##

# oX1eig = orderedeig5.getH()*oX1*orderedeig5

# oX2eig = orderedeig5.getH()*oX2*orderedeig5

# oXminchop = np.tril(oX1eig-oX2eig,k=-1)

# oXpluschop = np.tril(oX1eig+oX2eig,k=-1)

#

# coefx1 = np.multiply(oX1eig,P2eig)

# coefx2 = np.multiply(oX2eig,P2eig)

###

# coefx1chop = np.tril(coefx1,k=-1)

# coefx2chop = np.tril(coefx2,k=-1)

###

# omegaarray = np.repeat(ordereden5,2*N**2).reshape(2*N**2,2*N**2) - np.repeat(ordereden5,2*N**2).reshape(2*N**2,2*N**2).transpose()

# omegachop = np.tril(omegaarray,k=-1)

#

##np.save('rho_3ps_PE545',np.abs(rhoT[int(3*dtperps),:].reshape(50,50)))


# rho8ps = np.abs(rhoT[int(8*dtperps),:].reshape(50,50))


# %%


count4 = time.time()



sampleratecm = 1/(t_cm[1]-t_cm[0])

freqres1 = 0.5

ftlen = (t_cm[1]-t_cm[0])*np.arange(int(sampleratecm/freqres1))



anaX1array = np.zeros([2*N**2,2*N**2,np.size(ftlen)])

anaX2array = np.zeros([2*N**2,2*N**2,np.size(ftlen)])

anaX1 = np.zeros(np.size(ftlen))

anaX2 = np.zeros(np.size(ftlen))

#

for a in np.arange(np.size(ftlen)):

   anaX1array[:,:,a] = np.multiply((2*np.cos(omegachop*2*constant.pi*ftlen[a])),coefx1chop)

   anaX1[a] = np.sum(anaX1array[:,:,a]) + np.trace(coefx1)

   anaX2array[:,:,a] = np.multiply((2*np.cos(omegachop*2*constant.pi*ftlen[a])),coefx2chop)

   anaX2[a] = np.sum(anaX2array[:,:,a]) + np.trace(coefx2)



freqres2 = 0.1

pads = int((sampleratecm/freqres2)-np.shape(anaX1)[0]) #30000

x1pad = np.append(anaX1,np.zeros(pads))

x2pad = np.append(anaX2,np.zeros(pads))

fr10 = np.fft.rfft(x1pad,int(np.size(x1pad)))

fr20 = np.fft.rfft(x2pad,int(np.size(x2pad)))

freq_cm0 = sampleratecm*np.arange(0,1-1/np.size(x1pad),1/np.size(x1pad))



count5 = time.time()

print('Measurements =',count5-count4)


# %%

#FOURIER
st = int(1100/(sampleratecm/np.size(x1pad)))

en = int(1130/(sampleratecm/np.size(x1pad)))



plt.figure(30)

plt.plot(freq_cm0[st:en],np.real(fr10)[st:en],label=r'$\langle X_1\rangle$')

plt.plot(freq_cm0[st:en],np.real(fr20)[st:en],label=r'$\langle X_2\rangle$')

plt.ylabel('Real Part of FT')

plt.xlabel('Frequency ($cm^{-1}$)')

plt.legend()

plt.grid(True,which='both')

plt.minorticks_on()

plt.yticks([0])

plt.title(r'Components of $\langle X\rangle$ at $T=2ps$')

##plt.savefig('ReFT_2ps_PE545',bbox_inches='tight',dpi=300)


# plt.figure(31)

# plt.plot(freq_cm0[st:en],(np.abs(fr10)**2)[st:en],label='<$X_1$>')

# plt.plot(freq_cm0[st:en],(np.abs(fr20)**2)[st:en],label='<$X_2$>')

# plt.ylabel('Power')

# plt.xlabel('Frequency ($cm^{-1}$)')

# plt.legend()

# plt.grid()

# plt.yticks([])

##plt.savefig('PowerFT_2ps',bbox_inches='tight',dpi=300)


# %%

# plt.figure(21)

# en = 31100

# plt.plot(t_ps[np.arange(0,en,itvl)],c_p12[np.arange(0,en,itvl)],'o',markersize=1)

# plt.ylabel('$C_{<p_1><p_2>}$')

# plt.xlabel('Time / $ps$')

# %%

# plt.figure(20)

# plt.plot(t_ps[np.arange(0,np.size(c_p1x1),itvl)],c_p1x1[np.arange(0,np.size(c_p12),itvl)],'o',markersize=1)

# plt.ylabel('$C_{<p_1><x_1>}$')

# plt.xlabel('Time / $ps$')

# %%

# plt.figure(19)

# plt.plot(t_ps[np.arange(0,np.size(c_p12sq),itvl)],c_p12sq[np.arange(0,np.size(c_p12sq),itvl)],'o',markersize=1)

# plt.ylabel('$C_{<p_1^2><p_2^2>}$')

# plt.xlabel('Time / $ps$')

# %%

# plt.figure(17)

# en = 31100

# itvl = 1

# plt.plot(t_ps[np.arange(0,en,itvl)],c_X12[np.arange(0,en,itvl)],'o',markersize=1)

# plt.ylabel(r'$C_{\langle x_1\rangle\langle x_2\rangle}$',fontsize=12)

# plt.xlabel('Time $(ps)$')

##plt.savefig('Hevo_c_x12_initE2thth',bbox_inches='tight',dpi=300)

# %%

# plt.figure(16)

# plt.plot(t_ps[np.arange(0,np.size(c_X12sq),itvl)],c_X12sq[np.arange(0,np.size(c_X12sq),itvl)],'o',markersize=1)

# plt.ylabel('$C_{<x_1^2><x_2^2>}$')

# plt.xlabel('Time / $ps$')


# %% #####################################################

FigureA = plt.figure(14)

en = 13000

st = 0000

itvl = 5

axA = FigureA.add_subplot(111)

axA.plot(t_ps[np.arange(st, en, itvl)], x2[np.arange(st, en, itvl)], label=r'$\langle X_2\rangle$')

# plt.plot(t_ps[np.arange(st,en,itvl)],x2sq[np.arange(st,en,itvl)],label=r'$\langle X_2^2\rangle$')

axA.plot(t_ps[np.arange(st, en, itvl)], x1[np.arange(st, en, itvl)], label=r'$\langle X_1\rangle$')

# plt.plot(t_ps[np.arange(st,en,itvl)],x1sq[np.arange(st,en,itvl)],label=r'$\langle X_1^2\rangle$')

# plt.plot(t_ps[np.arange(0,en,itvl)],c_Xsq12[np.arange(0,en,itvl)],'o',markersize=1,label=r'$C_{\langle x_1^2\rangle\langle x_2^2\rangle}$')

# plt.plot(t_ps[np.arange(0,en,itvl)],c_nsq12[np.arange(0,en,itvl)],'o',markersize=1,label=r'$C_{\langle n_1^2\rangle\langle n_2^2\rangle}$')

# plt.plot(t_ps[np.arange(0,en,itvl)],c_n12[np.arange(0,en,itvl)],'o',markersize=1,label=r'$C_{\langle n_1\rangle\langle n_2\rangle}$')

# plt.plot(t_ps[np.arange(st,en,itvl)],v_x1[np.arange(st,en,itvl)],label=r'$V(X_1)$')

# plt.plot(t_ps[np.arange(st,en,itvl)],v_x2[np.arange(st,en,itvl)],label=r'$V(X_2)$')

# plt.plot(t_ps[np.arange(0,en,itvl)],c_vx12[np.arange(0,en,itvl)],'o',markersize=1,label=r'$C_{\langle V(x_1)\rangle\langle V(x_2)\rangle}$')

# plt.plot(t_ps[np.arange(st,en,itvl)],np.real(hINT_t)[np.arange(st,en,itvl)]/250,label=r'$H_{int}$')

# plt.ylabel('$<x>$')

# plt.ylabel('$C_{<X_1><X_2>}$',fontsize=12)

axA.set_xlabel('Time (ps)')

# axA.set_xlim([0,10])

axA.set_yticks([])

axB = axA.twinx()

axB.plot(t_ps[np.arange(st, en, itvl)], c_X12[np.arange(st, en, itvl)], 'r-o', markevery=0.05, markersize=5,
         label=r'$C_{\langle x_1\rangle\langle x_2\rangle}$')

axB.grid()

axA.grid()

axB.legend(bbox_to_anchor=(0.9, 0.6))

axA.legend()

# plt.savefig('cXX_dw004_QC.pdf',bbox_inches='tight',dpi=600,format='pdf',transparent=True)

#################################################################
# %%

# """ Zoom on X's to see phase between them. Pick starting t """

# FigureA = plt.figure()

# start_ps = 2

#

# st = np.int(np.round(start_ps*dtperps))

# en = st + 500

#

# axA = FigureA.add_subplot(111)

# axA.plot(t_ps[np.arange(st,en)],x2[np.arange(st,en)],label=r'$\langle X_2\rangle$')

##plt.ylabel('$C_{<X_1><X_2>}$',fontsize=12)

# axA.set_xlabel('Time (ps)')

##axA.set_xlim([0,10])

# axA.set_yticks([])

#

# axB = axA.twinx()

# axB.plot(t_ps[np.arange(st,en)],x1[np.arange(st,en)],'r',label=r'$\langle X_1\rangle$')

# axA.grid()

# axB.legend(bbox_to_anchor=(1,1))

# axA.legend(bbox_to_anchor=(1,0.85))

# axB.set_yticks([])

##plt.savefig('cXX_dw004_QC.pdf',bbox_inches='tight',dpi=600,format='pdf',transparent=True)


# %%


#QUANTUM PLOT
FigureA = plt.figure(14)

en = 49000

st = 000

itvl = 5



axA = FigureA.add_subplot(111)

axA.plot(corr_times,c_info,label=r'Classical Info')

axA.plot(corr_times,q_mutual,label=r'Q Mutual Info')

axA.plot(corr_times,q_discord,label=r'Discord')

axA.set_xlabel('Time (ps)')

axA.set_xlim([0,12])

#axA.set_yticks([])



axB = axA.twinx()

axB.plot(t_ps[np.arange(st,en,itvl)],c_X12[np.arange(st,en,itvl)],'r-o',markevery=0.05,markersize=5,label=r'$C_{\langle x_1\rangle\langle x_2\rangle}$')



axB.grid()

axA.grid()

axB.legend(bbox_to_anchor=([0.3,0.8]))

axA.legend(bbox_to_anchor=([0.9,0.8]))

#plt.legend()

#plt.savefig('cXX_dw004_100dt_QC.pdf',bbox_inches='tight',dpi=600,format='pdf',transparent=True)




# %%

# plt.figure(14)

# en = 6660

# st = 0000

# itvl = 5

# plt.plot(t_ps[np.arange(st,en,itvl)],x2[np.arange(st,en,itvl)],label=r'$\langle X_2\rangle$')

##plt.plot(t_ps[np.arange(st,en,itvl)],x2sq[np.arange(st,en,itvl)],label=r'$\langle X_2^2\rangle$')

# plt.plot(t_ps[np.arange(st,en,itvl)],x1[np.arange(st,en,itvl)],label=r'$\langle X_1\rangle$')

##plt.plot(t_ps[np.arange(st,en,itvl)],x1sq[np.arange(st,en,itvl)],label=r'$\langle X_1^2\rangle$')

##plt.plot(t_ps[np.arange(st,en,itvl)],c_X12[np.arange(st,en,itvl)],'o',markersize=1,label=r'$C_{\langle x_1\rangle\langle x_2\rangle}$')

##plt.plot(t_ps[np.arange(0,en,itvl)],c_Xsq12[np.arange(0,en,itvl)],'o',markersize=1,label=r'$C_{\langle x_1^2\rangle\langle x_2^2\rangle}$')

##plt.plot(t_ps[np.arange(0,en,itvl)],c_nsq12[np.arange(0,en,itvl)],'o',markersize=1,label=r'$C_{\langle n_1^2\rangle\langle n_2^2\rangle}$')

##plt.plot(t_ps[np.arange(0,en,itvl)],c_n12[np.arange(0,en,itvl)],'o',markersize=1,label=r'$C_{\langle n_1\rangle\langle n_2\rangle}$')

##plt.plot(t_ps[np.arange(st,en,itvl)],v_x1[np.arange(st,en,itvl)],label=r'$V(X_1)$')

##plt.plot(t_ps[np.arange(st,en,itvl)],v_x2[np.arange(st,en,itvl)],label=r'$V(X_2)$')

##plt.plot(t_ps[np.arange(0,en,itvl)],c_vx12[np.arange(0,en,itvl)],'o',markersize=1,label=r'$C_{\langle V(x_1)\rangle\langle V(x_2)\rangle}$')

##plt.ylabel('$<x>$')

##plt.ylabel('$C_{<X_1><X_2>}$',fontsize=12)

# plt.xlabel('Time $(ps)$')

# plt.grid()

##plt.legend(markerscale=4)

# plt.legend()

##plt.savefig('V_x_2cmdetun',bbox_inches='tight',dpi=600)


# %%

# plt.figure()

# en = 6500

# st = 000

# itvl = 5

# plt.plot(t_ps[np.arange(st,en,itvl)],c_X12[np.arange(st,en,itvl)],label=r'$C_{\langle x_1\rangle\langle x_2\rangle}$')

# plt.plot(t_ps[np.arange(0,en,itvl)],c_Xsq12[np.arange(0,en,itvl)],label=r'$C_{\langle x_1^2\rangle\langle x_2^2\rangle}$')

# plt.plot(t_ps[np.arange(st,en,itvl)],c_nsq12[np.arange(st,en,itvl)],label=r'$C_{\langle n_1^2\rangle\langle n_2^2\rangle}$')

##plt.plot(t_ps[np.arange(0,en,itvl)],c_n12[np.arange(0,en,itvl)],label=r'$C_{\langle n_1\rangle\langle n_2\rangle}$')

#

##plt.plot(t_ps[np.arange(0,en,itvl)],c_vx12[np.arange(0,en,itvl)],label=r'$C_{\langle Vx_1\rangle\langle Vx_2\rangle}$')

##plt.plot(t_ps[np.arange(0,en,itvl)],c_vn12[np.arange(0,en,itvl)],label=r'$C_{\langle Vn_1\rangle\langle Vn_2\rangle}$')

#

# plt.xlabel('Time $(ps)$')

# plt.grid()

# plt.legend()

##plt.savefig('synchs_50cmdetun',bbox_inches='tight',dpi=600)




sampleratecm = 1/(t_cm[1]-t_cm[0])

freqres = 0.5

pads = int((sampleratecm/freqres)-np.shape(x1)[0]) #30000

x1pad = np.append(x1,np.zeros(pads))

x2pad = np.append(x2,np.zeros(pads))

fr1 = np.fft.rfft(x1pad,int(np.size(x1pad)))

fr2 = np.fft.rfft(x2pad,int(np.size(x2pad)))

freq_cm = sampleratecm*np.arange(0,1-1/np.size(x1pad),1/np.size(x1pad))

# %%


st = int(1000/(sampleratecm/np.size(x1pad)))

en = int(1300/(sampleratecm/np.size(x1pad)))


plt.figure(13)

plt.plot(freq_cm[st:en],np.real(fr1)[st:en],label='<$X_1$>')

plt.plot(freq_cm[st:en],np.real(fr2)[st:en],label='<$X_2$>')

plt.ylabel('Real')

plt.xlabel('Frequency ($cm^{-1}$)')

plt.legend()

plt.grid()

#

# plt.figure(12)

# plt.plot(freq_cm[st:en],np.imag(fr1)[st:en],label='<$X_1$>')

# plt.plot(freq_cm[st:en],np.imag(fr2)[st:en],label='<$X_2$>')

# plt.ylabel('Imaginary')

# plt.xlabel('Frequency ($cm^{-1}$)')

# plt.grid()


plt.figure(19)

plt.plot(freq_cm[st:en],(np.abs(fr1)**2)[st:en],label='<$X_1$>')

plt.plot(freq_cm[st:en],(np.abs(fr2)**2)[st:en],label='<$X_2$>')

plt.ylabel('Power')

plt.xlabel('Frequency ($cm^{-1}$)')

plt.grid(True,which='both')

plt.minorticks_on()

plt.yticks([])

plt.title('Full time evolution FT')

plt.legend()


# %%

# plt.figure(11)

# en = 6000

# st = 00

# plt.plot(t_ps[st:en],(x2sq[st:en]),label='2')

# plt.plot(t_ps[st:en],(x1sq[st:en]),label='1')

# plt.ylabel('$<X^2>$')

# plt.xlabel('Time / $ps$')

# plt.legend()

# %%

# plt.figure(10)

# en = 31100

# st = 00

# plt.plot(t_ps[st:en],(x1+x2)[st:en],label=r'$\langle X_+ \rangle$')

# plt.plot(t_ps[st:en],(x1-x2)[st:en],label=r'$\langle X_- \rangle$')

##plt.ylabel('$<x>$')

# plt.xlabel('Time / $ps$')

# plt.legend()

# plt.grid()

##plt.savefig('Xpm_PE545_long',bbox_inches='tight',dpi=300)

# %%

# plt.figure(9)

# st = 1800

# en = 2500

# plt.plot(t_ps[np.arange(st,en)],p2[np.arange(st,en)],label='2')

# plt.plot(t_ps[np.arange(st,en)],p1[np.arange(st,en)],label='1')

# plt.ylabel('$<p>$')

# plt.xlabel('Time / $ps$')

# plt.legend()


# %%

# plt.figure(10)

# plt.plot(t_ps[0,:],(p2sq[:]),label='$2$')

# plt.plot(t_ps[0,:],(p1sq[:]),label='$1$')

# plt.ylabel('$<p^2>$')

# plt.xlabel('Time / $ps$')

# plt.legend()

# %%

# plt.figure(9)

# en = 4000

# st = 000

# plt.plot(t_ps[np.arange(st,en)],(p1+p2)[np.arange(st,en)],label='$p_+$')

# plt.plot(t_ps[np.arange(st,en)],(p1-p2)[np.arange(st,en)],label='$p_-$')

# plt.ylabel('$<p>$')

# plt.xlabel('Time / $ps$')

# plt.legend()

# %%

# plt.figure()

# plt.plot(t_ps[0,0:200],P_el[0,1,0:200],label='E1E2')

# plt.plot(t_ps[0,0:200],P_el[1,0,0:200],label='E2E1')

# plt.ylabel('Coherence')

# plt.xlabel('Time / $ps$')

# plt.legend()

# plt.savefig('Coherencedecay',dpi=300)


# %%

# plt.figure(8)

# en = 3300

# st = 0000

# itvl = 5

# plt.plot(t_ps[np.arange(st,en)],bb2[np.arange(st,en)],label=r'$\langle n_2 \rangle$')

# #plt.plot(t_ps[np.arange(st,en)],bb2sq[np.arange(st,en)],label=r'$\langle n_2^2 \rangle$')

# plt.plot(t_ps[np.arange(st,en)],bb1[np.arange(st,en)],label=r'$\langle n_1 \rangle$')

# plt.plot(t_ps[np.arange(st,en)],bb1sq[np.arange(st,en)],label=r'$\langle n_1^2 \rangle$')

# plt.plot(t_ps[np.arange(st,en)],(bb1sq-bb1**2)[np.arange(st,en)],label=r'$\langle n_1^2 \rangle - \langle n_1 \rangle^2$')

# plt.plot(t_ps[np.arange(st,en)],(bb2sq-bb2**2)[np.arange(st,en)],label=r'$\langle n_2^2 \rangle - \langle n_2 \rangle^2$')

# plt.plot(t_ps[np.arange(st,en)],q_n2[np.arange(st,en)],label=r'$Q(n_2)$')

# plt.plot(t_ps[np.arange(st,en)],q_n1[np.arange(st,en)],label=r'$Q(n_1)$')

# plt.plot(t_ps[np.arange(0,en,itvl)],c_n12[np.arange(0,en,itvl)],'o',markersize=1,label=r'$C_{\langle bb_1\rangle\langle bb_2\rangle}$')

# plt.plot(t_ps[np.arange(st,en,itvl)],c_nsq12[np.arange(st,en,itvl)],'o',markersize=1,label=r'$C_{\langle n_1^2 \rangle\langle n_2^2\rangle}$')

# plt.ylabel('Population of Modes')

# plt.xlabel('Time (ps)')

# #plt.yticks([-0.1,0,0.1])

# plt.legend()

# plt.grid()

##plt.savefig('Qn_2cmdetuning_1p75g',bbox_inches='tight',dpi=600)

# %%

# plt.figure()

# en = 20000

# st = 0

# plt.plot(t_ps[0,np.arange(st,en)],m1[np.arange(st,en)],label='$m_1$')

# plt.plot(t_ps[0,np.arange(st,en)],m1e0[np.arange(st,en)],label='$m_1(0)$')

# plt.plot(t_ps[0,np.arange(st,en)],m1e1[np.arange(st,en)],label='$m_1(1)$')

# plt.plot(t_ps[0,np.arange(st,en)],m1e2[np.arange(st,en)],label='$m_1(2)$')

# plt.plot(t_ps[0,np.arange(st,en)],m1e3[np.arange(st,en)],label='$m_1(3)$')

# plt.plot(t_ps[0,np.arange(st,en)],m1e4[np.arange(st,en)],label='$m_1(4)$')

# plt.ylabel('Population of $M_1$ levels')

# plt.xlabel('Time / $ps$')

# plt.legend()

# plt.grid()

# plt.savefig('M1_levels',dpi=300)


# %%

# plt.figure()

# en = 1000

# st = 0

##plt.plot(t_ps[0,np.arange(st,en)],m2[np.arange(st,en)],label='$m_1$')

# plt.plot(t_ps[0,np.arange(st,en)],m2g[np.arange(st,en)],label='$m_2(0)$')

# plt.plot(t_ps[0,np.arange(st,en)],m2e1[np.arange(st,en)],label='$m_2(1)$')

# plt.plot(t_ps[0,np.arange(st,en)],m2e2[np.arange(st,en)],label='$m_2(2)$')

# plt.plot(t_ps[0,np.arange(st,en)],m2e3[np.arange(st,en)],label='$m_2(3)$')

# plt.plot(t_ps[0,np.arange(st,en)],m2e4[np.arange(st,en)],label='$m_2(4)$')

# plt.ylabel('Population of $M_2$ levels')

# plt.xlabel('Time / $ps$')

# plt.legend()

# plt.grid()

# plt.savefig('M2_levels',dpi=300)


# %%
#ENERGY TRANSFER
plt.figure(7)

st = 0000

en = 13000  # P_el.shape[2]

itvl = 3

plt.plot(t_ps[0:en], ex1[0:en], label=r'$|E_{1}\rangle\langle E_{1}|$')

plt.plot(t_ps[0:en], ex2[0:en], label=r'$|E_{2}\rangle\langle E_{2}|$')

plt.plot(t_ps[0:en], ex12[0:en], label=r'$||E_{1}\rangle\langle E_{2}||$')

#plt.plot(t_ps[0:en],ex12_00[0:en],label=r'$|E_{1}00\rangle\langle E_{2}00|$')

plt.plot(t_ps[np.arange(0,en,itvl)],c_X12[np.arange(0,en,itvl)],'o',markersize=1,label=r'$C_{\langle x_1\rangle\langle x_2\rangle}$')

#plt.plot(t_ps[0:en],ex21[0:en],label='abs($E_{21}$)')

# plt.plot(t_ps[st:en],sigZ[st:en],label='$\sigma_Z$')

# plt.plot(t_ps[0:en],ex12m0m0[0:en],label='$E_{12}m_0m_0$')

# plt.ylabel('Population')

plt.xlabel('Time ($ps$)')

# plt.xlim([0,5])

plt.grid()

# plt.legend(bbox_to_anchor=[1,1])

plt.legend()

# plt.title(r'Exciton Coherence and Synchronisation $\eta=0.175$')

# plt.savefig('El_dw10_3V',dpi=300)

# plt.savefig('synch_1p75g_w2_1113',bbox_inches='tight',dpi=600)


# %%

# plt.figure(5)

# st = 000

# en = 10000 #P_el.shape[2]

# #plt.plot(t_ps[0:en],psi15[0:en],label='Re|$\psi_{1}$><$\psi_{5}$|')

# plt.plot(t_ps[st:en],psi14r[st:en],label='Re|$\psi_{1}$><$\psi_{4}$|')

# plt.plot(t_ps[st:en],psi02r[st:en],label='Re|$\psi_{0}$><$\psi_{2}$|')

# #plt.plot(t_ps[0:en],psi37[0:en],label='Re|$\psi_{3}$><$\psi_{7}$|')

# plt.plot(t_ps[st:en],psi03r[st:en],label='Re|$\psi_{0}$><$\psi_{3}$|')

# plt.plot(t_ps[st:en],psi01r[st:en],label='Re|$\psi_{0}$><$\psi_{1}$|')



# #plt.plot(t_ps[0:en],psi13[0:en],label='Re|$\psi_{1}$><$\psi_{3}$|')

# #plt.plot(t_ps[0:en],psi26[0:en],label='Re|$\psi_{2}$><$\psi_{6}$|')



# #plt.plot(t_ps[0:en],ex2[0:en],label='|$E_{2}$><$E_{2}$|')

# #plt.plot(t_ps[0:en],ex12[0:en],label='|$E_{1}$><$E_{2}$|')

# #plt.plot(t_ps[0:en],ex21[0:en],label='abs($E_{21}$)')

# #plt.plot(t_ps[st:en],sigZ[st:en],label='$\sigma_Z$')

# #plt.plot(t_ps[0:en],ex12m0m0[0:en],label='$E_{12}m_0m_0$')

# #plt.ylabel('Population')

# plt.xlabel('Time ($ps$)')

# plt.grid()

# plt.legend()

# plt.title('Eigenstate Coherence Oscillations')

#plt.savefig('El_dw10_3V',dpi=300)

#plt.savefig('X_coherences',bbox_inches='tight',dpi=300)


# %%

plt.figure(5)

st = 00

en = 13000

plt.plot(t_ps[st:en],oX1eig[0,1]*np.real(psi01[st:en]),label=r'$\Omega_{01} =$'+str(f01))

plt.plot(t_ps[st:en],oX1eig[0,2]*np.real(psi02[st:en]),label=r'$\Omega_{02} =$'+str(f02))

plt.plot(t_ps[st:en],oX1eig[0,3]*np.real(psi03[st:en]),label=r'$\Omega_{03} =$'+str(f03))

plt.plot(t_ps[st:en],oX1eig[1,4]*np.real(psi14[st:en]),label=r'$\Omega_{14} =$'+str(f14))

plt.plot(t_ps[st:en],oX1eig[1,5]*np.real(psi15[st:en]),label=r'$\Omega_{15} =$'+str(f15))

plt.plot(t_ps[st:en],oX1eig[3,7]*np.real(psi37[st:en]),label=r'$\Omega_{37} =$'+str(f37))

plt.plot(t_ps[st:en],oX1eig[3,8]*np.real(psi38[st:en]),label=r'$\Omega_{38} =$'+str(f38))

plt.plot(t_ps[st:en],oX1eig[1,3]*np.real(psi13[st:en]),label=r'$\Omega_{13} =$'+str(f13))







plt.xlabel('Time ($ps$)')

plt.grid()

plt.legend(bbox_to_anchor=([1,1]))

plt.title(r'$\omega_2$ = ' + np.str(np.round(w2,decimals=2))+ ' $\omega_1$ = ' + np.str(np.round(w1,decimals=2))) #$\omega=1530cm^{-1}$')

##plt.savefig('Eigcoherences_1p75g_w2_1113',bbox_inches='tight',dpi=600)


# %%

# plt.figure(5)

# st = 0000

# en = 12000

# plt.plot(t_ps[st:en], np.abs(oX1eig[0, 1]) * np.abs(psi01[st:en]), label=r'$\Omega_{01} =$' + str(f01))

# plt.plot(t_ps[st:en], np.abs(oX1eig[0, 2]) * np.abs(psi02[st:en]), label=r'$\Omega_{02} =$' + str(f02))

# plt.plot(t_ps[st:en], np.abs(oX1eig[0, 3]) * np.abs(psi03[st:en]), label=r'$\Omega_{03} =$' + str(f03))

# plt.plot(t_ps[st:en], np.abs(oX1eig[1, 4]) * np.abs(psi14[st:en]), label=r'$\Omega_{14} =$' + str(f14))

# plt.plot(t_ps[st:en], np.abs(oX1eig[1, 5]) * np.abs(psi15[st:en]), label=r'$\Omega_{15} =$' + str(f15))

# plt.plot(t_ps[st:en], np.abs(oX1eig[3, 7]) * np.abs(psi37[st:en]), label=r'$\Omega_{37} =$' + str(f37))

# plt.plot(t_ps[st:en], np.abs(oX1eig[3, 8]) * np.abs(psi38[st:en]), label=r'$\Omega_{38} =$' + str(f38))

# plt.plot(t_ps[st:en], np.abs(oX1eig[1, 3]) * np.abs(psi13[st:en]), label=r'$\Omega_{13} =$' + str(f13))

# plt.xlabel('Time ($ps$)')

# # plt.ylim(-0.005,0.1)

# plt.grid()

# plt.legend(bbox_to_anchor=([1, 1]))

# plt.title(r'$\omega_2$ = ' + np.str(np.round(w2, decimals=2)) + ' $\omega_1$ = ' + np.str(
#     np.round(w1, decimals=2)))  # $\omega=1530cm^{-1}$')

# # plt.savefig('Eigcoherences_1p75g_w2_1113',bbox_inches='tight',dpi=600)

plt.show()
# %%

# plt.figure(5)

# st = 2000

# en = 12000

# plt.plot(t_ps[st:en],np.abs(psi01[st:en]),label=r'$\Omega_{01} =$'+str(f01))

# plt.plot(t_ps[st:en],np.abs(psi02[st:en]),label=r'$\Omega_{02} =$'+str(f02))

# plt.plot(t_ps[st:en],np.abs(psi03[st:en]),label=r'$\Omega_{03} =$'+str(f03))

# plt.plot(t_ps[st:en],np.abs(psi14[st:en]),label=r'$\Omega_{14} =$'+str(f14))

# plt.plot(t_ps[st:en],np.abs(psi15[st:en]),label=r'$\Omega_{15} =$'+str(f15))

# plt.plot(t_ps[st:en],np.abs(psi37[st:en]),label=r'$\Omega_{37} =$'+str(f37))

# plt.plot(t_ps[st:en],np.abs(psi38[st:en]),label=r'$\Omega_{38} =$'+str(f38))

# plt.plot(t_ps[st:en],np.abs(psi13[st:en]),label=r'$\Omega_{13} =$'+str(f13))







# plt.xlabel('Time ($ps$)')

# #plt.ylim(-0.005,0.1)

# plt.grid()

# plt.legend(bbox_to_anchor=([1,1]))

# plt.title(r'$\omega_2$ = ' + np.str(np.round(w2,decimals=2))+ ' $\omega_1$ = ' + np.str(np.round(w1,decimals=2))) #$\omega=1530cm^{-1}$')

##plt.savefig('Eigcoherences_1p75g_w2_1113',bbox_inches='tight',dpi=600)


# %%

# plt.figure(5)

# st = 13200

# en = 16000

# plt.plot(t_ps[st:en],oX2eig[0,1]*np.real(psi01[st:en]),label=r'$\Omega_{01} =$'+str(f01))

# plt.plot(t_ps[st:en],oX2eig[0,2]*np.real(psi02[st:en]),label=r'$\Omega_{02} =$'+str(f02))

# plt.plot(t_ps[st:en],oX2eig[0,3]*np.real(psi03[st:en]),label=r'$\Omega_{03} =$'+str(f03))

# plt.plot(t_ps[st:en],oX2eig[1,4]*np.real(psi14[st:en]),label=r'$\Omega_{14} =$'+str(f14))

# plt.plot(t_ps[st:en],oX2eig[1,5]*np.real(psi15[st:en]),label=r'$\Omega_{15} =$'+str(f15))

# plt.plot(t_ps[st:en],oX2eig[3,7]*np.real(psi37[st:en]),label=r'$\Omega_{37} =$'+str(f37))

# plt.plot(t_ps[st:en],oX2eig[3,8]*np.real(psi38[st:en]),label=r'$\Omega_{38} =$'+str(f38))

# plt.plot(t_ps[st:en],oX2eig[1,3]*np.real(psi13[st:en]),label=r'$\Omega_{13} =$'+str(f13))







# plt.xlabel('Time ($ps$)')

# plt.grid()

# plt.legend(bbox_to_anchor=([1,1]))

# plt.title(r'$\omega_2$ = ' + np.str(np.round(w2,decimals=2))+ ' $\omega_1$ = ' + np.str(np.round(w1,decimals=2))) #$\omega=1530cm^{-1}$')

#plt.savefig('Eigcoherences_1p75g_w2_1113',bbox_inches='tight',dpi=600)


# %%

# plt.figure(5)

# st = 000

# en = 33000 #P_el.shape[2]

# plt.plot(t_ps[st:en],np.abs(psi13[st:en]),label=r'$\Omega_{13} =$'+str(f13)+' $\delta\phi=$'+str(ph13))

# plt.plot(t_ps[st:en],np.abs(psi01[st:en]),label=r'$\Omega_{01} =$'+str(f01)+' $\delta\phi=$'+str(ph01))

# plt.plot(t_ps[st:en],np.abs(psi02[st:en]),'--',label=r'$\Omega_{02} =$'+str(f02)+' $\delta\phi=$'+str(ph02)) # synch freq

# plt.plot(t_ps[st:en],np.abs(psi03[st:en]),label=r'$\Omega_{03} =$'+str(f03)+' $\delta\phi=$'+str(ph03))

# plt.plot(t_ps[st:en],np.abs(psi14[st:en]),label=r'$\Omega_{14} =$'+str(f14)+' $\delta\phi=$'+str(ph14))

# plt.plot(t_ps[st:en],np.abs(psi15[st:en]),'--',label=r'$\Omega_{15} =$'+str(f15)+' $\delta\phi=$'+str(ph15)) # synch freq

# plt.plot(t_ps[st:en],np.abs(psi37[st:en]),'--',label=r'$\Omega_{37} =$'+str(f37)+' $\delta\phi=$'+str(ph37)) # synch freq

# plt.plot(t_ps[st:en],np.abs(psi38[st:en]),label=r'$\Omega_{38} =$'+str(f38)+' $\delta\phi=$'+str(ph38))

#

##plt.plot(t_ps[st:en],psi13[0:en],label=r'$|\psi_{13}|$ $\omega =$'+str(f13))

##plt.plot(t_ps[st:en],psi13r[0:en],label=r'$|\psi_{1}\rangle\langle\psi_{3}$|')

#

##plt.plot(t_ps[st:en],psi16[0:en],label=r'$|\psi_{1}\rangle\langle\psi_{6}$|')

##plt.plot(t_ps[st:en],psi25[0:en],label=r'$|\psi_{2}\rangle\langle\psi_{5}$|')

#

# plt.xlabel('Time (ps)')

# plt.xlim([0,5])

# plt.grid()

# plt.legend(bbox_to_anchor=(1,1))

# plt.title(r'$\omega_2$ = ' + np.str(np.round(w2,decimals=2))+ ' $\omega_1$ = ' + np.str(np.round(w1,decimals=2))) #$\omega=1530cm^{-1}$')

##plt.savefig('Eigcoherences_1p75g_w2_1113',bbox_inches='tight',dpi=600)


# %%

# plt.figure(5)

# st = 30000

# en = 43000 #P_el.shape[2]

# plt.plot(t_ps[st:en],np.abs(0.5*(np.abs(oX1eig[1,3])+np.abs(oX2eig[1,3])))*np.abs(psi13[st:en]),label=r'$\Omega_{13} =$'+str(f13))#+' $\delta\phi=$'+str(phases[1,3]))

##plt.plot(t_ps[st:en],np.abs(oX1eig[1,3])*np.abs(psi01[st:en]),label=r'$\Omega_{01} =$'+str(f01)+' $\delta\phi=$'+str(ph01))

# plt.plot(t_ps[st:en],np.abs(0.5*(np.abs(oX1eig[0,2])+np.abs(oX2eig[0,2])))*np.abs(psi02[st:en]),label=r'$\Omega_{02} =$'+str(f02))#+' $\delta\phi=$'+str(phases[0,2])) # synch freq

# plt.plot(t_ps[st:en],np.abs(0.5*(np.abs(oX1eig[0,3])+np.abs(oX2eig[0,3])))*np.abs(psi03[st:en]),label=r'$\Omega_{03} =$'+str(f03))#+' $\delta\phi=$'+str(phases[0,3]))

# plt.plot(t_ps[st:en],np.abs(0.5*(np.abs(oX1eig[1,4])+np.abs(oX2eig[1,4])))*np.abs(psi14[st:en]),label=r'$\Omega_{14} =$'+str(f14))#+' $\delta\phi=$'+str(phases[1,4]))

# plt.plot(t_ps[st:en],np.abs(0.5*(np.abs(oX1eig[1,5])+np.abs(oX2eig[1,5])))*np.abs(psi15[st:en]),label=r'$\Omega_{15} =$'+str(f15))#+' $\delta\phi=$'+str(phases[1,5])) # synch freq

# plt.plot(t_ps[st:en],np.abs(0.5*(np.abs(oX1eig[3,7])+np.abs(oX2eig[3,7])))*np.abs(psi37[st:en]),label=r'$\Omega_{37} =$'+str(f37))#+' $\delta\phi=$'+str(phases[3,7])) # synch freq

# plt.plot(t_ps[st:en],np.abs(0.5*(np.abs(oX1eig[3,8])+np.abs(oX2eig[3,8])))*np.abs(psi38[st:en]),label=r'$\Omega_{38} =$'+str(f38))#+' $\delta\phi=$'+str(phases[3,8]))

#

##plt.plot(t_ps[st:en],psi13[0:en],label=r'$|\psi_{13}|$ $\omega =$'+str(f13))

##plt.plot(t_ps[st:en],psi13r[0:en],label=r'$|\psi_{1}\rangle\langle\psi_{3}$|')

#

##plt.plot(t_ps[st:en],psi16[0:en],label=r'$|\psi_{1}\rangle\langle\psi_{6}$|')

##plt.plot(t_ps[st:en],psi25[0:en],label=r'$|\psi_{2}\rangle\langle\psi_{5}$|')

#

# plt.xlabel('Time (ps)')

# plt.grid()

# plt.legend(bbox_to_anchor=(1,1))

##plt.title(r'$\omega_2$ = ' + np.str(np.round(w2,decimals=2))+ ' $\omega_1$ = ' + np.str(np.round(w1,decimals=2))) #$\omega=1530cm^{-1}$')

###plt.savefig('Eigcoherences_dw004.pdf',bbox_inches='tight',dpi=600,format='pdf',transparent=True)

##


# %%

# plt.figure(51)

# st = 000

# en = 30000 #P_el.shape[2]

##plt.plot(t_ps[st:en],psi00[st:en],label=r'$|\psi_{0}|$')

##plt.plot(t_ps[st:en],psi11[st:en],label=r'$|\psi_{1}|$')

# plt.plot(t_ps[st:en],psi22[st:en],label=r'$|\psi_{2}|$')

##plt.plot(t_ps[st:en],psi33[st:en],label=r'$|\psi_{3}|$')

# plt.plot(t_ps[st:en],psi44[st:en],label=r'$|\psi_{4}|$')

# plt.plot(t_ps[st:en],psi55[st:en],label=r'$|\psi_{5}|$')

# plt.plot(t_ps[st:en],psi66[st:en],label=r'$|\psi_{6}|$')

# plt.plot(t_ps[st:en],psi77[st:en],label=r'$|\psi_{7}|$')

# plt.plot(t_ps[st:en],psi88[st:en],label=r'$|\psi_{8}|$')

# plt.plot(t_ps[st:en],psi99[st:en],label=r'$|\psi_{9}|$')

#

#

# plt.xlabel('Time ($ps$)')

# plt.grid()

# plt.legend()

# plt.title(r'Eigenstates over time $\eta=0.175$')

# plt.savefig('Eigcoherences_0175eta',bbox_inches='tight',dpi=300)


# %%

# plt.figure(14) #CLOSE UP TO SHOW OSCILLATION SYNC

# plt.plot(t_ps[0,800:1300],(x2[800:1300]-0.45),label='X2')

# plt.plot(t_ps[0,800:1300],2*x1[800:1300],label='X1')

# plt.ylabel('$<X>$')

# plt.xlabel('Time / $ps$')

# plt.legend(bbox_to_anchor=(1, 1.1))

# plt.savefig('X_dw20_c09',dpi=300)


# %%

# plt.figure(6)

# st =0000

# en= 9000

# plt.plot(t_ps[st:en],sigX[st:en],label='$\sigma_X$')

# plt.plot(t_ps[st:en],sigZ[st:en],label='$\sigma_Z$')

##plt.ylabel('$<p>$')

# plt.xlabel('Time / $ps$')

# plt.legend()


# %%

# np.save('c_x12_N6_dw0',c_X12)

# np.save('exc12_05eta_w111.csv',ex12)

# np.save('c_x12_'+omg+'dw_'+ve+'V'+rtherm+rel,c_X12)


# %%

# eeta= '10rth_05rel'

# detun = ''

# np.save('cx12_'+eeta+detun,c_X12)

# np.save('psi01_'+eeta+detun,psi01)

# np.save('psi13_'+eeta+detun,psi13)

# np.save('psi02_'+eeta+detun,psi02)

# np.save('psi03_'+eeta+detun,psi03)

# np.save('psi14_'+eeta+detun,psi14)

# np.save('psi15_'+eeta+detun,psi15)

# np.save('psi37_'+eeta+detun,psi37)

# np.save('psi38_'+eeta+detun,psi38)

# np.save('ex1_'+eeta+detun,ex1)

# np.save('ex12_'+eeta+detun,ex12)

# np.save('t_ps_'+eeta+detun,t_ps)

# np.save('x1_'+eeta+detun,x1)

# np.save('x2_'+eeta+detun,x2)

# np.save('oX1eig_'+eeta+detun,oX1eig)

# np.save('oX2eig_'+eeta+detun,oX2eig)

# np.save('coh_measure_'+eeta,coh)


# %%

# endstate = np.matrix(rhoT[49000,:].reshape(2*N**2,2*N**2))

# endstatesq = endstate.dot(endstate.getH())

# initstatesq = P0.dot(P0.getH())

# spreadend = np.trace(endstatesq)

# spreadint = np.trace(initstatesq)


# %%

# plt.figure(19)

# en = 16600

# st = 0000

##plt.plot(t_ps[np.arange(st,en)],bbratio[np.arange(st,en)],label=r'$\frac{n_2}{n_1}$')

# plt.plot(t_ps[np.arange(st,en)],pr[np.arange(st,en)],label=r'$PR$')

#

# plt.xlabel('Time (ps)')

##plt.yticks([-0.1,0,0.1])

# plt.legend()

# plt.grid()

##plt.savefig('Qn_2cmdetuning_1p75g',bbox_inches='tight',dpi=600)


# %%

# np.save('pr_PE545_005dw',pr)

# np.save('nratio_PE545_005dw_SS',bbratio[49000])


# %%

# np.save('ex1_PE545_long',ex1)

# np.save('ex2_PE545_long',ex2)

# np.save('ex12_PE545_long',ex12)

# np.save('x1_dephasing',x1)

# np.save('x2_PE545_Hevo',x2)

# np.save('n1_PE545_long',bb1)

# np.save('n2_PE545_long',bb2)

# %%

# localenergies = [(enE1m0m0,'E100'),(enE2m0m0,'E200'),(enE1m0m1,'E101'),(enE1m1m0,'E110'),(enE2m1m0,'E210'),(enE2m0m1,'E201'),(enE1m1m1,'E111'),(enE1m2m0,'E120'),(enE1m0m2,'E102')]

# np.save('EMM_energies_PE545',localenergies)

# %%

# np.save('eig_energies_PE545_w1500',vals[0:9])
