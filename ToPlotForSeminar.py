# -*- coding: utf-8 -*-
"""
Created on Mon May  2 15:28:31 2022

@author: JK-WORK
"""


import numpy as np
from scipy.stats import norm
import matplotlib
import matplotlib.pyplot as plt
import time
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from CUSUM import *
import CUSUM_opt


def annotate_dim(ax,xyfrom,xyto,text=None):

    if text is None:
        text = str(np.sqrt( (xyfrom[0]-xyto[0])**2 + (xyfrom[1]-xyto[1])**2 ))

    ax.annotate("",xyfrom,xyto,arrowprops=dict(arrowstyle='<->'))
    ax.text((xyto[0]+xyfrom[0])/2,(xyto[1]+xyfrom[1])/2,text,fontsize=16)

matplotlib.rcParams['text.usetex'] = True
np.random.seed(2)

N=200;
std=1;
N_sw=int(N/2);
noise=np.random.normal(loc=0, scale=std, size=(N,))
true_signal=np.zeros((N,));true_signal[:N_sw]= -1;true_signal[N_sw:]= 1;
cont_signal=true_signal+noise;
time=np.arange(N);
mu=-1; delta=2; h= 50;
LLR_acc=np.cumsum(delta / std**2 * (cont_signal - mu - delta / 2))

min_LLR_acc=np.min(LLR_acc)
min_LLR_acc_idx=np.where(LLR_acc==min_LLR_acc)[0][0];
min_LLR_acc_det_idx=np.where(LLR_acc[min_LLR_acc_idx:]>=min_LLR_acc+h)[0][0] + min_LLR_acc_idx;
min_LLR_acc_det=LLR_acc[min_LLR_acc_det_idx]

fig, [ax1, ax2] = plt.subplots(2, 1, figsize=(10,10))
ax1.plot(time,true_signal, label='True signal')
ax1.plot(time,cont_signal, label='Contaminated signal')
ax1.legend()
ax1.set_xlabel(r'$k \; [samples]$')
ax1.set_ylabel(r'$x[k]$', fontsize=20)

ax2.plot(time,LLR_acc)
ax2.plot(min_LLR_acc_idx,min_LLR_acc,'ro')
ax2.plot(min_LLR_acc_det_idx,min_LLR_acc_det,'bo')
ax2.set_xlabel(r'$k \; [samples]$')
ax2.set_ylabel(r'$\sum_{i=1}^{k} ln \left(\frac{p(x_i | \theta_1)}{p(x_i |\theta_0)} \right)$', fontsize=20)

xyfrom=[min_LLR_acc_det_idx,min_LLR_acc]; xyto=[min_LLR_acc_det_idx,min_LLR_acc_det]
ax2.annotate("",xyfrom,xyto,arrowprops=dict(arrowstyle='<->'))
ax2.text((xyto[0]+xyfrom[0])/2+3,(xyto[1]+xyfrom[1])/2-2,r'$h$',fontsize=16)
xyfrom=[min_LLR_acc_idx,min_LLR_acc]; xyto=[min_LLR_acc_det_idx,min_LLR_acc]
# ax2.annotate("",xyfrom,xyto,
#              arrowprops={"arrowstyle" : "-", "linestyle" : "--",
#                          "linewidth" : 1.5, "shrinkA": 0, "shrinkB": 0})
ax2.annotate("",xyfrom,xyto,arrowprops=dict(arrowstyle='<->'))
ax2.text((xyto[0]+xyfrom[0])/2,(xyto[1]+xyfrom[1])/2+5,r'$d$',fontsize=16)