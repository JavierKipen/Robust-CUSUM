# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 11:04:34 2022

@author: JK-WORK
"""
import numpy as np
import copy


### Log likelihood ratios for the three models
def base_LLR(x,mu,std,delta):
    return delta / std**2 * (x - mu - delta / 2)

def bigaussian_LLR(x,mu,std,delta,std2,alpha):
    den=alpha * 1/np.sqrt(2*np.pi*std**2) * np.exp(-(x-mu)**2/(2*std**2))+ (1- alpha) * 1/np.sqrt(2*np.pi*std2**2) * np.exp(-(x-mu)**2/(2*std2**2))
    num=alpha * 1/np.sqrt(2*np.pi*std**2) * np.exp(-(x-mu-delta)**2/(2*std**2))+ (1- alpha) * 1/np.sqrt(2*np.pi*std2**2) * np.exp(-(x-mu-delta)**2/(2*std2**2))
    return np.log(num/den)  

def PWG_LLR_single(x,mu,std,delta,std2,KLLR,xc):
    retVal=np.nan;
    if x > mu + delta - xc and x < mu + xc:
        retVal = delta / std**2 * (x - mu - delta / 2)
    elif (x < mu-xc) or (x > mu + xc and x < mu + delta - xc) or (x > mu+ delta +xc):
        retVal = delta / std2**2 * (x - mu - delta / 2)
    elif x > mu-xc and x < mu + xc:
        retVal = -KLLR + (x-mu)**2/ (2* std**2) - (x-mu-delta)**2/ (2* std2**2) 
    elif x > mu + delta - xc and x < mu + delta + xc:
        retVal = KLLR + (x-mu)**2/ (2* std2**2) - (x-mu-delta)**2/ (2* std**2) 
    else:
        assert()
    return retVal;
    
def PWG_LLR(x,mu,std,std2,delta,KLLR,xc):
    if isinstance(x,list) or isinstance(x,np.ndarray):
        retVal=[];
        for i in x:
            retVal.append(PWG_LLR_single(i,mu,std,delta,std2,KLLR,xc))
    else:
        retVal=PWG_LLR_single(x,mu,std,delta,std2,KLLR,xc)
    return retVal
LLR_map={"Gaussian":base_LLR,"Bigaussian":bigaussian_LLR,"PWG":PWG_LLR_single};


def CUSUM(input, delta, h, std, LLR_callback=base_LLR,params=None, verbose=False): #This is from open Nanopore
    """
    Needs to be tested
    """
    # initialization
    Nd = k0 = 0
    kd = []
    krmv = []
    k = 1
    l = len(input)
    m = np.zeros(l)
    m[k0] = input[k0]
    accsum=m[k0];
    sp = np.zeros(l)
    Sp = np.zeros(l)
    gp = np.zeros(l)
    sn = np.zeros(l)
    Sn = np.zeros(l)
    gn = np.zeros(l)

    while k < l:
        accsum+=input[k]
        
        N=(k-k0+1)
        m[k] = accsum/N
        
        if params==None:
            sp[k] = LLR_callback(input[k],m[k],std,delta)
            sn[k] = LLR_callback(input[k],m[k],std,-delta)
        else:
            sp[k] = LLR_callback(input[k],m[k],std,delta,*params)
            sn[k] = LLR_callback(input[k],m[k],std,-delta,*params)
        

        Sp[k] = Sp[k - 1] + sp[k]
        Sn[k] = Sn[k - 1] + sn[k]

        gp_aux = gp[k - 1] + sp[k];
        gp[k] = gp_aux if gp_aux > 0 else 0;
        gn_aux = gn[k - 1] + sn[k]
        gn[k] = gn_aux if gn_aux > 0 else 0;

        if gp[k] > h or gn[k] > h:
            kd.append(k)
            if gp[k] > h:
                kmin = np.argmin(Sp[k0:k + 1])
                krmv.append(kmin + k0)
            else:
                kmin = np.argmin(Sn[k0:k + 1])
                krmv.append(kmin + k0)

            # Re-initialize
            k0 = kmin + k0 + 1
            m[k0] = input[k0]
            sp[k0] = Sp[k0] = gp[k0] = sn[k0] = Sn[k0] = gn[k0] = 0
            accsum=m[k0];
            k=copy.deepcopy(k0);
            Nd = Nd + 1
        k += 1
    if verbose:
        print('delta:' + str(delta))
        print('h:' + str(h))
        print('Nd: '+ str(Nd))
        print('krmv: ' + str(krmv))

    if Nd == 0:
        mc = np.mean(input) * np.ones(k)
    elif Nd == 1:
        mc = np.append(m[krmv[0]] * np.ones(krmv[0]), m[k - 1] * np.ones(k - krmv[0]))
    else:
        for change_idx in range(len(krmv)-1):
            change_sample= krmv[change_idx]
            prev_level=copy.deepcopy(m[change_sample])
            next_level=copy.deepcopy(m[krmv[change_idx+1]])
            while (abs(input[change_sample]-prev_level) < abs(input[change_sample]-next_level)) and change_sample < len(input) and change_sample < krmv[change_idx+1]-1:
                change_sample+=1;
                krmv[change_idx]=change_sample;
                m[change_sample]=m[change_sample-1];
                
        mc = m[krmv[0]] * np.ones(krmv[0])
        for ii in range(1, Nd):
            mc = np.append(mc, m[krmv[ii]] * np.ones(krmv[ii] - krmv[ii - 1]))
        mc = np.append(mc, m[k - 1] * np.ones(k - krmv[Nd - 1]))
    return (mc, kd, krmv)