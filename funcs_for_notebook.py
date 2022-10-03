# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 17:16:44 2022

@author: JK-WORK
"""

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import time
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from CUSUM import *
import CUSUM_opt

LLR_map_opt={"Gaussian":CUSUM_opt.base_LLR_opt,"Bigaussian":CUSUM_opt.bigaussian_LLR_opt,"PWG":CUSUM_opt.PWG_LLR_single_opt};
############PDFs ###########
def guassian_pdf(x,mu,std):
    if isinstance(x,list) or isinstance(x,np.ndarray):
        retVal=[1/np.sqrt(2*np.pi*std**2) * np.exp(-(i-mu)**2/(2*std**2)) for i in x]
    else:
        retVal=1/np.sqrt(2*np.pi*std**2) * np.exp(-(x-mu)**2/(2*std**2))
    return retVal;

def biguassian_pdf(x,mu,std,std2,alpha):
    if isinstance(x,list) or isinstance(x,np.ndarray):
        retVal=[alpha * 1/np.sqrt(2*np.pi*std**2) * np.exp(-(i-mu)**2/(2*std**2)) + (1-alpha) * 1/np.sqrt(2*np.pi*std2**2) * np.exp(-(i-mu)**2/(2*std2**2)) for i in x]
    else:
        retVal=alpha * 1/np.sqrt(2*np.pi*std**2) * np.exp(-(x-mu)**2/(2*std**2)) + (1-alpha) * 1/np.sqrt(2*np.pi*std2**2) * np.exp(-(x-mu)**2/(2*std2**2))
    return retVal;

def pw_gaussian_pdf(x,mu,std,std2,alpha,Kcont,Kint,xc):
    if isinstance(x,list) or isinstance(x,np.ndarray):
        retVal=[Kint * alpha * 1/np.sqrt(2*np.pi*std**2) * np.exp(-(i-mu)**2/(2*std**2)) if np.abs(i-mu) < xc else Kint * Kcont * 1/np.sqrt(2*np.pi*std2**2) * np.exp(-(i-mu)**2/(2*std2**2)) for i in x]
    else:
        retVal= Kint * alpha * 1/np.sqrt(2*np.pi*std**2) * np.exp(-(x-mu)**2/(2*std**2)) if np.abs(x-mu) < xc else Kint * Kcont * 1/np.sqrt(2*np.pi*std2**2) * np.exp(-(x-mu)**2/(2*std2**2));
    return retVal;

### To fit a pw gaussian to the bigaussian

def fit_pw_to_bigaussian(mu,std,std2,alpha):
    xc=np.sqrt( (2 * std**2 * std2 **2/(std2**2 - std **2))*np.log( alpha * std2 / ((1 - alpha) * std)));
    kcont = alpha * std2 / std * np.exp((xc**2/2) * (std**2 - std2**2)/(std**2 * std2**2))
    kint=1/(alpha*(1-2*norm().cdf(-xc/std))+2*kcont*norm().cdf(-xc/std2));
    return xc,kcont,kint

def fit_gaussian_to_bigaussian(mu,std,std2,alpha):
    mu_fit=mu;
    std_fit=np.sqrt(alpha * std**2 + (1-alpha) * std2**2)
    return mu_fit,std_fit


### General CUSUM for different likelihoods:
def CUSUM_for_ARL(signal_in, delta, h,mu,std,LLR_callback=base_LLR,params=None,mode="Return first false alarm sample"): 
    # initialization
    Nd = k0 = 0
    kd = []
    krmv = []
    k = 1
    l = len(signal_in)
    m = np.zeros(l)
    sp = np.zeros(l)
    Sp = np.zeros(l)
    gp = np.zeros(l)
    gn = np.zeros(l)

    while k < l:#It should be sample by sample because mu and std are estimated. In this case they are not estimated to improve performance time.
        N=(k-k0+1)
        if params==None:
            sp[k] = LLR_callback(signal_in[k],mu,std,delta)
        else:
            sp[k] = LLR_callback(signal_in[k],mu,std,delta,*params)

        Sp[k] = Sp[k - 1] + sp[k]

        gp_aux = gp[k - 1] + sp[k];
        gp[k] = gp_aux if gp_aux > 0 else 0;

        if gp[k] > h :
            kd.append(k)
            if gp[k] > h:   
                if mode=="Return first false alarm sample":
                    kmin = np.argmin(Sp[k0:k + 1])
                    krmv.append(kmin + k0)
                    k=l-1;
                if mode=="Return detection sample":
                    krmv.append(k) #Saves detection sample
                    k=l-1;
            # Re-initialize
            k0 = k
            m[k0] = signal_in[k0]

            Nd = Nd + 1
        k += 1
    if mode=="Return first false alarm sample" or mode=="Return detection sample":
        retVal=krmv[0] if krmv else None;
    return retVal

### Functions to compare functional properties:

def gen_bigaussian_noise(mu,std,std2,alpha,length):
    len_std1=int(alpha*length);
    len_std2=int(length-len_std1);
    all_noise=np.concatenate((np.random.normal(loc=mu, scale=std, size=(1,len_std1)),np.random.normal(loc=mu, scale=std2, size=(1,len_std2))),axis=1)
    np.random.shuffle(np.transpose(all_noise))
    return all_noise[0] #To have it in 1D

def gen_ARL1_data(mu,std,std2,alpha,delta,H0_samples,H1_samples,n_runs=1e3):
    n_samples_heavy_tailed=np.max((int((1-alpha)*H0_samples),1));
    normal_noise=np.random.normal(loc=mu, scale=std, size=(int(n_runs),H0_samples-n_samples_heavy_tailed))
    heavy_tailed_noise=np.random.normal(loc=mu, scale=std2, size=(int(n_runs),n_samples_heavy_tailed))
    H0_data=np.concatenate((normal_noise,heavy_tailed_noise),axis=1)
    np.apply_along_axis(np.random.shuffle, 1, H0_data);
    normal_noise=np.random.normal(loc=mu+delta, scale=std, size=(int(n_runs),int(alpha*H1_samples)))
    heavy_tailed_noise=np.random.normal(loc=mu+delta, scale=std2, size=(int(n_runs),np.max((int((1-alpha)*H1_samples),1))))
    H1_data=np.concatenate((normal_noise,heavy_tailed_noise),axis=1);
    np.apply_along_axis(np.random.shuffle, 1, H1_data);
    return np.concatenate((H0_data,H1_data),axis=1)

def estimate_ARL0(mu,std,std2,alpha,delta,h,noise_generated, pdf_type,ARL0_tol,base_len, minN):
    ARL0_mean_est=0;ARL0_mean_std_est=np.inf; N_false_alarms=0;idx=0;ARL0_values=[];computing_times=[];
    if pdf_type=="Gaussian":
        mu,std_est= fit_gaussian_to_bigaussian(mu,std,std2,alpha)
        params=None
    elif pdf_type=="Bigaussian":
        std_est=std
        params=(std2,alpha)
    elif pdf_type=="PWG":
        std_est=std
        xc,Kcont,Kint=fit_pw_to_bigaussian(mu,std,std2,alpha)
        KLLR=np.log((alpha * std2) / (Kcont*std));
        params=(std2,KLLR,xc)
    while not((ARL0_mean_std_est < ARL0_tol * ARL0_mean_est) and (N_false_alarms > minN)):
        start=time.time();
        #res=CUSUM_opt.CUSUM_opt(noise_generated[idx:], delta, h, std_est, LLR_callback=LLR_map_opt[pdf_type],params=params, mode="Return first false alarm sample");
        res=CUSUM_for_ARL(noise_generated[idx:],delta,h,mu,std_est,LLR_callback=LLR_map[pdf_type],params=params,mode="Return detection sample")
        computing_times.append(time.time()-start);
        if res==None and N_false_alarms==0:
            print("base length was very short to estimate the ARL! Try with a bigger value.")
            break;
        elif res==None: #There were not enough samples and it needs to continue programming.
            print("Warning: More noise needed to be generated to have a better estimate of the ARL0")
            noise_generated=gen_bigaussian_noise(mu,std,std2,alpha,base_len); idx=0;
        else:
            ARL0_values.append(res)
            idx=idx+res;
            N_false_alarms += 1;
            ARL0_mean_est = np.mean(ARL0_values);
            ARL0_mean_std_est = np.std(ARL0_values)/N_false_alarms;
    return ARL0_mean_est,ARL0_mean_std_est,ARL0_values,computing_times;
    
def estimate_ARL1(mu,std,std2,alpha,delta,h,pdf_type,ARL0_mean_est,ARL0_frac_for_ARL1=0.2,n_runs=1e3,minN=5,ARL1_tol=0.05):
    H0_samples=np.max((int(ARL0_frac_for_ARL1*ARL0_mean_est),20));
    H1_samples=int(5*ARL0_mean_est);
    ARL1_mean_est=0;ARL1_mean_std_est=np.inf; count=0;db_count=0;ARL1_values=[];computing_times=[];
    db=gen_ARL1_data(mu,std,std2,alpha,delta,H0_samples,H1_samples,n_runs=n_runs) ## Generates batch of runs to speed up data generation.
    if pdf_type=="Gaussian":
        mu,std_est= fit_gaussian_to_bigaussian(mu,std,std2,alpha)
        params=None
    elif pdf_type=="Bigaussian":
        std_est=std
        params=(std2,alpha)
    elif pdf_type=="PWG":
        std_est=std
        xc,Kcont,Kint=fit_pw_to_bigaussian(mu,std,std2,alpha)
        KLLR=np.log((alpha * std2) / (Kcont*std));
        params=(std2,KLLR,xc)
    #Main Loop
    regen_count=0;
    while not((ARL1_mean_std_est < ARL1_tol * ARL1_mean_est) and (count > minN)):
        data=db[db_count,:]
        start=time.time();
        #res=CUSUM_opt.CUSUM_opt(noise_generated[idx:], delta, h, std_est, LLR_callback=LLR_map_opt[pdf_type],params=params, mode="Return first false alarm sample");
        res=CUSUM_for_ARL(data,delta,h,mu,std_est,LLR_callback=LLR_map[pdf_type],params=params,mode="Return detection sample")
        if res != None and res-H0_samples>0: #Triggers before breaking point are considered false alarms
            computing_times.append(time.time()-start);
            ARL1_values.append(res-H0_samples);
            count+=1
            ARL1_mean_est = np.mean(ARL1_values);
            ARL1_mean_std_est = np.std(ARL1_values)/count;
        db_count += 1;
        if db_count == n_runs:#In case we run out of runs, we generate data again.
            print("Warning: More noise needed to be generated to have a better estimate of the ARL1")
            db=gen_ARL1_data(mu,std,std2,alpha,delta,H0_samples,H1_samples,n_runs=n_runs);db_count=0;regen_count+=1;
        if regen_count > 10: #Ten times regenerated then we finish the measurement throwing a result that is not posible
            ARL1_mean_est=0.5;ARL1_mean_std_est=0;ARL1_values=[0.5,0.5];computing_times=[1,1];
            break;
        
    return ARL1_mean_est,ARL1_mean_std_est,ARL1_values,computing_times,H0_samples;
def get_computing_time(ARL0_values,computing_timesARL0,ARL1_values,computing_timesARL1,H0_len):
    ARL0_comp_time=np.mean([computing_timesARL0[i]/ARL0_values[i] for i in range(len(ARL0_values))])
    ARL1_comp_time=np.mean([computing_timesARL1[i]/(ARL1_values[i]+H0_len) for i in range(len(ARL1_values))]) #ARL1 value can be zero or negative.
    return ARL0_comp_time,ARL1_comp_time
def analyze_configuration(mu,std,std2,alpha,delta,h,pdf_type="Gaussian",base_len=1e6, tol=0.05, minN=5):
    noise_generated=gen_bigaussian_noise(mu,std,std2,alpha,int(base_len))
    ARL0_mean_est,ARL0_mean_std_est,ARL0_values,computing_timesARL0=estimate_ARL0(mu,std,std2,alpha,delta,h,noise_generated, pdf_type,tol,base_len, minN)
    ARL1_mean_est,ARL1_mean_std_est,ARL1_values,computing_timesARL1,H0_len=estimate_ARL1(mu,std,std2,alpha,delta,h,pdf_type,ARL0_mean_est,ARL1_tol=tol)
    ARL0_comp_time,ARL1_comp_time=get_computing_time(ARL0_values,computing_timesARL0,ARL1_values,computing_timesARL1,H0_len)

    return ARL0_mean_est,ARL0_mean_std_est,ARL1_mean_est,ARL1_mean_std_est,ARL0_comp_time,ARL1_comp_time


##To pick the correct hs to obtain the desired ARLs:
def fit_to_line(ARL0_mean_est_log,hs):
    ARL0_vals=np.asarray(ARL0_mean_est_log);h_vals=np.asarray(hs)
    ARL0_vals=np.reshape(ARL0_vals,(-1,1));h_vals=np.reshape(h_vals,(-1,1))
    reg = LinearRegression().fit(h_vals,ARL0_vals)
    m=reg.coef_[0];
    b=reg.intercept_;
    r2=r2_score(ARL0_vals, reg.predict(h_vals))
    return m,b,r2

def get_h_lims_for_target_ARL(target_ARL0_range,m,b):
    return (np.log(target_ARL0_range)-b)/m

def get_low_std_estimates_for_hs(mu,std,std2,alpha,delta,h,noise_generated, pdf_type ,ext_tol,base_len, minN,h_sum=True):
    ARL0_vals_log=[];h_vals=[];
    for i in range(3): #3 points to throw a rect
        ARL0_mean_est,ARL0_mean_std_est,ARL0_values,computing_timesARL0=estimate_ARL0(mu,std,std2,alpha,delta,h,noise_generated, pdf_type,ext_tol,base_len, minN);
        ARL0_vals_log.append(np.log(ARL0_mean_est));h_vals.append(h);
        if h_sum==True:
            h += 0.1;
        else:
            h *= 1.1; 
    return ARL0_vals_log, h_vals
def estimate_h_range_for_all_types(mu,std,std2,alpha,delta,target_ARL0_range,start_h=2.5,base_len=1e6, fast_tol=0.05,ext_tol=0.001, minN=5,verbose=1):
    noise_generated=gen_bigaussian_noise(mu,std,std2,alpha,int(base_len))
    h=start_h-0.1;
    ARL0_mean_est=0;
    while ARL0_mean_est < 75:
        h=h*1.1;
        ARL0_mean_est,ARL0_mean_std_est,ARL0_values,computing_timesARL0=estimate_ARL0(mu,std,std2,alpha,delta,h,noise_generated, "Bigaussian",fast_tol,base_len, minN);
    if verbose:
        print("Base ARL0 found for bigaussian is :" + str(ARL0_mean_est) + " with H= " + str(h));
    ARL0_vals_log,h_vals=get_low_std_estimates_for_hs(mu,std,std2,alpha,delta,h,noise_generated, "Bigaussian" ,ext_tol,base_len, minN,h_sum=True)
    m_bigauss,b_bigauss,r2 = fit_to_line(ARL0_vals_log,h_vals);
    h_lims_bigauss = get_h_lims_for_target_ARL(target_ARL0_range,m_bigauss,b_bigauss)
    h_lims_pwg= h_lims_bigauss*1.1; #Seemed to work fine before
    if m_bigauss < 0 or h_lims_bigauss[0]<0:
        print("ARL0 fitted decaying with h or fitting error. Maybe due to the randomness of estimation")
        h_lims_bigauss=[h/2,h*5]
    if verbose:
        print("Hs range for bigaussian are: "+ str(h_lims_bigauss[0]) +" - "+ str(h_lims_bigauss[1]) + ". R2 val is: "+str(r2));
    h_g=h*1.01; #Starting point to guess for the gaussian
    mu,std_est= fit_gaussian_to_bigaussian(mu,std,std2,alpha)
    ARL0_mean_est=0;
    while ARL0_mean_est < 100:
        h_g=h_g*1.1;
        ARL0_mean_est,ARL0_mean_std_est,ARL0_values,computing_timesARL0=estimate_ARL0(mu,std,std2,alpha,delta,h_g,noise_generated, "Gaussian",fast_tol,base_len, minN);
    if verbose:
        print("Base ARL0 found for gaussian is :" + str(ARL0_mean_est) + " with H= " + str(h_g));
    ARL0_vals_log,h_vals=get_low_std_estimates_for_hs(mu,std,std2,alpha,delta,h_g,noise_generated, "Gaussian" ,ext_tol,base_len, minN,h_sum=False)
    m_gauss,b_gauss,r2 = fit_to_line(ARL0_vals_log,h_vals);
    h_lims_gauss = get_h_lims_for_target_ARL(target_ARL0_range,m_gauss,b_gauss)
    if m_gauss < 0 or h_lims_gauss[0]<0:
        print("ARL0 fitted decaying with h or fitting error. Maybe due to the randomness of estimation")
        h_lims_gauss=[h_g/2,h_g*5]
    if verbose:
        print("Hs range for Gaussian are: "+ str(h_lims_gauss[0]) +" - " + str(h_lims_gauss[1])+ ". R2 val is: "+str(r2));
    return h_lims_bigauss, h_lims_pwg,h_lims_gauss

def estimate_h_range_and_evaluate_for_all_types(mu,std,std2,alpha,delta,target_ARL0_range,verbose=1):
    h_lims_bigauss, h_lims_pwg, h_lims_gauss =estimate_h_range_for_all_types(mu,std,std2,alpha,delta,target_ARL0_range,verbose=verbose)
    #Checking that the results with the given hs!
    noise_len=1e6;
    noise_generated=gen_bigaussian_noise(mu,std,std2,alpha,int(noise_len))
    tol=0.001;
    ARL0_bigaussian_min,_,_,_=estimate_ARL0(mu,std,std2,alpha,delta,h_lims_bigauss[0],noise_generated, "Bigaussian",tol,noise_len, minN=5)
    ARL0_bigaussian_max,_,_,_=estimate_ARL0(mu,std,std2,alpha,delta,h_lims_bigauss[1],noise_generated, "Bigaussian",tol,noise_len, minN=5)
    ARL0_bigaussian=[ARL0_bigaussian_min,ARL0_bigaussian_max]
    if verbose:
        print("Bigaussian: Min ARL0 got: " +str(ARL0_bigaussian_min) + " And max ARL0:"+str(ARL0_bigaussian_max))
    ARL0_gaussian_min,_,_,_=estimate_ARL0(mu,std,std2,alpha,delta,h_lims_gauss[0],noise_generated, "Gaussian",tol,noise_len, minN=5)
    ARL0_gaussian_max,_,_,_=estimate_ARL0(mu,std,std2,alpha,delta,h_lims_gauss[1],noise_generated, "Gaussian",tol,noise_len, minN=5)
    ARL0_gaussian=[ARL0_gaussian_min,ARL0_gaussian_max]
    if verbose:
        print("Gaussian: Min ARL0 got: " +str(ARL0_gaussian_min) + " And max ARL0:"+str(ARL0_gaussian_max))
    ARL0_pwg_min,_,_,_=estimate_ARL0(mu,std,std2,alpha,delta,h_lims_pwg[0],noise_generated, "PWG",tol,noise_len, minN=5)
    ARL0_pwg_max,_,_,_=estimate_ARL0(mu,std,std2,alpha,delta,h_lims_pwg[1],noise_generated, "PWG",tol,noise_len, minN=5)
    ARL0_pwg=[ARL0_pwg_min,ARL0_pwg_max]
    if verbose:
        print("PWG: Min ARL0 got: " +str(ARL0_pwg_min) + " And max ARL0:"+str(ARL0_pwg_max))
    return h_lims_bigauss, h_lims_pwg, h_lims_gauss, ARL0_bigaussian, ARL0_gaussian, ARL0_pwg

    

### This part is to generate synthetic data and analyze it with cusum  
def plot_LLR_opt(std,std2,alpha,title="Log likelihood ratio plot [asssuming initial state -1]"):
    X=np.linspace(-2,2,1000);
    _,std_g= fit_gaussian_to_bigaussian(0,std,std2,alpha)
    xc,Kcont,Kint=fit_pw_to_bigaussian(0,std,std2,alpha)
    KLLR=np.log((alpha * std2) / (Kcont*std));
    g_LLR=[];bi_g_LLR=[];pwg_LLR=[];
    for xi in X:
        g_LLR.append(CUSUM_opt.base_LLR_opt(xi,-1,std_g,2))
        bi_g_LLR.append(CUSUM_opt.bigaussian_LLR_opt(xi,-1,std,2,std2,alpha))
        pwg_LLR.append(CUSUM_opt.PWG_LLR_single_opt(xi,-1,std,2,std2,KLLR,xc))
    plt.figure(figsize=(14,7))
    plt.plot(X,g_LLR,label="Assuming gaussian noise",color="red")
    plt.plot(X,bi_g_LLR,label="True LLR",color="blue")
    plt.plot(X,pwg_LLR,label="Approximated PWG LLR",color="green")
    plt.ylim(np.min(bi_g_LLR),np.max(bi_g_LLR))
    plt.title(title)
    plt.ylabel("Log likelihood ratio")
    plt.xlabel("x")
    plt.legend()
    plt.show()  


def generate_ss_true_data(length,expected_switching_time,fs=200e3,v_high=1,v_low=-1):
    out_arr=np.empty((length,))
    t=np.linspace(0,length/fs,length)
    expected_switching_samples=int(expected_switching_time * fs);
    n_durations_to_take=5*length/expected_switching_samples; #So its very likely that there will be enough durations to fill the array
    state_durations=np.floor(np.random.exponential(scale=expected_switching_samples-1,size=(int(n_durations_to_take),))+1);
    state_durations=state_durations.astype(int)
    out_idx=int(0);state = v_high; d_list=[];states=[]
    for d in state_durations:
        if out_idx+d < length :
            out_arr[out_idx:out_idx+d]=state;
            d_list.append(d);states.append(state)
            state = v_high if state == v_low else v_low;
            out_idx += d    
        else:
            out_arr[out_idx:]=state;
            d_list.append(len(out_arr)-out_idx);states.append(state)
            out_idx=length
            break;
    if out_idx != length:
        assert() #This should be very unlikely.
    return t,out_arr,d_list,states;
def get_noise_parameters(SNR,alpha,std2_rel):
    total_var=1/SNR; #True signal has power 1.
    std=np.sqrt(total_var/(alpha+ (1-alpha)*std2_rel**2))
    std2=std2_rel*std
    return std, std2;
def generate_synthetic_data(length,SNR,expected_switching_time=1e-3,alpha=0.9,std2_rel=6): #std2_rel indicates how high is the 2nd std compared to the first one.
    t,true_signal,d_list,states=generate_ss_true_data(length,expected_switching_time);
    std, std2=get_noise_parameters(SNR,alpha,std2_rel)
    noise=gen_bigaussian_noise(0,std,std2,alpha,length)
    cont_signal=true_signal+noise;
    return t,true_signal,cont_signal,d_list,states,std,std2,alpha;
def run_cusum(h,cont_signal,pdf_type,alpha,std,std2,delta=2,opt=True):
    params=None;
    if pdf_type=="Gaussian":
        _,std_to_use= fit_gaussian_to_bigaussian(0,std,std2,alpha)
    elif pdf_type=="Bigaussian":
        std_to_use=std
        params=(std2,alpha)
    elif pdf_type=="PWG":
        std_to_use=std
        xc,Kcont,Kint=fit_pw_to_bigaussian(0,std,std2,alpha)
        KLLR=np.log((alpha * std2) / (Kcont*std));
        params=(std2,KLLR,xc)
    else:
        assert();
    if opt:
        (mc, kd, krmv) = CUSUM_opt.CUSUM_opt(cont_signal, delta, h, std_to_use, LLR_callback=LLR_map_opt[pdf_type],params=params, verbose=False);
    else:
        (mc, kd, krmv) = CUSUM(cont_signal, delta, h, std_to_use, LLR_callback=LLR_map[pdf_type],params=params, verbose=False);
    return mc
def compute_results(true_signal,d_list,states,cont_signal,cusum_output):
    total_MSE=np.mean((true_signal-cusum_output)**2)
    class_output=np.sign(cusum_output);
    per_sample_clasification_score=np.sum(true_signal==class_output)/len(true_signal); #How many were classified correctly in samples
    total_MSE_out_q=np.sum((true_signal-np.round(cusum_output))**2)
    sample_idx=0;
    sum_squared_error_list=[];
    per_state_clasification=[];#For each state we consider it was correctly mapped if more than half of the samples were on the same sign
    for d in d_list:
        true_local=true_signal[sample_idx:sample_idx+d]
        cusum_out_local=cusum_output[sample_idx:sample_idx+d]
        class_output_local=class_output[sample_idx:sample_idx+d]
        sum_squared_error_list.append(np.sum((true_local-cusum_out_local)**2))
        per_state_clasification.append(1 if np.sum(true_local==class_output_local)/d >= 0.5 else 0)
        sample_idx+=d;
    return total_MSE,sum_squared_error_list,per_sample_clasification_score,per_state_clasification,total_MSE_out_q

def loop_config_through_hs(hs,SNR,n_runs=3,length=int(200e3),pdf_types=["Gaussian","Bigaussian", "PWG"],alpha=0.9,std2_rel=6,expected_switching_time=1e-3):
    n_pdfs=len(pdf_types);
    n_statistics=4; #Number of statistics to output (MSE, per sample class and per state class)
    output_results=np.zeros((n_pdfs,len(hs),n_statistics,n_runs))
    for h_idx in range(len(hs)):
        h=hs[h_idx]
        for i in range(n_runs):
            t,true_signal,cont_signal,d_list,states,std,std2,alpha=generate_synthetic_data(length,SNR,alpha=alpha,std2_rel=std2_rel,expected_switching_time=expected_switching_time)
            for pdf_type_idx in range(n_pdfs):
                pdf_type=pdf_types[pdf_type_idx];
                cusum_output=run_cusum(h,cont_signal,pdf_type,alpha,std,std2,delta=2,opt=True)
                total_MSE,_,per_sample_clasification_score,per_state_clasification,total_MSE_out_q=compute_results(true_signal,d_list,states,cont_signal,cusum_output)
                output_results[pdf_type_idx,h_idx,0,i]=total_MSE;
                output_results[pdf_type_idx,h_idx,1,i]=per_sample_clasification_score;
                output_results[pdf_type_idx,h_idx,2,i]=np.mean(per_state_clasification);
                output_results[pdf_type_idx,h_idx,3,i]=total_MSE_out_q;
    return output_results,pdf_types



def plot_loop_config_through_hs_output(hs,SNR,pdf_types,output_results,title=None):
    fig, (ax1,ax2,ax3,ax4) = plt.subplots(1, 4, figsize=(24,10))
    for pdf_type_idx in range(len(pdf_types)):
        pdf_type=pdf_types[pdf_type_idx]
        MSEs=np.mean(output_results[pdf_type_idx,:,0,:],axis=1);
        ax1.plot(hs,MSEs,label=pdf_type+" - Min: " + "{:.3g}".format(np.min(MSEs)))
        psacs=np.mean(output_results[pdf_type_idx,:,1,:],axis=1);
        ax2.plot(hs,psacs,label=pdf_type+" - Max: " + "{:.3g}".format(np.max(psacs)))
        pstcs=np.mean(output_results[pdf_type_idx,:,2,:],axis=1);
        ax3.plot(hs,pstcs,label=pdf_type+" - Max: " + "{:.3g}".format(np.max(pstcs)))
        MSEs_q=np.mean(output_results[pdf_type_idx,:,3,:],axis=1);
        ax4.plot(hs,MSEs_q,label=pdf_type+" - Min: " + "{:.3g}".format(np.min(MSEs_q)))
    ax1.legend();ax2.legend();ax3.legend();ax4.legend();
    ax1.set_xlabel("h");ax1.set_ylabel("MSE")
    ax2.set_xlabel("h");ax2.set_ylabel("Per sample class rate")
    ax3.set_xlabel("h");ax3.set_ylabel("Per state class rate")
    ax4.set_xlabel("h");ax4.set_ylabel("MSE signal quantized")
    if title is None:
        title="Results for SNR="+"{:.3g}".format(SNR)
    fig.suptitle(title, fontsize=18)
    
def simple_run_test(hs,pdf_types=["Gaussian","Bigaussian", "PWG"],SNR=1,length=int(200e3),alpha=0.9,std2_rel=6,expected_switching_time=1e-3):
    if not isinstance(hs,list):
        hs=[hs]*len(pdf_types)
    t,true_signal,cont_signal,d_list,states,std,std2,alpha=generate_synthetic_data(length,SNR,expected_switching_time=expected_switching_time,alpha=alpha,std2_rel=std2_rel)
    
    plt.figure(figsize=(20,10))
    plt.plot(t,true_signal,label="True signal")
    plt.plot(t,cont_signal,label="Contaminated signal",alpha=0.2)
    for pdf_type_idx in range(len(pdf_types)):
        pdf_type=pdf_types[pdf_type_idx]
        cusum_output=run_cusum(hs[pdf_type_idx],cont_signal,pdf_type,alpha,std,std2,delta=2,opt=True);
        plt.plot(t,cusum_output, label= pdf_type+ " h: "+ str(hs[pdf_type_idx]));
    plt.legend()
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.title("Run comparison with SNR=" + "{:.3g}".format(SNR)+ " Len=" + "{:.3g}".format(length)+ " alpha=" + "{:.3g}".format(alpha)+ " std2_rel=" + "{:.3g}".format(std2_rel)+ " expected_switching_time=" + "{:.3g}".format(expected_switching_time))
    plt.show();
        
## For the paper

def means_are_in_tol(MSE_lists,tol):
    retVal=True;
    means=[np.mean(i) for i in MSE_lists];
    stds=[np.std(i) for i in MSE_lists]
    Ns=[len(i) for i in MSE_lists]
    for mean_idx in range(len(means)):
        retVal= retVal and stds[mean_idx]/(Ns[mean_idx]*means[mean_idx])<tol;
    return retVal;

def loop_config_through_hs_MSE_w_tol(hs,SNR,tol=0.01,length=int(200e3),pdf_types=["Gaussian","Bigaussian", "PWG"],alpha=0.9,std2_rel=6,expected_switching_time=1e-3, minN=5):
    n_pdfs=len(pdf_types);
    output_results=np.zeros((n_pdfs,len(hs)))
    for h_idx in range(len(hs)):
        count=0;
        h=hs[h_idx]
        MSE_lists=[[]for i in range(n_pdfs)];
        while (not means_are_in_tol(MSE_lists,tol)) or count<minN:
            t,true_signal,cont_signal,d_list,states,std,std2,alpha=generate_synthetic_data(length,SNR,alpha=alpha,std2_rel=std2_rel,expected_switching_time=expected_switching_time)
            for pdf_type_idx in range(n_pdfs):
                if not means_are_in_tol([MSE_lists[pdf_type_idx]],tol) or count<minN:
                    pdf_type=pdf_types[pdf_type_idx];
                    cusum_output=run_cusum(h,cont_signal,pdf_type,alpha,std,std2,delta=2,opt=True)
                    total_MSE,_,_,_,_=compute_results(true_signal,d_list,states,cont_signal,cusum_output)
                    MSE_lists[pdf_type_idx].append(total_MSE)
            count+=1;
        for i in range(n_pdfs):
            output_results[i,h_idx]=np.mean(MSE_lists[i])
    return output_results,pdf_types

def plot_loop_MSE(hs,SNR,pdf_types,output_results):
    plt.figure()
    for pdf_type_idx in range(len(pdf_types)):
        pdf_type= pdf_types[pdf_type_idx]
        plt.plot(hs,output_results[pdf_type_idx,:],label=pdf_type+" - MMSE: " + "{:.3f}".format(np.min(output_results[pdf_type_idx,:])));
    plt.legend();
    plt.title("MSE analysis with SNR="+str(SNR))
    plt.xlabel("h value")
    plt.ylabel("MSE")
    plt.grid();
    plt.show();
    
def get_data_for_hist(h=15,SNR=0.5,base_len=int(1e6),n_runs=10,alpha=0.9,std2_rel=6,expected_switching_time=1e-3,nbins=30):
    n_pdfs=len(pdf_types);
    classification_scores=[[]for i in range(n_pdfs)];
    total_d_list=[];
    fs=200e3;
    for i in range(n_runs):
        t,true_signal,cont_signal,d_list,states,std,std2,alpha=generate_synthetic_data(length,SNR,alpha=alpha,std2_rel=std2_rel,expected_switching_time=expected_switching_time,pdf_types=["Gaussian","Bigaussian", "PWG"])
        total_d_list.append(d_list)
        for pdf_type_idx in range(n_pdfs):
            pdf_type=pdf_types[pdf_type_idx];
            cusum_output=run_cusum(h,cont_signal,pdf_type,alpha,std,std2,delta=2,opt=True)
            _,_,_,per_state_clasification,_=compute_results(true_signal,d_list,states,cont_signal,cusum_output)
            classification_scores[pdf_type_idx].append(per_state_clasification)
    total_d_list_t=[i/fs for i in total_d_list]
    return total_d_list_t,classification_scores

def plot_hist_classifications(total_d_list_t,classification_scores):
    bins=np.logspace(np.log10(np.min(total_d_list_t)),np.log10(np.max(total_d_list_t)), nbins)
    plt.hist(total_d_list_t, bins=bins)
    plt.gca().set_xscale("log")
    plt.show()
## For the re do of the plots of ARL0 vs ARL1:
def analyze_configuration_v2(h,SNR,alpha,std2_rel,pdf_type="Gaussian",base_len=1e6, tol=0.05, minN=20,delta=2):
    std, std2=get_noise_parameters(SNR,alpha,std2_rel)
    noise_generated=gen_bigaussian_noise(0,std,std2,alpha,int(base_len))
    ARL0_mean_est,ARL0_mean_std_est,ARL0_values,computing_timesARL0=estimate_ARL0(0,std,std2,alpha,delta,h,noise_generated, pdf_type,tol,base_len, minN)
    ARL1_mean_est,ARL1_mean_std_est,ARL1_values,computing_timesARL1,H0_len=estimate_ARL1(0,std,std2,alpha,delta,h,pdf_type,ARL0_mean_est,ARL1_tol=tol)
    ARL0_comp_time,ARL1_comp_time=get_computing_time(ARL0_values,computing_timesARL0,ARL1_values,computing_timesARL1,H0_len)
    return ARL0_mean_est,ARL0_mean_std_est,ARL1_mean_est,ARL1_mean_std_est,ARL0_comp_time,ARL1_comp_time

def estimate_h_range_for_all_types_v2(SNR,alpha,std2_rel,target_ARL0_range,delta=2,start_h=1.5,minN=30,inc=1.15,tol=0.01,base_len=int(1e6)):
    std, std2=get_noise_parameters(SNR,alpha,std2_rel)
    noise_generated=gen_bigaussian_noise(0,std,std2,alpha,int(base_len))
    pdfs=["Gaussian", "Bigaussian", "PWG"]
    hlims=[[] for i in range(3)]
    for pdf_idx in range(3):
        pdf=pdfs[pdf_idx]
        h=start_h;
        ARL0_mean_est=0;
        while ARL0_mean_est < target_ARL0_range[0]:
            h=h*inc;
            ARL0_mean_est,ARL0_mean_std_est,ARL0_values,computing_timesARL0=estimate_ARL0(0,std,std2,alpha,delta,h,noise_generated, pdf,tol,base_len, minN);
        hlims[pdf_idx].append(h/inc) #To be sure that the range is included.
        while ARL0_mean_est < target_ARL0_range[1]:
            h=h*inc;
            ARL0_mean_est,ARL0_mean_std_est,ARL0_values,computing_timesARL0=estimate_ARL0(0,std,std2,alpha,delta,h,noise_generated, pdf,tol,base_len, minN);
        hlims[pdf_idx].append(h)
    
    return hlims[0], hlims[1],hlims[2] #Gauss, bigauss, pwg

### Test functions

def test_pdfs_and_fits():
    mu=0;alpha=0.91;std=0.013;std2=0.074; #Fits for dev 17 on 0709 
    fit_pw_to_bigaussian(mu,std,std2,alpha)
    x_plot=np.linspace(-0.3,0.3,num=1000);
    mu_g,std_g=fit_gaussian_to_bigaussian(mu,std,std2,alpha)
    xc,Kcont,Kint=fit_pw_to_bigaussian(mu,std,std2,alpha)
    plt.figure(figsize=(14,7))
    plt.semilogy(x_plot,guassian_pdf(x_plot,mu_g,std_g),label="Gaussian fit")
    plt.semilogy(x_plot,biguassian_pdf(x_plot,mu,std,std2,alpha),label="True bigaussian dist")
    plt.semilogy(x_plot,pw_gaussian_pdf(x_plot,mu,std,std2,alpha,Kcont,Kint,xc),label="PWG approx")
    plt.ylim(1e-4,100)
    plt.ylabel("Log of prob density")
    plt.xlabel("x")
    plt.legend()
    plt.show()    
   
def test_log_likelihoods():
    print("In this case is shown when the expected jump is before the breaking point")
    mu=0;alpha=0.91;std=0.013;std2=0.074; #Fits for dev 17 on 0709 
    mean_current=0.150; 
    delta= 0.15*mean_current#Jumps in state switching devices are likely to be around 10-20% of the mean value
    fit_pw_to_bigaussian(mu,std,std2,alpha)
    x_plot=np.linspace(-0.3,mu+delta+0.3,num=1000);
    mu_g,std_g=fit_gaussian_to_bigaussian(mu,std,std2,alpha)
    xc,Kcont,Kint=fit_pw_to_bigaussian(mu,std,std2,alpha)
    plt.figure(figsize=(14,7))
    plt.semilogy(x_plot,biguassian_pdf(x_plot,mu,std,std2,alpha),label="Hyp 0",color="red")
    plt.semilogy(x_plot,biguassian_pdf(x_plot,mu+delta,std,std2,alpha),label="Hyp 1",color="blue")
    plt.ylim(1e-4,100)
    plt.ylabel("Log of prob density")
    plt.xlabel("x")
    plt.legend()
    plt.show()  
    #Log likelihood plot
    x_plot=np.linspace(-0.1,mu+delta+0.1,num=1000);
    plt.figure(figsize=(14,7))
    KLLR=np.log((alpha * std2) / (Kcont*std));
    plt.plot(x_plot,base_LLR(x_plot,mu_g,std_g,delta),label="Assuming gaussian noise",color="red")
    plt.plot(x_plot,bigaussian_LLR(x_plot,mu,std,delta,std2,alpha),label="True LLR",color="blue")
    plt.plot(x_plot,PWG_LLR(x_plot,mu,std,std2,delta,KLLR,xc),label="Approximated PWG LLR",color="green")
    plt.ylim(-5,5)
    plt.title("Log likelihood ratio plot")
    plt.ylabel("Log likelihood ratio")
    plt.xlabel("x")
    plt.legend()
    plt.show()  
    print("In this case is shown when the expected jump is after the breaking point")
    mu=0;alpha=0.91;std=0.013;std2=0.074; #Fits for dev 45 on 0709 
    mean_current=1.03; 
    delta= 0.15*mean_current#Jumps in state switching devices are likely to be around 10-20% of the mean value
    fit_pw_to_bigaussian(mu,std,std2,alpha)
    x_plot=np.linspace(-0.3,mu+delta+0.3,num=1000);
    mu_g,std_g=fit_gaussian_to_bigaussian(mu,std,std2,alpha)
    xc,Kcont,Kint=fit_pw_to_bigaussian(mu,std,std2,alpha)
    plt.figure(figsize=(14,7))
    plt.semilogy(x_plot,biguassian_pdf(x_plot,mu,std,std2,alpha),label="Hyp 0",color="red")
    plt.semilogy(x_plot,biguassian_pdf(x_plot,mu+delta,std,std2,alpha),label="Hyp 1",color="blue")
    plt.ylim(1e-4,100)
    plt.ylabel("Log of prob density")
    plt.xlabel("x")
    plt.legend()
    plt.show()  
    #Log likelihood plot
    KLLR=np.log((alpha * std2) / (Kcont*std));
    x_plot=np.linspace(-0.2,mu+delta+0.2,num=1000);
    plt.figure(figsize=(14,7))
    plt.plot(x_plot,base_LLR(x_plot,mu_g,std_g,delta),label="Assuming gaussian noise",color="red")
    plt.plot(x_plot,bigaussian_LLR(x_plot,mu,std,delta,std2,alpha),label="True likelihood",color="blue")
    plt.plot(x_plot,PWG_LLR(x_plot,mu,std,std2,delta,KLLR,xc),label="Approximated PWG LLR",color="green")
    plt.ylim(-10,10)
    plt.title("Log likelihood ratio plot")
    plt.ylabel("Log likelihood ratio")
    plt.xlabel("x")
    plt.legend()
    plt.show()  
    
def test_cusum():
    mu=0;alpha=0.91;std=0.013;std2=0.074;  
    mean_current=0.150; h=7;
    delta= 0.15*mean_current
    data=gen_bigaussian_noise(mu,std,std2,alpha,10000)
    pdf_type="Gaussian"
    mu,std_est= fit_gaussian_to_bigaussian(mu,std,std2,alpha)
    params=None
    res=CUSUM_for_ARL(data,delta,h,mu,std_est,LLR_callback=LLR_map[pdf_type],params=params,mode="Return detection sample")
    pdf_type="Bigaussian"
    std_est=std
    params=(std2,alpha)
    res=CUSUM_for_ARL(data,delta,h,mu,std_est,LLR_callback=LLR_map[pdf_type],params=params,mode="Return detection sample")
    pdf_type="PWG"
    std_est=std
    xc,Kcont,Kint=fit_pw_to_bigaussian(mu,std,std2,alpha)
    KLLR=np.log((alpha * std2) / (Kcont*std));
    params=(std2,KLLR,xc)
    res=CUSUM_for_ARL(data,delta,h,mu,std_est,LLR_callback=LLR_map[pdf_type],params=params,mode="Return detection sample")
    print(res)

def test_ARL_estimations():
    mu=0;alpha=0.91;std=0.013;std2=0.074;  
    mean_current=0.15; h=10;
    delta= 0.15*mean_current
    ARL0_mean_est,ARL0_mean_std_est,ARL1_mean_est,ARL1_mean_std_est,ARL0_comp_time,ARL1_comp_time=analyze_configuration(mu,std,std2,alpha,delta,h,pdf_type="Gaussian")
    h=3;
    ARL0_mean_est,ARL0_mean_std_est,ARL1_mean_est,ARL1_mean_std_est,ARL0_comp_time,ARL1_comp_time=analyze_configuration(mu,std,std2,alpha,delta,h,pdf_type="Bigaussian")
    ARL0_mean_est,ARL0_mean_std_est,ARL1_mean_est,ARL1_mean_std_est,ARL0_comp_time,ARL1_comp_time=analyze_configuration(mu,std,std2,alpha,delta,h,pdf_type="PWG")
    
def test_h_range_predictions():
    mu=0;alpha=0.91;std=0.013;std2=0.074;  
    mean_current=0.15; h=10;
    delta= 0.15*mean_current
    target_ARL0_range=[50, 500]
    h_lims_bigauss, h_lims_pwg, h_lims_gauss, ARL0_bigaussian, ARL0_gaussian, ARL0_pwg = estimate_h_range_and_evaluate_for_all_types(mu,std,std2,alpha,delta,target_ARL0_range,verbose=1)
    # h_lims_bigauss, h_lims_pwg,h_lims_gauss= estimate_h_range_for_all_types(mu,std,std2,alpha,delta,target_ARL0_range,start_h=2.5,base_len=1e6, fast_tol=0.05,ext_tol=0.001, minN=5,verbose=1)
    # #Checking that the results with the given hs!
    # noise_len=1e6;
    # noise_generated=gen_bigaussian_noise(mu,std,std2,alpha,int(noise_len))
    # tol=0.001;
    # ARL_min,_,_,_=estimate_ARL0(mu,std,std2,alpha,delta,h_lims_bigauss[0],noise_generated, "Bigaussian",tol,noise_len, minN=5)
    # ARL_max,_,_,_=estimate_ARL0(mu,std,std2,alpha,delta,h_lims_bigauss[1],noise_generated, "Bigaussian",tol,noise_len, minN=5)
    # print("Bigaussian: Min ARL0 got: " +str(ARL_min) + " And max ARL0:"+str(ARL_max))
    # ARL_min,_,_,_=estimate_ARL0(mu,std,std2,alpha,delta,h_lims_gauss[0],noise_generated, "Gaussian",tol,noise_len, minN=5)
    # ARL_max,_,_,_=estimate_ARL0(mu,std,std2,alpha,delta,h_lims_gauss[1],noise_generated, "Gaussian",tol,noise_len, minN=5)
    # print("Gaussian: Min ARL0 got: " +str(ARL_min) + " And max ARL0:"+str(ARL_max))
    # ARL_min,_,_,_=estimate_ARL0(mu,std,std2,alpha,delta,h_lims_pwg[0],noise_generated, "PWG",tol,noise_len, minN=5)
    # ARL_max,_,_,_=estimate_ARL0(mu,std,std2,alpha,delta,h_lims_pwg[1],noise_generated, "PWG",tol,noise_len, minN=5)
    # print("PWG: Min ARL0 got: " +str(ARL_min) + " And max ARL0:"+str(ARL_max))
    
def test_synth_data_cusum():
    length=int(200e3);# would be 1 second long
    SNR=5;h=50;pdf_type="PWG"
    #np.random.seed(1)
    t,true_signal,cont_signal,d_list,states,std,std2,alpha=generate_synthetic_data(length,SNR)
    s=time.time();
    cusum_output=run_cusum(h,cont_signal,pdf_type,alpha,std,std2,delta=2,opt=False)
    print(time.time()-s)
    s=time.time();
    cusum_output_opt=run_cusum(h,cont_signal,pdf_type,alpha,std,std2,delta=2,opt=True)
    print(time.time()-s)
    total_MSE,sum_squared_error_list,per_sample_clasification_score,per_state_clasification=compute_results(true_signal,d_list,states,cont_signal,cusum_output)
    #This was to show that the output for the opt is the same, but is much more faster
    plt.figure()
    plt.plot(true_signal,label="True signal")
    plt.plot(cont_signal,label="Contaminated signal",alpha=0.2)
    plt.plot(cusum_output,label="CUSUM out")
    plt.plot(cusum_output_opt,label="CUSUM out opt",linestyle='--')
    plt.legend()
    plt.show()
    
def test_loop_h_output_synth_data():
    SNR=1;
    hs=np.linspace(2,40,num=10)
    output_results,pdf_types=loop_config_through_hs(hs,SNR,length=int(10e3))
    plot_loop_config_through_hs_output(hs,SNR,pdf_types,output_results,title=None)
    
def test_synth_noise():
    length=int(200e3);# would be 1 second long
    h=50;pdf_type="PWG"
    SNRs=[0.5,1,2,5,10];
    #np.random.seed(1)
    alpha=0.75;
    for SNR in SNRs:
        t,true_signal,cont_signal,d_list,states,std,std2,alpha=generate_synthetic_data(length,SNR,alpha=alpha)
        n,bins_plot,_=plt.hist(true_signal-cont_signal,bins=300,density=1,alpha=0.5)
        X=np.linspace(np.min(bins_plot),np.max(bins_plot),num=1000)
        bi_g_pdf=[];
        for xi in X:
            bi_g_pdf.append(biguassian_pdf(xi,0,std,std2,alpha))
        plt.plot(X,bi_g_pdf)
        plt.gca().set_yscale("log")
        plt.title("SNR: "+  "{:.3g}".format(SNR))
        plt.show();
        
def test_curves_v2():
    target_ARL0_range=[50,1000]
    SNR=1;alpha=0.9;std2_rel=6;
    #a,b,c=estimate_h_range_for_all_types_v2(SNR,alpha,std2_rel,target_ARL0_range)
    ARL0_mean_est,ARL0_mean_std_est,ARL1_mean_est,ARL1_mean_std_est,ARL0_comp_time,ARL1_comp_time=analyze_configuration_v2(10,SNR,alpha,std2_rel,pdf_type="Gaussian", minN=20)

def test_functions_for_paper():
    hs=np.linspace(2,30,num=3);
    SNR=0.2
    out,pdf_types=loop_config_through_hs_MSE_w_tol(hs,SNR,tol=0.05,length=int(50e3))
    plot_loop_MSE(hs,SNR,pdf_types,out)
    

if __name__ == "__main__":
    #test_pdfs_and_fits();
    #test_log_likelihoods();    
    #test_cusum();
    #test_ARL_estimations();
    #test_h_range_predictions();
    # ARL0_mean_est,ARL0_mean_std_est,ARL1_mean_est,ARL1_mean_std_est,ARL0_comp_time,ARL1_comp_time=analyze_configuration(0,1,6,0.8,1.5,4,tol=0.01,pdf_type="Gaussian",base_len=1e5)
    # ARL0_mean_est,ARL0_mean_std_est,ARL1_mean_est,ARL1_mean_std_est,ARL0_comp_time,ARL1_comp_time=analyze_configuration(0,1,6,0.8,1.5,4,tol=0.01,pdf_type="Bigaussian",base_len=1e5)
    # ARL0_mean_est,ARL0_mean_std_est,ARL1_mean_est,ARL1_mean_std_est,ARL0_comp_time,ARL1_comp_time=analyze_configuration(0,1,6,0.8,1.5,4,tol=0.01,pdf_type="PWG",base_len=1e5)
    #test_synth_data_cusum();
    #test_loop_h_output_synth_data();
    #stest_synth_noise();
    test_curves_v2();
    #test_functions_for_paper();