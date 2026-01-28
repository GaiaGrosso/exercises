import glob, json, h5py, math, time, os, torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from scipy.stats import norm, expon, chi2, uniform, chisquare
    
def plot_loss_curves(epoch_losses):
    # -----------------------------
    # Plot 1: Training loss curves
    # -----------------------------
    fig = plt.figure(figsize=(10, 10)) 
    fig.patch.set_facecolor('white')  
    losses = [l.item().squeeze() if torch.is_tensor(l) else l for l in epoch_losses]
    losses = [l.reshape((-1,)) for l in losses]
    plt.plot(losses)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.close()
    return

def plot_training_data(data, weight_data, ref, weight_ref, feature_labels, bins_code, xlabel_code, ymax_code={},
                       save=False, save_path='', file_name=''):
    '''
    Plot distributions of the input variables for the training samples.
    
    data:            (numpy array, shape (None, n_dimensions)) data training sample (label=1)
    weight_data:     (numpy array, shape (None,)) weights of the data sample (default ones)
    ref:             (numpy array, shape (None, n_dimensions)) reference training sample (label=0)
    weight_ref:      (numpy array, shape (None,)) weights of the reference sample
    feature_labels:  (list of string) list of names of the training variables
    bins_code:       (dict) dictionary of bins edge for each training variable (bins_code.keys()=feature_labels)
    xlabel_code:     (dict) dictionary of xlabel for each training variable (xlabel.keys()=feature_labels)
    ymax_code:       (dict) dictionary of maximum value for the y axis in the ratio panel for each training variable (ymax_code.keys()=feature_labels)
    '''
    plt_i = 0
    for key in feature_labels:
        bins = bins_code[key]
        plt.rcParams["font.family"] = "serif"
        plt.style.use('classic')
        fig = plt.figure(figsize=(10, 10)) 
        fig.patch.set_facecolor('white')  
        ax1= fig.add_axes([0.1, 0.43, 0.8, 0.5])        
        hD = plt.hist(data[:, plt_i],weights=weight_data, bins=bins, label='DATA', color='black', lw=1.5, histtype='step', zorder=4)
        hR = plt.hist(ref[:, plt_i], weights=weight_ref, color='#a6cee3', ec='#1f78b4', bins=bins, lw=1, label='REFERENCE')
        plt.errorbar(0.5*(bins[1:]+bins[:-1]), hD[0], yerr= np.sqrt(hD[0]), color='black', ls='', marker='o', ms=5, zorder=3)
        font = font_manager.FontProperties(family='serif', size=16)
        l    = plt.legend(fontsize=18, prop=font, ncol=2)
        font = font_manager.FontProperties(family='serif', size=18) 
        plt.tick_params(axis='x', which='both',    labelbottom=False)
        plt.yticks(fontsize=16, fontname='serif')
        plt.xlim(bins[0], bins[-1])
        plt.ylabel("events", fontsize=22, fontname='serif')
        plt.yscale('log')
        ax2 = fig.add_axes([0.1, 0.1, 0.8, 0.3]) 
        x   = 0.5*(bins[1:]+bins[:-1])
        plt.errorbar(x, hD[0]/hR[0], yerr=np.sqrt(hD[0])/hR[0], ls='', marker='o', label ='DATA/REF', color='black')
        font = font_manager.FontProperties(family='serif', size=16)
        plt.legend(fontsize=18, prop=font)
        plt.xlabel(xlabel_code[key], fontsize=22, fontname='serif')
        plt.ylabel("ratio", fontsize=22, fontname='serif')
        if key in list(ymax_code.keys()):
            plt.ylim(0., ymax_code[key])
        plt.yticks(fontsize=16, fontname='serif')
        plt.xticks(fontsize=16, fontname='serif')
        plt.xlim(bins[0], bins[-1])
        plt.grid()
        if save:
            if save_path=='': print('argument save_path is not defined. The figure will not be saved.')
            else:
                if file_name=='': file_name = 'InputVariable_%s'%(key)
                else: file_name += '_InputVariable_%s'%(key)
                fig.savefig(save_path+file_name+'.pdf')
        plt.show()
        plt.close()
        plt_i+=1
    return

def plot_reconstruction(df, data, weight_data, ref, weight_ref, tau_OBS, output_tau_ref,  
                        feature_labels, bins_code, xlabel_code, ymax_code={}, delta_OBS=None, output_delta_ref=None,
                        save=False, save_path='', file_name=''):
    '''
    Reconstruction of the data distribution learnt by the model.
    
    df:              (int) chi2 degrees of freedom
    data:            (numpy array, shape (None, n_dimensions)) data training sample (label=1)
    weight_data:     (numpy array, shape (None,)) weights of the data sample (default ones)
    ref:             (numpy array, shape (None, n_dimensions)) reference training sample (label=0)
    weight_ref:      (numpy array, shape (None,)) weights of the reference sample
    tau_OBS:         (float) value of the tau term after training
    output_tau_ref:  (numpy array, shape (None, 1)) tau prediction of the reference training sample after training
    feature_labels:  (list of string) list of names of the training variables
    bins_code:       (dict) dictionary of bins edge for each training variable (bins_code.keys()=feature_labels)
    xlabel_code:     (dict) dictionary of xlabel for each training variable (xlabel.keys()=feature_labels)
    ymax_code:       (dict) dictionary of maximum value for the y axis in the ratio panel for each training variable (ymax_code.keys()=feature_labels)
    delta_OBS:       (float) value of the delta term after training (if not given, only tau reconstruction is plotted)
    output_delta_ref:(numpy array, shape (None, 1)) delta prediction of the reference training sample after training (if not given, only tau reconstruction is plotted)
    '''
    Zscore=None
    if delta_OBS==None:
        if df!=None:
            Zscore=norm.ppf(chi2.cdf(tau_OBS, df))
    else:
        if df!=None:
            Zscore=norm.ppf(chi2.cdf(tau_OBS-delta_OBS, df))
    plt_i = 0
    for key in feature_labels:
        bins = bins_code[key]
        plt.rcParams["font.family"] = "serif"
        plt.style.use('classic')
        fig = plt.figure(figsize=(10, 10)) 
        fig.patch.set_facecolor('white')  
        ax1= fig.add_axes([0.1, 0.43, 0.8, 0.5])        
        hD = plt.hist(data[:, plt_i],weights=weight_data, bins=bins, label='DATA', color='black', lw=1.5, histtype='step', zorder=2)
        hR = plt.hist(ref[:, plt_i], weights=weight_ref, color='#a6cee3', ec='#1f78b4', bins=bins, lw=1, label='REFERENCE', zorder=1)
        hN = plt.hist(ref[:, plt_i], weights=np.exp(output_tau_ref[:, 0])*weight_ref, histtype='step', bins=bins, lw=0)
        if not delta_OBS==None:
            hN2= plt.hist(ref[:, plt_i], weights=np.exp(output_delta_ref[:, 0])*weight_ref, histtype='step', bins=bins, lw=0)
        plt.errorbar(0.5*(bins[1:]+bins[:-1]), hD[0], yerr= np.sqrt(hD[0]), color='black', ls='', marker='o', ms=5, zorder=3)
        plt.scatter(0.5*(bins[1:]+bins[:-1]),  hN[0], edgecolor='black', label=r'$\tau$ RECO', color='#b2df8a', lw=1, s=30, zorder=4)
        if not delta_OBS==None:
            plt.scatter(0.5*(bins[1:]+bins[:-1]), hN2[0], edgecolor='black', label=r'$\Delta$ RECO', color='#33a02c', lw=1, s=30, zorder=4)
        font = font_manager.FontProperties(family='serif', size=16)
        l    = plt.legend(fontsize=18, prop=font, ncol=2)
        font = font_manager.FontProperties(family='serif', size=18) 
        title  = r'$\tau(D,\,A)$='+str(np.around(tau_OBS, 2))
        if not delta_OBS==None:
            title += r', $\Delta(D,\,A)$='+str(np.around(delta_OBS, 2))
        if Zscore !=None:
            title += ', Z-score='+str(np.around(Zscore, 2))
        l.set_title(title=title, prop=font)
        plt.tick_params(axis='x', which='both',    labelbottom=False)
        plt.yticks(fontsize=16, fontname='serif')
        plt.xlim(bins[0], bins[-1])
        plt.ylabel("events", fontsize=22, fontname='serif')
        plt.yscale('log')
        ax2 = fig.add_axes([0.1, 0.1, 0.8, 0.3]) 
        x   = 0.5*(bins[1:]+bins[:-1])
        plt.errorbar(x, hD[0]/hR[0], yerr=np.sqrt(hD[0])/hR[0], ls='', marker='o', label ='DATA/REF', color='black')
        plt.plot(x, hN[0]/hR[0], label =r'$\tau$ RECO/REF', color='#b2df8a', lw=3)
        if not delta_OBS==None:
            plt.plot(x, hN2[0]/hR[0], ls='--', label =r'$\Delta$ RECO/REF', color='#33a02c', lw=3)
        font = font_manager.FontProperties(family='serif', size=16)
        plt.legend(fontsize=18, prop=font)
        plt.xlabel(xlabel_code[key], fontsize=22, fontname='serif')
        plt.ylabel("ratio", fontsize=22, fontname='serif')
        if key in list(ymax_code.keys()):
            plt.ylim(0., ymax_code[key])
        plt.yticks(fontsize=16, fontname='serif')
        plt.xticks(fontsize=16, fontname='serif')
        plt.xlim(bins[0], bins[-1])
        plt.grid()
        if save:
            if save_path=='': print('argument save_path is not defined. The figure will not be saved.')
            else:
                if file_name=='': file_name = 'Reconstruction'
                else: file_name += '_Reconstruction'
                fig.savefig(save_path+file_name+'.pdf')
        plt.show()
        plt.close()
        plt_i+=1
    return