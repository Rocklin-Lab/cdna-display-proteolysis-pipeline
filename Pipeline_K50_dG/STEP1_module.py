#!/usr/bin/env python
# coding: utf-8

# In[1]:


# get library

import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
from jax import random, vmap
import numpyro
numpyro.enable_x64()

from numpyro.diagnostics import hpdi
import numpyro.distributions as dist
from numpyro import handlers
from numpyro.infer import MCMC, NUTS,initialization
import jax.numpy

from scipy import stats

from jax.lib import xla_bridge
print(xla_bridge.get_backend().platform)

gpu_info = get_ipython().getoutput('nvidia-smi')
gpu_info = '\n'.join(gpu_info)
if gpu_info.find('failed') >= 0:
    print('Not connected to a GPU')
else:
    print(gpu_info)
get_ipython().system('XLA_PYTHON_CLIENT_PREALLOCATE=false')


# In[18]:





# In[24]:





# In[1]:


def selection_model(counts1,counts2,counts3,counts4,counts5):
###############################
#
#  Variables shared between all 4 libraries
#
###############################
    # the number of each set of overlapped sequences
    n1 = len(counts1[0,:])
    n2 = len(counts2[0,:])
    n3 = len(counts3[0,:])
    n4 = len(counts4[0,:])
    n5 = len(counts5[0,:])
    
    
    kmax_times_t = 10**0.65
    
    # total_countsX: observed total NGS counts for each condition [# of conditions (48)]
    total_count1 = np.array([int(x) for x in np.sum(counts1, axis=1)])
    total_count2 = np.array([int(x) for x in np.sum(counts2, axis=1)])
    total_count3 = np.array([int(x) for x in np.sum(counts3, axis=1)])
    total_count4 = np.array([int(x) for x in np.sum(counts4, axis=1)])
    total_count5 = np.array([int(x) for x in np.sum(counts5, axis=1)])
    
    # expected_protease_conc1/2: experimental (expected) protease concentrations (μM) [number of conditions (11) except for no-protease condition]
    expected_protease_conc1 = np.array(list([25/3**x for x in range(10,-1,-1)])) # Protease concentrations for the 1st replicate
    expected_protease_conc2 = np.array(list([25/3**x*3**0.5 for x in range(10,-1,-1)])) #Protease concentrations for the 2nd replicate (offset by 3**0.5)
    
    # protease_vX: protease concentration for each library  (μM) [# of conditions (24) including no-protease sample]
    # Protease concentrations were sampled in truncated normal distribution, but no-protease concentration is fixed at 0
    con_ratio_lim = 1.5 
    con_ratio_sigma = 1.5
    protease_v1 = jax.numpy.concatenate([jax.numpy.array([0]), 
                                         numpyro.sample("protease1_1", dist.TruncatedDistribution(dist.Normal(expected_protease_conc1,expected_protease_conc1/con_ratio_sigma),expected_protease_conc1/con_ratio_lim, expected_protease_conc1*con_ratio_lim)),
                                         jax.numpy.array([0]), 
                                         numpyro.sample("protease1_2", dist.TruncatedDistribution(dist.Normal(expected_protease_conc2,expected_protease_conc2/con_ratio_sigma),expected_protease_conc2/con_ratio_lim,expected_protease_conc2*con_ratio_lim))])
    protease_v2 = jax.numpy.concatenate([jax.numpy.array([0]), 
                                         numpyro.sample("protease2_1", dist.TruncatedDistribution(dist.Normal(expected_protease_conc1,expected_protease_conc1/con_ratio_sigma),expected_protease_conc1/con_ratio_lim, expected_protease_conc1*con_ratio_lim)),
                                         jax.numpy.array([0]), 
                                         numpyro.sample("protease2_2", dist.TruncatedDistribution(dist.Normal(expected_protease_conc2,expected_protease_conc2/con_ratio_sigma),expected_protease_conc2/con_ratio_lim, expected_protease_conc2*con_ratio_lim))])
    protease_v3 = jax.numpy.concatenate([jax.numpy.array([0]), 
                                         numpyro.sample("protease3_1", dist.TruncatedDistribution(dist.Normal(expected_protease_conc1,expected_protease_conc1/con_ratio_sigma),expected_protease_conc1/con_ratio_lim, expected_protease_conc1*con_ratio_lim)),
                                         jax.numpy.array([0]), 
                                         numpyro.sample("protease3_2", dist.TruncatedDistribution(dist.Normal(expected_protease_conc2,expected_protease_conc2/con_ratio_sigma),expected_protease_conc2/con_ratio_lim, expected_protease_conc2*con_ratio_lim))])
    
    # Protease concentration for lib4 is  fixed at experimental concentration
    protease_v4 = jax.numpy.concatenate([jax.numpy.array([0]), 
                                         expected_protease_conc1,
                                         jax.numpy.array([0]),
                                         expected_protease_conc2])

    protease_v1 = numpyro.deterministic('protease_v1', protease_v1)
    protease_v2 = numpyro.deterministic('protease_v2', protease_v2)
    protease_v3 = numpyro.deterministic('protease_v3', protease_v3)
    protease_v4 = numpyro.deterministic('protease_v4', protease_v4)
    
    
###############################
#
#  Sampling A0, K50, and protease concentrations
#  for sequences that overlap between libraries 1 and 2 (overlap sequence set #1)
#
###############################

    # log10_A0_xy: initial fraction for each sequence in log10 [# of sequences (n1)]
    # x: index of overlapped sequence sets (set #1 in this section)
    # y: index of experiments, 1=1st replicate of 1st library, 2=2nd replicate of 1st library, 3=1st replicate of 2nd library, 4=2nd replicate of 2nd library
    # sampled in truncated normal distribution
    log10_A0_11 = numpyro.sample("log10_A0_11", dist.Normal(np.resize(np.log10(1/n1),n1), 1))
    log10_A0_12 = numpyro.sample("log10_A0_12", dist.Normal(np.resize(np.log10(1/n1),n1), 1))
    log10_A0_13 = numpyro.sample("log10_A0_13", dist.Normal(np.resize(np.log10(1/n1),n1), 1))
    log10_A0_14 = numpyro.sample("log10_A0_14", dist.Normal(np.resize(np.log10(1/n1),n1), 1))
    # A0_1: initial fraction for each sequence [ # of conditions (48),# of sequences (n1)]
    A0_1 = 10**jax.numpy.concatenate([jax.numpy.resize(log10_A0_11,(12,n1)),jax.numpy.resize(log10_A0_12,(12,n1)),jax.numpy.resize(log10_A0_13,(12,n1)),jax.numpy.resize(log10_A0_14,(12,n1))],axis=0)


    # log10_K50_1: log10 K50 values [# of sequences (n1)]
    # sampled in wide normal distribution, shared between lib1 and lib2
    log10_K50_1 = numpyro.sample("log10_K50_1", dist.Normal(np.resize(0,n1), 4) ) 
    
    # protease_1: combined protease concentration for lib1 + lib2 [# of conditions (48)]
    protease_1 = jax.numpy.concatenate([protease_v1,protease_v2])  
    
    # survival_1: relative ratio of each sequence for each condition to initial condition (no protease) [# of sequences (n1), # of conditions (48)]
    # survival = exp(- kmax*t*[protease]/(K50+[protease]))
    survival_1 = jax.numpy.exp(-jax.numpy.outer(kmax_times_t,  protease_1)/((jax.numpy.resize(10.0**log10_K50_1,(48,n1)).T)+jax.numpy.resize( protease_1,(n1,48))))

        
    # nonnorm_fraction_1: relative ratio of each sequence for each condition [ # of conditions (48),# of sequences (n1)]
    # nonnorm_fraction = initial ratio (A0) * survival
    nonnorm_fraction_1 = survival_1.T*A0_1
 
    # fraction_1: normalized ratio of each sequence for each condition [# of conditions (48),# of sequences (n1)]
    # fraction = nonnorm_fraction/sum(nonnorm_fraction)
    fraction_1=nonnorm_fraction_1 / np.reshape(jax.numpy.sum(nonnorm_fraction_1, axis=1), (48, 1))
    
    # obs_counts_1: observed NGS count number [ # of conditions (48), # of sequences (n1)]
    # The observed NGS counts are sampled using multinomial distribution
    obs_counts_1 = numpyro.sample("counts_1", dist.Multinomial(total_count = total_count1,probs=fraction_1),obs=jax.numpy.array(counts1))
    
###############################
#
#  Sampling A0, K50, and protease concentrations
#  for sequences that overlap between libraries 2 and 3 (overlap sequence set #2)
#
###############################
    log10_A0_21 = numpyro.sample("log10_A0_21", dist.Normal(np.resize(np.log10(1/n2),n2), 1))
    log10_A0_22 = numpyro.sample("log10_A0_22", dist.Normal(np.resize(np.log10(1/n2),n2), 1))
    log10_A0_23 = numpyro.sample("log10_A0_23", dist.Normal(np.resize(np.log10(1/n2),n2), 1))
    log10_A0_24 = numpyro.sample("log10_A0_24", dist.Normal(np.resize(np.log10(1/n2),n2), 1))
    A0_2 = 10**jax.numpy.concatenate([jax.numpy.resize(log10_A0_21,(12,n2)),jax.numpy.resize(log10_A0_22,(12,n2)),jax.numpy.resize(log10_A0_23,(12,n2)),jax.numpy.resize(log10_A0_24,(12,n2))],axis=0)
    
    log10_K50_2 = numpyro.sample("log10_K50_2", dist.Normal(np.resize(0,n2), 4) ) 
    protease_2 = jax.numpy.concatenate([protease_v2,protease_v3])    

    survival_2 = jax.numpy.exp(-jax.numpy.outer(kmax_times_t, protease_2)/((jax.numpy.resize(10.0**log10_K50_2,(48,n2)).T)+jax.numpy.resize(protease_2,(n2,48))))

    nonnorm_fraction_2 = survival_2.T * A0_2
    fraction_2 = nonnorm_fraction_2 / np.reshape(jax.numpy.sum(nonnorm_fraction_2, axis=1), (48, 1))
    obs_counts_2 = numpyro.sample("counts_2", dist.Multinomial(total_count = total_count2,probs=fraction_2),obs=jax.numpy.array(counts2))
    
###############################
#
#  Sampling A0, K50, and protease concentrations
#  for sequences that overlap between libraries 1 and 4 (overlap sequence set #3)
#
############################### 
    log10_A0_31 = numpyro.sample("log10_A0_31", dist.Normal(np.resize(np.log10(1/n3),n3), 1))
    log10_A0_32 = numpyro.sample("log10_A0_32", dist.Normal(np.resize(np.log10(1/n3),n3), 1))
    log10_A0_33 = numpyro.sample("log10_A0_33", dist.Normal(np.resize(np.log10(1/n3),n3), 1))
    log10_A0_34 = numpyro.sample("log10_A0_34", dist.Normal(np.resize(np.log10(1/n3),n3), 1))
    A0_3 = 10**jax.numpy.concatenate([jax.numpy.resize(log10_A0_31,(12,n3)),jax.numpy.resize(log10_A0_32,(12,n3)),jax.numpy.resize(log10_A0_33,(12,n3)),jax.numpy.resize(log10_A0_34,(12,n3))],axis=0)
    
    log10_K50_3 = numpyro.sample("log10_K50_3", dist.Normal(np.resize(0,n3), 4) ) 
    protease_3 = jax.numpy.concatenate([protease_v1,protease_v4])    
    survival_3 = jax.numpy.exp(-jax.numpy.outer(kmax_times_t, protease_3)/((jax.numpy.resize(10.0**log10_K50_3,(48,n3)).T)+jax.numpy.resize(protease_3,(n3,48))))
    
    nonnorm_fraction_3 = survival_3.T * A0_3

    fraction_3 = nonnorm_fraction_3 / np.reshape(jax.numpy.sum(nonnorm_fraction_3, axis=1), (48, 1))
    obs_counts_3 = numpyro.sample("counts_3", dist.Multinomial(total_count = total_count3,probs=fraction_3),obs=jax.numpy.array(counts3))


###############################
#
#  Sampling A0, K50, and protease concentrations
#  for sequences that overlap between libraries 2 and 4 (overlap sequence set #4)
#
############################### 
    log10_A0_41 = numpyro.sample("log10_A0_41", dist.Normal(np.resize(np.log10(1/n4),n4), 1))
    log10_A0_42 = numpyro.sample("log10_A0_42", dist.Normal(np.resize(np.log10(1/n4),n4), 1))
    log10_A0_43 = numpyro.sample("log10_A0_43", dist.Normal(np.resize(np.log10(1/n4),n4), 1))
    log10_A0_44 = numpyro.sample("log10_A0_44", dist.Normal(np.resize(np.log10(1/n4),n4), 1))
    A0_4 = 10**jax.numpy.concatenate([jax.numpy.resize(log10_A0_41,(12,n4)),jax.numpy.resize(log10_A0_42,(12,n4)),jax.numpy.resize(log10_A0_43,(12,n4)),jax.numpy.resize(log10_A0_44,(12,n4))],axis=0)
    
    log10_K50_4 = numpyro.sample("log10_K50_4", dist.Normal(np.resize(0,n4), 4) ) 
    protease_4 = jax.numpy.concatenate([protease_v2,protease_v4])    
    survival_4 = jax.numpy.exp(-jax.numpy.outer(kmax_times_t, protease_4)/((jax.numpy.resize(10.0**log10_K50_4,(48,n4)).T)+jax.numpy.resize(protease_4,(n4,48))))
    
    nonnorm_fraction_4 = survival_4.T * A0_4
    
    fraction_4=nonnorm_fraction_4 / np.reshape(jax.numpy.sum(nonnorm_fraction_4, axis=1), (48, 1))
    obs_counts_4 = numpyro.sample("counts_4", dist.Multinomial(total_count = total_count4,probs=fraction_4),obs=jax.numpy.array(counts4))


###############################
#
#  Sampling A0, K50, and protease concentrations
#  for sequences that overlap between libraries 3 and 4 (overlap sequence set #5)
#
############################### 
    log10_A0_51 = numpyro.sample("log10_A0_51", dist.Normal(np.resize(np.log10(1/n5),n5), 1))
    log10_A0_52 = numpyro.sample("log10_A0_52", dist.Normal(np.resize(np.log10(1/n5),n5), 1))
    log10_A0_53 = numpyro.sample("log10_A0_53", dist.Normal(np.resize(np.log10(1/n5),n5), 1))
    log10_A0_54 = numpyro.sample("log10_A0_54", dist.Normal(np.resize(np.log10(1/n5),n5), 1))
    A0_5 = 10**jax.numpy.concatenate([jax.numpy.resize(log10_A0_51,(12,n5)),jax.numpy.resize(log10_A0_52,(12,n5)),jax.numpy.resize(log10_A0_53,(12,n5)),jax.numpy.resize(log10_A0_54,(12,n5))],axis=0)
    
    log10_K50_5 = numpyro.sample("log10_K50_5", dist.Normal(np.resize(0,n5), 4) ) 
    protease_5 = jax.numpy.concatenate([protease_v3,protease_v4])    
    survival_5 = jax.numpy.exp(-jax.numpy.outer(kmax_times_t, protease_5)/((jax.numpy.resize(10.0**log10_K50_5,(48,n5)).T)+jax.numpy.resize(protease_5,(n5,48))))
    
    nonnorm_fraction_5 = survival_5.T * A0_5
    fraction_5=nonnorm_fraction_5 / np.reshape(jax.numpy.sum(nonnorm_fraction_5, axis=1), (48, 1))
    obs_counts_5 = numpyro.sample("counts_5", dist.Multinomial(total_count = total_count5,probs=fraction_5),obs=jax.numpy.array(counts5))


# In[ ]:



def protease_calibration(counts1,counts2,counts3,counts4,counts5):
    # run the model

    rng_key = random.PRNGKey(1)
    rng_key, rng_key_ = random.split(rng_key)

    kernel = NUTS(selection_model)
    mcmc = MCMC(kernel, num_warmup=200, num_samples=50, num_chains=1)
    mcmc.run(rng_key_,counts1=counts1,counts2=counts2,counts3=counts3,counts4=counts4,counts5=counts5)
    samples=mcmc.get_samples()
    
    return samples

