import os
import signal
import time
import pickle
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt

from LOST.kflib import createDataAR
from LOST.ARlib import dcyCovMat
from LOST.mcmcARspk import mcmcARspk
from LOST.monitor_gibbs import stationary_test
from LOST.commdefs import __COMP_REF__, __NF__
from LOST.ARcfSmplFuncs import ampAngRep, dcmpcff

cython_inv_v = 5  # Determines the Cython version to use
if cython_inv_v == 5:
    from LOST.kfARlibMPmv_ram5 import armdl_FFBS_1itrMP as armdl_sampler

cython_arc = True
if cython_arc:
    from LOST.ARcfSmplNoMCMC_ram import ARcfSmpl as ar_cf_sampler

# Global flag for handling interruptions
interrupted = False

def signal_handler(signal, frame):
    global interrupted
    print("******  INTERRUPT")
    interrupted = True

signal.signal(signal.SIGINT, signal_handler)

class mcmcARp(mcmcARspk):
    """
    Implements MCMC for autoregressive (AR) models with Gibbs sampling.
    Inherits from mcmcARspk and extends functionality for time series inference.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_prior = __COMP_REF__  # Prior reference
        self.ARord = __NF__  # AR order
        self.interrupted = False  # Internal flag for interruption handling
        
    def getComponents(self):
        """Extracts real and imaginary components from Gibbs samples."""
        skpdITER = self.wts.shape[0]
        N = self.smpx.shape[1] - 2
        self.rts = np.zeros((skpdITER, self.TR, N, self.R))
        self.zts = np.zeros((skpdITER, self.TR, N, self.C))
        
        for it in range(skpdITER):
            if np.all(self.allalfas[it * self.BsmpxSkp] != 0):
                b, c = dcmpcff(alfa=self.allalfas[it * self.BsmpxSkp])
                for tr in range(self.TR):
                    for r in range(self.R):
                        self.rts[it, tr, :, r] = b[r] * self.uts[it, tr, r]
                    for z in range(self.C):
                        cf1, cf2 = 2 * c[2 * z].real, 2 * (c[2 * z].real * self.allalfas[it, self.R + 2 * z].real + c[2 * z].imag * self.allalfas[it, self.R + 2 * z].imag)
                        self.zts[it, tr, :, z] = cf1 * self.wts[it, tr, z, 1:N+2] - cf2 * self.wts[it, tr, z, :N+1]
        
    def gibbsSamp(self):
        """Runs Gibbs sampling for MCMC inference of AR model parameters."""
        global interrupted
        self.interrupted = False
        
        for it in range(self.ITERS):
            if interrupted:
                print("Gibbs sampling interrupted at iteration", it)
                break
            
            # Sample latent variables
            self.sample_latent_state(it)
            
            # Sample AR coefficients if applicable
            if not self.noAR:
                self.sample_AR_coefficients(it)
            
            # Convergence test
            if it > self.minITERS and self.check_convergence(it):
                break
        
    def sample_latent_state(self, it):
        """Samples the latent state using Polya-Gamma augmentation."""
        # Compute Polya-Gamma random variables
        # Use devroye method for efficient sampling
        self.ws = self.pg_sampler()  # Placeholder for PG sampler
        
    def sample_AR_coefficients(self, it):
        """Samples AR coefficients using Cython-optimized functions."""
        smpx_contiguous = self.smpx[:, 1:, :]
        self.uts[it], self.wts[it] = ar_cf_sampler(self.N + 1, self.k, self.TR, self.AR2lims, smpx_contiguous, self.q2, self.R, self.Cs, self.Cn, self.F_alfa_rep[:self.R], self.F_alfa_rep[self.R:], self.sig_ph0L, self.sig_ph0H, 0.2 * 0.2)
        self.F_alfa_rep[:self.R], self.F_alfa_rep[self.R:] = self.uts[it], self.wts[it]

    def check_convergence(self, it):
        """Checks if the MCMC chain has converged using stationarity tests."""
        return stationary_test(self.amps[:it+1, 0], self.fs[:it+1, 0], self.mnStds[:it+1], it+1, blocksize=self.mg_blocksize, points=self.mg_points)
    
    def run(self, data_filename, run_dir, trials=None, psth_run=False):
        """
        Main execution function to load data and run Gibbs sampling.
        """
        self.load_data(run_dir, data_filename, trials)
        if not psth_run:
            start_time = time.time()
            self.gibbsSamp()
            print("Total execution time: {:.3f} seconds".format(time.time() - start_time))
