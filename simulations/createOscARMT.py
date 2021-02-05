import scipy.optimize as _sco
from shutil import copyfile
from LOST.utildirs import setFN
from LOST.mcmcARpPlot import plotWFandSpks
import matplotlib.pyplot as _plt

from LOST.kflib import createDataPPl2, savesetMT, savesetMTnosc, createDataAR, createFlucOsc
from LOST.LOSTdirs import resFN, datFN
import numpy as _N
import pickle as _pk
import warnings
import numpy.polynomial.polynomial as _Npp
import utilities as _U

TR         = None;     N        = None;      dt       = 0.001
trim       = 50;
nzs        = None;     nRhythms = None;
rs         = None;     ths      = None;      alfa     = None;
lambda2    = None;     psth     = None
lowQpc     = 0;        lowQs    = None
isis       = None;     rpsth    = None
us         = None
csTR       = None;     #  coupling strength trend
etme       = None;
absrefr    = 0
prbs       = None
latst      = None
prbsWithHist= None

bGenOscUsingAR = True;  bGenOscUsingSines = False;
#   These params if osc. not generated by AR but by Sines
f0VAR      = None;     f0         = None;     Bf         = None;     Ba       = None;      amp      = 1;   amp_nz  = 0;
dSA        = 5;     dSF        = 5;

def create(outdir):
    # _plt.ioff()
    global dt, lambda2, rpsth, isis, us, csTR, etme, bGenOscUsingAR, f0VAR, f0, Bf, Ba, amp, amp_nz, dSA, dSF, psth, prbs, prbsWithHist, latst
    latst = _N.empty((TR, N))
    prbsWithHist = _N.empty((TR, N))
    lowQs = []

    if csTR is None:
        csTR = _N.ones(TR)

    if bGenOscUsingAR:
        ARcoeff = _N.empty((nRhythms, 2))
        for n in range(nRhythms):
            ARcoeff[n]          = (-1*_Npp.polyfromroots(alfa[n])[::-1][1:]).real
        stNzs   = _N.empty((TR, nRhythms))
        for tr in range(TR):
            if _N.random.rand() < lowQpc:
                lowQs.append(tr)
                stNzs[tr] = nzs[:, 0]
            else:
                stNzs[tr] = nzs[:, 1]    #  high
    elif bGenOscUsingSines:
        if f0VAR is None:
            f0VAR   = _N.zeros(TR)
        sig = 0.1
        x, y = createDataAR(100000, Bf, sig, sig)
        stdf  = _N.std(x)   #  choice of 4 std devs to keep phase monotonically increasing
        x, y = createDataAR(100000, Ba, sig, sig)
        stda  = _N.std(x)   #  choice of 4 std devs to keep phase monotonically increasing
        stNzs   = _N.empty((TR, 1))
        for tr in range(TR):
            print("!!!!  %.2f" % lowQpc)
            if _N.random.rand() < lowQpc:
                lowQs.append(tr)
                csTR[tr] = 0
            else:
                csTR[tr] = 1





    #  x, prbs, spks    3 columns
    nColumns = 3
    spkdat  = _N.empty((N, TR))
    gtdat  = _N.empty((N, TR*2))
    probNOsc  = _N.empty((N, TR))
    spksPT  = _N.empty(TR)

    isis   = []
    rpsth  = []

    if etme is None:
        etme = _N.ones((TR, N))
    if us is None:
        us = _N.zeros(TR)
    elif (type(us) is float) or (type(us) is int):
        us = _N.zeros(TR) * us
    for tr in range(TR):
        if bGenOscUsingAR:
            #x, dN, prbs, fs, prbsNOsc = createDataPPl2(TR, N, dt, ARcoeff, psth + us[tr], stNzs[tr], lambda2=lambda2, p=1, nRhythms=nRhythms, cs=csTR[tr], etme=etme[tr])
            #  psth is None.  Turn it off for now
            x, dN, prbs, fs, prbsNOsc = createDataPPl2(TR, N, dt, ARcoeff, us[tr], stNzs[tr], lambda2=lambda2, p=1, nRhythms=nRhythms, cs=csTR[tr], etme=etme[tr], offset=psth)
        else:
            print("here 00000")
            xosc = createFlucOsc(f0, _N.array([f0VAR[tr]]), N, dt, 1, Bf=Bf, Ba=Ba, amp=amp, amp_nz=amp_nz, stdf=stdf, stda=stda, sig=sig, smoothKer=5, dSA=dSA, dSF=dSF) * etme[tr]  # sig is arbitrary, but we need to keep it same as when stdf, stda measured
            #x, dN, prbs, fs, prbsNOsc = createDataPPl2(TR, N, dt, None, psth + us[tr], None, lambda2=lambda2, p=1, nRhythms=1, cs=csTR[tr], etme=etme[tr], x=xosc[0])
            x, dN, latst[tr], prbsWithHist[tr], fs, prbsNOsc = createDataPPl2(TR, N, dt, None, us[tr], stNzs[tr], lambda2=lambda2, p=1, nRhythms=1, cs=csTR[tr], etme=etme[tr], x=xosc, offset=psth)

        spksPT[tr] = _N.sum(dN)
        rpsth.extend(_N.where(dN == 1)[0])
        gtdat[:, 2*tr] = latst[tr]#_N.sum(x, axis=0).T*etme[tr]*csTR[tr]
        gtdat[:, 2*tr+1] = prbsWithHist[tr]
        spkdat[:, tr] = dN
        probNOsc[:, tr] = prbsNOsc
        isis.extend(_U.toISI([_N.where(dN == 1)[0].tolist()])[0])

    savesetMT(TR, spkdat, gtdat, model, lambda2, psth, outdir)
    #savesetMTnosc(TR, probNOsc, setname)

    arfs = ""
    xlst = []

    if bGenOscUsingAR:
        for nr in range(nRhythms):
            arfs += "%.1fHz " % (500*ths[nr]/_N.pi)
            xlst.append(x[nr])
    else:
        xlst.append(x[0])
    sTitle = "AR2 freq %(fs)s    spk Hz %(spkf).1fHz   TR=%(tr)d   N=%(N)d" % {"spkf" : (_N.sum(spksPT) / (N*TR*0.001)), "tr" : TR, "N" : N, "fs" : arfs}

    #plotWFandSpks(N-1, dN, xlst, sTitle=sTitle, sFilename=resFN("generative", dir=setname))


    """
    fig = _plt.figure(figsize=(8, 4))
    _plt.hist(isis, bins=range(100), color="black")
    _plt.grid()
    _plt.savefig(resFN("ISIhist", dir=setname))
    #_plt.close()


    fig = _plt.figure(figsize=(13, 4))
    _plt.plot(spksPT, marker=".", color="black", ms=8)
    _plt.ylim(0, max(spksPT)*1.1)
    _plt.grid()
    _plt.suptitle("avg. Hz %.1f" % (_N.mean(spksPT) / (N*0.001)))
    _plt.savefig(resFN("spksPT", dir=setname))
    _plt.close()

    if (lambda2 is None) and (absrefr > 0):
        lambda2 = _N.array([0.0001] * absrefr)
    if lambda2 is not None:
        _N.savetxt(resFN("lambda2.dat", dir=setname), lambda2, fmt="%.7f")
    """
    #  if we want to double bin size
    #lambda2db = 0.5*(lambda2[1::2] + lambda2[::2])
    #_N.savetxt(resFN("lambda2db.dat", dir=setname), lambda2db, fmt="%.7f")
    #_plt.ion()

    print(lowQs)
    if lowQpc > 0:
        _N.savetxt("%(od)s/lowQtrials" % {"od" : outdir}, lowQs, fmt="%d")

