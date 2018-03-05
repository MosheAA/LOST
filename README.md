##  Latent Oscillation in Spike Train (LOST)
#  (Arai and Kass, PLoS computational biology 2017).  
The LOST code is in the file mcmcARp_ram.py.  As explained in the paper, LOST automatically chooses TAE and spike history knot locations.  As such, it assumes a particular directory structure.

Packages
PyPG
poly-gamma sampler
arai_utils
PP-AR



Directory structure:
RESDIR=$(basedir)/Results/
PYDIR=$(basedir)/pyscripts/

basedir is an environment variable.

In $RESDIR, there is a generator of a run script.  

cpRunTemplate <wp or np> <tr#1> <tr#2> <# C> <# R> 

creates a file called
wp_tr1-tr2_#C_#R.py 

Edit this file to change # of Gibbs iterations, length of each trial, how often to plot intermediate results etc.

~$ cd $(RESDIR)
$(RESDIR)$ python wp_tr1-tr2_#C_#R.py 

This creates directory

$(RESDIR)/wp_tr1-tr2_#C_#R



Data format:
spike train in file named "xprbsdN.dat"
File format is 3 * (total trials) columns x N time points per trial



Output:
$(RESDIR)/wp_tr1-tr2_#C_#R/smpls-N.dump

