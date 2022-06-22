import numpy as np

n_w = 1000
wini = -0.2
wfin = 0.2
n_orb = 3
Sigma_w = np.load('ED/Sigma_w.npy')
Sigma_w = np.ascontiguousarray(Sigma_w)
freq_w = np.linspace(wini, wfin ,n_w)
data = np.zeros((n_w, 2*n_orb*n_orb+1), dtype='double')
data[:,0] = freq_w
count = 0
for i in range(n_orb):
    for j in range(n_orb):
        data[:, 2*count+1:2*count+3] = Sigma_w[0][0][i][j].view(float).reshape(-1,2)[:,:]
        count += 1
filename = 'SVOi_DS3Self_ra-omega_iatom0001_isppol1'
np.savetxt(filename, data)
