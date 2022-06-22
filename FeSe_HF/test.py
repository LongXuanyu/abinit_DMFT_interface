from mpi4py import MPI
import netCDF4
from netCDF4 import Dataset
import numpy as np
import os
from scipy import interpolate
import scipy.linalg as LA 
from triqs.gf import *
import sys
sys.path.append(os.getcwd())
from tools import *

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nc_obj=Dataset('test.nc', 'a')
iatom = np.array(nc_obj.variables['iatom'])[0]
comm.Barrier()
if iatom == 1:
    omega = np.array(nc_obj.variables['omega'])
    beta = float(np.array(nc_obj.variables['beta'])[0])
    n_orb = np.array(nc_obj.variables['norb'])[0]
    n_iw = 2000
    freqs = np.array([(2*i+1) * np.pi / beta for i in range(n_iw)])
    bname = ['up', 'down']
    gf_struct = dict([(spn, range(n_orb)) for spn in bname])
    G0_iw = make_block_gf(GfImFreq, gf_struct, beta, n_iw)
    Gloc_iw = make_block_gf(GfImFreq, gf_struct, beta, n_iw)
    Sigma_iw = make_block_gf(GfImFreq, gf_struct, beta, n_iw)
    Gimp_iw = make_block_gf(GfImFreq, gf_struct, beta, n_iw)
    
    u_mat = np.array(nc_obj.variables['u_mat']).transpose()

    g0r = np.array(nc_obj.variables['real_g0_iw']).transpose()
    g0i = np.array(nc_obj.variables['imag_g0_iw']).transpose()
    g0 = g0r + 1j * g0i
    g0_fun = interpolate.interp1d(omega, g0, kind = 'cubic', fill_value='extrapolate')
    g0_tmp = [g0_fun(freqs)]*2
    assign_from_numpy_array(G0_iw, g0_tmp, bname)
    
    gr = np.array(nc_obj.variables['real_g_iw']).transpose()
    gi = np.array(nc_obj.variables['imag_g_iw']).transpose()
    g = gr + 1j * gi
    g_fun = interpolate.interp1d(omega, g, kind = 'cubic', fill_value = 'extrapolate')
    g_tmp = [g_fun(freqs)]*2
    assign_from_numpy_array(Gloc_iw, g_tmp, bname)

    comm.Barrier()

    # Hartree-Fock impurity solver
    # static solver parameters
    mode = 'simple'
    tol = 1e-4
    maxstep = 40
    if mode == 'simple':
        maxstep = 1
    amix = 0.5

    Gimp_iw << Gloc_iw
    Sigma = np.zeros((2, n_orb, n_orb, n_iw), dtype='complex')
    HFPot = np.zeros((2, n_orb, n_orb), dtype='complex')

    n_iter = 0
    while n_iter < maxstep:
        n_iter += 1
        dens_mat = Gimp_iw.density()
        dens_mat_tot = sum(dens_mat.values())  # spin-sum of density matrix
        HFPot_new = np.zeros((2, n_orb, n_orb), dtype='complex')
        for i,sp1 in enumerate(bname):
            HFPot_new[i] += np.einsum("ijkl, jl->ik", u_mat, dens_mat_tot, optimize=True)  # Hartree
            HFPot_new[i] -= np.einsum("ijkl, jk->il", u_mat, dens_mat[sp1], optimize=True)  # Fock
            if n_iter == 1:
                HFPot[i] = HFPot_new[i]
            else:
                HFPot[i] = amix * HFPot_new[i] + (1 - amix) * HFPot[i]
            Sigma[i] = np.dstack([HFPot[i]] * n_iw)
        res = np.linalg.norm(HFPot_new-HFPot)
        assign_from_numpy_array(Sigma_iw, Sigma, bname)
        Gimp_iw << dyson(G0_iw=G0_iw, Sigma_iw=Sigma_iw)
        if n_iter > 5 and res < tol: break
    if rank == 0:
        print('-------------------static solver---------------------')
        print('number of iterations: %d' % n_iter)
        print('residule: %f' % res)
        print('Hartree-Fock potential:')
        for i,sp1 in enumerate(bname):
            print("\n    Spin ", sp1)
            for i1 in range(n_orb):
                print("          ", end="")
                for i2 in range(n_orb):
                    print("{0:.3f} ".format(HFPot[i][i1, i2]), end="")
                print("")
        print('-----------------------------------------------------')

    # Extrapolate the impurity GF to the log mesh
    Gimp = get_data(Gimp_iw)
    gimp_fun = interpolate.interp1d(freqs, Gimp[0], kind = 'cubic', bounds_error=False, fill_value=0)
    gimp = gimp_fun(omega)
    bound = 0
    while(omega[bound] <= freqs[-1]):
        bound += 1
    known_mom = make_zero_tail(Gimp_iw, 2)
    for i in range(2):
        known_mom[i][0] = 0
        known_mom[i][1] = np.identity(n_orb)
    moments = fit_hermitian_tail(Gimp_iw, known_mom)[0][0]
    for n, moment in enumerate(moments):
        gimp[:,:,bound:] += moment[:,:,np.newaxis] / (1j * omega[bound:]) ** n
    assert abs(freqs[0]-omega[0]) < 1e-7
    gimp[:,:,0] = Gimp[0][:,:,0]

    # the G0 is kept unchanged
    weiss = g0

    # Save GF and self-energy data
    if rank == 0:
        np.save('gimp.npy', gimp)
        np.save('weiss.npy', weiss)

else:  
    # iatom > 1, construct the new GF by symmetrization 
    symm_U = np.diag([1,-1,1,-1,1])
    if os.path.exists('gimp.npy'):
        gimp_tmp = np.load('gimp.npy')
        gimp = np.einsum('ab, bcw, cd->adw', symm_U.transpose().conjugate(), gimp_tmp, symm_U)
    else:
        print("error: impurity GF data do not exist!")
        exit(1)
    if os.path.exists('weiss.npy'):
        weiss_tmp = np.load('weiss.npy')
        weiss = np.einsum('ab, bcw, cd->adw', symm_U.transpose().conjugate(), weiss_tmp, symm_U)
    else:
        print("error: non-interacting impurity GF data do not exist!")
        exit(1)

# Write GF to the .nc file
if rank == 0:
    real_gimp_iw = nc_obj.createVariable('real_gimp_iw', 'f8', ('nw', 'norb', 'norb',))
    imag_gimp_iw = nc_obj.createVariable('imag_gimp_iw', 'f8', ('nw', 'norb', 'norb',))
    real_gimp_iw[:,:,:] = gimp.real.transpose()
    imag_gimp_iw[:,:,:] = gimp.imag.transpose()
    print("Gimp written!")
    real_g0imp_iw = nc_obj.createVariable('real_g0imp_iw', 'f8', ('nw', 'norb', 'norb',))
    imag_g0imp_iw = nc_obj.createVariable('imag_g0imp_iw', 'f8', ('nw', 'norb', 'norb',))
    real_g0imp_iw[:,:,:] = weiss.real.transpose()
    imag_g0imp_iw[:,:,:] = weiss.imag.transpose()
    print("Weiss written!")

comm.Barrier()
nc_obj.close()
