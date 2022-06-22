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
    n_iw = 1000
    freqs = np.array([(2*i+1) * np.pi / beta for i in range(n_iw)])
    bname = ['up', 'down']
    gf_struct = dict([(spn, range(n_orb)) for spn in bname])
    Gimp_iw = make_block_gf(GfImFreq, gf_struct, beta, n_iw)
    G0_iw = make_block_gf(GfImFreq, gf_struct, beta, n_iw)
    Delta_iw = make_block_gf(GfImFreq, gf_struct, beta, n_iw)
    Sigma_iw = make_block_gf(GfImFreq, gf_struct, beta, n_iw)
    
    u_mat = np.array(nc_obj.variables['u_mat']).transpose()
    
    g0r = np.array(nc_obj.variables['real_g0_iw']).transpose()
    g0i = np.array(nc_obj.variables['imag_g0_iw']).transpose()
    g0 = g0r + 1j * g0i
    g0_fun = interpolate.interp1d(omega, g0, kind = 'cubic', fill_value = 'extrapolate')
    g0_tmp = [g0_fun(freqs)]*2
    assign_from_numpy_array(G0_iw, g0_tmp, bname)

    comm.Barrier()

    params_kw = {
        'n_bath'         : 6,
        'n_spin'         : 1,
        'bath_type'      : 'normal',
        'finite_T'       : 'T',
        'n_iter'         : 512,
        'lanc_tol'       : 1e-7,  
        'nstates_sector' : 20,
        'nstates_total'  : 300,
        'gs_threshold'   : 1e-5,
        'cutoff'         : 1e-5,
        'chiexct'        : 'F',
        'eps'            : 0.001,
    }

    inputED = """\
    NORB = {0}                                      !Number of impurity orbitals (max 5).
    NBATH = {1}                                     !Number of bath sites:(normal=>Nbath per orb)(hybrid=>Nbath total)(replica=>Nbath=Nreplica)
    ULOC = {2}                                      !Values of the local interaction per orbital (max 5)
    UST = {3}                                       !Value of the inter-orbital interaction term
    JH = {4}                                        !Hunds coupling
    JX = {4}                                        !S-E coupling
    JP = {4}                                        !P-H coupling
    BETA = {5}                                      !Inverse temperature, at T=0 is used as a IR cut-off.
    LMATS = {6}                                     !Number of Matsubara frequencies.
    ED_FINITE_TEMP = {7}                            !flag to select finite temperature method. note that if T then lanc_nstates_total must be > 1
    NSPIN = {8}                                     !Number of spin degeneracy (max 2)
    ED_TWIN = {9}                                   !flag to reduce (T) or not (F,default) the number of visited sector using twin symmetry.
    LANC_NITER = {10}                               !Number of Lanczos iteration in spectrum determination.
    LANC_TOLERANCE = {11}                           !Tolerance for the Lanczos iterations as used in Arpack and plain lanczos.
    LANC_NSTATES_SECTOR = {12}                      !Initial number of states per sector to be determined.
    LANC_NSTATES_TOTAL = {13}                       !Initial number of total states to be determined.
    GS_THRESHOLD = {14}                             !Energy threshold for ground state degeneracy loop up
    CUTOFF = {15}                                   !Spectrum cut-off, used to determine the number states to be retained.
    BATH_TYPE = {16}                                !flag to set bath type: normal (1bath/imp), hybrid(1bath), replica(1replica/imp)
    ED_SECTORS = {17}                               !flag to reduce sector scan for the spectrum to specific sectors +/- ed_sectors_shift.
    ED_SECTORS_SHIFT = {18}                         !shift to ed_sectors
    ED_VERBOSE = {19}                               !Verbosity level: 0=almost nothing --> 5:all. Really: all
    CHIEXCT_FLAG = {20}                             !Flag to activate excitonis susceptibility calculation.
    CHIPAIR_FLAG = {21}                             !Flag to activate pair susceptibility calculation.
    HFMODE = F                                      !Flag to set the Hartree form of the interaction (n-1/2). see xmu.
    WINI = {22}                                     !Smallest real-axis frequency
    WFIN = {23}                                     !Largest real-axis frequency
    LREAL = {24}                                    !Number of real-axis frequencies.
    EPS = {25}                                      !Broadening on the real-axis.
    ED_PRINT_SIGMA = F                              !flag to print impurity Self-energies
    ED_PRINT_G = F                                  !flag to print impurity Greens function
    ED_PRINT_G0 = F                                 !flag to print non-interacting impurity Greens function
    """    

    n_bath = params_kw.get('n_bath', 0)  # 0 for Hubbard-I approximation
    finite_T = params_kw.get('finite_T', 'F')
    n_spin = params_kw.get('n_spin', 2)
    nstates_sector = params_kw.get('nstates_sector', 1)
    nstates_total = params_kw.get('nstates_total', 1)
    n_iter = params_kw.get('n_iter', 512)
    lanc_tol = params_kw.get('lanc_tol', 1e-12)
    gs_threshold = params_kw.get('gs_threshold', 1e-5)
    cutoff = params_kw.get('cutoff', 1e-5)
    bath_type = params_kw.get('bath_type', 'normal')
    ed_sectors = params_kw.get('ed_sectors', 'F')
    sectors_shift = params_kw.get('sectors_shift', 1)
    ed_verbose = params_kw.get('ed_verbose', 2)
    chiexct = params_kw.get('chiexct', 'F')
    chipair = params_kw.get('chipair', 'F')
    wini = params_kw.get('wini', -0.2)
    wfin = params_kw.get('wfin', 0.2)
    Lreal = params_kw.get('Lreal', 1000)
    eps = params_kw.get('eps', 0.0004)
        
    if finite_T not in ['T', 'F']:
        print("error: finite_T must be T or F!")
        exit(1)
    if n_spin not in [1, 2]:
        print("error: n_spin must be 1 or 2!")
        exit(1)
    if bath_type not in ['normal', 'hybrid']:
        print("error: bath_type must be normal or hybrid!")
        exit(1)    
    if ed_sectors not in ['T', 'F']:
        print("error: ed_sectors must be T or F!")
        exit(1)
    if ed_verbose not in range(6):
        print("error: ed_verbose must be 0~5!")
        exit(1)
    if chiexct not in ['T', 'F']:
        print("error: chiexct must be T or F!")
        exit(1)
    if chipair not in ['T', 'F']:
        print("error: chipair must be T or F!")
        exit(1)
                          
    if bath_type == 'normal':
        if n_bath%n_orb != 0:
            print("error: n_bath/n_orb must be an integer!")
            exit(1)
        n_bath_input = n_bath//n_orb
        if n_spin == 1:
            bath_flag = 0 #normal bath, spin degenerate
        else:
            bath_flag = 1 #normal bath, spinful
    else:
        n_bath_input = n_bath
        if n_spin == 1:
            bath_flag = 2 #hybrid bath, spin degenerate
        else:
            bath_flag = 3 #hybrid bath, spinful            
    if n_spin == 1:
        ed_twin = 'T'
    else:
        ed_twin = 'F'

    # ED bath fitting
    if not os.path.exists('ED/') and rank == 0:
        os.makedirs('ED/')
    comm.Barrier()
    os.chdir('ED/')
    fit_params = {'fit_gtol': 1e-6}
    h0_mat = extract_H0(G0_iw, bname)
    assert h0_mat.shape == (2*n_orb, 2*n_orb)

    Delta_iw = delta(G0_iw)
    bath_levels, bath_hyb = extract_bath_params(comm, Delta_iw, beta, bname, n_bath, bath_flag, **fit_params)
    assert bath_levels.shape == (2*n_bath,)
    assert bath_hyb.shape == (2*n_orb, 2*n_bath)

    Hloc = h0_mat.reshape((2, n_orb, 2, n_orb)).transpose((0,2,1,3))
    if n_spin == 1:
        Hloc = Hloc[0][0].reshape((1, 1, n_orb, n_orb))
    imag = LA.norm(Hloc.imag)
    if imag > 1e-7 and rank == 0:
        print("Warning: Hloc has a imaginary part %f larger than 1e-7!" %imag)
    Hloc = Hloc.real
            
    bath = []

    levels = 2*n_bath
    if n_spin == 1:
        levels = n_bath
    for bath_cnt in range(levels):
        bath.append(bath_levels[bath_cnt])

    bath_hyb_is = bath_hyb.reshape((2, n_orb, 2, n_bath))    
    for spin_cnt in range(n_spin):
        for orb_cnt in range(n_orb):
            if bath_type == "normal":
                for bath_cnt in range(n_bath//n_orb):
                    bath.append(bath_hyb_is[spin_cnt, orb_cnt, spin_cnt, orb_cnt*n_bath//n_orb+bath_cnt])
            else:
                for bath_cnt in range(n_bath):
                    bath.append(bath_hyb_is[spin_cnt, orb_cnt, spin_cnt, bath_cnt])
                    
    bath = np.array(bath)
    imag = LA.norm(bath.imag)
    if imag > 1e-7 and rank == 0:
        print("Warning: bath has a imaginary part %f larger than 1e-7!" %imag)
    bath = bath.real

    U = np.zeros(5, dtype = float)
    for orb in range(n_orb):
        U[orb] = u_mat[0][0][0][0]
    U_str = np.array2string(U, separator=', ')[1:-1]
    if n_orb > 1:
        U_ = u_mat[0][1][0][1]
        J = u_mat[0][1][1][0]
    else:
        U_ = 0.0
        J = 0.0
            
    inputED_file = "inputED.conf"
    if rank == 0:
        with open(inputED_file, 'w') as f:
            print(inputED.format(n_orb, n_bath_input, U_str, U_, J, beta, n_iw, \
            finite_T, n_spin, ed_twin, n_iter, lanc_tol, nstates_sector, nstates_total, \
            gs_threshold, cutoff, bath_type, ed_sectors, sectors_shift, ed_verbose, \
            chiexct, chipair, wini, wfin, Lreal, eps), end="", file=f)

    # Solve impurity model by ED
    comm.Barrier()
    if 'edipy' not in sys.modules:
        from edipy import *
        import edipy as ed
        ed.read_input(inputED_file)
        Nb = ed.get_bath_dimension()
        bath_ = np.zeros(Nb,dtype='float',order='F')
        ed.init_solver(bath_)
    ed.solve(bath, Hloc)
    Gimp = np.zeros((ed.Nspin, ed.Nspin, ed.Norb, ed.Norb, ed.Lmats), dtype='complex', order='F')
    Sigma = np.zeros((ed.Nspin, ed.Nspin, ed.Norb, ed.Norb, ed.Lmats), dtype='complex', order='F')
    Sigma_w = np.zeros((ed.Nspin, ed.Nspin, ed.Norb, ed.Norb, ed.Lreal), dtype='complex', order='F')
    ed.get_gimp_matsubara(Gimp)
    ed.get_sigma_matsubara(Sigma)
    ed.get_sigma_realaxis(Sigma_w) 
    if n_spin == 1:
        Gimp_tmp = np.zeros((2, 2, n_orb, n_orb, n_iw), dtype='complex', order='F')
        Gimp_tmp[0][0] = Gimp[0][0]
        Gimp_tmp[1][1] = Gimp[0][0]
        Gimp = Gimp_tmp
        Sigma_tmp = np.zeros((2, 2, n_orb, n_orb, n_iw), dtype='complex', order='F')
        Sigma_tmp[0][0] = Sigma[0][0]
        Sigma_tmp[1][1] = Sigma[0][0]
        Sigma = Sigma_tmp
    Gimp = np.einsum("ssijw->sijw", Gimp)
    Sigma = np.einsum("ssijw->sijw", Sigma)
    assert Gimp.shape == Sigma.shape == (2, n_orb, n_orb, n_iw)
    assign_from_numpy_array(Gimp_iw, Gimp, bname)
    assign_from_numpy_array(Sigma_iw, Sigma, bname)

    # Save txt data to check the fit quality
    G0_imp = dyson(Sigma_iw=Sigma_iw, G_iw=Gimp_iw)
    if rank == 0:
        save_data_comp(Delta_iw, delta(G0_imp), 'delta_comp.txt')
        save_data_comp(G0_iw, G0_imp, 'g0_comp.txt')
    
    # Extrapolate the impurity GF to the log mesh
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

    # Obtain the non-interacting impurity GF from fitting results on the log mesh
    _eps = bath_levels[0:n_bath].real
    _hyb = bath_hyb_is[0,:,0,:].real
    denom = 1j*omega[:, None] - _eps[None, :]
    delta_fit = np.einsum('al, bl, wl->wab', _hyb, _hyb.conj(), np.reciprocal(denom))
    n_iw_log = len(omega)
    weiss = np.zeros((n_orb, n_orb, n_iw_log), dtype='complex')
    for i in range(n_iw_log):
        weiss[:,:,i] = LA.inv(1j*omega[i]*np.identity(n_orb) - Hloc[0][0] - delta_fit[i])

    # Save GF and self-energy data
    if rank == 0:
        np.save('gimp.npy', gimp)
        np.save('weiss.npy', weiss)
        np.save('Sigma_w.npy', Sigma_w)
    os.chdir('../')

else:  
    # should not get here
    # iatom > 1, construct the new GF by symmetrization 
    symm_U = np.diag([1,-1,1,-1,1])
    if os.path.exists('ED/gimp.npy'):
        gimp_tmp = np.load('ED/gimp.npy')
        gimp = np.einsum('ab, bcw, cd->adw', symm_U.transpose().conjugate(), gimp_tmp, symm_U)
    else:
        print("error: impurity GF data do not exist!")
        exit(1)
    if os.path.exists('ED/weiss.npy'):
        weiss_tmp = np.load('ED/weiss.npy')
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
