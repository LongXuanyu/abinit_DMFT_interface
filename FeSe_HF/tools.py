from triqs.gf import *
import numpy
import os
import sys

def make_block_gf(gf_class, gf_struct, beta, n_points):
    """
    Make a BlockGf object

    :param gf_class: GfImFreq, GfImTime or GfLegendre
    :param gf_struct: structure of Green's function
    :param beta: inverse temperature
    :param n_points: number of points
    :return: object of BlockGf
    """

    assert isinstance(gf_struct, dict)

    blocks = []
    block_names = []
    for name, indices in list(gf_struct.items()):
        assert isinstance(name, str)
        block_names.append(name)
        indices_str = list(map(str, indices))
        blocks.append(gf_class(indices=indices_str, beta=beta, n_points=n_points, name=name))
    return BlockGf(name_list=block_names, block_list=blocks, make_copies=True)

def assign_from_numpy_array(g_block, data, block_names):

    for i, bname in enumerate(block_names):
        gf = g_block[bname]

        # number of positive Matsubara freq
        n_iw = data[i].shape[2]
        assert gf.data.shape[0] == 2*n_iw

        # data(o1, o2, iw) --> (iw, o1, o2)
        gf_iw_o1_o2 = data[i].transpose(2, 0, 1)

        # copy data in the positive freq side
        gf.data[n_iw:, :, :] = gf_iw_o1_o2.copy()

        # copy data in the negative freq side (complex conjugate)
        gf.data[0:n_iw, :, :] = gf_iw_o1_o2.transpose(0, 2, 1)[::-1, :, :].conjugate().copy()

def extract_H0_from_tail(G0_iw):
    if isinstance(G0_iw, BlockGf):
        return {name:extract_H0_from_tail(b) for name, b in G0_iw}
    elif isinstance(G0_iw.mesh, MeshImFreq):
       import triqs.gf.gf_fnt as gf_fnt
       import triqs.gf.descriptors as descriptors
       assert len(G0_iw.target_shape) in [0,2], "extract_H0_from_tail(G0_iw) requires a matrix or scalar_valued Green function"
       assert gf_fnt.is_gf_hermitian(G0_iw), "extract_H0_from_tail(G0_iw) requires a Green function with the property G0_iw[iw][i,j] = conj(G0_iw[-iw][j,i])"
       delta_iw = G0_iw.copy()
       delta_iw << descriptors.iOmega_n - inverse(G0_iw)
       tail, err = gf_fnt.fit_hermitian_tail(delta_iw)
       if err > 1e-5:
           print("WARNING: delta extraction encountered a sizeable tail-fit error: ", err)
       return tail[0]
    else:
        raise RuntimeError('extract_H0_from_tail does not support type {}'.format(type(G0_iw)))

def extract_H0(G0_iw, block_names, hermitianize=True):
    """
    Extract non-interacting Hamiltonian elements from G0_iw
    """

    assert isinstance(block_names, list)

    H0_dict  = extract_H0_from_tail(G0_iw)
    H0 = [H0_dict[b] for b in block_names]

    n_spin_orb = numpy.sum([b.shape[0] for b in H0])

    if G0_iw.n_blocks > 2:
        raise RuntimeError("n_blocks must be 1 or 2.")

    data = numpy.zeros((n_spin_orb, n_spin_orb), dtype=complex)
    offset = 0
    for block in H0:
        block_dim = block.shape[0]
        data[offset:offset + block_dim, offset:offset + block_dim] = block
        offset += block_dim

    if hermitianize:
        data = 0.5 * (data.transpose().conj() + data)

    return data

def fit_delta_iw_core(delta_iw, beta, n_bath, n_w_fit, verbose, index, **fit_params):

    from scipy import optimize

    n_w = delta_iw.shape[0]
    if n_w < n_w_fit:
        n_w_fit = n_w
    n_orb = delta_iw.shape[1]

    # fermionic Matsubara freqs
    freqs = numpy.array([1j * (2*i+1) * numpy.pi / beta for i in range(n_w_fit)])
    weight = 1 / numpy.sqrt(freqs.imag)

    # Define distance between delta_iw and delta_fit
    # delta_fit = sum_{l=1}^{n_bath} V_{o1, l} * V_{l, o2} / (iw - eps_{l})
    def distance(x):
        _eps = x[0:n_bath]
        _hyb = x[n_bath:].reshape(n_orb, n_bath)

        # denom[i,j] = (freqs[i] - eps[j])
        denom = freqs[:, None] - _eps[None, :]

        # sum over bath index l
        delta_fit = numpy.einsum('al, bl, wl->wab', _hyb, _hyb.conj(), numpy.reciprocal(denom))

        # squared error
        res = numpy.einsum('wab,w->wab', delta_iw[:n_w_fit] - delta_fit, weight)
        return numpy.square(numpy.linalg.norm(res))

    # Determine eps and V which minimize the distance between delta_iw and delta_fit
    # [0:n_bath] -> eps_{l},  [n_bath:n_orb*n_bath] -> V_{o,l}
    # initial guess, random values in the range [-1:1]
    local_state = numpy.random.RandomState()
    x0 = local_state.uniform(-1, 1, n_bath + n_orb * n_bath)
    filename = 'fit_%d.npy' %index
    if os.path.exists(filename):
        x0 = numpy.load(filename)

    # fitting
    result = optimize.fmin_bfgs(distance, x0, **fit_params)
    if(verbose):
        print(" ", result)
    dis = distance(result)
    return [dis, result]

def fit_delta_iw(comm, delta_iw, beta, n_bath, n_w_fit, verbose, index, **fit_params):
    """
    Fit Delta(iw) using scipy

    scipy.optimize.fmin_bfgs
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_bfgs.html

    Parameters
    ----------
    delta_iw: [numpy.ndarray] (n_w, n_orb, n_orb)
    beta: [float] 1/T
    n_bath: [int] number of bath
    **fit_params: [dict] optional parameters to the fitting function

    Returns
    -------
    eps: [numpy.ndarray] (n_bath,) bath levels
    hyb: [numpy.ndarray] (n_orb, n_bath) hybridization parameters

    """
    n_orb = delta_iw.shape[1]
    assert delta_iw.shape[2] == n_orb
    result = fit_delta_iw_core(delta_iw, beta, n_bath, n_w_fit, verbose, index, **fit_params)
    result = comm.gather(result)
    rank = comm.Get_rank()
    filename = 'fit_%d.npy' %index
    if rank == 0:
        result.sort(key = lambda r: r[0])
        numpy.save(filename, result[0][1])
    comm.Barrier()
    result = numpy.load(filename)
    eps = result[0:n_bath]
    hyb = result[n_bath:].reshape(n_orb, n_bath)
    return eps, hyb

def extract_bath_params(comm, delta_iw, beta, block_names, n_bath, bath_flag=3, n_w_fit=1000, fit_gtol=1e-5, verbose=False):
    """
    Determine bath parameters by fitting Delta(iw)

    Parameters
    ----------
    delta_iw: [block Gf] Delta(iw)
    beta: [float] 1/T
    block_names: [list] block names
    n_bath: [int] number of bath
    fit_gtol: [float] A fitting parameter: Gradient norm must be less than gtol before successful termination.

    Returns
    -------
    eps: [numpy.ndarray] (2*n_bath,) bath levels
    hyb: [numpy.ndarray] (2*n_orb, 2*n_bath) hybridization parameters

    """

    rank = comm.Get_rank()
    n_orb = delta_iw[block_names[0]].data.shape[1]
    n_blocks = len(block_names)

    # These arrays will be returned
    eps_full = numpy.zeros((n_bath * n_blocks,), dtype=float)
    hyb_full = numpy.zeros((n_orb * n_blocks, n_bath * n_blocks), dtype=float)

    if n_bath == 0:
        return eps_full, hyb_full

    # fitting parameters
    fit_params = {
        "gtol": fit_gtol,
        "disp": verbose,
    }

    if rank == 0:
        print("\nDetermine bath parameters by fitting Delta(iw)")
        for key, val in list(fit_params.items()):
            print("  {} : {}".format(key, val))

    # bath parameters for each block
    eps_list = []
    hyb_list = []
    count = 0
    for i, b in enumerate(block_names):
        # do not fit for the spin down block with spin degeneracy
        if not ((bath_flag == 0 or bath_flag == 2) and i == 1):
            # fit Delta(iw)
            if(verbose):
                print("\nblock =", b)
            # data.shape == (n_w, n_orb, n_orb)
            n_w = delta_iw[b].data.shape[0]
            assert delta_iw[b].data.shape[1] == delta_iw[b].data.shape[2] == n_orb
            # use only positive Matsubara freq
            if bath_flag == 0 or bath_flag == 1: # normal bath
                assert n_bath%n_orb == 0
                eps = numpy.zeros((n_orb, n_bath//n_orb), dtype=float)
                hyb = numpy.zeros((n_orb, n_orb, n_bath//n_orb), dtype=float)
                for orb in range(n_orb):
                    delta_orb = delta_iw[b].data[n_w//2:n_w, orb, orb].reshape(n_w//2, 1, 1)
                    eps_, hyb_ = fit_delta_iw(comm, delta_orb, beta, n_bath//n_orb, n_w_fit, verbose, count, **fit_params)
                    count = count+1
                    eps[orb] = eps_
                    hyb[orb][orb] = hyb_[0]
                eps = eps.reshape(n_bath)
                hyb = hyb.reshape((n_orb, n_bath))           
            else: # hybrid bath
                eps, hyb = fit_delta_iw(comm, delta_iw[b].data[n_w//2:n_w, :, :], beta, n_bath, n_w_fit, verbose, count, **fit_params)
                count = count+1
            assert eps.shape == (n_bath,)
            assert hyb.shape == (n_orb, n_bath)
        eps_list.append(eps)
        hyb_list.append(hyb)

    # Combine spin blocks
    # eps_full = {eps[up], eps[dn]}
    for i, block in enumerate(eps_list):
        n = block.shape[0]
        eps_full[n*i:n*(i+1)] = block

    # hyb_full = {{hyb[up], 0}, {0, hyb[dn]}}
    for i, block in enumerate(hyb_list):
        m, n = block.shape
        hyb_full[m*i:m*(i+1), n*i:n*(i+1)] = block

    if rank == 0:
        print("\nfitting results")
        print("  eps[l]    hyb[0,l]  hyb[1,l]  ...")
        for l in range(eps_full.size):
            print(" %9.5f" %eps_full[l], end="")
            for orb in range(hyb_full.shape[0]):
                print(" %9.5f" %hyb_full[orb, l], end="")
            print("")
        sys.stdout.flush()
    
    return eps_full, hyb_full

def get_data(GF):
    data = []
    for bname in GF.indices:
        n_iw = GF[bname].data.shape[0]//2
        data_ = GF[bname].data[n_iw:, :, :].copy()
        data.append(data_.transpose((1,2,0)))
    return numpy.array(data)

def save_data_comp(GF_ori, GF_fit, filename):
    data_ori = get_data(GF_ori)
    data_fit = get_data(GF_fit)
    count = 0
    n_iw = data_ori[0].shape[-1]
    n_orb = data_ori[0].shape[0]
    data_comp = numpy.zeros((n_iw, 2*n_orb*(n_orb+1)), dtype='double')
    for i in range(n_orb):
        for j in range(i,n_orb):
            data_comp[:, 4*count:4*count+2] = data_ori[0][i][j].view(float).reshape(-1,2)[:,:]
            data_comp[:, 4*count+2:4*count+4] = data_fit[0][i][j].view(float).reshape(-1,2)[:,:]
            count = count+1
    numpy.savetxt(filename, data_comp)
