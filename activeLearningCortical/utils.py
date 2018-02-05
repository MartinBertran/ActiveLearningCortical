import numpy as np
import os

import pkg_resources


def load_datasets(dataset):

    """
    utility function for loading the provided datasets
    :param dataset: any dataset from the following list: ['lt3_000_002','lt3_000_003','lt3_001_003','lt3_001_004','lt3_002_000','lt3_002_001']
    :return:

    stimuli: #r, phi encoded stimuli in one hot vector format
    spikes: #detected spikes
    stimuli_raw: # unaltered A, ky and ky values presented each frame
    r_disp, phi_disp # center of bin values for r and phi coding
    """

    dataset_list=['lt3_000_002','lt3_000_003','lt3_001_003','lt3_001_004','lt3_002_000','lt3_002_001']
    DATA_PATH = pkg_resources.resource_filename('activeLearningCortical', 'data')

    if dataset not in dataset_list:
        print('requested dataset does not exist, please choose one of the following datasets:')
        print(dataset_list)
        return


    spikes = np.genfromtxt(
        os.path.join(DATA_PATH, dataset + '_spikes.csv'),
        delimiter=",", skip_header=1)
    stimuli = np.genfromtxt(
        os.path.join(DATA_PATH, dataset + '_stimuli.csv'),
        delimiter=",", skip_header=1)
    subset = np.genfromtxt(
        os.path.join(DATA_PATH, dataset + '_subset.csv'),
        delimiter=",", skip_header=0)


    # expanding stimuli into real number of frames (only stimuli onset is stored in csv
    delta_t = np.diff(stimuli[:, -1])
    delta_t = np.append(delta_t, delta_t[-1])  # repeating last observed diff
    max_frame = np.ceil(stimuli[-1, -1] + delta_t[-1]).astype('int')
    new_stimuli = np.zeros([max_frame, stimuli.shape[1]])

    for j in range(stimuli.shape[0]):
        ini_frm = int(stimuli[j, -1])
        end_frm = int(ini_frm + delta_t[j])
        new_stimuli[ini_frm:end_frm] = stimuli[j]

    stimuli_raw = new_stimuli[:, 1:4] # unmodified stimulation sequence containing A, kx, and ky values

    # drop sign value A and codify stimuli into kx_ky one hot vector
    aux_stimuli = new_stimuli[:, 2:4]  # trim frame states and sign

    def arctan2_custom(kx,ky):
        if ky !=0:
            return np.arctan(kx/ky)
        elif kx>0:
            return np.pi/2
        else:
            return -np.pi/2

    n_states = (aux_stimuli.max(0) - aux_stimuli.min(0) + 1).astype('int')
    min_k = -np.abs(aux_stimuli).max(0)
    reparam_stimuli = np.zeros([new_stimuli.shape[0], *n_states])
    RPhi_stimuli = np.zeros([new_stimuli.shape[0], 2])
    for j in range(reparam_stimuli.shape[0]):
        kx = int(aux_stimuli[j, 0])
        ky = int(aux_stimuli[j, 1])
        RPhi_stimuli[j, 0] = np.sqrt(kx**2+ky**2)
        RPhi_stimuli[j, 1] = arctan2_custom(kx,ky)
        kx_ind = int(kx - min_k[0])
        ky_ind = int(ky - min_k[1])
        reparam_stimuli[j, kx_ind, ky_ind] = 1

    #compute r and phi transforms, and bin them into 7 bins each
    kx = np.linspace(min_k[0], -min_k[0], -2 * min_k[0] + 1)
    ky = np.linspace(min_k[0], -min_k[0], -2 * min_k[0] + 1)

    r = np.zeros([kx.size, ky.size])
    fi = np.zeros([kx.size, ky.size])
    for i in np.arange(kx.size):
        for j in np.arange(ky.size):
            r[i, j] = np.sqrt((kx[i] ** 2 + ky[j] ** 2))
            fi[i, j] = arctan2_custom(kx[i], ky[j])


    quant_r = 7
    quant_fi = 7
    q_r_th = np.linspace(0, 100, quant_r+1)
    q_phi_th = np.linspace(0, 100, quant_fi + 1)
    r_th = np.percentile(r.flatten(), q_r_th)
    phi_th = np.percentile(fi.flatten(), q_phi_th)

    r_disp = r_th[:-1]+np.diff(r_th)/2
    phi_disp = phi_th[:-1]+np.diff(phi_th)/2

    r_th=r_th[1:]
    phi_th = phi_th[1:]


    reparam_stimuli_phi = np.zeros([RPhi_stimuli.shape[0], quant_r, quant_fi])

    for j in range(RPhi_stimuli.shape[0]):
        r = RPhi_stimuli[j, 0]
        fi = RPhi_stimuli[j, 1]
        r_ind = np.min(np.where(r_th >= r)[0])
        phi_ind = np.min(np.where(phi_th >= fi)[0])
        reparam_stimuli_phi[j, r_ind, phi_ind] = 1
    reparam_stimuli =reparam_stimuli_phi
    reparam_stimuli = np.reshape(reparam_stimuli, [reparam_stimuli.shape[0], -1])

    stimuli = reparam_stimuli
    spikes = spikes[:stimuli.shape[0], subset.astype('int')]

    return stimuli, spikes, stimuli_raw, r_disp, phi_disp

def generate_spikes(W, H, b, I, kappa=200, expected=False):
    """
    This function generates the spike traces (or expected spiking rates) for a given model with parameters W, H and b
    :param W: inter-neuron connectivity kernel, delays x cells x cells
    :param H: stimuli response connectivity kernel, delays x n_stimuli x cells
    :param b: neuron bias for firing rate
    :param I: input stimulation sequence, samples x n_stimuli
    :param kappa: kappa parameter in spiking rate (lambda) function
    :param expected: boolean, if True, function will instead compute the expected spiking rate of the stimulation sequence
    :return: X, I, a spiking trace for the modeled neurons, and its generating input stimulation sequence
    """
    W_window = W.shape[0]
    H_window = H.shape[0]
    cells = W.shape[2]
    n_samples = I.shape[0] - H_window
    X = np.zeros([n_samples + W_window, cells])
    for i in np.arange(n_samples):

        # select previous stimuli and retile it into delays x n_s x cells (same shape as H)
        I_prev = I[i:H_window + i, :]
        I_prev = np.tile(I_prev, (cells, 1, 1)).transpose(1,2,0)

        # compute instantaneous contribution of all inputs across every cell
        # (sum over delays and stimuli)
        factor_input = np.sum(H * I_prev, axis=(0,1))

        # select previous spikes and retile it into delays x n_c x cells (same shape as W)
        X_prev = X[i:W_window + i, :]
        X_prev = np.tile(X_prev, (cells, 1, 1)).transpose(1,2,0)

        # NETWORK TERM#
        factor_cells = np.sum(W * X_prev, axis=(0,1))

        eta = b + factor_cells + factor_input
        L = np.logaddexp(0, kappa * eta) / kappa

        if expected:
            X[W_window + i, :] = L
        else:
            X[W_window + i, :] = np.random.poisson(lam=L)

    X = X[W_window:, :]
    I = np.array(I[W_window:, :])
    return X, I

def get_kernels_WH(W, H, dw, dh):
    size_dw = -1 * dw[1] + dw[0] + 1
    size_dh = -1 * dh[1] + dh[0] + 1
    H_kernel = np.zeros([-1 * dh[1] + 1, H.shape[0], H.shape[1]])
    W_kernel = np.zeros([-1 * dw[1] + 1, W.shape[0], W.shape[1]])
    H_kernel[-1 * dh[0]:-1 * dh[1] + 1, :, :] = np.tile(H, (size_dh, 1, 1))
    W_kernel[-1 * dw[0]:-1 * dw[1] + 1, :, :] = np.tile(W, (size_dw, 1, 1))

    return W_kernel, H_kernel