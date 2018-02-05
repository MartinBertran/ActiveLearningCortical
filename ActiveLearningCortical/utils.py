import numpy as np
import os

import pkg_resources
DATA_PATH = pkg_resources.resource_filename('<package name>', 'data/')

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

    if dataset not in dataset_list:
        print('requested dataset does not exist, please choose one of the following datasets:')
        print(dataset_list)
        return


    spikes = np.genfromtxt(
        os.join(DATA_PATH, dataset + '_spikes.csv'),
        delimiter=",", skip_header=1)
    stimuli = np.genfromtxt(
        os.join(DATA_PATH, dataset + '_stimuli.csv'),
        delimiter=",", skip_header=1)
    subset = np.genfromtxt(
        os.join(DATA_PATH, dataset + '_subset.csv'),
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
    min_k = -np.abs(aux_stimuli).max()
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

def generate_spikes(W_kernel, H_r, b_c, stimuli, kappa=500):
    W_window = W_kernel.shape[0]
    H_window = H_r.shape[0]
    n_samples = stimuli.shape[0] - H_window
    cells = W_kernel.shape[2]
    output_i = np.zeros([n_samples + W_window, cells])
    lambdaout = np.zeros([n_samples + W_window, cells])
    dlambdaout = np.zeros([n_samples + W_window, cells])
    for i in np.arange(n_samples):

        # STIMULI ANTERIOR
        i_ant = stimuli[i:H_window + i, :]
        i_ant = np.tile(i_ant, (cells, 1, 1))
        i_ant = np.swapaxes(i_ant, 1, 2)
        i_ant = np.swapaxes(i_ant, 0, 2)

        factor_input = H_r * i_ant
        factor_input = np.swapaxes(factor_input, 0, 2)
        factor_input = np.sum(factor_input, axis=2)
        factor_input = np.sum(factor_input, axis=1)

        # Y ANTERIOR
        y_ant = output_i[i:W_window + i, :]
        y_ant = np.tile(y_ant, (cells, 1, 1))
        y_ant = np.swapaxes(y_ant, 1, 2)
        y_ant = np.swapaxes(y_ant, 0, 2)

        # NETWORK TERM#
        factor_cells = W_kernel * y_ant
        factor_cells = np.swapaxes(factor_cells, 0, 2)
        factor_cells = np.sum(factor_cells, axis=2)
        factor_cells = np.sum(factor_cells, axis=1)

        eta = b_c + factor_cells + factor_input

        dL = 1 - 1 / (1 + np.exp(kappa * eta))
        L = np.logaddexp(0, k * eta) / kappa


        output_i[W_window + i, :] = np.random.poisson(lam=L)
        lambdaout[W_window + i, :] = L
        dlambdaout[W_window + i, :] = dL

    output = output_i[W_window:, :]
    lambdaout = lambdaout[W_window:, :]
    dlambdaout = dlambdaout[W_window:, :]
    stimul = np.array(stimuli[W_window:, :])
    return output, lambdaout, dlambdaout, stimul