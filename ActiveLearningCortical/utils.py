import numpy as np


def load_datasets(dataset):

    dataset_list=['lt3_000_002','lt3_000_003','lt3_001_003','lt3_001_004','lt3_002_000','lt3_002_001']

    if dataset not in dataset_list:
        print('requested dataset does not exist, please choose one of the following datasets:')
        print(dataset_list)
        return



    # this will become a function

    spikes = np.genfromtxt('../Datasets/'+dataset + '_spikes.csv', delimiter=",", skip_header=1)
    stimuli = np.genfromtxt('../Datasets/'+dataset + '_stimuli.csv', delimiter=",", skip_header=1)
    subset = np.genfromtxt('../Datasets/'+dataset + '_subset.csv', delimiter=",", skip_header=0)



    # first i'm gonna expand stimuli into real number of frames
    delta_t = np.diff(stimuli[:, -1])
    delta_t = np.append(delta_t, delta_t[-1])  # repeating last observed diff

    # making new stimuli variable
    max_frame = np.ceil(stimuli[-1, -1] + delta_t[-1]).astype('int')

    new_stimuli = np.zeros([max_frame, stimuli.shape[1]])
    for j in range(stimuli.shape[0]):
        ini_frm = int(stimuli[j, -1])
        end_frm = int(ini_frm + delta_t[j])
        new_stimuli[ini_frm:end_frm] = stimuli[j]

    # print(stimuli.shape, new_stimuli.shape)

    aux_stimuli = new_stimuli[:, 2:4]  # trim useless frame states and sign

    def arctan2_custom(kx,ky):
        if ky !=0:
            return np.arctan(kx/ky)
        elif kx>0:
            return np.pi/2
        else:
            return -np.pi/2


    n_states = (aux_stimuli.max(0) - aux_stimuli.min(0) + 1).astype('int')
    min_k = aux_stimuli.min(0)
    print("min_k",min_k)
    reparam_stimuli = np.zeros([new_stimuli.shape[0], *n_states])
    RPhi_stimuli = np.zeros([new_stimuli.shape[0], 2])
    for j in range(reparam_stimuli.shape[0]):
        kx = int(aux_stimuli[j, 0])
        ky = int(aux_stimuli[j, 1])

        # print(kx,ky)
        RPhi_stimuli[j, 0] = np.sqrt(kx**2+ky**2)
        RPhi_stimuli[j, 1] = arctan2_custom(kx,ky)

        kx_ind = int(kx - min_k[0])
        ky_ind = int(ky - min_k[1])
        # print(kx, ky)

        reparam_stimuli[j, kx_ind, ky_ind] = 1


    print("badly named r variable, its actually spatial frequency, not wavelength")
    print('RPhi')

    # This one is very different
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
    q_fi_th = np.linspace(0, 100, quant_fi + 1)
    r_th = np.percentile(r.flatten(), q_r_th)
    fi_th = np.percentile(fi.flatten(), q_fi_th)

    r_disp = r_th[:-1]+np.diff(r_th)/2
    fi_disp = fi_th[:-1]+np.diff(fi_th)/2

    r_th=r_th[1:]
    fi_th = fi_th[1:]


    reparam_stimuli_phi = np.zeros([RPhi_stimuli.shape[0], quant_r, quant_fi])

    for j in range(RPhi_stimuli.shape[0]):
        r = RPhi_stimuli[j, 0]
        fi = RPhi_stimuli[j, 1]
        r_ind = np.min(np.where(r_th >= r)[0])
        fi_ind = np.min(np.where(fi_th >= fi)[0])
        reparam_stimuli_phi[j, r_ind, fi_ind] = 1
    reparam_stimuli =reparam_stimuli_phi


    n_states = reparam_stimuli.shape[1:]  # update n_states
    print(n_states)


    reparam_stimuli = np.reshape(reparam_stimuli, [reparam_stimuli.shape[0], -1])

    stimuli = reparam_stimuli
    spikes = spikes[:stimuli.shape[0], subset.astype('int')]

    return stimuli, spikes, n_states, r_disp, fi_disp
