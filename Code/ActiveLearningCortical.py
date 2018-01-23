import numpy as np


class ClassModel():
    '''
    Missing a really nice descriptor with .. math:: environment

    Parameters
    ----------


    Reference
    ---------
    ref to paper
    '''

    def __init__(self, D_c_l, D_c_u, D_s_l, D_s_u, k, X, I, gamma, nu, n_splits):

        self.D_c_l = D_c_l
        self.D_c_u = D_c_u
        self.D_s_l = D_s_l
        self.D_s_u = D_s_u
        self.k = k
        self.gamma = gamma
        self.nu = nu
        self.n_splits = n_splits

        X, I, X_hat, I_hat = boxcar(X,I)

        self.X = X
        self.I = I
        self.X_hat = X_hat
        self.I_hat = I_hat
        self.R_hat = np.concatenate(X_hat,I_hat,axis=0)

    def boxcar(self):
        # windows
        win_c = self.D_c_u - self.D_c_l + 1
        win_i = self.D_s_u - self.D_s_l + 1

        # half lengths (window size virtually upscaled to nearest odd value
        hl_i = int(np.floor((win_i + 1) / 2))
        hl_c = int(np.floor((win_c + 1) / 2))

        # kernels
        ker_i = np.zeros(2 * hl_i + 1);
        ker_i[:win_i] = 1
        ker_c = np.zeros(2 * hl_c + 1);
        ker_c[:win_c] = 1

        # phase differences
        roll_i = hl_i - self.D_s_u  # self.D_s_l + hl_i
        roll_c = hl_c - self.D_c_u  # self.D_c_l + hl_c

        ## each cell
        X_hat = np.zeros(spikes.shape)
        I_hat = np.zeros(stimuli.shape)

        # conv xc
        for i in np.arange(X_hat.shape[1]):
            xc = spikes[:, i]
            xcc = np.convolve(xc, ker_c, mode='full')[hl_c:-hl_c]
            X_hat[:, i] = xcc

        # conv xi
        for i in np.arange(I_hat.shape[1]):
            xi = stimuli[:, i]
            xic = np.convolve(xi, ker_i, mode='full')[hl_i:-hl_i]
            # print(xic.shape)
            I_hat[:, i] = xic

        # roll back convolved spikes
        X_hat = np.roll(X_hat, shift=roll_c, axis=0)
        # roll back convolved input
        I_hat = np.roll(I_hat, shift=roll_i, axis=0)

        X = np.array(spikes)
        I = np.array(stimuli)

        # trim "corrupted" frames
        trim = np.maximum(abs(roll_c), abs(roll_i))

        X = X[trim:, :]
        I = I[trim:, :]
        I_hat = I_hat[trim:, :]
        X_hat = X_hat[trim:, :]

        return I_hat, X_hat, X, I