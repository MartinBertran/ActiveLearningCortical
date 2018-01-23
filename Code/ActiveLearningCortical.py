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

        X, I, X_hat, I_hat = self.boxcar(X,I)

        self.X = X
        self.I = I
        self.X_hat = X_hat
        self.I_hat = I_hat
        self.R_hat = np.concatenate(X_hat,I_hat,axis=0)

    def boxcar(self,X,I):
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
        X_hat = np.zeros(X.shape)
        I_hat = np.zeros(I.shape)

        # conv xc
        for i in np.arange(X_hat.shape[1]):
            xc = X[:, i]
            xcc = np.convolve(xc, ker_c, mode='full')[hl_c:-hl_c]
            X_hat[:, i] = xcc

        # conv xi
        for i in np.arange(I_hat.shape[1]):
            xi = I[:, i]
            xic = np.convolve(xi, ker_i, mode='full')[hl_i:-hl_i]
            I_hat[:, i] = xic

        # roll back convolved spikes and visual stimuli
        X_hat = np.roll(X_hat, shift=roll_c, axis=0)
        I_hat = np.roll(I_hat, shift=roll_i, axis=0)

        # trim "corrupted" frames
        trim = np.maximum(abs(roll_c), abs(roll_i))

        X = X[trim:, :]
        I = I[trim:, :]
        I_hat = I_hat[trim:, :]
        X_hat = X_hat[trim:, :]

        return I_hat, X_hat, X, I

    def addData(self,X,I):

        X, I, X_hat, I_hat = self.boxcar(X, I)

        R_hat = np.concatenate(X_hat, I_hat, axis=0)

        self.X = np.append(self.X,X, axis=1)
        self.I = np.append(self.I,I, axis=1)
        self.X_hat = np.append(self.X_hat,X_hat, axis=1)
        self.I_hat = np.append(self.I_hat,I_hat, axis=1)
        self.R_hat = np.append(self.I_hat, I_hat, axis=1)
