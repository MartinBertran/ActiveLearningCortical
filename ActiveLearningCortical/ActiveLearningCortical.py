import numpy as np
from scipy.optimize import minimize


class ClassModel():
    '''
    Missing a really nice descriptor with .. math:: environment

    Parameters
    ----------


    Reference
    ---------
    ref to paper
    '''

    def __init__(self, D_c_l, D_c_u, D_s_l, D_s_u, k,kappa, X, I, gamma, nu, n_splits):

        self.D_c_l = D_c_l
        self.D_c_u = D_c_u
        self.D_s_l = D_s_l
        self.D_s_u = D_s_u
        self.k = k
        self.kappa = kappa
        self.gamma = gamma
        self.nu = nu
        self.n_splits = n_splits

        X, I, X_hat, I_hat = self.boxcar(X,I)

        self.X = X
        self.I = I
        self.X_hat = X_hat
        self.I_hat = I_hat
        self.R_hat = np.append(X_hat,I_hat,axis=1)

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

        R_hat = np.concatenate(X_hat, I_hat, axis=1)

        self.X = np.append(self.X,X, axis=0)
        self.I = np.append(self.I,I, axis=0)
        self.X_hat = np.append(self.X_hat,X_hat, axis=0)
        self.I_hat = np.append(self.I_hat,I_hat, axis=0)
        self.R_hat = np.append(self.R_hat, R_hat, axis=0)

    def MAP_likelihood(self,X_c, R_c, theta, kappa):

        nabla = np.dot(R_c, theta)
        p_lambda = np.logaddexp(0, kappa * nabla) / kappa + 1e-20
        likelihood = X_c * np.log(p_lambda) - p_lambda
        likelihood = np.sum(likelihood)
        return -likelihood

    def grad_MAP_likelihood(self,X_c, R_c, theta, kappa):

        nabla = np.dot(R_c, theta)
        p_lambda = np.logaddexp(0, kappa * nabla) / kappa + 1e-20
        d_like_d_lambda = (X_c / p_lambda - 1)
        d_lambda_d_nabla = 1 - (1 / (1 + np.exp(nabla * kappa)))
        d_nabla_d_theta = R_c
        d_lambda_d_theta = (d_like_d_lambda * d_lambda_d_nabla)[:, np.newaxis] * d_nabla_d_theta
        d_lambda_d_theta = np.sum(d_lambda_d_theta, axis=0)
        return -d_lambda_d_theta

    def computeMAP(self, c, PA_c,theta_ini=None, index_mask=None):

        #build regressors, intial values, and select target variable
        if index_mask is None:
            R_c = self.R_hat[:,PA_c]
            X_c = self.X[:,c]
        else:
            R_c = self.R_hat[index_mask,PA_c]
            X_c = self.X[index_mask, c]

        #append bias vector
        R_c = np.append(R_c,np.ones([R_c.shape[0],1]),axis=1)

        #build initialization
        if theta_ini is None:
            theta_ini = np.random.uniform(0.001, 0.01, self.R_hat.shape[1]+1)
        theta_ini_local = theta_ini[np.append(PA_c,[True]).astype('bool')]



        # Minimization
        f = lambda theta: self.MAP_likelihood(X_c,R_c, theta, self.kappa)
        df = lambda theta: self.grad_MAP_likelihood(X_c,R_c, theta, self.kappa)


        options = {}
        options['maxiter'] = 1000
        options['disp'] = False
        options['ftol'] = 1e-14

        theta_MAP_local = minimize(f, theta_ini_local, jac=df, options=options)
        theta_MAP_local=theta_MAP_local.x

        theta_MAP  = np.zeros(theta_ini.shape)
        theta_MAP[PA_c] = theta_MAP_local
        return theta_MAP

    def forwardModelProposal(self,c,PA_c, split_indexes):
        '''
        :param c:
        :param PA_c:
        :param split_indexes: n_splits x total_samples boolean masks of each spliting subset
        :return:
        '''

        BIC_split = np.inf([self.n_splits, self.R_hat.shape[1]])
        pval_split = np.ones([self.n_splits, self.R_hat.shape[1]])

        BIC_full = np.inf([self.R_hat.shape[1]])
        pval_full = np.ones([self.R_hat.shape[1]])


        # go through all regressors not currently in the model
        for j in np.where(PA_c==False)[0]:
            PA_c_r = np.array(PA_c)
            PA_c_r[j] = True

            # evaluate results for every split
            for split in np.arange(self.n_splits):
                split_idx = split_indexes[split,:]

                BIC, Likelihood, pval, theta_map, fisher = evaluateRegressors(c, PA_c_r,index_mask=split_idx)

                BIC_split[split,j] = BIC
                pval_split[split,j] = pval

            # evaluate results over full dataset
            BIC, Likelihood, pval, theta_map, fisher = evaluateRegressors(c, PA_c_r, index_mask=None)

            BIC_full[j]=BIC
            pval_full[j]=pval

        #Finally, compute score

        pval_score = np.maximum(np.median(pval_split,axis=0), pval_full)
        BIC_score = np.maximum(np.median(BIC_split,axis=0), BIC_full)


        # get index of regressors that simultaneoulsy satisfy BIC_score < 0 and pval_score < gamma
        index_satisfactory = np.where(BIC_score<0 and pval_score<self.gamma)[0]

        #sort remaining regressors in ascending order of BIC score, and select the first k regressors
        sorted_remainder =  np.argsort(BIC_score[index_satisfactory])[:self.k]

        best_candidates = index_satisfactory[sorted_remainder]

        return best_candidates





