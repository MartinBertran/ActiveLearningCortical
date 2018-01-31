import numpy as np
from scipy.optimize import minimize
import scipy as scipy


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

        I_hat, X_hat, X, I = self.boxcar(X,I)

        self.X = X
        self.I = I
        self.X_hat = X_hat
        self.I_hat = I_hat
        self.R_hat = np.append(X_hat,I_hat,axis=1)

        self.n_samples = self.X_hat.shape[0]
        self.n_c = self.X_hat.shape[1]
        self.n_r = self.R_hat.shape[1]
        self.n_s = self.I_hat.shape[1]

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

        I_hat, X_hat, X, I = self.boxcar(X, I)

        R_hat = np.concatenate(X_hat, I_hat, axis=1)

        self.X = np.append(self.X,X, axis=0)
        self.I = np.append(self.I,I, axis=0)
        self.X_hat = np.append(self.X_hat,X_hat, axis=0)
        self.I_hat = np.append(self.I_hat,I_hat, axis=0)
        self.R_hat = np.append(self.R_hat, R_hat, axis=0)

        self.n_samples = self.X_hat.shape[0]
        self.n_c = self.X_hat.shape[1]
        self.n_r = self.R_hat.shape[1]
        self.n_s = self.I_hat.shape[1]

    def mapLikelihood(self,X_c, R_c, theta, kappa):

        nabla = np.dot(R_c, theta)
        p_lambda = np.logaddexp(0, kappa * nabla) / kappa + 1e-20
        likelihood = X_c * np.log(p_lambda) - p_lambda
        likelihood = np.sum(likelihood)
        return -likelihood

    def gradMapLikelihood(self,X_c, R_c, theta, kappa):

        nabla = np.dot(R_c, theta)
        p_lambda = np.logaddexp(0, kappa * nabla) / kappa + 1e-20
        d_like_d_lambda = (X_c / p_lambda - 1)
        d_lambda_d_nabla = 1 - (1 / (1 + np.exp(nabla * kappa)))
        d_nabla_d_theta = R_c
        d_lambda_d_theta = (d_like_d_lambda * d_lambda_d_nabla)[:, np.newaxis] * d_nabla_d_theta
        d_lambda_d_theta = np.sum(d_lambda_d_theta, axis=0)
        return -d_lambda_d_theta

    def computeMap(self, c, PA_c,theta_ini=None, index_mask=None):

        #build regressors, intial values, and select target variable
        if index_mask is None:
            R_c = self.R_hat[:,PA_c]
            X_c = self.X[:,c]
        else:
            R_c = self.R_hat[index_mask,PA_c]
            X_c = self.X[index_mask, c]

        #append bias vector
        R_c = np.append(R_c,np.ones([R_c.shape[0],1]),axis=1)
        PA_c_with_bias = np.append(PA_c,[True]).astype('bool')

        #build initialization
        if theta_ini is None:
            theta_ini = np.random.uniform(-0.01, 0.01, self.R_hat.shape[1]+1)
        theta_ini_local = theta_ini[PA_c_with_bias]



        # Minimization
        f = lambda theta: self.mapLikelihood(X_c,R_c, theta, self.kappa)
        df = lambda theta: self.gradMapLikelihood(X_c,R_c, theta, self.kappa)


        options = {}
        options['maxiter'] = 1000
        options['disp'] = False
        options['ftol'] = 1e-14

        theta_MAP_local = minimize(f, theta_ini_local, jac=df, options=options)
        theta_MAP_local=theta_MAP_local.x

        theta_MAP  = np.zeros(theta_ini.shape)
        theta_MAP[PA_c_with_bias] = theta_MAP_local
        return theta_MAP

    def forwardModelProposal(self,c,PA_c, index_masks):
        '''
        :param c:
        :param PA_c:
        :param index_masks: n_splits x total_samples boolean masks of each spliting subset
        :return:
        '''

        BIC_split = np.full([self.n_splits, self.R_hat.shape[1]], np.inf)
        pval_split = np.ones([self.n_splits, self.R_hat.shape[1]])

        BIC_full = np.full([self.R_hat.shape[1]],np.inf)
        pval_full = np.ones([self.R_hat.shape[1]])


        # go through all regressors not currently in the model
        for j in np.where(PA_c==False)[0]:
            PA_c_r = np.array(PA_c)
            PA_c_r[j] = True

            # evaluate results for every split
            for split in np.arange(self.n_splits):
                split_idx = index_masks[split,:]

                BIC, Likelihood, pval, theta_map, fisher = self.evaluateRegressors(c, PA_c_r,index_samples=split_idx)

                BIC_split[split,j] = BIC
                pval_split[split,j] = pval

            # evaluate results over full dataset
            BIC, Likelihood, pval, theta_map, fisher = self.evaluateRegressors(c, PA_c_r, index_samples=None)

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

    def nabla(self, W, H, b):
        R_hat = np.array(self.R_hat)
        R_hat = np.concatenate([R_hat, np.ones([R_hat.shape[0], 1])], axis=1)
        # Concatenate with 1's
        theta = np.concatenate([W, H, b], axis=0)
        nabla = np.dot(R_hat, theta)
        return nabla

    def logexpLambda(self, nabla):
        L = np.logaddexp(0, nabla * self.kappa) / self.kappa
        dL = 1 - (1 / (1 + np.exp(nabla * self.kappa)))
        # dL2 = (self.k/(1 + np.exp(nabla * self.k)))*(1 - (1/(1 + np.exp(nabla * self.k))))
        dL2 = (self.kappa / (1 + np.exp(nabla * self.kappa))) * dL
        return L, dL, dL2

    def logexpObservedFisherInfo(self, c, PA_c, L, dL, dL2):
        # Load regressors and spikes of neuron c
        R_hat = np.array(self.R_hat)
        R_hat = R_hat[:,PA_c]
        if R_hat.shape[1]>0:
            R_hat = np.concatenate([R_hat, np.ones([R_hat.shape[0], 1])], axis=1)
        else:
            R_hat = np.ones([R_hat.shape[0], 1])
        X_c = np.array(self.X[:, c])
        # Build Hessian
        diag_values = -X_c * ((dL / (L+1e-20)) ** 2) + ((X_c / (L+1e-20)) - 1) * dL2
        I = R_hat.transpose() * diag_values[np.newaxis, :]
        I = np.dot(I, R_hat)
        return -I

    def evaluateRegressors(self, c, PA_c, theta_ini=None, index_samples=None):

        # number of possible regressors
        nr = PA_c.shape[0]

        # Regressors
        R_hat = np.array(self.R_hat)

        # Neuron to model
        X_c = np.array(self.X[:, c])

        print(index_samples.shape)
        print(R_hat.shape)
        print(X_c.shape)
        if index_samples is not None:
            R_hat = R_hat[index_samples, :]
            X_c = X_c[index_samples, :]

        ## Check that all regressors are acceptables - remove ill cases ##

        mask_R = np.zeros(R_hat.shape)
        mask_R[R_hat != 0] = 1
        mask_Xc = np.tile(X_c, (R_hat.shape[1], 1)).transpose()
        mask_Xc[mask_Xc != 0] = 1
        index_PA_c = np.sum(mask_R, axis=0) * np.sum(mask_R * mask_Xc, axis=0)

        index_clean = np.zeros(PA_c.shape)
        index_clean[(index_PA_c > 0) & PA_c] = 1
        nr_ef = np.sum(index_clean > 0)

        # Effective PAc
        PAc_ef = np.zeros(PA_c.shape)
        PAc_ef[index_clean > 0] = 1
        PAc_ef = PAc_ef.astype('bool')

        # Compute MAP
        theta_full_c = self.computeMap(c, PAc_ef, theta_ini=theta_ini, index_mask=index_samples)

        # Save in full theta vector
        # theta_full_c = np.zeros([nr])
        # if nr_ef > 0:
        #     theta_full_c[PAc_ef] = theta_c[0:-1]
        # theta_full_c = np.concatenate([theta_full_c,theta_c[-1:]],axis = 0) # bias

        ## FISHER ##
        # Build nabla
        nc = self.X_hat.shape[1]
        W_c = theta_full_c[0:nc]
        H_c = theta_full_c[nc:-1]
        b_c = theta_full_c[-1:]
        nabla_c = self.nabla(W_c, H_c, b_c)
        L_c, dL_c, dL2_c = self.logexpLambda(nabla_c)

        # Hessian of considered Parents
        H_c = self.logexpObservedFisherInfo(c,PAc_ef, L_c, dL_c, dL2_c)
        I_c = np.linalg.pinv(H_c)
        var_c = np.diagonal(I_c)

        p_vals_c = scipy.stats.chi2.sf((theta_full_c[np.append(PAc_ef,True)] ** 2)/var_c, df=1)

        # Save in full variance vector
        var_full_c = np.zeros([nr])
        var_full_c[:] = np.inf
        if nr_ef > 0:
            var_full_c[PAc_ef] = var_c[0:-1]
        var_full_c = np.concatenate([var_full_c,var_c[-1:]],axis = 0)  #bias

        # Save in full p_vals vector
        p_vals_full_c = np.ones([nr])
        if nr_ef > 0:
            p_vals_full_c[PAc_ef] = p_vals_c[0:-1]
        p_vals_full_c = np.concatenate([p_vals_full_c,p_vals_c[-1:]],axis = 0)  #bias

        #Get BIC + Likelihood :)
        L_c = -1*self.mapLikelihood(X_c, np.concatenate([R_hat, np.ones([R_hat.shape[0], 1])], axis=1), theta_full_c, self.kappa)
        BIC_c = np.log(R_hat.shape[0])*(np.sum(PA_c == True)+1) - 2*L_c
        return theta_full_c,p_vals_full_c,var_full_c,L_c,BIC_c

    def elasticForwardSelection(self, c):

        #initialize set of active regressors to empty set
        r_prime = np.zeros([self.n_r]).astype('bool')
        theta = None

        #make index_masks once
        split_samples = int(self.n_samples*self.nu)
        index_masks = np.zeros([self.n_splits, self.n_samples], dtype=np.bool)
        for j in np.arange(self.n_splits):
            aux = np.random.choice(np.arange(self.n_samples),split_samples, replace=False).astype('int')
            index_masks[j,aux]=True

        index_masks = np.zeros([self.n_splits,self.n_samples], dtype=np.bool)


        while True:

            theta, likelihood, fisherInformation, pvals, BIC = self.evaluateRegressors(c, r_prime, theta_ini=theta, index_samples=None)
            best_candidates = self.forwardModelProposal(c=c,PA_c=r_prime, index_masks=index_masks)

            print(best_candidates)

            if len(best_candidates)==0:
                self.theta =theta
                return theta, likelihood, fisherInformation, pvals, BIC

            n = len(best_candidates)
            BIC_best = BIC
            r_best = r_prime
            while True:
                r_ddag = np.array(r_best)
                r_ddag[best_candidates[:n]]=True
                print(r_best.sum(), r_ddag.sum())

                #evaluate new regressor set
                theta_ddag, likelihood_ddag, fisherInformation_ddag, pvals_ddag, BIC_ddag = self.evaluateRegressors(
                                        c, r_ddag, theta_ini=theta,
                                        index_samples=None)

                if (np.max(pvals_ddag)<= self.gamma) and (BIC_ddag<= BIC_best): #found better regressor subset
                    r_best = r_ddag
                    BIC_best = BIC_ddag

                if (BIC_ddag>= BIC_best) and (r_best!= r_prime): #Already got a better set in the descending sequence, update and exit loop
                    r_prime = r_best
                    break

                n -=1








