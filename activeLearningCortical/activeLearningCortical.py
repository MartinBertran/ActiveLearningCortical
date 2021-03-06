import numpy as np
from scipy.optimize import minimize
import scipy
import scipy.io
import scipy.stats
import scipy.special
from .utils import *
import logging
import dill


class ClassModel():
    '''
    Missing a really nice descriptor with .. math:: environment

    Parameters
    ----------


    Reference
    ---------
    ref to paper
    '''

    def __init__(self,X, I, D_c_l=-2, D_c_u=-5, D_s_l=-2, D_s_u=-5,
                 k=1,kappa = 10,  gamma=1e-3,
                 nu=1, n_splits=1, beta=1/4,
                 sensitivity_th=0.1, verbose= False, logfile=None, checkpoint=None,
                 seed=None, additional_restrictions=None, approx_gamma=None, AL_th=2):

        self.verbose = verbose
        self.logfile = logfile
        # if logfile is not None:
        #     logging.basicConfig(filename=logfile, level=logging.DEBUG)


        self.D_c_l = D_c_l
        self.D_c_u = D_c_u
        self.D_s_l = D_s_l
        self.D_s_u = D_s_u
        self.k = k
        self.kappa = kappa
        self.gamma = gamma
        self.nu = nu
        self.n_splits = n_splits
        self.beta = beta

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

        self.W = np.zeros([self.n_c, self.n_c])
        self.W_pval = np.zeros([self.n_c, self.n_c])
        self.W_fisher = np.zeros([self.n_c, self.n_c])

        self.H = np.zeros([self.n_s, self.n_c])
        self.H_pval = np.zeros([self.n_s, self.n_c])
        self.H_fisher = np.zeros([self.n_s, self.n_c])

        self.b = np.zeros([1, self.n_c])
        self.b_pval = np.zeros([1, self.n_c])
        self.b_fisher = np.zeros([1, self.n_c])

        self.BIC_model = np.zeros([1, self.n_c])
        self.likelihood_model = np.zeros([1, self.n_c])
        self.checkpoint=checkpoint

        self.sensitivity_th = sensitivity_th
        self.seed = seed
        self.additional_restrictions=additional_restrictions
        if approx_gamma is not None:
            self.approx_gamma=approx_gamma
        else:
            self.approx_gamma=self.gamma
        self.AL_th=AL_th

    def verbose_print(self,*args):
        if self.verbose:
            print(*args)
        if self.logfile is not None:
            # logging.info(*args[1:])
            with open(self.logfile, 'a+') as f:
                print(*args, file=f)


    def boxcar(self,X,I):
        # windows
        win_c = self.D_c_u - self.D_c_l + 1
        win_i = self.D_s_u - self.D_s_l + 1

        # half lengths (window size virtually upscaled to nearest odd value
        hl_i = int(np.floor((win_i + 1) / 2))
        hl_c = int(np.floor((win_c + 1) / 2))

        # kernels
        ker_i = np.zeros(2 * hl_i + 1)
        ker_i[:win_i] = 1
        ker_c = np.zeros(2 * hl_c + 1)
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

        R_hat = np.append(X_hat, I_hat, axis=1)

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
        # d_lambda_d_nabla = 1 - (1 / (1 + np.exp(nabla * kappa)))
        d_lambda_d_nabla = 1 - scipy.special.expit(-nabla*kappa)
        d_nabla_d_theta = R_c
        d_lambda_d_theta = (d_like_d_lambda * d_lambda_d_nabla)[:, np.newaxis] * d_nabla_d_theta
        d_lambda_d_theta = np.sum(d_lambda_d_theta, axis=0)
        return -d_lambda_d_theta

    def computeMap(self, c, PA_c,theta_ini=None, index_mask=None):

        #build regressors, intial values, and select target variable
        R_c = self.R_hat[:, PA_c]
        X_c = self.X[:, c]

        if index_mask is not None:
            R_c = R_c[index_mask,:]
            X_c = X_c[index_mask]

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
        # options['tol'] = 1e-14
        # theta_MAP_local = minimize(f, theta_ini_local, jac=df, tol = 1e-14, options=options)
        theta_MAP_local = minimize(f, theta_ini_local, jac=df, options=options)
        theta_MAP_local=theta_MAP_local.x

        theta_MAP  = np.zeros(theta_ini.shape)
        theta_MAP[PA_c_with_bias] = theta_MAP_local
        return theta_MAP

    def forwardModelProposal(self,c,PA_c, index_masks, BIC_base=np.inf, exclusion_list=None):
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

            if exclusion_list is not None:
                if exclusion_list[j]:
                    # self.verbose_print( 'regressor {} was excluded'.format(j))
                    continue

            PA_c_r = np.array(PA_c)
            PA_c_r[j] = True
            PA_c_r_with_bias = np.append(PA_c_r,[True]).astype('bool')

            # evaluate results for every split
            for split in np.arange(self.n_splits):
                split_idx = index_masks[split,:]

                theta_map, pvals, fisher, Likelihood, BIC = self.evaluateRegressors(c, PA_c_r,index_samples=split_idx)
                max_pval = np.max(pvals[PA_c_r_with_bias])


                BIC_split[split,j] = BIC
                pval_split[split,j] = max_pval

            # evaluate results over full dataset
            theta_map, pvals, fisher, Likelihood, BIC = self.evaluateRegressors(c, PA_c_r, index_samples=None)
            max_pval = np.max(pvals[PA_c_r_with_bias])
            BIC_full[j]=BIC
            pval_full[j]=max_pval

        #Finally, compute score

        pval_score = np.maximum(np.median(pval_split,axis=0), pval_full)
        BIC_score = np.maximum(np.median(BIC_split,axis=0), BIC_full)


        # get index of regressors that simultaneoulsy satisfy BIC_score < 0 and pval_score < gamma
        index_satisfactory = np.where(np.logical_and(BIC_score<BIC_base,pval_score<self.gamma))[0]

        #sort remaining regressors in ascending order of BIC score, and select the first k regressors
        sorted_remainder =  np.argsort(BIC_score[index_satisfactory])[:self.k]

        best_candidates = index_satisfactory[sorted_remainder]

        return best_candidates, pval_score, BIC_score

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

        if index_samples is not None:
            R_hat = R_hat[index_samples, :]
            X_c = X_c[index_samples]

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

        exclusion_list = self.getApproximatePvals(c)[0]

        if self.additional_restrictions is not None:
            additional_exclusion = self.additional_restrictions[:,c]
            exclusion_list = np.logical_or(exclusion_list,additional_exclusion)

        self.verbose_print('excluded regressors', np.where(exclusion_list)[0])

        self.verbose_print('####### Elastic Forward cell : ', c, '  ########')
        while True:
            self.verbose_print('starting primary loop', np.where(r_prime)[0])

            theta, pvals, fisherInformation, likelihood, BIC = self.evaluateRegressors(c, r_prime,
                                                                                       theta_ini=theta,
                                                                                       index_samples=None)
            best_candidates,_,_ = self.forwardModelProposal(c=c,PA_c=r_prime,
                                                            index_masks=index_masks, BIC_base=BIC,
                                                            exclusion_list=exclusion_list)

            self.verbose_print("best candidates elastic forward selection",best_candidates)

            if len(best_candidates)==0:
                self.verbose_print("no suitable candidate regressors found")
                # self.theta =theta
                return theta, likelihood, fisherInformation, pvals, BIC

            n = len(best_candidates)
            BIC_best = BIC
            r_best = r_prime
            while True:
                r_ddag = np.array(r_best)
                r_ddag[best_candidates[:n]]=True
                r_ddag_with_bias = np.append(r_ddag,[True]).astype('bool')

                # self.verbose_print("best regressor number so far",r_best.sum(), "trying model with n parameters", r_ddag.sum())

                #evaluate new regressor set
                theta_ddag, pvals_ddag, fisherInformation_ddag, likelihood_ddag, BIC_ddag = self.evaluateRegressors(
                                        c, r_ddag, theta_ini=theta,
                                        index_samples=None)

                # self.verbose_print('r_best: ', r_best)
                self.verbose_print('PVALS: ' , pvals_ddag[r_ddag_with_bias])
                # self.verbose_print('BIC_best: ', BIC_best)

                if (np.max(pvals_ddag[r_ddag_with_bias])<= self.gamma) and (BIC_ddag<= BIC_best): #found better regressor subset
                    r_best = r_ddag
                    BIC_best = BIC_ddag
                    self.verbose_print('Found a better regressor set with BIC ', BIC_best)

                if (BIC_ddag>= BIC_best) and not((r_best == r_prime).all()): #Already got a better set in the descending sequence, update and exit loop
                    r_prime = r_best
                    self.verbose_print('Already got a better set in the descending sequence, update and exit loop')
                    break

                n -=1
                if n==0:
                    self.verbose_print("tried all candidates and none was satisfactory")
                    break

    def updateModel(self):
        self.saveCheckpoint()
        np.random.seed(self.seed)
        for c in np.arange(self.n_c):
            theta, likelihood, fisherInformation, pvals, BIC = self.elasticForwardSelection(c)

            # W, H, b MLE
            self.W[:,c] = theta[0:self.n_c]
            self.H[:,c] = theta[self.n_c:-1]
            self.b[:,c] = theta[-1:]

            # W, H, b pvals
            self.W_pval[:,c] = pvals[0:self.n_c]
            self.H_pval[:,c] = pvals[self.n_c:-1]
            self.b_pval[:,c] = pvals[-1:]

            # W, H, b fisher info
            self.W_fisher[:,c] = fisherInformation[0:self.n_c]
            self.H_fisher[:,c] = fisherInformation[self.n_c:-1]
            self.b_fisher[:,c] = fisherInformation[-1:]

            #Model Likelihood and BIC
            self.BIC_model[:,c] = BIC
            self.likelihood_model[:,c] = likelihood

        #also update the W and H kernels with the trailing time dimension
        dw = [self.D_c_u, self.D_c_l]
        dh = [self.D_s_u, self.D_s_l]
        W_kernel, H_kernel = get_kernels_WH(self.W, self.H, dw, dh)

        self.W_kernel = W_kernel
        self.H_kernel = H_kernel
        self.saveCheckpoint()

    def getExpectedSpikingRates(self, p, N=2000,duration=4):

        #sample stimuli according to p for N samples, all stimulations persist for duration

        sampled_I = getSampledStimuli(N, duration, p)

        # ix = np.random.choice(self.n_s, int(N / duration), p=p)
        # sampled_I = np.zeros([int(N / duration) * duration, self.n_s])
        # duration_kernel = np.ones(duration)
        # for i in np.arange(self.n_s):
        #     ix_i = np.zeros(ix.shape)
        #     ix_i[ix == i] = 1
        #     sampled_I[::duration, i] = ix_i
        #     sampled_I[:, i] = np.convolve(sampled_I[:, i], duration_kernel, mode='same')
        # sampled_I = sampled_I[0:-duration,:]


        # generate expected spiking traces
        expected_X, expected_I = generate_spikes(self.W_kernel, self.H_kernel, self.b, sampled_I, kappa=self.kappa, expected=True)

        # return only the time average of the traces

        return expected_X.mean(0), expected_I.mean(0)

    def computeStimuliImpact(self):

        #set up variables
        impact_matrix_X = np.zeros([self.n_s, self.n_c])
        impact_matrix_I = np.zeros([self.n_s, self.n_s])
        N=4000
        duration=4


        #get base rates
        p_base= np.ones(self.n_s)/self.n_s

        rate_X_base, rate_I_base = self.getExpectedSpikingRates(p_base, N=N, duration=duration)

        #get rates for every stimuli and compute impact ratio
        for i in np.arange(self.n_s):

            p = np.ones(self.n_s)/self.n_s* self.beta
            p[i] += (1-self.beta)
            rate_X_i, rate_I_i = self.getExpectedSpikingRates(p, N=N, duration=duration)

            impact_matrix_X[i,:] = rate_X_i/rate_X_base

            impact_matrix_I[i, :] = p / p_base

        return impact_matrix_X, impact_matrix_I

    def computeLogLikelihoodDifference(self):

        # make index_masks once
        split_samples = int(self.n_samples * self.nu)
        index_masks = np.zeros([self.n_splits, self.n_samples], dtype=np.bool)
        for j in np.arange(self.n_splits):
            aux = np.random.choice(np.arange(self.n_samples), split_samples, replace=False).astype('int')
            index_masks[j, aux] = True

        # setting up the variables
        likelihood_Delta = np.zeros([self.n_c, self.n_r, self.n_splits])

        theta = np.concatenate((np.array(self.W),np.array(self.H),self.b),axis=0)
        PA = (theta[:-1,:])!=0

        for c in np.arange(self.n_c): #for every target cell
            # get PA_c, and theta_c initialization
            theta_c = theta[:,c]
            PA_c = PA[:,c]
            # self.verbose_print("computing log likelihood difference for all potential additional parents of cell {:d}".format(c))
            for split in np.arange(self.n_splits): #for every split
                #do current split and current model
                current_split = index_masks[split,:]
                _, _, _, likelihood_base, _ = self.evaluateRegressors( c, PA_c, theta_ini=theta_c, index_samples=current_split)

                for r in np.arange(self.n_r): # for every possible regressor
                    if PA_c[r]: #if regressor os already included, do nothing
                        continue
                    PA_c_plus_r = np.array(PA_c)
                    PA_c_plus_r[r]=True

                    _, _, _, likelihood_r, _ = self.evaluateRegressors(c, PA_c_plus_r, theta_ini=theta_c,
                                                                          index_samples=current_split)
                    likelihood_Delta[c,r,split] = likelihood_r - likelihood_base


        likelihood_Delta = np.median(likelihood_Delta,axis=2) # compute median difference over splits

        likelihood_score = np.zeros([self.n_r])
        for r in np.arange(self.n_r): # compute mean over potential child cells
            likelihood_score[r] = np.mean(likelihood_Delta[np.logical_not(PA[r,:]),r])
        return likelihood_score

    def getActiveLearningDistribution(self):
        self.saveCheckpoint()
        #get logLikelihood Difference score over all regressors
        likelihood_score = self.computeLogLikelihoodDifference()

        #compute stimuli impact over all regressors
        impact_matrix_X, impact_matrix_I = self.computeStimuliImpact()

        # compute raw score over all stimuli
        impact_matrix = np.append(impact_matrix_X, impact_matrix_I, axis=1)
        score = np.dot(impact_matrix, likelihood_score)

        # compute truncated z-score of the score vector
        score_mean = np.mean(score)
        score_std = np.std(score)
        z_score_truncated = np.maximum(np.minimum((score - score_mean) / score_std, self.AL_th), -self.AL_th)

        # finally, translate truncated z-scores into a probability distribution using the softmax function
        p_AL = np.exp(z_score_truncated)
        p_AL /= p_AL.sum()

        self.p_AL = p_AL
        self.saveCheckpoint()
        return p_AL

    def saveCheckpoint(self):
        if self.checkpoint is not None:
            with open(self.checkpoint, 'wb') as f:
                f.write(dill.dumps(self))

    def loadCheckpoint(self):
        if self.checkpoint is not None:
            with open(self.checkpoint, 'rb') as f:
                oldModel = dill.load(f)
                self.__dict__ = oldModel.__dict__.copy()

    def getApproximatePvals(self,c):

        #set up variables
        exclusion_list = np.ones([self.n_r]).astype('bool')
        approximate_pvals = np.ones([self.n_r])
        sensitivity = np.ones([self.n_r])
        z2_vector = np.ones([self.n_r])


        #loop through all possible regressors
        for j in np.arange(self.n_r):
            reg_active = self.R_hat[:,j]!=0
            #get conditional spike counts and influence lengths
            N_pi = np.sum(reg_active)
            N_mi = np.sum(np.logical_not(reg_active))
            if (N_pi*N_mi)==0:
                continue

            s_pi = np.sum(self.X[reg_active,c])
            s_mi = np.sum(self.X[np.logical_not(reg_active),c])

            lam_pi = s_pi / N_pi
            lam_mi = s_mi / N_mi

            if (s_pi*s_mi)==0:
                continue

            #sensitivity sieve
            Df_f = np.sqrt(s_mi**2+s_pi**2)/(s_pi*s_mi)
            sensitivity[j]=Df_f

            if Df_f > self.sensitivity_th:
                continue

            # get z-score bound
            z2 = self.getApproximateZ2(self.kappa,lam_pi,lam_mi,N_pi)
            z2_vector[j]=z2

            p_val = scipy.stats.chi2.sf(z2, df=1)
            approximate_pvals[j] = p_val
            exclusion_list[j] = p_val> (self.approx_gamma)




        return exclusion_list, approximate_pvals, sensitivity, z2_vector

    def getApproximateZ2(self,k, L1, L0, N=1):
        C1 = np.exp(k * L1) - 1
        C0 = np.exp(k * L0) - 1
        term1 = (np.log(C1 / C0)) ** 2
        term2 = (C1 ** 2) / ((1 + C1) ** 2)
        term3 = 1 / L1
        Z2 = term1 * term2 * term3 * N / (k ** 2)
        return Z2