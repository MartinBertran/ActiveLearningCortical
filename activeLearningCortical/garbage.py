def get_stimuli_random2(ninput, nsamples, dur, p, down=0):
    ix = np.random.choice(ninput, int(nsamples / dur), p=p)
    stimuli = np.zeros([int(nsamples / dur) * dur, ninput])
    for i in np.arange(ninput):
        ix_i = np.zeros(ix.shape)
        ix_i[ix == i] = 1
        stimuli[::dur, i] = ix_i
        stimuli[:, i] = np.convolve(stimuli[:, i], np.ones(dur - down), mode='same')
    return stimuli[0:-dur, :], ix


def input_impact(W_est, nsamples, dw, dh, k=500, op='bias', dur=4):
    win_w = -dw[1] + dw[0] + 1
    win_h = -dh[1] + dh[0] + 1
    ncells = W_est.shape[1]
    W_kernel, H_kernel = get_kernels_WH(W_est[0:ncells, :] / win_w, W_est[ncells:-1, :] / win_h, dw, dh)
    b_kernel = W_est[-1:, :]
    H_impact = np.zeros([H_kernel.shape[1], H_kernel.shape[2] + H_kernel.shape[1]])

    # Only bias behaviour
    stimuli = np.zeros([nsamples, H_kernel.shape[1]])
    if op == 'bias':
        output_b, lambdaout_b, dlambdaout_b, s_b = simul_lambda_prediction(W_kernel, H_kernel, b_kernel, stimuli,
                                                                           nsamples)
    else:
        down = 0
        p = np.ones(stimuli.shape[1]) / stimuli.shape[1]
        stimuli, ix = get_stimuli_random2(p.shape[0], nsamples, dur, down=down, p=p)
        #         stimuli,ix = get_stimuli_random(stimuli.shape[1],nsamples,dur)
        output_b, lambdaout_b, dlambdaout_b, s_b = simul_lambda_prediction(W_kernel, H_kernel, b_kernel, stimuli,
                                                                           nsamples)

    for i in np.arange(H_kernel.shape[1]):
        #         dt = 100
        #         ix = np.arange(nsamples)
        #         stimuli = np.zeros([nsamples,H_kernel.shape[1]])
        #         stimuli[ix[5::2*dur],i] = 1
        #         stimuli[:,i]  = np.convolve(stimuli[:,i], np.ones(dur), mode='same')
        Kp = 8
        down = 0
        p = np.ones(stimuli.shape[1]) / ((Kp + 1) * stimuli.shape[1])
        p[i] = (1 + Kp * stimuli.shape[1]) / ((Kp + 1) * stimuli.shape[1])
        stimuli, ix = get_stimuli_random2(p.shape[0], nsamples, dur, down=down, p=p)
        #         print(i)
        #         print(np.sum(p))
        #         print(p)

        # Stimuli behaviour
        output_s1, lambdaout_s1, dlambdaout_s1, s1 = simul_lambda_prediction(W_kernel, H_kernel, b_kernel, stimuli,
                                                                             nsamples)
        #         impact_i = np.mean(lambdaout_s1,axis = 0)/np.mean(lambdaout_b,axis = 0) - 1
        impact_i = np.mean(lambdaout_s1, axis=0) / np.mean(lambdaout_b, axis=0)
        if np.min(impact_i < 0):
            print('#### EXTREME DANGEROUS ZONE HELP MOBBY ######')
        # print(impact_i)
        H_impact[i, 0:ncells] = impact_i
        #         H_impact[i,ncells+i] = np.sum(stimuli[:,i])/(stimuli.shape[0]/H_kernel.shape[1])-1
        H_impact[i, ncells:] = p * stimuli.shape[1]
    return H_impact


def input_scores_2(edges_scores, input_impact):
    I = np.array(input_impact)
    #     I[I<0] = 0
    #     I[I == 0] = 1e-20
    #     I = 1/I
    I_matrix_sc = np.zeros(I.shape)
    I_scores = np.zeros([I.shape[0]])

    for i in np.arange(I.shape[0]):
        for c in np.arange(I.shape[1]):
            #             print(I[i,c])
            #             print(edges_scores[c,:])
            if I[i, c] > 0:
                edges_scores_c = edges_scores[c, :]
                #                 value = np.sum(edges_scores[c,:]*I[i,c])
                value = np.mean(edges_scores_c[edges_scores_c != 0] * I[i, c])
            else:
                value = 0
            I_matrix_sc[i, c] = value
        # print('Input ', i)
        #         print(I_matrix_sc[i,:])
        I_scores[i] = np.sum(I_matrix_sc[i, I_matrix_sc[i, :] != 0])
    # print(np.sum(I_matrix_sc[i,I_matrix_sc[i,:] != 0 ]))
    return I_matrix_sc, I_scores


def AL_code_v2(stimuli, output, W_est, H_kernel, W_kernel, b_kernel, dw, dh, dur=4, down=0, niter=1, nsamples=1000,
               K=500, th_rank=0, th_fish=0.01, tol=0.01, op=1, indexes_input=None):
    W_est_dic = {}
    BIC_est_dic = {}
    F_max_dic = {}
    F_est_dic = {}
    stimul_sel = {}
    I_scores_dic = {}
    I_matrix_scores_dic = {}

    W_est_dic[0] = W_est
    #     BIC_est_dic[0] = BIC

    W_adj = np.sum(W_kernel, axis=0)
    H_adj = np.sum(H_kernel, axis=0)
    W_GT = np.concatenate([W_adj, H_adj], axis=0)
    W_GT[W_GT != 0] = 1

    ## BIC SCORE ##
    if op <= 2:
        BIC_score, Fisher_score = get_add_scores(output, stimuli, dw, dh, W_est_dic[0])
        F_max_dic[0] = Fisher_score
        BIC_est_dic[0] = BIC_score

    for i in np.arange(niter):
        print('### ITER : ', i)

        if op <= 2:
            # INPUT IMPACT
            HI = input_impact(W_est_dic[i], nsamples, dw, dh, k=K, op='random')
            HI = np.array(HI)
            #             plt.plot(HI[0,0:18],'*--')
            #             plt.show()
            if np.min(HI < 0):
                print('## TROUBLE INPUT IMPACT!!!##')
            # HI[HI < 0] = 0

            # BIC SCORE
            BIC = np.array(BIC_est_dic[i])
            BIC[BIC >= 0] = 0
            l_scores, Isc = input_scores_2(BIC, HI)
            I_scores_dic[i] = Isc
            I_matrix_scores_dic[i] = l_scores

            #             print('SCORES :: ',Isc)
            # SELECT 2 BEST INPUTS
            indexes = np.arange(Isc.shape[0])
            indexes = indexes[Isc < 0]
        # indexes = indexes[np.argsort(Isc[Isc<0])]

        if op == 1:
            #             indx_s = indexes[-3:]
            indx_s = indexes
            stimul_sel[i + 1] = indx_s
            #             print(indx_s)

            # Generate Stimuli -.-
            stimuli_g, ix = get_stimuli_random(indx_s.shape[0], nsamples, dur)
        else:
            if op == 2:
                if np.sum(Isc < 0) > 0:

                    indx_s = indexes
                    I_zscores = (Isc[indexes] - np.mean(Isc[indexes])) / np.std(Isc[indexes])
                    I_zscores = np.minimum(I_zscores, 2)
                    I_zscores = np.maximum(I_zscores, -2)
                    #                     I_zscores[I_zscores>0] = 0
                    Ki = 4
                    beta = 1 / 4
                    alfa = 1
                    #                     p1 = np.abs(I_zscores)
                    p1 = np.exp(-1 * I_zscores)
                    #                     p1 = (p1**alfa)/np.sum(p1**alfa)
                    #                     p2 = np.ones([I_zscores.shape[0]])/(I_zscores.shape[0]*Ki)
                    #                     p = p1+p2
                    p = np.array(p1)
                    p /= np.sum(p)
                    #                     p = p*(1-beta) + beta/p.shape[0]


                    plt.title('Iteration ' + str(i))
                    plt.plot(indexes, p, '*')
                    plt.show()

                # Ki = 3
                #                     indx_s = indexes
                #                     p1 = np.abs(Isc[indexes])
                #                     p1 = (p1**2)/np.sum(p1**2)
                #                     p2 = np.ones([indx_s.shape[0]])/(indx_s.shape[0]*Ki)
                #                     p = p1+p2
                #                     p /= np.sum(p)



                else:
                    indexes = np.arange(Isc.shape[0])
                    p = np.ones([indexes.shape[0]]) / indexes.shape[0]

                stimul_sel[i + 1] = indx_s
                print(indx_s)
                print(p)

                # Generate Stimuli -.
                stimuli_g, ix = get_stimuli_random2(indx_s.shape[0], nsamples, dur, down=down, p=p)
                print('DANGEROUS!!!!!')
                print(p)
                print(np.mean(stimuli_g, axis=0))
            # stimuli_g,ix = get_stimuli_random(indx_s.shape[0],nsamples,dur)
            #                 dt = dur*2
            #                 indx_s = indexes[-1:]
            #                 print(indx_s)
            #                 ix = np.arange(nsamples)
            #                 stimuli_g = np.zeros([nsamples])
            #                 stimuli_g[ix[5::dt]] = 1
            #                 stimuli_g = np.convolve(stimuli_g, np.ones(dur), mode='same')
            #                 stimuli_g = stimuli_g[:,np.newaxis]
            else:
                indx_s = indexes_input
                stimul_sel[i + 1] = indx_s
                #                 print(indx_s)
                p = np.ones([indx_s.shape[0]]) / indx_s.shape[0]
                stimuli_g, ix = get_stimuli_random2(indx_s.shape[0], nsamples, dur, down=down, p=p)

                # Generate Stimuli -.-
                #                 stimuli_g,ix = get_stimuli_random(indx_s.shape[0],nsamples,dur)

        ### SIMULATION ###

        # simul net behaviour to new stimuli
        stimuli_n = np.zeros([stimuli_g.shape[0], H_kernel.shape[1]])
        #         print(stimuli_n.shape)
        print(indx_s)
        #         print(stimuli_g.shape)
        stimuli_n[:, indx_s] = stimuli_g

        print(np.mean(stimuli_n, axis=0))
        plt.title('Iteration ' + str(i))
        plt.plot(np.mean(stimuli_n, axis=0), '*')
        plt.show()
        # get kernels
        win_w = -dw[1] + dw[0] + 1
        win_h = -dh[1] + dh[0] + 1
        ncells = W_est.shape[1]
        #         W_kernel,H_kernel = get_kernels_WH(W_est[0:ncells,:]/win_w,W_est[ncells:-1,:]/win_h,dw,dh)
        #         b_kernel = W_est[-1:,:]
        #         print(W_kernel.shape,H_kernel.shape,b_kernel.shape)

        # Simuli
        output_ni, lambdaout_ni, dlambdaout_ni, stimuli_ni = nrs.generate_spikes(W_kernel, H_kernel, b_kernel,
                                                                                 stimuli_n, link='Logexp', k=K)
        output = np.concatenate([output, output_ni], axis=0)
        stimuli = np.concatenate([stimuli, stimuli_ni], axis=0)

        # INFERENCE
        # Get preprocessed Data
        input_array, past_spikes_array, spikes_array, input_i = nwidf.get_spikes_processedM(output, stimuli, dc_p=dw[1],
                                                                                            dc_f=dw[0], di_p=dh[1],
                                                                                            di_f=dw[0])

        # Learning the network...
        W_est_dic[i + 1], F_est_dic[i + 1], F_max_dic[i + 1], BIC_est_dic[
            i + 1], indx_est = network_learning_BIC_el_forward_CROSS(input_array, past_spikes_array, spikes_array,
                                                                     th_fish=th_fish, th_rank=th_rank, tol=tol,
                                                                     regul=True, beta=0)

        # check TP,FP,FN
        W_i2 = np.array(W_est_dic[i + 1])
        W_i2[W_i2 != 0] = 1
        WF = 2 * W_i2[:-1, :] + W_GT
        print('TP : ' + str(np.sum(WF == 3)) + '; FP : ' + str(np.sum(WF == 2)) + '; FN : ' + str(np.sum(WF == 1)))

        if op <= 2:
            ## BIC SCORE ##
            BIC_score, Fisher_score = get_add_scores(output, stimuli, dw, dh, W_est_dic[i + 1])
            F_max_dic[i + 1] = Fisher_score
            BIC_est_dic[i + 1] = BIC_score

        performance_print(W_GT, W_est_dic[i + 1], ncells, 'Options')

        ## SAVES
        save_data = {}
        save_data['stimuli'] = stimuli
        save_data['output'] = output
        save_data['W_est'] = W_est
        save_data['W_est_dic'] = W_est_dic
        save_data['F_est_dic'] = F_est_dic
        save_data['F_max_dic'] = F_max_dic
        save_data['BIC_est_dic'] = BIC_est_dic
        save_data['stimul_sel'] = stimul_sel

        file_uid = 'our_spikes_database/simulSW_AL_CHECKPOINT.pkl'
        with open(file_uid, 'wb') as f:
            pickle.dump(save_data, f)

    return W_est_dic, BIC_est_dic, F_est_dic, stimul_sel, stimuli, output, I_scores_dic, I_matrix_scores_dic


def get_add_scores(output0, stimuli0, dw, dh, W_est, n_sets=5, p_set=0.7):
    input_array, past_spikes_array, spikes_array, input_i = nwidf.get_spikes_processedM(output0, stimuli0, dc_p=dw[1],
                                                                                        dc_f=dw[0], di_p=dh[1],
                                                                                        di_f=dw[0])
    ncells = spikes_array.shape[1]
    nstimuli = input_array.shape[1]
    BIC_score = np.zeros([ncells + nstimuli + 1, ncells])
    Fisher_score = np.zeros([ncells + nstimuli + 1, ncells, 2])
    W_out_score = np.zeros([ncells + nstimuli + 1, ncells])
    #     print(BIC_score.shape)
    for cell in np.arange(ncells):
        y = spikes_array[:, cell]
        x_past = np.array(past_spikes_array)
        i_past = np.array(input_array)
        b_ones = np.ones([i_past.shape[0]])
        inputs = np.concatenate([x_past, i_past, b_ones[:, np.newaxis]], axis=1)
        #         print(W_est.shape)
        W_est_cell = np.array(W_est[:, cell])
        indx_reg = np.arange(W_est.shape[0])
        indx_reg = indx_reg[W_est_cell != 0]
        #         print(indx_reg)

        # shuffle
        s_set = int(y.shape[0] * p_set)
        index_samples_array = np.zeros([n_sets, s_set])
        for n in np.arange(n_sets):
            index_samples_array[n, :] = np.random.choice(y.shape[0], size=s_set, replace=False).astype('int')

        BIC_score_sets = np.zeros([n_sets, inputs.shape[1]])
        LK_score_sets = np.zeros([n_sets, inputs.shape[1]])
        FISH_score_sets = np.zeros([n_sets, inputs.shape[1], 2])

        for n in np.arange(n_sets):
            index_samples = index_samples_array[n, :].astype('int')
            # print(index_samples)
            W_out1, F_out1, L_out1, BIC_out1, F_max_out1 = nwidf.eval_adding_scores(y[index_samples],
                                                                                    inputs[index_samples, :], indx_reg,
                                                                                    regul=True, k=500,
                                                                                    beta=0)
            BIC_score_sets[n, :] = np.array(BIC_out1)
            FISH_score_sets[n, :, :] = np.array(F_max_out1[:, :])
            # Likelihood
            LT = -2 * np.sum(np.tile(y[index_samples], (L_out1.shape[0], 1)) * np.log(L_out1 + 1e-20) - L_out1, axis=1)
            LT = LT - LT[indx_reg[0]]
            LK_score_sets[n, :] = LT

        W_out, F_outf, L_outf, BIC_outf, F_max_outf = nwidf.eval_adding_scores(y, inputs, indx_reg, regul=True, k=500,
                                                                               beta=0)
        # print(FISH_score_sets)
        BIC_out = np.nanmedian(
            BIC_score_sets - np.tile(BIC_score_sets[:, indx_reg[0]], (BIC_score_sets.shape[1], 1)).transpose(),
            axis=0) - np.log(index_samples.shape[0])
        BIC_out[indx_reg] += np.log(index_samples.shape[0])

        LT_out = np.nanmedian(LK_score_sets, axis=0)
        #         np.array(BIC_out[indx_rem] - BIC_out[indx_reg[0]])
        # BIC_out = np.maximum(BIC_out, BIC_outf)
        BIC_out2 = BIC_outf - BIC_outf[indx_reg[0]] - np.log(y.shape[0])
        BIC_out2[indx_reg] += np.log(y.shape[0])

        #          -2*np.sum(y*np.log(L+1e-20) - L)
        LT = -2 * np.sum(np.tile(y, (L_outf.shape[0], 1)) * np.log(L_outf + 1e-20) - L_outf, axis=1)
        LT_out2 = LT - LT[indx_reg[0]]

        #         print('######  LT SCORES INDEXES ######')
        #         print(LT.shape)
        # #         print(LT)
        #         print(BIC_out2)
        #         print(LT_out2)


        #         print('######  BIC SCORES INDEXES ######')
        #         print(indx_reg)
        #         print(BIC_out)
        #         print(LT_out)

        #         print('###### MAXIMOS WUT (???) ######')
        #         print(np.max(BIC_out), np.argmax(BIC_out))
        #         print(np.max(BIC_out2), np.argmax(BIC_out2))
        #         print(np.max(LT), np.argmax(LT))

        BIC_out = np.maximum(BIC_out, BIC_out2)
        LT_out = np.maximum(LT_out, LT_out2)
        LT_out[LT_out >= 0] = -1 * 1e-20
        LT_out[indx_reg] = 0
        #         print('############')
        #         print(np.max(BIC_out), np.argmax(BIC_out))
        #         print(np.max(LT_out), np.argmax(LT_out))

        #         LT_out = LT_out2
        #         print(BIC_out)


        #         print(BIC_out[indx_reg])

        F_max_out = np.nanmedian(FISH_score_sets, axis=0)
        F_max_out = np.maximum(F_max_outf, F_max_out)
        BIC_score[:, cell] = LT_out
        Fisher_score[:, cell, :] = F_max_out
    return BIC_score, Fisher_score