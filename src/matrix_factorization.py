#from collections import defaultdict, Counter
#import urllib

import numpy as np
import pandas as pd 
import nimfa
import mdl
from progressbar import ProgressBar
import time 


def min_max_scaler(target_list):
    min_target_list = min(target_list)
    max_target_list = max(target_list)

    result = [(item -min_target_list)/(max_target_list - min_target_list) for item in target_list]

    return result 

def vanilla_mf(feature_matrix,rank,max_iter = 1000):
    
    '''
    vanilla matrix factorization, without any constraints
    '''
    start_time = time.time()
    lsnmf = nimfa.Lsnmf(feature_matrix, rank=rank, max_iter=max_iter)
    lsnmf_fit = lsnmf()
    W = np.asarray(lsnmf_fit.basis())
    H = np.asarray(lsnmf_fit.coef())
    end_time = time.time()
    
    print "***---***---***---***---***"
    print("rank is %d" % rank)
    print('Rss: %5.4f' % lsnmf_fit.fit.rss())
    print('Evar: %5.4f' % lsnmf_fit.fit.evar())
    print('K-L divergence: %5.4f' % lsnmf_fit.distance(metric='kl'))
    print('Sparseness, W: %5.4f, H: %5.4f' % lsnmf_fit.fit.sparseness())
    print("time elasped: %5.4f s" % (end_time - start_time))

    return (W,H)

def mf(feature_matrix,save_W_file = "outputs/nodeRoles.txt",save_H_file = "outputs/roleFeatures.txt"):

    '''
    input: 
        feature_matrix: feature matrix 2d
        save_W_file: file name of first output matrix 
        save_H_file: file name of second output matrix  
    output:
        saved to file 
    '''

    actual_fx_matrix = np.array(feature_matrix)
    
    n, f = actual_fx_matrix.shape
    
    print 'Number of Features: ' + str(f)
    print 'Number of Nodes: ' + str(n)   

    number_bins = int(np.log2(n))
    max_roles = min([n, f])
    best_W = None
    best_H = None

    mdlo = mdl.MDL(number_bins)
    minimum_description_length = 1e20
    min_des_not_changed_counter = 0


    model_cost_list = []
    loglikelihood_list = []

    # ProgressBar: Text progress bar library for Python
    pbar = ProgressBar()
    for rank in pbar(xrange(1,max_roles + 1)):

        W,H = vanilla_mf(actual_fx_matrix,rank=rank,max_iter=1000)
        estimated_matrix = np.asarray(np.dot(W, H))
        code_length_W = mdlo.get_huffman_code_length(W)
        code_length_H = mdlo.get_huffman_code_length(H)

        print str(W.shape[0]) + " " + str(W.shape[1]) +  " " + str(H.shape[0]) +  " "  + str(H.shape[1])

        model_cost = code_length_W * (W.shape[1]) + code_length_H * (H.shape[0])
        #model_cost = code_length_W * (W.shape[0] + W.shape[1]) + code_length_H * (H.shape[0] + H.shape[1])
        loglikelihood = mdlo.get_log_likelihood(actual_fx_matrix, estimated_matrix)

        print("model_cost: %5.4f" % model_cost)
        print "loglikelihood: %5.4f" % loglikelihood)

        model_cost_list.append(model_cost)
        loglikelihood_list.append(-loglikelihood)

    
    model_cost_list = min_max_scaler(model_cost_list)
    
    loglikelihood_list = min_max_scaler(loglikelihood_list)

    description_length_list = np.add(model_cost_list,loglikelihood_list)

    best_k = np.argmin(description_length_list) + 1

    best_W,best_H = vanilla_mf(actual_fx_matrix,rank = best_k, max_iter = 1000)

    np.savetxt(save_W_file, X=best_W)
    np.savetxt(save_H_file, X=best_H)
    



if __name__ == "__main__":
	

	feature_matrix = np.random.rand(4096,32)

	print feature_matrix.shape

	mf(feature_matrix)




