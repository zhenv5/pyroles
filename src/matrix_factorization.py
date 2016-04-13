import numpy as np
import pandas as pd 
import nimfa
import mdl
from progressbar import ProgressBar
import time 
import cvxpy as cvx 

def min_max_scaler(target_list):
    min_target_list = min(target_list)
    max_target_list = max(target_list)

    result = [(item -min_target_list)/(max_target_list - min_target_list) for item in target_list]

    return result 

def nmf(feature_matrix,W,H,rank,max_iter = 1000,sparsity = False, diversity = False):
    '''
    non-negative matrix factorization

    feature_matrix = W * H 

    sparsity: sparsity constraint 
    
    diversity: diversity constraint 

    '''
    
    m,n = feature_matrix.shape

    # sparsity threshold is num_nodes / num_roles
    sparsity_threshold = 2*float(n)*float(m) / float(rank)

    diversity_threshold = 0.8

    pbar = ProgressBar()

    best_W = np.copy(W)
    best_H = np.copy(H)

    residual = []

    for iter_num in pbar(xrange(1,1 + max_iter)):
        # for odd iterations, treat W as constant, optimize over H
        if iter_num % 2 == 1:
            H = cvx.Variable(rank,n)
            constraint = [H >= 0]
            if sparsity:
                constraint += [cvx.sum_entries(H) <= sparsity_threshold]
            if diversity:
                for i in xrange(rank):
                    for j in xrange(i+1,rank):
                        constraint += [H[i,:]*best_H[j,:].T/(cvx.sum_squares(best_H[j,:])) <= diversity_threshold]
        else:
            # for even iterations, treat H as constant, optimize over W 
            W = cvx.Variable(m,rank)
            constraint = [W >= 0]
            if sparsity:
                constraint += [cvx.sum_entries(W) <= sparsity_threshold]
            if diversity:
                for i in xrange(rank):
                    for j in xrange(i+1,rank):
                        constraint += [W[:,i].T*best_W[:,j]/cvx.sum_squares(best_W[:,j]) <= diversity_threshold]

        # solve the problem
        obj = cvx.Minimize(cvx.norm(feature_matrix - W*H,"fro"))
        prob = cvx.Problem(obj,constraint)
        prob.solve(solver = cvx.SCS)

        print 'Iteration {}, residual norm {}'.format(iter_num, prob.value)
        residual.append(prob.value)

        if iter_num % 2 == 1:
            H = H.value
            best_H = H
        else:
            W = W.value
            best_W = W

        if prob.status == cvx.OPTIMAL:
            break
    if prob.status != cvx.OPTIMAL:
        raise Exception("Solver did not converge!")
    else:
        return (best_W,best_H)



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

def mf(feature_matrix,save_W_file = "outputs/nodeRoles.txt",save_H_file = "outputs/roleFeatures.txt",sparsity = False, diversity = False,max_rank = 30):

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
    max_rank = min(max_roles,max_rank)
    best_W = None
    best_H = None

    mdlo = mdl.MDL(number_bins)
    minimum_description_length = 1e20
    min_des_not_changed_counter = 0


    model_cost_list = []
    loglikelihood_list = []

    # ProgressBar: Text progress bar library for Python
    pbar = ProgressBar()
    for rank in pbar(xrange(1,(max_roles + 1)/2)):

        W,H = vanilla_mf(actual_fx_matrix,rank=rank,max_iter=1000)

        W,H = nmf(actual_fx_matrix,W,H,rank,sparsity = sparsity,diversity = diversity)


        estimated_matrix = np.asarray(np.dot(W, H))
        code_length_W = mdlo.get_huffman_code_length(W)
        code_length_H = mdlo.get_huffman_code_length(H)

        #print str(W.shape[0]) + " " + str(W.shape[1]) +  " " + str(H.shape[0]) +  " "  + str(H.shape[1])

        model_cost = code_length_W * (W.shape[1]) + code_length_H * (H.shape[0])
        #model_cost = code_length_W * (W.shape[0] + W.shape[1]) + code_length_H * (H.shape[0] + H.shape[1])
        loglikelihood = mdlo.get_log_likelihood(actual_fx_matrix, estimated_matrix)

        #print("model_cost: %5.4f" % model_cost)
        #print("loglikelihood: %5.4f" % loglikelihood)

        model_cost_list.append(model_cost)
        loglikelihood_list.append(-loglikelihood)

    
    model_cost_list = min_max_scaler(model_cost_list)
    
    loglikelihood_list = min_max_scaler(loglikelihood_list)

    description_length_list = np.add(model_cost_list,loglikelihood_list)

    best_k = np.argmin(description_length_list) + 1

    best_W,best_H = vanilla_mf(actual_fx_matrix,rank = best_k, max_iter = 1000)
    best_W,best_H = nmf(actual_fx_matrix,best_W,best_H,best_k, max_iter = 1000,sparsity = sparsity,diversity = diversity)

    np.savetxt(save_W_file, X=best_W)
    np.savetxt(save_H_file, X=best_H)
    return (best_W,best_H)



if __name__ == "__main__":

	
    np.random.seed(0)
    m = 10000
    n = 1500
    k = 10

    feature_matrix = np.random.rand(m,k).dot(np.random.rand(k,n))
    best_W,best_H = mf(feature_matrix,sparsity = False,diversity = False,max_rank = 30)
	#feature_matrix = np.random.rand(1248,45)
	#print feature_matrix.shape
	#mf(feature_matrix)
    #print feature_matrix
    #print best_W.dot(best_H)
    #print sum(best_W)
    #print sum(best_H)




