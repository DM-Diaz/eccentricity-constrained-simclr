import numpy as np
import pandas as pd
import torch
import warnings
import sys

def split_normalize_feats(f, trn_inds, val_inds, nest_inds):

    # In this step, we're going to z-score the features within each column. 
    # This is helpful because the different features can have very different variances
    # and normalizing these variances helps stabilize the resulting fits.

    f_trn = f[trn_inds,:]
    f_val = f[val_inds,:]
    f_nest = f[nest_inds,:]

    # I'm computing the normalization parameters (mean and std) on my training data only
    # (plus the nested held-out partition), but not the val set.
    # this helps reduce leakage of data between train and val partitions.
    # then apply those same normalization parameters to the val set too.
    f_concat = np.concatenate([f_trn, f_nest], axis=0)
    # f_concat = f_trn
    
    features_m = np.mean(f_concat, axis=0, keepdims=True) #[:trn_size]
    # print(features_m[0,0:10])
    features_s = np.std(f_concat, axis=0, keepdims=True) + 1e-6
    
    f_trn -= features_m
    f_trn /= features_s
    f_nest -= features_m
    f_nest /= features_s
    f_val -= features_m
    f_val /= features_s
    
    return f_trn, f_val, f_nest

def split_feats(f, trn_inds, val_inds, nest_inds):

    # Split only, skipping z-score.
    
    f_trn = f[trn_inds,:]
    f_val = f[val_inds,:]
    f_nest = f[nest_inds,:]

    return f_trn, f_val, f_nest
    
def solve_ridge(xtrn, vtrn, xnest, vnest, lambdas, eps=1e-12, return_loss=False):

    
    # xtrn = training features [n_images x n_feat]
    # vtrn = training voxel data [n_images x n_voxels]
    # xnest = nested held-out set features [n_images x n_feat]
    # vnest = nested held-out set voxel data [n_images x n_voxels]
    # lambdas = list of your candidate lambdas

    # Convert everything to float64 at the start
    # xtrn = xtrn.double()
    # vtrn = vtrn.double()
    # xnest = xnest.double()
    # vnest = vnest.double()
    # with torch.no_grad():
        
    lambdas = [float(l) for l in lambdas]  # ensure lambdas are float
    device = xtrn.device
    n_vox = vtrn.shape[1]
    
    n_features = xtrn.shape[1]
    XtX = xtrn.T @ xtrn
    XtV = xtrn.T @ vtrn

    ridge_term = torch.eye(n_features, device=device, dtype=torch.float64)

    weights_list = []
    for li, l in enumerate(lambdas):

        # Solve for weights that correspond to this lambda.

        # first computing this matrix: (X^T X + λI)
        ridge_matrix = XtX + ridge_term * (l + eps)
    
        # want to compute w.
        # express as:
        # A * w = B
        # where A = (X^T X + λI) ("ridge_matrix" above)
        # and where B = X^T y ("XtV" above)
        # so we have:
        # w = (X^T X + λI)^(-1) X^T y
        
        # the below code implements this solve for w:
        try:
            # Cholesky decomp should be faster and more numerically stable, generally works well for ridge.
            L = torch.linalg.cholesky(ridge_matrix)
            w = torch.cholesky_solve(XtV, L)
        except:
            # fallback to linalg.solve
            print('Cholesky decomposition failed (lambda = %.12f), trying linalg.solve..'%(l))
            w = torch.linalg.solve(ridge_matrix, XtV)
            
        weights_list.append(w)

    weights = torch.stack(weights_list, dim=0)  # [#lambdas, #features, #voxels]

    # print(XtX.shape, XtV.shape, xtrn.shape, vtrn.shape, ridge_matrix.shape)
    del XtX, XtV, xtrn, vtrn, ridge_matrix
    torch.cuda.empty_cache()

    # print('Memory usage just before making pred:')
    # print_gpu_memory()  # Check after each epoch

    # predict the response on nested held-nest data, using features from nested held-nest data (xnest)
    pred = torch.tensordot(xnest, weights, dims=[[1],[1]]) 
    # inputs are [#samples, #feature], [#lambdas, #feature, #voxel]
    # yields [#samples, #lambdas, #voxels]
    # this is an expensive step in terms of memory

    # compute loss for nested held-nest data
    # this will tell us the loss for each of the possible lambda values
    loss = torch.sum(torch.pow(vnest[:,None,:] - pred, 2), dim=0) # [#lambdas, #voxels]
    loss = loss.cpu().numpy()
    
    weights_use = torch.zeros((n_features, n_vox),device=device, dtype=torch.float64)
    best_lambda_inds = np.zeros((n_vox,), dtype=np.float32)
    best_nest_loss = np.zeros((n_vox,), dtype=np.float32)
    
    # for each voxel, find its best weights
    for vi in range(n_vox):
        # choose the best lambda value, based on min loss
        best_lambda_ind = np.argmin(loss[:,vi])
        best_lambda_inds[vi] = best_lambda_ind
        weights_use[:, vi] = weights[best_lambda_ind,:,vi]
        best_nest_loss[vi] = loss[best_lambda_ind, vi] # loss is [lambdas x voxels]

    if return_loss:
        # best_nest_loss is the loss for the best lambda, on the nested heldout set.
        # this can be used for choosing best pRF. 
        return weights_use, best_lambda_inds, best_nest_loss
    else:
        return weights_use, best_lambda_inds


def print_gpu_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        free = total - reserved
    
        print(f"GPU Memory:")
        print(f"  Allocated: {allocated:.2f} GB")
        print(f"  Reserved:  {reserved:.2f} GB")
        print(f"  Total:     {total:.2f} GB")
        print(f"  Free:      {free:.2f} GB")

def get_r2(actual,predicted):
    """
    This computes the coefficient of determination (R2).
    Always goes along first dimension (i.e. the trials/samples dimension)
    MAKE SURE INPUTS ARE ACTUAL AND THEN PREDICTED, NOT FLIPPED
    """
    ssres = np.sum(np.power((predicted - actual),2), axis=0);
    sstot = np.sum(np.power((actual - np.mean(actual, axis=0)),2), axis=0);
    r2 = 1-(ssres/sstot)
    
    return r2

def get_r2_torch(actual,predicted):
    """
    This computes the coefficient of determination (R2).
    Always goes along first dimension (i.e. the trials/samples dimension)
    MAKE SURE INPUTS ARE ACTUAL AND THEN PREDICTED, NOT FLIPPED
    """
    ssres = torch.sum(torch.pow(predicted - actual, 2), dim=0)
    sstot = torch.sum(torch.pow(actual - torch.mean(actual, dim=0), 2), dim=0)
    r2 = 1-(ssres/sstot)
    
    return r2

def get_corrcoef(actual,predicted,dtype=np.float32):
    """
    This computes the linear correlation coefficient.
    Always goes along first dimension (i.e. the trials/samples dimension)
    Assume input is 2D.
    """
    assert(len(actual.shape)==2)
    vals_cc = np.full(fill_value=0, shape=(actual.shape[1],), dtype=dtype)
    for vv in range(actual.shape[1]):
        # vals_cc[vv] = numpy_corrcoef_warn(actual[:,vv], predicted[:,vv])[0,1] 
        vals_cc[vv] = np.corrcoef(actual[:,vv], predicted[:,vv])[0,1] 
    return vals_cc


# def get_corrcoef_torch(actual, predicted, dtype=torch.float32):
#     """
#     This computes the linear correlation coefficient.
#     Always goes along first dimension (i.e. the trials/samples dimension)
#     Assume input is 2D.
#     """
#     assert len(actual.shape) == 2
    
#     # Stack actual and predicted as rows: shape (2*n_features, n_samples)
#     combined = torch.cat([actual.T, predicted.T], dim=0)
    
#     # Compute correlation matrix
#     corr_matrix = torch.corrcoef(combined)
    
#     # Extract diagonal from the off-diagonal block
#     # corr_matrix is (2*n_features, 2*n_features)
#     # We want correlations between actual[i] and predicted[i]
#     n_features = actual.shape[1]
#     vals_cc = torch.diagonal(corr_matrix[:n_features, n_features:])
    
#     return vals_cc.to(dtype)

def get_corrcoef_torch(actual, predicted, dtype=torch.float32):
    """
    Compute correlation coefficient for each feature pair efficiently.
    Goes along first dimension (trials/samples).
    Input: actual, predicted are (n_samples, n_features)
    """
    assert len(actual.shape) == 2
    assert actual.shape == predicted.shape
    
    # Center the data
    actual_centered = actual - actual.mean(dim=0, keepdim=True)
    predicted_centered = predicted - predicted.mean(dim=0, keepdim=True)
    
    # Compute correlation for each feature
    numerator = (actual_centered * predicted_centered).sum(dim=0)
    
    # Compute standard deviations
    actual_std = torch.sqrt((actual_centered ** 2).sum(dim=0))
    predicted_std = torch.sqrt((predicted_centered ** 2).sum(dim=0))
    
    # Correlation coefficient
    vals_cc = numerator / (actual_std * predicted_std + 1e-8)  # Add epsilon to avoid division by zero
    
    return vals_cc.to(dtype)


def get_featsens(predicted_resp, feat, dtype=np.float32):

    """
    Get a measure of voxels' "feature sensitivity" for each feature, based on how correlated 
    the predicted responses of the encoding model are with the activation in each feature channel.
    From our paper: https://doi.org/10.1167/jov.23.4.
    """
    
    n_voxels = predicted_resp.shape[1]
    n_features = feat.shape[1]
    assert(predicted_resp.shape[0]==feat.shape[0])
    
    featsens = np.zeros((n_voxels, n_features),dtype=dtype)

    # this loop is slow, see torch version below
    for vv in np.arange(n_voxels):
        for ff in np.arange(n_features):
            # featsens[vv,ff] = numpy_corrcoef_warn(predicted_resp[:,vv], feat[:,ff])[0,1] 
            featsens[vv,ff] = np.corrcoef(predicted_resp[:,vv], feat[:,ff])[0,1] 

    return featsens

def get_featsens_torch(predicted_resp, feat, dtype=np.float32):

    """
    Get a measure of voxels' "feature sensitivity" for each feature, based on how correlated 
    the predicted responses of the encoding model are with the activation in each feature channel.
    From our paper: https://doi.org/10.1167/jov.23.4.
    """
    
    n_voxels = predicted_resp.shape[1]
    n_features = feat.shape[1]
    assert(predicted_resp.shape[0]==feat.shape[0])

    # # Stack all voxels and features as rows
    # combined = torch.cat([predicted_resp.T, feat.T], dim=0)  # (n_voxels + n_features, n_samples)
    
    # # Compute full correlation matrix
    # corr_matrix = torch.corrcoef(combined)  # (n_voxels + n_features, n_voxels + n_features)
    
    # # Extract the voxel-feature correlation block
    # featsens = corr_matrix[:n_voxels, n_voxels:]  # (n_voxels, n_features)

    # Center the data
    pred_centered = predicted_resp - predicted_resp.mean(dim=0, keepdim=True)
    feat_centered = feat - feat.mean(dim=0, keepdim=True)
    
    # Compute cross-correlation: (n_voxels, n_features)
    numerator = pred_centered.T @ feat_centered  # (n_voxels, n_features)
    
    # Compute standard deviations
    pred_std = torch.sqrt((pred_centered ** 2).sum(dim=0, keepdim=True))  # (1, n_voxels)
    feat_std = torch.sqrt((feat_centered ** 2).sum(dim=0, keepdim=True))  # (1, n_features)
    
    # Correlation matrix
    denominator = pred_std.T @ feat_std  # (n_voxels, n_features)
    featsens = numerator / (denominator + 1e-8)
    
    return featsens
    

def numpy_corrcoef_warn(a,b):
    
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            cc = np.corrcoef(a,b)
        except RuntimeWarning as e:
            print('Warning: problem computing correlation coefficient')
            print('shape a: ',a.shape)
            print('shape b: ',b.shape)
            print('sum a: %.9f'%np.sum(a))
            print('sum b: %.9f'%np.sum(b))
            print('std a: %.9f'%np.std(a))
            print('std b: %.9f'%np.std(b))
            print(e)
            warnings.filterwarnings('ignore')
            cc = np.corrcoef(a,b)
            
    if np.any(np.isnan(cc)):
        print('There are nans in correlation coefficient')
    
    return cc
   