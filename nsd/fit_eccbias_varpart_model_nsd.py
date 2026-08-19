
import sys
import os
import time
import numpy as np
import argparse
import distutils.util
import gc
import h5py
import pandas as pd
import torch
import gc

device = "cuda" if torch.cuda.is_available() else "cpu"
if device=="cuda":
    print('\nUsing GPU device:')
    print(torch.cuda.get_device_name(0))
    print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# where my preprocessed NSD files live
nsd_path = '/lab_data/hendersonlab/datasets/nsd_preproc'
data_folder = os.path.join(nsd_path, 'data')
labels_folder = os.path.join(nsd_path, 'labels')
stim_folder = os.path.join(nsd_path, 'stimuli')
rois_folder = os.path.join(nsd_path, 'rois')

# where the pre-computed DNN features are placed
# different models are organized in sub-folders in here
# features_folder = '/lab_data/hendersonlab/features/NSD/eccbias/'
features_folder = '/lab_data/hendersonlab/features/NSD/'

# where you want to save the model fits.
save_fits_folder = '/lab_data/hendersonlab/projects/eccbias/model_fits/'

import model_fitting_utils

def fit_model(args):

    ########## LOADING THE DATA #############################################################################

    voxel_data, good_values = load_nsd_data(args.subject)

    # by default, fitting all voxels at once. but could sub-select voxels here if needed
    n_voxels = voxel_data.shape[1]
    voxel_inds = np.ones((n_voxels,),dtype=bool)

    voxel_data = voxel_data[:,voxel_inds]

    trn_inds, val_inds, nest_inds = load_nsd_splits(args.subject, good_values)

    # load info about rois and voxel mask
    fn = os.path.join(rois_folder, 'S%d_voxel_roi_info.npy'%args.subject)
    rinfo = np.load(fn, allow_pickle=True).item()

    # v is our fMRI data, these are the 3 different splits
    vtrn = torch.Tensor(voxel_data[trn_inds,:]).to(device).to(torch.float64)
    vnest = torch.Tensor(voxel_data[nest_inds,:]).to(device).to(torch.float64)
    vval = torch.Tensor(voxel_data[val_inds,:]).to(device).to(torch.float64)
    

    ########## LOADING THE FEATURES #############################################################################
    
    # pre-computed features were made ahead of time, saved as .npy files
    features_file_list1, features_file_list2 = get_features_filename(args)
    
    f1 = []
    for file in features_file_list1:
        print('loading features from: %s'%file)
        sys.stdout.flush()
        ftmp = np.load(file, allow_pickle = True).astype(np.float64)
        print(ftmp.shape)
        f1 += [ftmp]

    # concatenate all features together, one big array
    f1 = np.concatenate(f1, axis=1)
    print(f1.shape)

    # make sure we only take features that have valid fMRI data
    f1 = f1[good_values,:]
    print(f1.shape)

    f2 = []
    for file in features_file_list2:
        print('loading features from: %s'%file)
        sys.stdout.flush()
        ftmp = np.load(file, allow_pickle = True).astype(np.float64)
        print(ftmp.shape)
        f2 += [ftmp]

    # concatenate all features together, one big array
    f2 = np.concatenate(f2, axis=1)
    print(f2.shape)

    # make sure we only take features that have valid fMRI data
    f2 = f2[good_values,:]
    print(f2.shape)

    # each model alone, and both combined.
    # create the model list here:
    varpart_models = [f1, f2, np.concatenate([f1, f2], axis=1)]
    varpart_model_names = ['%s-only'%args.model_name1, \
                         '%s-only'%args.model_name2, \
                         'combined']

    ########## FITTING THE MODEL #############################################################################
    
    # define lambda values
    # lambda is the ridge penalty, bigger = more regularization
    n_lambdas = 20
    small_value = 0.0001
    max_lambda = 10**10
    # max_lambda = 10**20
    print('max_lambda = %.2f'%max_lambda)
    lambdas = np.logspace(np.log(small_value),np.log(max_lambda+small_value),n_lambdas, \
                          dtype=np.float32, base=np.e) - small_value

    # preallocate dictionaries to store results from each variance partition model type
    r2_varpart = dict([])
    corr_varpart = dict([])
    weights_varpart = dict([])
    best_lambda_inds_varpart = dict([])

    for f, varpart_name in zip(varpart_models, varpart_model_names):

        print('\nfitting %s'%varpart_name)
        print('features are size:')
        print(f.shape)
        
        # divide into three splits
        # z-scoring happens in here as well.
        f_trn, f_val, f_nest = model_fitting_utils.split_normalize_feats(f, trn_inds, val_inds, nest_inds)
    
        # add the intercept: a column of ones
        f_trn = np.concatenate([f_trn, np.ones(shape=(len(f_trn), 1), dtype=f_trn.dtype)], axis=1)
        f_nest = np.concatenate([f_nest, np.ones(shape=(len(f_nest), 1), dtype=f_nest.dtype)], axis=1)
        f_val = np.concatenate([f_val, np.ones(shape=(len(f_val), 1), dtype=f_val.dtype)], axis=1)
        
        print('Size of features matrices:')
        print(f_trn.shape, f_val.shape, f_nest.shape)
    
        # convert features to tensors, send to gpu:
        xtrn = torch.Tensor(f_trn).to(device).to(torch.float64)
        xnest = torch.Tensor(f_nest).to(device).to(torch.float64)
        xval = torch.Tensor(f_val).to(device).to(torch.float64)

        print('Memory usage just before fitting function')
        model_fitting_utils.print_gpu_memory()  # Check after each epoch
        sys.stdout.flush()
        
        # here is where we actually solve for the weights. 
        st_fit = time.time()
        weights, best_lambda_inds = model_fitting_utils.solve_ridge(xtrn, vtrn, xnest, vnest, lambdas)
        elapsed_fit = time.time() - st
        print('Model fitting time elapsed: %.5f s'%elapsed_fit)
    
        # print('Memory usage just after fitting function')
        # model_fitting_utils.print_gpu_memory()  # Check after each epoch
        sys.stdout.flush()
    
        # predict voxel response in held-out validation data here.
        # yhat = X @ W
        pred = xval @ weights

        r2 = model_fitting_utils.get_r2_torch(vval, pred)
        corr = model_fitting_utils.get_corrcoef_torch(vval, pred)
        
        # remember to turn these back into numpy, from torch.
        # sometimes tensors will give errors in your subsequent numpy code.
        weights = weights.cpu().numpy()
        r2 = r2.cpu().numpy()
        corr = corr.cpu().numpy()

        del xtrn, xnest, xval
        torch.cuda.empty_cache()
    
        r2_varpart[varpart_name] = r2
        corr_varpart[varpart_name] = corr
        weights_varpart[varpart_name] = weights
        best_lambda_inds_varpart[varpart_name] = best_lambda_inds
    
    # Now save.
    # Will make a dictionary of things to save.
    dict2save = {'subject': args.subject, \
                 'model1': args.model_name1, \
                 'model2': args.model_name2, \
                 'features_file_list1': features_file_list1, \
                 'features_file_list2': features_file_list2, \
                 'lambdas': lambdas, \
                 'voxel_mask': rinfo['voxel_mask'], \
                 'voxel_index': rinfo['voxel_idx'], \
                 'voxel_nc': rinfo['noise_ceiling_avgreps'], \
                 'brain_nii_shape': rinfo['brain_nii_shape'], \
                 'weights_varpart': weights_varpart, \
                 'r2_varpart': r2_varpart, \
                 'corr_varpart': corr_varpart, \
                 'best_lambda_inds_varpart': best_lambda_inds_varpart, \
                    }

    save_folder = os.path.join(save_fits_folder, \
                            'varpart_%s_vs_%s'%(args.model_name1, args.model_name2))
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    fn2save = os.path.join(save_folder, \
                           'NSD_S%d_varpart_%s_vs_%s.npy'%(args.subject, args.model_name1, args.model_name2))
    print('saving to %s'%fn2save)
    np.save(fn2save, dict2save, allow_pickle=True)


def load_nsd_data(ss):

    # load the preprocessed data files, made using code in nsd_preproc/code 
    
    # load info about images on each trial
    info_fn = os.path.join(labels_folder, 'S%d_image_info.csv'%(ss))
    print(info_fn)
    info = pd.read_csv(info_fn)

    image_order = np.array(info['unique_ims'])
    n_reps = np.array(info['n_reps'])

    # load fmri data
    data_filename = os.path.join(data_folder, 'S%d_betas_avg_bigmask.hdf5'%ss)
    print(data_filename)

    t = time.time()
    with h5py.File(data_filename, 'r') as data_set:
        values = np.copy(data_set['/betas'])
        data_set.close() 
    elapsed = time.time() - t
    print('Took %.5f seconds to load file'%elapsed)
    # data is organized as:
    # [images x voxels]

    # Some of these values may be nans, only for some subjects
    # this is for subjects who didn't complete all 40 sessions of NSD experiment.
    # make sure we remove the nans now.
    good_values = ~np.isnan(values[:,0])
    print(values.shape)
    print(np.sum(~good_values))

    # check that nans are exactly where we expect
    # nans happen when n_reps=0
    assert(np.all(good_values[n_reps>0]))
    assert(np.all(~good_values[n_reps==0]))

    voxel_data = values[good_values,:]
    print(voxel_data.shape)
    
    return voxel_data, good_values
    

def load_nsd_splits(ss, good_values):

    # I computed the data splits ahead of time, so that the random seed is reproducible
    # Always holding out 1000 shared images as val. 
    # Then a random 10% as the "nested held-out" set that is used to choose ridge parameters.
    splits_filename = os.path.join(stim_folder, 'Image_data_partitions.npy')
    splits = np.load(splits_filename, allow_pickle=True).item()

    si = ss-1
    trn_inds = splits['is_trn'][good_values,si]
    val_inds = splits['is_val'][good_values,si]
    nest_inds = splits['is_holdout'][good_values,si]

    return trn_inds, val_inds, nest_inds
    

def get_features_filename(args):

    # this is how we can figure out the full path to the features file used here.
    # can modify this for any set of features, depending on how they were named.

    f = []
    
    for model_name in [args.model_name1, args.model_name2]:
    
        if model_name=='RN50-CLIP-vision':
            feat_path = os.path.join(features_folder, model_name)
            assert(args.model_layer=='concat')
            layers_do = [0,1,2,3,4,5,6]
            features_file_list = [ os.path.join(feat_path, \
                                                'NSD_S%d'%args.subject, \
                                                'vision_layer_resnet_%d.npy'%(layer)) \
                                   for layer in layers_do ]
    
        elif 'resnet18' in model_name:
            feat_path = os.path.join(features_folder, 'eccbias',model_name)
        
            if args.model_layer=='concat':
                layers_do = ['conv1',
                             'layer1-1',
                             'layer2-1',
                             'layer3-1',
                             'layer4-1',
                             'avgpool']
                
                features_file_list = [ os.path.join(feat_path, 'NSD_S%d_ims224pix_%s.npy'%(args.subject, layer)) for layer in layers_do ]
            else:
                features_file = os.path.join(feat_path, 'NSD_S%d_ims224pix_%s.npy'%(args.subject, args.model_layer))
                features_file_list = [features_file]
        elif 'resnet50' in model_name:
            feat_path = os.path.join(features_folder, 'eccbias',model_name)
            assert(args.model_layer=='concat')
            if args.model_layer=='concat':
                    layers_do = ['conv1',
                    'layer1-2',
                    'layer2-3',
                    'layer3-5',
                    'layer4-2',
                    'avgpool',
                ]
            # layers_do = ['block1','block3','block5','block7']
            features_file_list = [ os.path.join(feat_path, 'NSD_S%d_ims224pix_%s.npy'%(args.subject, layer)) for layer in layers_do ]

        f += [features_file_list]
            
    return f

    
if __name__ == '__main__':

    # this is just a function that helps with argument parsing
    def nice_str2bool(x):
        return bool(distutils.util.strtobool(x))

    # this part handles the arguments 
    parser = argparse.ArgumentParser()

    # we can add as many arguments here as needed, using this same format.
    parser.add_argument("--subject", type=int,default=1,
                    help="number of the subject, 1-8")
    parser.add_argument("--debug",type=nice_str2bool,default=False,
                    help="want to run a fast test version of this script to debug? 1 for yes, 0 for no")
    
    parser.add_argument("--model_name1",type=str,default='',
                    help="which model are the features from?")
    parser.add_argument("--model_name2",type=str,default='',
                    help="which model are the features from?")
    parser.add_argument("--model_layer",type=str,default='',
                    help="which model layer are the features from?")
    
    args = parser.parse_args()

    st = time.time()
    
    # then we call the main function. arguments gets passed in.
    fit_model(args)

    elapsed = time.time() - st
    print('fitting took %.5f s total'%elapsed)
    sys.stdout.flush()


    

    