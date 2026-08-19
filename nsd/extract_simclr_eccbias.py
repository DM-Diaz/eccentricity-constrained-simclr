import numpy as np
import sys, os
import argparse
import gc
import torch
import time
import h5py
import copy
from collections import OrderedDict
import torchvision.models as models
import torch.nn as nn
from sklearn.decomposition import PCA

# need this for pulling out intermediate activations
# pip install torchextractor
import torchextractor as tx

dtype=np.float32


# Based on code/paths copied from:
# /user_data/dylandia/eb_notebooks/SimCLR-master/run_updated_fixed_v5.py
# /user_data/dylandia/eb_notebooks/SimCLR-master/simclr.py
# /user_data/dylandia/eb_notebooks/SimCLR-master/models/resnet_simclr.py

simclr_code_path = '/user_data/dylandia/eb_notebooks/SimCLR-master'
sys.path.append(simclr_code_path)
from models.resnet_simclr import ResNetSimCLR
from simclr import SimCLR
import fusion_simclr_normal_v2


device = "cuda" if torch.cuda.is_available() else "cpu"
if device=="cuda":
    print('\nUsing GPU device:')
    print(torch.cuda.get_device_name(0))
    print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")


# where preproc NSD data lives, images are all in here
save_preproc_path = '/lab_data/hendersonlab/datasets/nsd_preproc/'
stim_folder = os.path.join(save_preproc_path, 'stimuli')
        
def extract_NSD_features(ss, training_type = 'FoveaGaze', debug=0, batch_size=100, model_architecture='resnet18'):

    # ss is my NSD subject number, extracting resnet features for this NSD subject, all images.
    # training type can be clip or imgnet
    
    debug = (debug==1)

    print('Subject = %d, Training Type = %s, Debug = %d'%(ss, training_type, debug))
    
    sys.stdout.flush()

    # where i would like to put feature files when created.
    features_folder = '/lab_data/hendersonlab/features/NSD/eccbias'

    # this is the final save path
    feat_path = os.path.join(features_folder, '%s-%s'%(model_architecture,training_type))
    if not os.path.exists(feat_path):
        os.makedirs(feat_path)
    
    
    dtype=np.float32

    layers_to_do, layer_names = get_layers(model_architecture)
    n_layers = len(layers_to_do)
    
    # loading my images
    n_pix = 224
    
    image_filename = os.path.join(stim_folder, 'S%d_stimuli_%d.h5py'%(ss, n_pix))
    
    t = time.time()
    with h5py.File(image_filename, 'r') as data_set:
        values = np.copy(data_set['/stimuli'])
        data_set.close() 
    elapsed = time.time() - t
    print('Took %.5f seconds to load file'%elapsed)
    sys.stdout.flush()

    # image_data = values
    image_data = normalize_ims(values)
    # need this step - puts the values in correct range.
    
    n_images = image_data.shape[0]

    n_batches = int(np.ceil(n_images/batch_size))

    # store my big feature matrices as i extract them
    features_big = [[] for ll in layers_to_do]

    # subsampling size: this sets how much we want to downsample the feature maps, from each conv layer
    # natively they're too large to handle, so need to reduce size partially before PCA
    subsampling_size = 5000

        
    with torch.no_grad():
    
        for bb in range(n_batches):

            if debug and (bb>1):
                # debug mode, we stop early and save to make sure it's working
                continue
        
            batch_inds = np.arange(batch_size * bb, np.min([batch_size * (bb+1), n_images]))
        
            image_batch = image_data[batch_inds,:,:,:]
            print(image_batch.shape)
            sys.stdout.flush()
            
            # this returns a list over layers, each one is the activations for same batch.
            activ_batch = get_resnet_activations_batch(image_batch, \
                                                       layers_to_do, \
                                                       model_architecture = model_architecture, \
                                                       training_type = training_type, \
                                                       device=device)
            print(activ_batch[0].shape)
            
            # loop over the layers, do the maxpool here
            for li, a in enumerate(activ_batch):

                print('\n%d'%li)

                print(a.size())

                # if it's a convolutional layer, has [C x H x W] dims
                if (len(a.size()) == 4) and ( a.shape[2] > 1 ):

                    print('reducing size w avgpool')
                    
                    c = a.data.shape[1]  # number of channels
                    k = int(np.floor(np.sqrt(subsampling_size / c)))
                    # k is how big you want the output maps to be
                    
                    # applying average pooling: this reduces size of feature maps
                    tmp = nn.functional.adaptive_avg_pool2d(a.data, (k, k))
                    
                    print(tmp.size())

                # if it's a fully connected layer, no need to downsample
                else:
                    tmp = a

                # then flatten
                feat_flat = np.reshape(tmp.cpu().numpy(), [tmp.shape[0], -1])

                print(feat_flat.shape)
                sys.stdout.flush()

                if bb==0:
                    features_big[li] = np.zeros((n_images, feat_flat.shape[1]),dtype=dtype)

                features_big[li][batch_inds,:] = feat_flat

    
    # Save my results here, across all batches
    
    for ii, layer in enumerate(layer_names):
    
        # layer = resnet_block_names[ll]
        f = features_big[ii]
        print('Saving layer %s, features are: [%d x %d]'%(layer, f.shape[0], f.shape[1]))
        features_filename = os.path.join(feat_path, 'NSD_S%d_ims%dpix_%s_prePCA.npy'%(ss, n_pix, layer))
       
        print('Saving to: %s'%features_filename)
        np.save(features_filename, f)
        sys.stdout.flush()
        

        # t = time.time()
        # with h5py.File(features_filename, 'w') as data_set:
        #     dset = data_set.create_dataset("features", np.shape(f), dtype=np.float32)
        #     data_set['/features'][:,:] = f
        #     data_set.close()  
        # elapsed = time.time() - t
        # print('Took %.2f seconds to save'%elapsed)


def pca_NSD_features(ss, training_type = 'FoveaGaze', n_components = 100, model_architecture='resnet18'):

    print('PCA: Subject = %d, Training Type = %s'%(ss, training_type))
    sys.stdout.flush()
    
    features_folder = '/lab_data/hendersonlab/features/NSD/eccbias'

    feat_path = os.path.join(features_folder, '%s-%s'%(model_architecture,training_type))

    layers_to_do, layer_names = get_layers(model_architecture)
    
    n_pix = 224
    
    for ii, layer in enumerate(layer_names):

        # layer = resnet_block_names[ll]

        big_filename = os.path.join(feat_path, 'NSD_S%d_ims%dpix_%s_prePCA.npy'%(ss, n_pix, layer))
       
        if os.path.exists(big_filename):
            f = np.load(big_filename)
        else:
            print('File not found: %s'%(big_filename))
            sys.exit(1)

        # Reduce the dimensionality with PCA here.
        print("Running PCA")
        print("feature shape: ")
        print(f.shape)
        print('n_components = %d'%n_components)
        sys.stdout.flush()
        pca = PCA(n_components=min(f.shape[0], n_components), svd_solver="auto")

        fp = pca.fit_transform(f)
        print("Feature %s has shape of:" % layer)
        print(fp.shape)

        # now we have a smaller set of features, save it now.
        pca_filename = os.path.join(feat_path, 'NSD_S%d_ims%dpix_%s.npy'%(ss, n_pix, layer))
        print('Saving to: %s'%pca_filename)
        np.save(pca_filename, fp)
        sys.stdout.flush()
        
    
def get_resnet_activations_batch(image_batch, \
                               layers_to_do, \
                               model_architecture, \
                               training_type, \
                               device=None):

    """
    Get activations for images in NSD, passed through pretrained resnet model.
    Specify which NSD images to look at, and which layers to return.
    """

    if device is None:
        device = torch.device('cpu:0')

    if (training_type=='pretrained-imgnet') and (model_architecture=='resnet18'):

        # this is an imagenet-trained model, provided by pytorch
        # https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html
        print('Using pytorch imagenet-pretrained resnet18 model')
        model = models.resnet18(weights='IMAGENET1K_V1')

    elif (training_type=='pretrained-imgnet') and (model_architecture=='resnet50'):

        # this is an imagenet-trained model, provided by pytorch
        # https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html
        print('Using pytorch imagenet-pretrained resnet50 model')
        model = models.resnet50(weights='IMAGENET1K_V2')
        
    elif (training_type=='pretrained-clip') and (model_architecture=='resnet50'):

        # clip implemented in this package, from:
        # https://github.com/openai/CLIP
        import clip

        # this is an imagenet-trained model, provided by pytorch
        # https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html
        print('Using pytorch imagenet-pretrained resnet50 model')
        model = models.resnet50(weights='IMAGENET1K_V2')
        
    elif (training_type=='pretrained-simclr') and (model_architecture=='resnet18'):

        # this is a pretrained resnet-18 model, accessed from:
        # https://github.com/Spijkervet/SimCLR
        # It's pretrained on STL-10
        checkpoint_file = '/lab_data/hendersonlab/features/model_ckpts/simclr-pretrained/resnet18/checkpoint_100-Spijkervet.tar' 
        print(checkpoint_file)
        checkpoint = torch.load(checkpoint_file, map_location = device)
        # state dict needs to be adjusted to match my current model:
        old_state_dict = checkpoint
        new_state_dict = {}
        for key, value in old_state_dict.items():
            # Replace 'encoder.' with 'backbone.'
            new_key = key.replace('encoder.', 'backbone.')
            # if 'projector.0' in key:
            if 'projector' in key:
                new_key = key.replace('projector.', 'backbone.fc.')
                print(key)
                print(old_state_dict[key].shape)
            new_state_dict[new_key] = value

        out_dim=64
        model = ResNetSimCLR(base_model=model_architecture, out_dim=out_dim)
        # remove bias, this makes it match the saved checkpoint file which doesn't have bias
        model.backbone.fc[0].bias.data.zero_(); 
        model.backbone.fc[2].bias.data.zero_();

        model_keys = set(model.state_dict().keys())
        checkpoint_keys = set(new_state_dict.keys())
        
        print("Keys in model but not in checkpoint:", model_keys - checkpoint_keys)
        print("Keys in checkpoint but not in model:", checkpoint_keys - model_keys)

        model.load_state_dict(new_state_dict, strict=False)
        model = model.backbone

    elif (training_type=='pretrained-simclr2') and (model_architecture=='resnet18'):

        # this is a pretrained resnet-18 model, accessed from:
        # https://github.com/sthalles/SimCLR
        # It's pretrained on STL-10
        checkpoint_file = '/lab_data/hendersonlab/features/model_ckpts/simclr-pretrained/resnet18/checkpoint_100-sthalles/checkpoint_0100.pth.tar' 
        print(checkpoint_file)
        checkpoint = torch.load(checkpoint_file, map_location = device)
        checkpoint = checkpoint['state_dict']
        # state dict needs to be adjusted to match my current model:
        old_state_dict = checkpoint
        new_state_dict = {}
        for key, value in old_state_dict.items():
            new_key=key
            new_state_dict[new_key] = value
        
        out_dim=128
        model = ResNetSimCLR(base_model=model_architecture, out_dim=out_dim)
        # remove bias, this makes it match the saved checkpoint file which doesn't have bias
        model.backbone.fc[0].bias.data.zero_(); 
        model.backbone.fc[2].bias.data.zero_();
        
        model_keys = set(model.state_dict().keys())
        checkpoint_keys = set(new_state_dict.keys())
        
        print("Keys in model but not in checkpoint:", model_keys - checkpoint_keys)
        print("Keys in checkpoint but not in model:", checkpoint_keys - model_keys)
        
        model.load_state_dict(new_state_dict, strict=True)
        model = model.backbone

    elif (training_type=='simclr-imgnet100') and (model_architecture=='resnet18'):

        checkpoint_file = '/lab_data/hendersonlab/ckpts/checkpoint_120-resnet18-simclr-imagenet100.ckpt'
        print('Loading from:')
        print(checkpoint_file)
        checkpoint = torch.load(checkpoint_file, map_location=device)
        
        # fix the state dict to match what is in the model names
        old_state_dict = checkpoint['state_dict']
        
        index_to_name = {
            '0': 'conv1',
            '1': 'bn1',
            '4': 'layer1',
            '5': 'layer2',
            '6': 'layer3',
            '7': 'layer4',
        }
        
        new_state_dict = {}
        for key, value in old_state_dict.items():
            parts = key.split('.')
            if parts[0] == 'projection_head':
                continue
            if parts[0] == 'backbone':
                idx = parts[1]
                if idx in index_to_name:
                    parts[1] = index_to_name[idx]
                    new_state_dict['.'.join(parts)] = value
            else:
                new_state_dict[key] = value
        
        out_dim = 128
        model_architecture = 'resnet18'
        model = ResNetSimCLR(base_model=model_architecture, out_dim=out_dim)
        
        model_keys = set(model.state_dict().keys())
        new_keys = set(new_state_dict.keys())
        print("Missing from checkpoint:", model_keys - new_keys)
        print("Unexpected in checkpoint:", new_keys - model_keys)
        
        model.load_state_dict(new_state_dict, strict=False)
        
        model = model.backbone


    elif (training_type=='simclr-imgnet1k') and (model_architecture=='resnet18'):

        checkpoint_file = '/lab_data/hendersonlab/ckpts/checkpoint_100-resnet18-simclr-imagenet1k.ckpt'
        print('Loading from:')
        print(checkpoint_file)
        checkpoint = torch.load(checkpoint_file, map_location=device)
        
        # fix the state dict to match what is in the model names
        old_state_dict = checkpoint['state_dict']
        
        index_to_name = {
            '0': 'conv1',
            '1': 'bn1',
            '4': 'layer1',
            '5': 'layer2',
            '6': 'layer3',
            '7': 'layer4',
        }
        
        new_state_dict = {}
        for key, value in old_state_dict.items():
            parts = key.split('.')
            if parts[0] == 'projection_head':
                continue
            if parts[0] == 'backbone':
                idx = parts[1]
                if idx in index_to_name:
                    parts[1] = index_to_name[idx]
                    new_state_dict['.'.join(parts)] = value
            else:
                new_state_dict[key] = value
        
        out_dim = 128
        model_architecture = 'resnet18'
        model = ResNetSimCLR(base_model=model_architecture, out_dim=out_dim)
        
        model_keys = set(model.state_dict().keys())
        new_keys = set(new_state_dict.keys())
        print("Missing from checkpoint:", model_keys - new_keys)
        print("Unexpected in checkpoint:", new_keys - model_keys)
        
        model.load_state_dict(new_state_dict, strict=False)
        
        model = model.backbone
    
    else:
    
        # here we will load one of the VED-trained models    
        
        if training_type=='FoveaGaze':
            checkpoint_file = '/user_data/dylandia/eb_notebooks/SimCLR-master/runs/Fovea_Gaze_SimCLR_Jan282026/checkpoint_0120.pth.tar'
        elif training_type=='Baseline':
            checkpoint_file = '/user_data/dylandia/eb_notebooks/SimCLR-master/runs/Baseline_SimCLR_Jan282026/checkpoint_0120.pth.tar'
        elif training_type=='PeriphNonTTM':
            checkpoint_file = '/user_data/dylandia/eb_notebooks/SimCLR-master/runs/Periph_NonTTM_SimCLR_Jan282026/checkpoint_0120.pth.tar' 
        elif training_type=='PeriphTTM':
            checkpoint_file = '/user_data/dylandia/eb_notebooks/SimCLR-master/runs/PeriphTTM_SimCLR_Feb042026/checkpoint_0120.pth.tar'
        elif training_type=='Fusion':
            checkpoint_file = '/user_data/dylandia/eb_processedData/simclr_fusion_runs/fusion_simclr_NEW/late_20260207_124501/ckpt_epoch_0120.pt'
        else:
            raise ValueError('training type %s not recognized'%training_type)

        print(checkpoint_file)
        checkpoint = torch.load(checkpoint_file, map_location = device)

        out_dim = 128
        
        if training_type!="Fusion":
            model = ResNetSimCLR(base_model=model_architecture, out_dim=out_dim)
            model.load_state_dict(checkpoint['state_dict'])
            model = model.backbone
        else:
            assert(model_architecture=='resnet50-fusion')
            model = fusion_simclr_normal_v2.LateFusionSimCLR(out_dim=out_dim)
            model.load_state_dict(checkpoint['model'])
            model = model
    
    model = model.to(device)
    model.eval()
    
    # check that everything i want to extract is in the named modules here.
    # for a different implementation, we may need to tweak names.
    module_names = list(dict(model.named_modules()).keys())
    assert np.all(np.isin(layers_to_do, module_names))

    # create the extractor object: this will handle extracting intermediate feature activs.
    model_extractor = tx.Extractor(model, layers_to_do)

    image_tensors = torch.Tensor(image_batch).to(device)
    _, features = model_extractor(image_tensors)

    activ = []
    for l in list(features.keys()):
        print(l)
        print(features[l].shape)
        activ += [features[l]]
    
    return activ

def normalize_ims(image_data):

    m = np.array([0.485, 0.456, 0.406])
    s = np.array([0.229, 0.224, 0.225])
    m = m.reshape(1, 3, 1, 1)
    s = s.reshape(1, 3, 1, 1)
    # ^ these values are approx mean and std of imagenet
    # want data rescaled to this range, to use w/ imagenet-trained models.
    # will generally work ok for other models too (but should check)
    
    image_data_norm = (image_data/255) # rescale to 0-1
    
    image_data_norm = (image_data_norm-m)/s
    
    return image_data_norm
    

def get_layers(model_architecture):

    # always choosing output of each conv block
    # plus first conv1 layer and last avgpool layer.
    if model_architecture=='resnet18':
        layers_to_do = ['conv1', 
                'layer1.1', \
                'layer2.1', \
                'layer3.1', \
                'layer4.1', \
                'avgpool']
        
    elif model_architecture=='resnet50':
        layers_to_do = ['conv1',
                'layer1.2',
                'layer2.3',
                'layer3.5',
                'layer4.2',
                'avgpool']
        
    elif model_architecture=='resnet50-fusion':
        layers_to_do = \
            ['fovea_enc.0',
             'fovea_enc.4.2',
             'fovea_enc.5.3',
             'fovea_enc.6.5',
             'fovea_enc.7.2',
             'periph_enc.0',
             'periph_enc.4.2',
             'periph_enc.5.3',
             'periph_enc.6.5',
             'periph_enc.7.2',
             'fuse']
    
    layer_names = [l.replace('.','-') for l in layers_to_do]

    return layers_to_do, layer_names


        
