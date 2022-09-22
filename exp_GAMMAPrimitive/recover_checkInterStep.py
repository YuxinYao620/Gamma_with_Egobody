import numpy as np
import argparse
import os
import sys
import pickle
import csv
import torch

sys.path.append(os.getcwd())
from utils import *
from exp_GAMMAPrimitive.utils.config_creator import ConfigCreator
from exp_GAMMAPrimitive.utils.batch_gen_openpose import BatchGeneratorOpenposeCanonicalized
from models.model_GAMMA_primitive_rec_checkInter import GAMMAPrimitiveComboRecOP as RecOP
# from models.models_GAMMA_longer_primitive import GAMMAPrimitiveComboGenOP as GenOP
import pdb

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default=None) # specify the model to evaluate
    parser.add_argument('--testdata') # which dataset to evaluate? choose only one
    parser.add_argument('--gpu_index', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    '''''dont touch these two, used for all exps'''
    N_SEQ = 3 #for each gender
    N_GEN = 3

    """setup"""
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dtype = torch.float32
    torch.set_default_dtype(dtype)
    cfgall = ConfigCreator(args.cfg)
    modelcfg = cfgall.modelconfig
    losscfg = cfgall.lossconfig
    traincfg = cfgall.trainconfig
    predictorcfg = ConfigCreator(modelcfg['predictor_config'])
    regressorcfg = ConfigCreator(modelcfg['regressor_config'])
    t_his = predictorcfg.modelconfig['t_his']
    
    load_pretrained_model=True
    testcfg = {}
    testcfg['gpu_index'] = args.gpu_index
    testcfg['ckpt_dir'] = traincfg['save_dir']
    # testcfg['testdata'] = 'canicalized-camera-wearer'
    # print('testcfg[\'testdata\']:', testcfg['testdata'])
    testcfg['result_dir'] = predictorcfg.cfg_result_dir if load_pretrained_model else cfgall.cfg_result_dir
    # print("testcfg.result_dir:{}".format(testcfg['result_dir']))
    testcfg['seed'] = args.seed
    testcfg['log_dir'] = cfgall.cfg_log_dir
    testcfg['training_mode'] = False
    testcfg['batch_size'] = N_GEN
    # testcfg['ckpt'] = args.ckpt


    """data"""
    testing_data = ['canicalized-camera-wearer-grab-openpose']

    if len(testing_data)>1:
        raise NameError('performing testing per dataset please.')
    from exp_GAMMAPrimitive.utils import config_env
    amass_path = config_env.get_amass_canonicalized_path()
    # print("amass_path:{}".format(amass_path))
    batch_gen = BatchGeneratorOpenposeCanonicalized(amass_data_path=amass_path,
                                                 amass_subset_name=testing_data,
                                                 sample_rate=1,
                                                 body_repr=predictorcfg.modelconfig['body_repr'],
                                                 read_to_ram=False)

    batch_gen.get_rec_list(shuffle_seed=args.seed)
    print('[INFO] load rec_list from dataset. This is not recommended for comparison')
    
    """models"""
    testop = RecOP(predictorcfg, regressorcfg, testcfg)
    testop.build_model(load_pretrained_model=load_pretrained_model)

    import pdb
    """similarity pair"""
    testop.recover_primitive_to_files(batch_gen,
                    n_seqs=N_SEQ, n_gens=testcfg['batch_size'],t_his=t_his)