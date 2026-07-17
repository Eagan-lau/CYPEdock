import numpy as np
import mdtraj as md
import os
import datetime
from models import MQAModel
from utils import process_strucs, predict_on_xtals
import argparse

parser = argparse.ArgumentParser(description='Process data')
parser.add_argument('-source_path', '--source_path', type=str, required=True, help='Folder of pdb')
parser.add_argument('-target_path', '--target_path', type=str, required=True, help='output folder')

# 
args = parser.parse_args()
# # 
source_path = args.source_path
target_path = args.target_path



def make_predictions(pdb_paths, model, nn_path, debug=False, output_basename=None):
    '''
        pdb_paths : list of pdb paths
        model : MQAModel corresponding to network in nn_path
        nn_path : path to checkpoint files
    '''
    strucs = [md.load(s) for s in pdb_paths]
    try:
        X, S, mask = process_strucs(strucs)
        predictions = predict_on_xtals(model, nn_path, X, S, mask)
        return predictions
    except Exception as e:
        print(f"Error processing data: {e}")
        return None



# main method
if __name__ == '__main__':
    # Load MQA Model used for selected NN network
    nn_path = "../model/pocketminer"
    DROPOUT_RATE = 0.1
    NUM_LAYERS = 4
    HIDDEN_DIM = 100
    model = MQAModel(node_features=(8, 50), edge_features=(1, 32),
                     hidden_dim=(16, HIDDEN_DIM),
                     num_layers=NUM_LAYERS, dropout=DROPOUT_RATE)
    # debugging mode can be turned on to output protein features and sequence
    debug = False
    # TO DO - provide input pdb(s), output name, and output folder
    #source_path = "/media/yugengliu/DATA/whli/Pocekt/Ori_PDB2"
    #target_path = "/media/yugengliu/DATA/whli/Pocekt/tokens/all_tokens"
    for file_name in os.listdir(source_path):
        strucs = [source_path + '/' + file_name]
        output_name = file_name.split('.')[0]
        output_folder = target_path
        print(f"Start time: {datetime.datetime.now()}")
        print(f"Processing {file_name}")
        if debug:
            output_basename = f'{output_folder}/{output_name}'
            predictions = make_predictions(strucs, model, nn_path, debug=True, output_basename=output_basename)
        else:
            predictions = make_predictions(strucs, model, nn_path)
        print(f"Done time: {datetime.datetime.now()}")
        # output filename can be modified here
        if predictions is not None:
            np.save(f'{output_folder}/{output_name}-tokens.npy', predictions)
        # np.save(f'{output_folder}/{output_name}-preds.npy', predictions)
        # np.savetxt(os.path.join(output_folder, f'{output_name}-predictions.txt'), predictions, fmt='%.4g', delimiter='\n')


