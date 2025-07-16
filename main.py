import sys
import os         
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)
from sklearn.model_selection import train_test_split
import argparse
import numpy as np
import deepxde as dde # version 0.11 or higher
from generate_plot import plot_results
import utils
import pinn
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys
import os         
from sklearn.model_selection import train_test_split
import argparse
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec

import fire
import os
from utils_fire import get_config, get_logger, make_result_dir
import shutil


## General Params
noise = 0.1 # noise factor
test_size = 0.9 # precentage of testing data


def main(config_file: str, name: str) -> None:
    """Run training script

    Parameters
    ----------
    config_file : str
        Path to config file.
    """

    # task = Task.init(project_name="Frontiers2024", task_name=name)
    # task.set_resource_monitor_iteration_timeout(180)
    #  set random seed
    # np.random.seed(42)

    #  setup logger
    logger = get_logger()

    #  read config file
    config = get_config(config_file)

    #track config with ClearML
    # _ = task.connect_configuration(config)

    #  print config info into log
    log_message =\
        f"Run training for config: {config}"
    logger.info(log_message)

    result_dir_name = name
    result_dir = make_result_dir(result_dir_name)
    
    #  save config file into result directory
    shutil.copyfile(config_file, os.path.join(result_dir, result_dir_name+'.yaml'))
    
    ## Get Dynamics Class
    dynamics = utils.system_dynamics(**config)
    
    ## Parameters to inverse (if needed)
    params = dynamics.params_to_inverse(**config)
    
    ## Generate Data 
    v_file_name = config['data']['core_name'] +config['data']['v_file_name']
    w_file_name = config['data']['core_name'] +config['data']['w_file_name']
    pt_file_name = config['data']['core_name'] +config['data']['pt_file_name']  
    scenario_name = config['data']['scenario_name']
    ionic_model_name = config['data']['ionic_model_name']
    observe_train, observe_test, v_train, v_test, w_train, w_test, Vsav, V, len_t, idx_data_larger, Wsav, W = dynamics.generate_data(v_file_name, w_file_name, pt_file_name, scenario_name, ionic_model_name)  #Temporary becasue we dont have W yet!! Plz add w_file_name later

    
    ## Add noise to training data if needed
    if config['data']['noise']:
        v_train = v_train + noise*np.random.randn(v_train.shape[0], v_train.shape[1])

    ## Geometry and Time domains
    geomtime = dynamics.geometry_time()
    ## Define Boundary Conditions
    bc = dynamics.BC_func(geomtime)
    ## Define Initial Conditions
    ic = dynamics.IC_func(observe_train, v_train)
    
    ## Model observed data
    observe_v = dde.PointSetBC(observe_train, v_train, component=0)
    input_data = [bc, ic, observe_v]
    if config['data']['w_input']: ## If W required as an input
        observe_w = dde.PointSetBC(observe_train, w_train, component=1)
        input_data = [bc, ic, observe_v, observe_w]
    
    ## Select relevant PDE (Heterogeneity) and define the Network
    model_pinn = pinn.PINN(dynamics, config['data']['heter'], config['data']['inverse'])
    model_pinn.define_pinn(geomtime, input_data, observe_train)
            
    ## Train Network
    out_path = config['data']['core_name'] + result_dir #config['data']['model_folder_name']
    model, losshistory, train_state = model_pinn.train(out_path, params)
    
    ## Compute rMSE
    pred = model.predict(observe_test)   
    v_pred, w_pred = pred[:,0:1], pred[:,1:2]
    rmse_v = np.sqrt(np.square(v_pred - v_test).mean())
    print('--------------------------')
    print("V rMSE for test data:", rmse_v)
    print('--------------------------')
    # print("Arguments: ", args)
    
    ## Save predictions, data
    np.savetxt(os.path.join(result_dir, "train_data.dat"), np.hstack((observe_train, v_train, w_train)),header="observe_train,v_train, w_train")
    np.savetxt(os.path.join(result_dir, "test_pred_data.dat"), np.hstack((observe_test, v_test,v_pred, w_test, w_pred)),header="observe_test,v_test, v_pred, w_test, w_pred")


    file_name = result_dir + "_W" # config['data']['model_folder_name'] + "_W"

    plot_results(Wsav, W, observe_train, w_train, observe_test, w_pred, len_t, idx_data_larger, file_name, False)
    #save Vsav and V
    v_pred_train = np.concatenate((observe_train, v_train), axis=1)
    v_pred_test = np.concatenate((observe_test, v_pred), axis=1)
    v_pred_min = np.min(v_pred)
    v_pred_max = np.max(v_pred)

    stacked_array = np.vstack((v_pred_train, v_pred_test)) #Combining the training & testing data into one array
    sorted_array = stacked_array[np.lexsort((stacked_array[:, 2], stacked_array[:, 0], stacked_array[:, 1]))] #Rearranging the array's order based on x,y and t
    V_pred = np.array(sorted_array[:, 3],) #Only include the arrray of V in the correct order 
    Vsav_pred = V_pred.reshape(Vsav.shape)
    
    np.save(os.path.join(result_dir, "Vpred.npy"), Vsav_pred)
    np.save(os.path.join(result_dir, "Vsav.npy"), Vsav)
    np.save(os.path.join(result_dir, "V.npy"), V)
    plot_results(Vsav, V, observe_train, v_train, observe_test, v_pred, len_t, idx_data_larger, out_path, config['data']['animation'])

## Run main code
# model = main(args)

if __name__ == '__main__':
    fire.Fire(main)
