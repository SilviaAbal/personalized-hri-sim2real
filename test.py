#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: sabal

Test script for the Personalized HRI Sim2Real framework.

This module evaluates trained RL policies in the simulation environment.

Main responsibilities:
- Load trained checkpoints
- Run inference (no exploration)
- Collect performance metrics
- Save evaluation statistics

NOTE:
This script runs ONLY simulation-based evaluation.
No real robot or human experiments are included.
"""


import os
from sim_env import SimulatedEnv
import random
import matplotlib
matplotlib.use('Agg')

# Ensure deterministic CUDA behavior (important for reproducibility)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from DQN import DQN_LSTM_LateFusion
from statTracker import StatTracker
from utils import (
    saveConfiguration, loadConfiguration, creationSavePath,
    resumeTraining, initialize_replay_memory, save_model,
    updateHidden, saveCurrentData, loadCurrentData,
    updateTargetNet, load_policy_net, load_pretrained_model
)

import copy 
import argparse
import gc 
import dill as pickle
import re

# --------------------------------------------------
# Utility: get available trained models
# --------------------------------------------------
def get_model_numbers(folder_path):
    """
    Extracts and returns sorted model numbers from a given folder.
    """
    files = os.listdir(folder_path)
    pattern = re.compile(r'model_(\d+).pt')
    
    # Extract and sort model numbers
    file_numbers = [int(match.group(1)) for file in files if (match := pattern.match(file))]
    file_numbers_sorted = sorted(file_numbers)
    print(f"Models found: {file_numbers_sorted}")
    
    return file_numbers_sorted

# --------------------------------------------------
# Action selection (NO exploration)
# --------------------------------------------------

def selectAction(state, exp, env):
    """
    This function implements and e-greedy policy for action taking
    """

    # 1 - Compute the epsilonTh
    policyNet = exp['policyNet']
    targetNet = exp['targetNet']
    cfg = exp['cfg']
    
    epsTh = cfg.EPS_TEST
   
    #Make decisions and advance states for both policy and target
    with torch.no_grad():
        out = policyNet(state)[0]
        _ = targetNet(state)[0]
        
    # Mask invalid actions (objects already used)
    if cfg.NO_IMPOSSIBLE_ACTIONS:
        objects_in_table = copy.deepcopy(env.oit)
        manipulable_objects = list(cfg.ROBOT_ACTIONS_2_OBJECTS.values())[:-1]
        mani_obj_table = [objects_in_table[obj] for obj in manipulable_objects]
        mani_obj_table.append(0) # la ultima accion es no hacer nada y siempre se puede hacer
       
        #Remove changes of objects that are in table
        out[0,np.array(mani_obj_table)>0]=-torch.inf

    #Simply choosing the best option
    action = out.max(1)[1].view(1,1)

    # we need to return the original hidden state of the policyNet for the replay memory
    pHidden = (policyNet.prevHidden[0].clone(), policyNet.prevHidden[1].clone())

    return action, pHidden, epsTh 


def test(epoch, exp, env, st, nfold=None, args= None):
    """
    Run a test epoch on the policy network, tracking performance metrics
    and storing action decisions.
    """
    
    print('\tTest epoch {%d/%d}' %(epoch+1, exp['cfg'].EPOCHS))

    cfg = exp['cfg']

    # Load model if only testing
    policyNet,robot_execution_times = load_policy_net(epoch, exp, nfold, args) 
    exp['policyNet'] = policyNet
    exp['robot_execution_times'] = robot_execution_times
        

    for epi in range(env.numEpisodes):
        
        if cfg.TEST_VERBOSE:
            print('************')
            print('epi: ',env.episode.folderName)

        # 0 - reset the initial hidden states of the models
        exp['policyNet'].resetHidden()
        # 1 - Get initial state
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=cfg.DEVICE).unsqueeze(0)
        
        # for each time step until end of episode
        t = 0
        while True:
            # 3.1 select an action based on current state and policy
            action, pHidden, epsTh = selectAction(state, exp, env)

            # 3.2 take the step
            (nextState, reward, done, _, Rt, Re) = stepTest(exp, action.item(), epoch, epi, t, env, st)

            # 3.4 move to the next state
            state = nextState

            # 3.7 track some useful stats
            st.add(epoch, epi,t,'instant_reward', reward)
            st.add(epoch, epi,t,'time_reward', Rt)
            st.add(epoch, epi,t,'energy_reward', Re)
            
            # st.add(epoch, epi,t, 'anticipationCAFin', env.anticipationCorrectRobotSinceFinishing)
            st.add(epoch, epi,t, 'stateActionPrediction', env.trackerActPred.get_intervals())
            st.add(epoch, epi,t, 'stateActionRecognition', env.trackerActRec.get_intervals())
            st.add(epoch, epi,t, 'stateVideoInteraction', env.trackerVideo.get_intervals())
           
            t += 1
            # 3.8 check if the episode is finished
            if done:    
                break
            
        st.add(epoch, epi,0,'video_prob')
        st.add(epoch, epi,0,'min_robot_time', env.episode.minRobotTime)
        st.add(epoch, epi,0,'episode_delay', env.episode.delay)
        st.add(epoch, epi,0,'reactive_total_reward', env.episode.reactiveReward)
        st.add(epoch, epi,0,'reactive_time_reward', env.episode.reactiveTimeReward)
        st.add(epoch, epi,0,'reactive_energy_reward', env.episode.reactiveEnergyReward)
        st.add(epoch, epi,0,'min_total_reward', env.episode.minReward)
        st.add(epoch, epi,0,'min_time_reward', env.episode.minTimeReward)
        st.add(epoch, epi,0,'min_energy_reward', env.episode.minEnergyReward)
        st.add(epoch, epi,0,'episode_name', env.episode.folderName)
        st.add(epoch, epi,0, 'min_time', env.episode.minTime)
        st.add(epoch, epi,0,'reactive_delay', env.episode.reactiveDelay)
        st.add(epoch, epi,0,'min_delay', env.episode.minDelay)
        st.add(epoch, epi,0, 'interaction_time', env.episode.interactionTime)
        st.add(epoch, epi,0,'reactive_time', env.episode.reactiveTime)
        st.add(epoch, epi,0, 'healthy_human', env.episode.healthyHuman)
        st.add(epoch, epi,0, 'CA_intime', env.CA_intime)
        st.add(epoch, epi,0, 'CA_late', env.CA_late)
        st.add(epoch, epi,0, 'IA_intime', env.IA_intime)
        st.add(epoch, epi,0, 'IA_late', env.IA_late)
        st.add(epoch, epi,0, 'CI', env.CI)
        st.add(epoch, epi,0, 'II', env.II)
        st.add(epoch, epi,0, 'IMPOSA', env.IMPOSA)
        st.add(epoch, epi, 0, 'anticipationCAandASR', env.anticipationCAandASR)
        
    torch.cuda.empty_cache()
    return

# --------------------------------------------------
# Experiment initialization
# --------------------------------------------------
def initExperiment(cfg, fold, step, save_path, mode='train'):
    """
    Initialize variables for the experiment.
    
    Parameters:
        cfg: Configuration object containing hyperparameters.
        fold: Path to the training/testing data file.
        step: The current step of the experiment.
        mode: train, ft, test
    Returns:
        exp: A dictionary containing all initialized experiment variables.
    """

    print("Initializing experiment variables...")

    # 1 - Initialize enviroment
    envTest = SimulatedEnv(mode='test', testFile=fold,configuration=cfg)
    num_actions = len(cfg.ROBOT_ACTIONS_MEANINGS)

    state, info = envTest.reset()
    state_dimensionality = len(state)
    print(state_dimensionality)
    
    # 'latefusion_lstm' (tu RL actual)
    policyNet = DQN_LSTM_LateFusion(state_dimensionality, num_actions, cfg).to(cfg.DEVICE)
    targetNet = DQN_LSTM_LateFusion(state_dimensionality, num_actions, cfg).to(cfg.DEVICE)

    # 2 - Initialize DQN networks, optimizer and loss function
    try:
        optimizer = optim.AdamW(policyNet.parameters(), lr = cfg.LR,amsgrad=True,weight_decay=cfg.WD) #1e-4 
    except:
        optimizer = optim.AdamW(policyNet.parameters(), lr = cfg.LR,amsgrad=True,weight_decay=1e-4) #1e-4 

    stepsDone = 0

    if cfg.LOAD_MODEL:
        pt_model, robot_execution_times, optimizer = load_pretrained_model(cfg.MODEL_PATH, optimizer, state)

    #assign target network
    targetNet.load_state_dict(policyNet.state_dict())
    #Set target network to eval
    targetNet.eval()

    #Loss
    criterion = nn.SmoothL1Loss()
    
    #Stat-tracker test
    stTest = StatTracker()

    # 4 - Initialize replay memory
    updateTarget = 0
    
    # Compute the maximum reward for reward normalization
    maxReward=cfg.ROBOT_TIME_BETA*(1+cfg.FACTOR_ENERGY_PENALTY)
    cfg.maxReward=maxReward

    print('Normalizer for reward {}'.format(cfg.maxReward))    
    exp = {'cfg': cfg,
           'fold': fold,
           'envTest': envTest,
           'stepsDone': stepsDone,
           'policyNet': policyNet,
           'targetNet': targetNet,
           'optimizer': optimizer,
           'criterion': criterion,
           'statTrackerTest': stTest,
           'step':step,
           'updateTarget': updateTarget,
           'robot_execution_times': robot_execution_times,
    }
           
    
    print("\tDone!")

    return exp

def stepTest(exp, action, epoch, epi, ti, env, st):

    """
   This function performs a single step of the environment during testing
   """
   
    cfg = exp['cfg']
    actionsTaken = {}
    env.episode.clearActionsASR()
    
    # Record the current frame before taking the provided action
    prev_frame = copy.deepcopy(env.episode.currentFrame)
    prev_delay = copy.deepcopy(env.episode.delay)
    # Take the action in the environment and retrieve the resulting observation, reward, and termination status
    observation, reward, terminated, truncated, infoDict, _ = env.step(action)
    
    actionsTaken[prev_frame+prev_delay] = {'action': action, 'duration': env.durationActionRobot}

    # If hidden state updates are enabled, update the hidden state with frames between previous and current frames
    if cfg.UPDATE_HIDDEN:
        updateHidden(prev_frame, env.episode.currentFrame, exp,env)

    t = 1  # Time step increment
    
    # check if the last n_step ended on a terminal state
    if terminated:
        nextState = None
    else:
        nextState = torch.tensor(observation, dtype=torch.float32, device=cfg.DEVICE).unsqueeze(0)

    done = terminated or truncated
    
    # Log the action taken to the statistics tracker
    st.add(epoch, epi, ti, 'actions_taken', actionsTaken)
    st.add(epoch, epi, ti, 'actions_ASR', env.episode.actionsASR)
    
    return (nextState, reward, done, t, infoDict['timeReward'], infoDict['energyReward'])

def runTest(cfg, args):
    """
    Run the test phase of the experiment by loading models, testing them, 
    and saving relevant statistics and results.
    
    Arguments:
    cfg -- Configuration object containing experiment parameters
    """
    
    for nf,fold in enumerate(cfg.FOLDS):
        print("Starting fold: %s" %(fold))

        if cfg.FIVE_FOLD == False:
            nfold = cfg.FOLDS[0].split('_')[0].split('fold')[-1]
            folder_path=cfg.save_path + f"_fold{nfold}/"

        else:
            nfold = (5-len(cfg.FOLDS))+nf
            if nfold == 0:
                pre_save_path = cfg.save_path
            folder_path  =pre_save_path + '_fold{}'.format(nfold+1) +'/'

        cfg.save_path = folder_path
        print(f"save path: {cfg.save_path}")

        exp = {}
        
        # 1 - Init experiment env, models, opt, loss, stat objects, etc
        exp_i = initExperiment(cfg, fold, cfg.N_STEP, folder_path, mode='test')
        
        print("Loading experiment: %s" % (folder_path))
        print('fold test: ', fold)
        print('N STEP: ', cfg.N_STEP)

        # Get sorted model numbers
        file_numbers_sorted = get_model_numbers(folder_path)

        # Test loop
        for i, model in enumerate(file_numbers_sorted):
            test(model, exp_i, exp_i['envTest'], exp_i['statTrackerTest'],nfold=nfold, args=args)

        with open(folder_path+'/stat_tracker_test.pkl', 'wb') as f:
            pickle.dump(exp_i['statTrackerTest'], f)

        torch.cuda.empty_cache()
        gc.collect()
       
        print("\tDone!")

    return exp

if( __name__ == '__main__' ):

    print('using CUDA GPU') if torch.cuda.is_available()  else print('using CPU')

    # 1 - Load the experiment config
    # Argument parser for command-line arguments
    parser = argparse.ArgumentParser(description="Run experiment with a specified config file.")
    
   # Add argument for config file with a default value
    parser.add_argument(
        '--config',
        type=str,
        default='2',  # Default config number (corresponds to config2.py)
        help='Specify the config file number to use (e.g., 1 for config1.py). Default is 2 (config2.py)'
    )

    parser.add_argument(
        '--name',
        type=str,
        default='exp',  # Default config number (corresponds to config2.py)
        help='Name of the experiment'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=97,  # Default config number (corresponds to config2.py)
        help='Seed of the experiment'
    )
    

    parser.add_argument(
        '--path_exp',
        type=str,
        default='Checkpoints',  # Default config number (corresponds to config2.py)
        help='Name of the folder experiment'
    )
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Construct the config file name based on the argument (e.g., config1, config2, etc.)
    config_name = f"config{args.config}"  # This will create strings like 'config1', 'config2', etc.
    
    # Load the config module dynamically
    cfg = loadConfiguration(config_name)
    
    if args.seed != 97:
        cfg.SEED = args.seed 

    # Creation save path 
    save_path=creationSavePath(cfg, args)
    cfg.save_path = save_path
    print(save_path)
   
    print('***** Test Mode *****')
    print('SEED: ', cfg.SEED)
  
    # Now you can use the configuration settings from the specified config file
    print(f"Using configuration: {config_name}.py")
    
    torch.manual_seed(cfg.SEED)
    random.seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    torch.cuda.manual_seed_all(cfg.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    # 2 - Run test process
    exp = runTest(cfg, args)