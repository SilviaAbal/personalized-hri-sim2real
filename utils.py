#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions for training and evaluating the Personalized HRI Sim2Real framework.

This module groups together helper functions used across the training pipeline,
including:

- configuration loading and saving
- experiment checkpointing and resume logic
- replay memory initialization/loading
- model checkpoint saving/loading
- target-network updates
- hidden-state maintenance for recurrent models
- temporary environment/model state snapshots

These utilities are used by the training script to keep the main training loop
clean and focused on the RL logic.
"""

import pandas as pd 
import copy 
import dill as pickle
import importlib
import os
import torch 
from memory import ReplayMemory
import numpy as np

def saveConfiguration(exp_i, save_path):
    """
    Save the experiment configuration to disk as a CSV file.

    The saved file contains:
    - the active fold
    - all non-callable configuration attributes

    This is useful for reproducibility and for tracking which settings were used
    for each run.
    """
    data_cfg = {'FOLDS': [exp_i['fold']]}

    cfg = exp_i['cfg']
    attributes = [
        a for a in dir(cfg)
        if not a.startswith('__') and not callable(getattr(cfg, a))
    ]

    for attribute in attributes:
        data_cfg[attribute] = [getattr(cfg, attribute)]

    df_cfg = pd.DataFrame(data_cfg)
    df_cfg.to_csv(save_path +'/cfg.csv')

    return 

def loadConfiguration(config_name):
    """
    Dynamically load a configuration module from the configs package.

    Parameters
    ----------
    config_name : str
        Configuration module name without the .py extension
        (e.g., 'config2').

    Returns
    -------
    module
        Imported configuration module.
    """
    cfg = importlib.import_module('configs.' + config_name)
    return cfg


def creationSavePath(cfg, args):
    """
    Create the root checkpoint directory for the current experiment.

    Parameters
    ----------
    cfg : module/object
        Configuration object containing the experiment name.
    args : argparse.Namespace
        Command-line arguments (kept for compatibility, even if not used here).

    Returns
    -------
    str
        Absolute path to the checkpoint directory.
    """
    root = os.getcwd()
    save_path = os.path.join(root, "Checkpoints", cfg.EXPERIMENT_NAME)
    os.makedirs(save_path, exist_ok=True)

    return save_path


def resumeTraining(cfg,save_path, policyNet,optimizer):
    """
    Resume training from the most recent checkpoint found in save_path.

    This function restores:
    - model weights
    - optimizer state
    - replay memory
    - training and validation statistics
    - robot execution times (if available)
    - the epoch/step counter

    If no checkpoint is found, the function returns the provided model and
    optimizer unchanged, together with None placeholders.
    """
    rm = None
    st = None
    stVal = None 
    robotExecutionTimes = None
    stepsDone = 0 
    try:
        # Search checkpoints from the last epoch backwards.
        for e in range(cfg.EPOCHS - 1, -1, -1):
            model_path = os.path.join(save_path, f'model_{e}.pt')

            if os.path.exists(model_path):
                print(f'Resuming from epoch {e}')

                # Load replay memory
                replay_memory_path = os.path.join(save_path, 'replay_memory.pkl')
                with open(replay_memory_path, 'rb') as f:
                    rm = pickle.load(f)

                # Load training statistics
                train_st_path = os.path.join(save_path, 'stat_tracker_train.pkl')
                with open(train_st_path, 'rb') as f:
                    st = pickle.load(f)
                f.close()
                print(' St train loaded ')

                # Load validation statistics
                val_st_path = os.path.join(save_path, 'stat_tracker_val.pkl')
                with open(val_st_path, 'rb') as f:
                    stVal = pickle.load(f)
                f.close()
                print(' St val loaded ')

                # Restore model and optimizer
                checkpoint = torch.load(model_path, weights_only=False)
                policyNet.load_state_dict(checkpoint['model_state_dict'])
                #Load the optimizer
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

                # Optional field in newer checkpoints
                robotExecutionTimes = checkpoint.get('robot_execution_times', None)

                # Restore training progress
                stepsDone = checkpoint['steps']
                break

        print('Resuming has finished')
        
        return policyNet, optimizer, rm, st, stVal, robotExecutionTimes, stepsDone

    except Exception:
        print('No previous models. Starting from scratch.')
        return policyNet, optimizer, None, None, None, None,0
    
def initialize_replay_memory(cfg, memory_size):
    """
    Initialize the replay memory used during RL training.

    Parameters
    ----------
    cfg : module/object
        Configuration object.
    memory_size : int
        Maximum replay memory capacity.

    Returns
    -------
    ReplayMemory
        Initialized replay buffer.
    """

    return ReplayMemory(cfg, memory_size)



def load_replay_memory(path, cfg):
    """
    Load replay memory from a previous run.

    Parameters
    ----------
    path : str
        Directory containing replay_memory.pkl.
    cfg : module/object
        Configuration object (kept for compatibility).

    Returns
    -------
    ReplayMemory or None
        Loaded replay memory, or None if it does not exist.
    """
    # GLOBAL REPLAY MEMORY
    try:
        with open(os.path.join(path, 'replay_memory.pkl'), 'rb') as f:
            rm = pickle.load(f)
    except Exception:
        print('No rm found')
        rm = None
    return rm


def save_model(exp, epoch, loss, save_path):
    """
    Save the current training checkpoint.

    The checkpoint includes:
    - policy network weights
    - optimizer state
    - current epoch
    - latest loss
    - number of training steps performed
    - current robot execution time estimates
    """
    model_name = f'model_{epoch}.pt'
    
    full_path = os.path.join(save_path, model_name)
    print(f"Saving model at {full_path}")
    
    # Save model state
    torch.save({
        'model_state_dict': exp['policyNet'].state_dict(),
        'optimizer_state_dict': exp['optimizer'].state_dict(),
        'epoch': epoch,
        'loss': loss,
        'steps': exp['stepsDone'],
        'robot_execution_times': exp['env'].get_robot_execution_times()
    }, full_path)
    
    
    
def updateHidden(prev_frame, current_frame, exp,env):
    
    """
    Update recurrent hidden states when the simulator skips frames.

    Since the environment may jump several frames at once, the LSTM would miss
    intermediate temporal context if we only fed the final state. This function
    reconstructs the skipped states and forwards them through both policy and
    target networks so their hidden states remain temporally consistent.
    """
    cfg = exp['cfg']
    policyNet = exp['policyNet']
    targetNet = exp['targetNet']
    
    # List to store states between prev_frame and current_frame
    states = []

    # Nothing to update if no temporal jump occurred.
    if prev_frame == current_frame:
        return

    # If the current frame is beyond the episode end, stop early.
    if current_frame >= env.episode.lastFrame:
        return

    # Rebuild skipped states at decision-rate intervals.
    for frame in range(prev_frame + cfg.DECISION_RATE, current_frame, cfg.DECISION_RATE):
        state = env.buildStateHidden(frame)
        states.append(state)
    
    if states:
        states_np = np.array(states, dtype=np.float32)
        states_tensor = torch.tensor(states_np, dtype=torch.float32, device=cfg.DEVICE)

        with torch.no_grad():
            _ = policyNet(states_tensor)[0]
            _ = targetNet(states_tensor)[0]


def saveCurrentData(env, state, targetNet, policyNet):
    """
    Snapshot the current simulator and network hidden-state information.

    This is used when the code temporarily branches the environment evolution
    (for example, to generate additional replay-memory samples such as ASR-based
    corrections) and later needs to restore the original state exactly.
    """
    curr_frame = env.episode.currentFrame
    curr_delay = env.episode.delay
    curr_oit = copy.deepcopy(env.oit)
    curr_asr = copy.deepcopy(env.asr)
    curr_state = copy.deepcopy(state)
    curr_THidden = (targetNet.hidden[0].clone(), targetNet.hidden[1].clone())
    curr_PHidden = (policyNet.hidden[0].clone(), policyNet.hidden[1].clone())
    curr_terminated = env.episode.terminated

    curr_data = {
        'frame': curr_frame,
        'delay': curr_delay,
        'oit': curr_oit,
        'asr': curr_asr,
        'state': curr_state,
        'THidden': curr_THidden,
        'PHidden': curr_PHidden,
        'terminated': curr_terminated
    }
    return curr_data


def loadCurrentData(data, env, exp):
    """
    Restore a simulator/network snapshot previously created by saveCurrentData().

    Returns
    -------
    torch.Tensor
        Restored environment state tensor.
    """
    env.episode.currentFrame = data['frame']
    env.episode.delay = data['delay']
    env.oit = copy.deepcopy(data['oit'])
    env.asr = copy.deepcopy(data['asr'])
    state = data['state']

    exp['policyNet'].hidden = (
        data['PHidden'][0].clone(),
        data['PHidden'][1].clone()
    )
    exp['targetNet'].hidden = (
        data['THidden'][0].clone(),
        data['THidden'][1].clone()
    )

    env.episode.terminated = data['terminated']
    
    return state
def updateTargetNet( exp ):

    """
    Soft-update the target network from the policy network.

    Uses the standard DQN-style interpolation:
        target = TAU * policy + (1 - TAU) * target
    """
    targetNet = exp['targetNet']
    policyNet = exp['policyNet']
    TAU = exp['cfg'].TAU

    targetStateDict = targetNet.state_dict()
    policyStateDict = policyNet.state_dict()

    # soft update implementation
    for key in policyStateDict:
        targetStateDict[key] = policyStateDict[key]*TAU + targetStateDict[key]*(1-TAU)

    # finally, update the targetNet weights
    targetNet.load_state_dict(targetStateDict)
    return


def load_policy_net(epoch, exp, cfg, nfold=None):
    """
    Load a policy network checkpoint for evaluation or testing.

    Parameters
    ----------
    epoch : int
        Epoch number to load.
    exp : dict
        Experiment dictionary containing the network instance.
    cfg : module/object
        Configuration object containing save_path.
    nfold : optional
        Reserved for fold-specific logic.

    Returns
    -------
    tuple
        (loaded policy network, robot_execution_times)
    """
    if nfold is None:
        file = os.path.join(cfg.save_path, f'model_{epoch}.pt')
    else:
            file = os.path.join(cfg.save_path, f'model_{epoch}.pt')
            
    checkpoint = torch.load(file, weights_only=False)

    policyNet = exp['policyNet']
    policyNet.load_state_dict(checkpoint['model_state_dict'])
    policyNet.eval()
    if 'robot_execution_times' in checkpoint.keys():
        robot_execution_times = checkpoint['robot_execution_times']
    else:
        robot_execution_times = np.array(exp['cfg'].HUMAN_AVERAGE_DURATIONS[:-1])
        
    return policyNet, robot_execution_times

def load_pretrained_model(model_path, optimizer, state):
    """
    Load a pretrained model checkpoint for initialization.

    Parameters
    ----------
    model_path : str
        Directory containing model.pt.
    optimizer : torch.optim.Optimizer
        Optimizer object (kept for compatibility with caller logic).
    state : torch.Tensor
        Current state (unused here, kept for compatibility).

    Returns
    -------
    tuple
        (model_state_dict, robot_execution_times, optimizer)
    """
    #1.- Load model
    model_file = os.path.join(model_path, 'model.pt')
    # checkpoint = torch.load(model_file)
    checkpoint = torch.load(model_file, weights_only=False)

    model=checkpoint['model_state_dict']
    
    if 'robot_execution_times' in checkpoint.keys():
        robot_execution_times = checkpoint['robot_execution_times']
    else:
        robot_execution_times = None
    
    return model,robot_execution_times, optimizer    

