"""
Main training and evaluation script for the Personalized HRI Sim2Real framework, 
from "Reducing the Sim2Real Gap in Personalized HRI: Simulation-Based Pretraining and Real-World Adaptation and Validation" article.

This script supports:
- training RL agents in simulation
- evaluation of trained models

Usage:
    python train.py --config 2 

"""

import os
from sim_env import SimulatedEnv
import random
import matplotlib
matplotlib.use('Agg')
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from DQN2 import DQN_LSTM_LateFusion,DQN_MLP_LateFusion
from statTracker import StatTracker
from memory import ReplayMemory, createTransition
from utils import saveConfiguration,loadConfiguration,creationSavePath,resumeTraining, initialize_replay_memory, load_replay_memory,save_model, updateHidden, saveCurrentData, loadCurrentData, updateTargetNet, load_policy_net, load_pretrained_model

import pandas as pd
import re

# import pickle
import copy 
import argparse
import importlib
import gc 
import dill as pickle



def get_num_optimizations(memory_size, batch_size=128, 
                       min_optim=5, max_optim=20, 
                       memory_max=10000):
     """
     Computes the number of optimization steps based on the replay memory size.
     """
     # Normalize memory into [0, 1] range
     memory_ratio = min(1.0, memory_size / memory_max)
     # Compute number of optimizations using linear interpolation
     num_optim = int(min_optim + memory_ratio * (max_optim - min_optim))
 
     return num_optim
 



def selectAction(state, exp, env, test=False):
    """
    This function implements an epsilon-greedy policy
    """

    # 1 - Compute the epsilonTh
    policyNet = exp['policyNet']
    targetNet = exp['targetNet']
    cfg = exp['cfg']
    
    if test:
        epsTh = cfg.EPS_TEST
    else:
        epsTh = max(cfg.EPS_END, cfg.EPS_START - (cfg.EPS_START - cfg.EPS_END) * (exp['stepsDone'] / cfg.EPS_DECAY)) / cfg.EPS_START

    # 2 - with probability epsTh we take the random action
    r = random.random()

    #Make decisions and advance states for both policy and target
    with torch.no_grad():
        out = policyNet(state)[0]
        _ = targetNet(state)[0]
        
    # if NO_IMPOSSIBLE_ACTIONS is True, the system cannot perfom actions that were previouly taken in the same episode 
    # (current actions only involve taking elements from the fridge, so whenever a object is taken it is impossible to take it again in the same episode)
    if cfg.NO_IMPOSSIBLE_ACTIONS:
        objects_in_table = copy.deepcopy(env.oit)
        manipulable_objects = list(cfg.ROBOT_ACTIONS_2_OBJECTS.values())[:-1]
        mani_obj_table = [objects_in_table[obj] for obj in manipulable_objects]
        mani_obj_table.append(0) # la ultima accion es no hacer nada y siempre se puede hacer
        index_posible_actions = [i for i, x in enumerate(mani_obj_table) if x == 0]
        # Mask actions corresponding to objects already on the table
        out[0,np.array(mani_obj_table)>0]=-torch.inf
    else:
        index_posible_actions = list(range(0,len(env.action_space)))
        mani_obj_table = np.zeros(len(env.action_space))
    
    
    #If e-greedy exploration and conditions holds
    if cfg.EXPLORATION_MODE == 'EPS_GREEDY' and r < epsTh:   
        randomAction = random.choice(index_posible_actions)
        action = torch.tensor([[randomAction]], dtype=torch.long, device=cfg.DEVICE)

    # if prob-based exploration
    elif cfg.EXPLORATION_MODE == 'PROB_BASED' and epsTh>0:
         weights = F.softmax(out[0,...]/epsTh,0)
         randomAction = random.choices(env.action_space, weights, k=1)[0]
         action = torch.tensor([[randomAction]], dtype=torch.long, device=cfg.DEVICE)
         
    #Simply choosing the best option
    else:
         action = out.max(1)[1].view(1,1)

    # we need to return the original hidden state of the policyNet for the replay memory
    pHidden = (policyNet.prevHidden[0].clone(), policyNet.prevHidden[1].clone())

    return action, pHidden, epsTh 


def nStep(env,exp, action, epoch, epi, ti):
    
    """
    The environment is in a state and we are going to take n-actions (n-env.steps())
    If we want to implement an n-step, the first action is the one passed as
    argument to this function (the eps-greedy-possibly-random action). After the
    action passed as an argument, the next actions taken up to the n-step should
    all be following optimal policy. This is, using max_a(policyNet(newStates)).
    Once we had taken n-step actions, we can acumulate the discounted-reward 
    in the reward variable, and set as next state the ending state given back
    by the environment. The rest of the code outside this function shouldn't 
    change. With this approach we will take n-actions in a single time-step of
    the outside for t in count() loop for the episode. So episodes are likely to
    finish earlier and less updates to the models will be done as n-increases.

    If before finishing the n-steps we reach a terminal state, then, that particular
    entry por the memory won't be n-step long.
    """
    
    cfg = exp['cfg']
    policyNet = exp['policyNet']
    targetNet = exp['targetNet']

    N_STEP = exp['step']
    GAMMA  = cfg.GAMMA
    
    # G is the inmediate discounted n_step reward: 
    # Rt+1 + gamma*Rt+2 + gamma^2*Rt+3 ...
    G = 0
    t = 0
    actionsTaken = {}
    # actionsASR = {}
    
    # the first action is the epsilon greedy action
    sec_actions = []
    sec_actions.append(action)
    currentF = copy.deepcopy(env.episode.currentFrame)
    prev_delay = copy.deepcopy(env.episode.delay)
    #Take the first action and update the state
    observation, reward ,terminated, truncated, infoDict, actionASR = env.step(action)
    
    # If hidden state updates are enabled, update the hidden state with frames between previous and current frames
    if cfg.UPDATE_HIDDEN:
        updateHidden(currentF, env.episode.currentFrame, exp,env)
        
    actionsTaken[currentF+prev_delay] = {'action': action, 'duration': env.durationActionRobot}
    instant_reward = reward
    t += 1
    
    # Clip the reward if specified in the configuration
    if cfg.NORMALIZE_REWARD:
        G += instant_reward/cfg.maxReward
    else:
        G += instant_reward

    updateDelay = 0
    #The updated state is the state after the first action
    updateState = torch.tensor(observation, dtype=torch.float32, device=cfg.DEVICE).unsqueeze(0)
    
    # save several variables if the condition cfg.STEP_BY_STEP is True, saving the exact next time instant
    updateObjectsInTable = copy.deepcopy(env.oit)
    updateASR = copy.deepcopy(env.asr)
    updateCurrentFrame = env.episode.currentFrame
    updateTHidden = (targetNet.hidden[0].clone(), targetNet.hidden[1].clone())
    updateDelay = env.episode.delay
    updateTargetStateDict = targetNet.state_dict()
    updateBehaviourStateDict = policyNet.state_dict()
        

    # the rest of the N-step actions, have to be optimal
    if not terminated or truncated:
        with torch.no_grad():
            for n in range(1,N_STEP):
                # 1 - check if env has ended
                if terminated or truncated:
                    env.episode.terminated = False 
                    env.episode.truncated = False
                    terminated = False
                    truncated = False
                    break 
                nextState = torch.tensor(observation, dtype=torch.float32, device=cfg.DEVICE).unsqueeze(0)
                    
                # 2 - find the next optimal action (also advance targetNet)
                out = policyNet(nextState)[0]
                _ = targetNet(nextState)[0]
                
                if cfg.NO_IMPOSSIBLE_ACTIONS:
                    objects_in_table = copy.deepcopy(env.oit)
                    manipulable_objects = list(cfg.ROBOT_ACTIONS_2_OBJECTS.values())[:-1]
                    mani_obj_table = [objects_in_table[obj] for obj in manipulable_objects]
                    mani_obj_table.append(0) # la ultima accion es no hacer nada y siempre se puede hacer
                    #Remove changes of objects that are in table
                    out[0,np.array(mani_obj_table)>0]+=-torch.inf
                
                #3. Choose the next action as the optimal one
                nextAction = out.max(1)[1].view(1,1)
                # print(f'nextAction: {cfg.ROBOT_ACTIONS_MEANINGS[nextAction.item()]}')
                # 4 - take the a new step
                sec_actions.append(nextAction.item())
                prev_frame = copy.deepcopy(env.episode.currentFrame)
                observation, instant_reward, terminated, truncated, infoDict, _ = env.step(nextAction.item())
                
                # If hidden state updates are enabled, update the hidden state with frames between previous and current frames
                if cfg.UPDATE_HIDDEN:
                    updateHidden(prev_frame, env.episode.currentFrame, exp,env)
                
                t += 1
                # 4 - update the discounted reward
                if cfg.NORMALIZE_REWARD: # Clip the reward if specified in the configuration
                    instant_reward = instant_reward/cfg.maxReward
                G += (GAMMA**n) * instant_reward
                # print(f'partial G: {G}')

    # check if the last n_step ended on a terminal state
    if terminated:
        nextState = None
    else:
        nextState = torch.tensor(observation, dtype=torch.float32, device=cfg.DEVICE).unsqueeze(0)
            
    reward = torch.tensor([G], device=cfg.DEVICE)
    done = terminated or truncated

    # 1-step: we need the corresponding hidden of the targetLSTM to be store along nextState
    if exp['step']==1:
        updateTHidden = (targetNet.hidden[0].clone(), targetNet.hidden[1].clone())
        updateState = nextState
    
    # N-step: other options...
    # finally, update the targetNet weights
    else:
        updateTHidden = (targetNet.hidden[0].clone(), targetNet.hidden[1].clone())
        targetNet.load_state_dict(updateTargetStateDict)
        policyNet.load_state_dict(updateBehaviourStateDict)
       
        if not terminated or truncated:
            env.updateCurrentFrame(updateCurrentFrame)
            env.updateOit(updateObjectsInTable)
            env.updateASR(updateASR)
            env.episode.advanceTo(updateCurrentFrame)
    
    return (nextState, updateTHidden, reward, done, t, infoDict['timeReward'], infoDict['energyReward'], instant_reward, updateState, updateDelay,sec_actions, actionsTaken, actionASR)


def optimizeModel( exp , epoch):

    """
    In this method we are going to compute two things:
        - The target values for each transition in the replay memory
        - The current estimation of our model for Q(St,At)
    
    Note that the taget values Q(St+n,a) should be zero for entries
    in the memory whose next_state = None. This is because after taking
    the action Q(St,a) the episode ends, and no more future rewards will
    follow. So the target is: Rt+1 + 0 (i.e., the Q(St,At) because that
    particular action At at state St has led to a terminal state, you
    only get the Rt+1 and nothign else to account for the future). Thus
    max_a(Q(St+1),a) = 0.
    """

    rm = exp['replayMemory']
    cfg = exp['cfg']
    policyNet = exp['policyNet']
    targetNet = exp['targetNet']
    criterion = exp['criterion']
    optimizer = exp['optimizer']
    ratio_new = None
    

    transitions = rm.sample(cfg.BATCH_SIZE)
    Transition = createTransition(cfg)
    batch = Transition(*zip(*transitions))
    BATCH_SIZE = cfg.BATCH_SIZE
    lr_factor = np.minimum(len(rm) / (cfg.MIN_MEMORY_SIZE),1.0)
        

    # 3 - We need to create batches as input for the models. The problem is 
    # that when creating the nextState batch there might be None's. Thus, 
    # we need some mask to avoid passing those to the target network
    nonFinalMask = torch.tensor( 
        tuple(map(lambda s: s is not None, batch.next_state)), 
        device=cfg.DEVICE, dtype=torch.bool)
    try:
        #The next state for all samples in the batch
        nextStateBatch = torch.cat([s for s in batch.next_state if s is not None])   
    except:
        nextStateBatch = None
    
    stateBatch     = torch.cat(batch.state)
    actionBatch    = torch.cat(batch.action)
    rewardBatch    = torch.cat(batch.reward)

    ph0 = torch.stack([t[0] for t in batch.pHidden], dim=1)
    pc0 = torch.stack([t[1] for t in batch.pHidden], dim=1)
    th0 = torch.stack([t[0] for t in batch.tHidden], dim=1)
    tc0 = torch.stack([t[1] for t in batch.tHidden], dim=1)
    th0 = th0[nonFinalMask.unsqueeze(0)].unsqueeze(0) 
    tc0 = tc0[nonFinalMask.unsqueeze(0)].unsqueeze(0)

    # 4 - Use St,At to get the current estimation for Q(St,At) that the 
    # policyNet model is currently giving
    out = policyNet(stateBatch.unsqueeze(1), (ph0, pc0))[0].squeeze(1)
    stateActionValues = out.gather(1, actionBatch)
    avgMaxQ = out.max(1)[0].mean()
    

    # 5 - Compute the state action value term of the target equation. Note 
    # that we have to take max_a{(Qt+n,a)}
    nextStateActionValues = torch.zeros(BATCH_SIZE, device=cfg.DEVICE)
    if nextStateBatch != None:
        with torch.no_grad():
            tma = targetNet(nextStateBatch.unsqueeze(1), (th0, tc0))[0].squeeze(1)
            #Desactivamos las salidas de las imposibles => Tenemos que sacar objects-in-table del estado
            if cfg.NO_IMPOSSIBLE_ACTIONS:
                if (cfg.NOT_PRED and cfg.NOT_REC):
                    objects_in_table = nextStateBatch[:, 46:101]
                elif cfg.NOT_PRED or cfg.NOT_REC:
                    objects_in_table = nextStateBatch[:, 78:101]
                else:
                    objects_in_table = nextStateBatch[:, 110:133]
                manipulable_objects = list(cfg.ROBOT_ACTIONS_2_OBJECTS.values())[:-1]
                all_objects=list(cfg.OBJECTS_MEANINGS.values())
                idx_manipulable_objects = [all_objects.index(obj) for obj in manipulable_objects]
                #Add the background, which is always available
                idx_manipulable_objects.append(0)
                mani_obj_table = objects_in_table[:,idx_manipulable_objects]
                tma[mani_obj_table>0]=-torch.inf
            
            nextStateActionValues[nonFinalMask] = tma.max(1)[0]

    # 6 - Now we can compute the targets for the loss function. Here we have to
    # take care of the n-step discounts only for the model prediction (the instant rewards
    # are done in the n_step function)
        expectedStateActionValues = ( (cfg.GAMMA**exp['step']) * nextStateActionValues) + rewardBatch
    else:
        expectedStateActionValues = rewardBatch
   
    # 7 - Finally, let's compute the loss and update the model
    loss = criterion(stateActionValues, expectedStateActionValues.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()

    real_lr = optimizer.param_groups[0]['lr']
    lr_eff = real_lr * lr_factor
    optimizer.param_groups[0]['lr'] = lr_eff
    optimizer.step()
    optimizer.param_groups[0]['lr'] = real_lr

    return loss.item(), avgMaxQ.item(), ratio_new, lr_factor, lr_eff

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



def train(epoch, exp, save_path):
    
    """
    This function implements the training process for a single epoch. 
    It resets the environment for each episode, selects actions based on the policy, 
    stores transitions in replay memory, optimizes the policy network, updates the target network, 
    and tracks various statistics during the training process.
    
    """

    print('Training epoch {%d/%d}' %(epoch+1, exp['cfg'].EPOCHS))

    # Unpack experiment settings and modules
    env = exp['env']
    st  = exp['statTracker']
    cfg = exp['cfg']
    policyNet = exp['policyNet'].train()
    targetNet = exp['targetNet']
    exp['stepsDone'] += 1

    
    rm = exp['replayMemory']

    # Set division factor for saving model depending on the number of epochs
    if cfg.EPOCHS < 100:
        DIV = 1/cfg.EPOCHS
    else:
        DIV = 0.01
    
    for epi in range(env.numEpisodes):
        # 0 - reset the initial hidden states of the models
        policyNet.resetHidden()
        targetNet.resetHidden()

        # 1 - Get initial state
        observation, _ = env.reset()
        state = torch.tensor(observation, dtype=torch.float32, device=cfg.DEVICE).unsqueeze(0)
        
        t = 0
        while True:
            
            #Save current data
            init_data=saveCurrentData(env,state,exp['targetNet'],exp['policyNet'])
                
            # 3.1 select an action based on current state and policy
            action, pHidden, epsTh = selectAction(state, exp,env)
            
            # 3.2 this is the section in charge of generating entries for the replayMemory
            # Here is where you can implement n-step strategies
            (nextState, tHidden, reward, done, stepsTaken, Rt, Re, instant_reward,nextInstant, updateDelay,sec_actions, actionsTaken, actionASR) = nStep(env,exp, action.item(), epoch, epi, t)
            
            
            exp['updateTarget'] += 1
            
            # 3.3 store this new transition in memory
            if len(sec_actions) <  cfg.N_STEP:
                rm.push(pHidden, state, action, tHidden, None, reward,env.episode.folderName, env.episode.currentFrame, env.get_percentage_video_done())
            else:
                rm.push(pHidden, state, action, tHidden, nextState, reward,env.episode.folderName, env.episode.currentFrame, env.get_percentage_video_done() )
                
            # 3.7 track some useful stats
            st.add(epoch, epi, t, 'actions_taken', actionsTaken) ######################################
            st.add(epoch, epi,t,'epsilon', epsTh)
            st.add(epoch, epi,t,'G', reward.item())
            st.add(epoch, epi,t,'instant_reward', instant_reward)
            st.add(epoch, epi,t,'energy_reward', Re)
            st.add(epoch, epi,t,'time_reward',Rt)
            st.add(epoch, epi, t, 'actions_ASR', env.episode.actionsASR)
            st.add(epoch,epi, t, 'stateActionPrediction', env.trackerActPred.get_intervals())
            st.add(epoch,epi, t, 'stateActionRecognition', env.trackerActRec.get_intervals())
            st.add(epoch,epi, t, 'stateVideoInteraction', env.trackerVideo.get_intervals())
            #If there is an ASR => Then, we generate a new data bby running nStep again
            if actionASR is not None:
                #Save updated data
                updated_data=saveCurrentData(env,nextInstant,exp['targetNet'],exp['policyNet'])
                
                _ = loadCurrentData(init_data,env,exp)
                # Now re-run nStep to generate a sample from the ASR => Fasten the training process!!!!!
                (nextState, tHidden, reward, _, _, _, _, _,_, _,sec_actions, _, _) = nStep(env,exp, actionASR, epoch, epi, t)
                
                #Store ASR transition in memory
                if cfg.NOT_ASR_RM == False:
                    if len(sec_actions) <  cfg.N_STEP:
                        rm.push(pHidden, state, torch.tensor([[actionASR]],device=cfg.DEVICE), tHidden, None, reward,env.episode.folderName, env.episode.currentFrame, env.get_percentage_video_done() )

                    else:
                        rm.push(pHidden, state, torch.tensor([[actionASR]],device=cfg.DEVICE), tHidden, nextState, reward,env.episode.folderName, env.episode.currentFrame, env.get_percentage_video_done() )

                # Reload the real data
                nextInstant = loadCurrentData(updated_data,env,exp)
                
            # UPdate the environment
            state = nextInstant 
            if exp['step']!=1:
                env.updateDelay(updateDelay)
            
            t += 1
                    
            # 3.8 check if the episode is finished
            if done:    
                break
        

        num_optimizations = cfg.NUM_OPTIMIZATION_PER_STEP
        for it in range(num_optimizations):
        
            # 3.5 perform policyNet learning (model optimization)
            # if the replay memory is at least half full --> optimize 3 times per step
            loss, avgMaxQ, ratio_new, lr_factor, lr_eff = optimizeModel(exp, epoch)

        st.add(epoch, epi,0,'batch_loss', loss)
        st.add(epoch, epi,0,'batch_avgMaxQ', avgMaxQ)
        st.add(epoch, epi,0, 'healthy_human', env.episode.healthyHuman)
        st.add(epoch, epi,0,'reactive_time', env.episode.reactiveTime)
        st.add(epoch, epi,0,'min_time', env.episode.minTime)
        st.add(epoch, epi,0,'CA_intime', env.CA_intime)
        st.add(epoch, epi,0,'CA_late', env.CA_late)
        st.add(epoch, epi,0,'IA_intime', env.IA_intime)
        st.add(epoch, epi,0,'IA_late', env.IA_late)
        st.add(epoch, epi,0,'CI', env.CI)
        st.add(epoch, epi,0,'II', env.II)
        st.add(epoch, epi,0,'IMPOSA', env.IMPOSA)
        st.add(epoch, epi,0,'episode_name', env.episode.folderName)
        st.add(epoch, epi,0,'reactive_delay', env.episode.reactiveDelay)
        st.add(epoch, epi,0,'min_delay', env.episode.minDelay)
        st.add(epoch, epi,0,'interaction_time', env.episode.interactionTime)
        st.add(epoch, epi,0,'episode_delay', env.episode.delay)
        st.add(epoch, epi, 0, 'anticipationCAandASR', env.anticipationCAandASR)

        # 3.6 update the targetNet if we have enough steps
        if exp['updateTarget'] > (cfg.UPDATE_TARGET) and len(rm) > cfg.BATCH_SIZE:
            updateTargetNet(exp)
            # print('-- target net updated')
            exp['updateTarget'] = 0
    # Model saving
    if epoch % round(cfg.EPOCHS * DIV) == 0:
        # Save model state
        save_model(exp, epoch, loss, save_path)
    torch.cuda.empty_cache()             
    return 


def test(epoch, exp, env, st, nfold=None, args= None):
    """
    Run a test epoch on the policy network, tracking performance metrics
    and storing action decisions.
    """
    
    print('\tTest epoch {%d/%d}' %(epoch+1, exp['cfg'].EPOCHS))

    cfg = exp['cfg']

    # Load model if only testing
    if cfg.ONLY_TEST:
        # Transition = createTransition(cfg)
        policyNet,robot_execution_times = load_policy_net(epoch, exp,cfg, nfold) 
        exp['policyNet'] = policyNet
        exp['robot_execution_times'] = robot_execution_times
        
   
    for epi in range(env.numEpisodes):
        if cfg.ONLY_TEST:
            if cfg.TEST_VERBOSE:
                print('************')
                print('epi: ',env.episode.folderName)

        # 0 - reset the initial hidden states of the models
        # policyNet.resetHidden()
        exp['policyNet'].resetHidden()
        # 1 - Get initial state
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=cfg.DEVICE).unsqueeze(0)
        
        # for each time step until end of episode
        t = 0
        while True:
            # 3.1 select an action based on current state and policy
            action, pHidden, epsTh = selectAction(state, exp, env, test=True)
            # 3.2 take the step
            (nextState, reward, done, _, Rt, Re) = stepTest(exp, action.item(), epoch, epi, t, env, st)
            
            # 3.4 move to the next state
            state = nextState
            
            # 3.7 track some useful stats
            st.add(epoch, epi,t,'instant_reward', reward)
            st.add(epoch, epi,t,'time_reward', Rt)
            st.add(epoch, epi,t,'energy_reward', Re)
            st.add(epoch, epi,t, 'stateActionPrediction', env.trackerActPred.get_intervals())
            st.add(epoch, epi,t, 'stateActionRecognition', env.trackerActRec.get_intervals())
            st.add(epoch, epi,t, 'stateVideoInteraction', env.trackerVideo.get_intervals())

            t += 1
            # 3.8 check if the episode is finished
            if done:    
                break
            
        st.add(epoch, epi,0,'min_robot_time', env.episode.minRobotTime)
        st.add(epoch, epi,0,'episode_delay', env.episode.delay)
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

    # 1 - Initialize environment
    env = SimulatedEnv(mode='train', testFile=fold,configuration=cfg)
    envVal = SimulatedEnv(mode='val', testFile=fold,configuration=cfg)
    
    num_actions = len(cfg.ROBOT_ACTIONS_MEANINGS)
    env.reset()
    envVal.reset()

    state, info = env.reset()
    state_dimensionality = len(state)
    print(state_dimensionality)
    
    if cfg.DQN == 'mlp_latefusion':
        
        policyNet = DQN_MLP_LateFusion(state_dimensionality, num_actions, cfg).to(cfg.DEVICE)
        targetNet = DQN_MLP_LateFusion(state_dimensionality, num_actions, cfg).to(cfg.DEVICE)
    elif cfg.DQN == 'DQN':
        policyNet = DQN_LSTM_LateFusion(state_dimensionality, num_actions, cfg).to(cfg.DEVICE)
        targetNet = DQN_LSTM_LateFusion(state_dimensionality, num_actions, cfg).to(cfg.DEVICE)

    else:  # 'latefusion_lstm' 
        policyNet = DQN_LSTM_LateFusion(state_dimensionality, num_actions, cfg).to(cfg.DEVICE)
        targetNet = DQN_LSTM_LateFusion(state_dimensionality, num_actions, cfg).to(cfg.DEVICE)
    # 2 - Initialize DQN networks, optimizer and loss function
    optimizer = optim.AdamW(policyNet.parameters(), lr = cfg.LR,amsgrad=True,weight_decay=1e-4) #1e-4 
    stepsDone = 0
    
    #If training, check if we cam resume
    if mode=='train':
        policyNet, optimizer,rm, st, stVal, robot_execution_times,stepsDone=resumeTraining(cfg,save_path,policyNet,optimizer)

    
    if cfg.LOAD_MODEL:
        pt_model, robot_execution_times, optimizer = load_pretrained_model(cfg.MODEL_PATH, optimizer, state)


    #If no previous executions, then initialize stat_trackers
    if stepsDone==0:
        st = StatTracker()
        stVal = StatTracker()
        robot_execution_times = np.array(cfg.HUMAN_AVERAGE_DURATIONS[:-1])
        rm = initialize_replay_memory(cfg, cfg.MEMORY_SIZE)
    #assign target network
    targetNet.load_state_dict(policyNet.state_dict())
    #Set target network to eval
    targetNet.eval()

    #Only at the begining of the train, we set the weights of the output layer of the target to zero
    if stepsDone==0 and mode=='train' and cfg.LOAD_MODEL == False:
        targetNet.init_zero_weights()

    #Loss
    criterion = nn.SmoothL1Loss()
    
    # 4 - Initialize replay memory
    updateTarget = 0
    
    #Compute the maximum reward for reward normalization
    # avg_durations=cfg.ROBOT_AVERAGE_DURATIONS
    maxReward=cfg.ROBOT_TIME_BETA*(1+cfg.FACTOR_ENERGY_PENALTY)
    cfg.maxReward=maxReward
    print('Normalizer for reward {}'.format(cfg.maxReward))    
    exp = {'cfg': cfg,
           'fold': fold,
           'env': env,
           'envVal': envVal,
           'stepsDone': stepsDone,
           'policyNet': policyNet,
           'targetNet': targetNet,
           'optimizer': optimizer,
           'criterion': criterion,
           'statTracker': st,
           'statTrackerVal': stVal,
           'replayMemory': rm,
           'step':step,
           'updateTarget': updateTarget,
           'robot_execution_times': robot_execution_times,
    }
           
    
    print("\tDone!")

    return exp



def runExperiment(cfg):
    """
    Runs the experiment for the given configuration.

    Parameters:
    - cfg: Configuration object containing experiment parameters.

    Returns:
    - exp: Dictionary containing the results of the experiments for each fold.
    """
    

    for i,fold in enumerate(cfg.FOLDS):
        print("Starting experiment: %s" %(fold))
        
        if cfg.FIVE_FOLD == False:
            nfold = cfg.FOLDS[0].split('_')[0].split('fold')[-1]
            save_path=cfg.save_path + f"_fold{nfold}/"
          
        else:
            if i == 0:
                pre_save_path = cfg.save_path
            save_path =pre_save_path + '_fold{}'.format(i+1) + '/'
            
        cfg.save_path = save_path
        # 1 - Init experiment env, models, opt, loss, stat objects, etc
        exp = {}
        exp_i = initExperiment(cfg, fold, cfg.N_STEP,save_path,mode='train')
        print("Experiment started: %s" %(fold))
        
        init_epoch = exp_i['stepsDone']
        print('Start training on epoch %d'%init_epoch)
        
        # Loop over the number of epochs for training
        for e in range(init_epoch,cfg.EPOCHS):
            #Run an epoch of training
            train(e, exp_i, save_path) # Train the model for the current epoch
            
            #Copy estimations of robot times
            if cfg.TEMPORAL_CONTEXT:
                exp_i['envVal'].set_robot_execution_times(exp_i['env'].get_robot_execution_times())
                
            # test the model in both for the training data (no exploration)
            test(e, exp_i, exp_i['envVal'], exp_i['statTrackerVal'])
            
            # Plot reward statistics for validation
            # #Save validation statistics
            with open(save_path+'/stat_tracker_val.pkl', 'wb') as f:
                pickle.dump(exp_i['statTrackerVal'], f)
            
            # Save training statistics
            with open(save_path+'/stat_tracker_train.pkl', 'wb') as f:
                pickle.dump(exp_i['statTracker'], f)
                
            
            #Save the replay memory 
            with open(save_path+'/replay_memory.pkl', 'wb') as archivo_pickle:
                pickle.dump(exp_i['replayMemory'], archivo_pickle)
                
            # Clear GPU cache
            torch.cuda.empty_cache()
            gc.collect()

        # Generate various plots for results
        exp[i+1] = exp_i
        
        saveConfiguration(exp_i, save_path)
        torch.cuda.empty_cache()
        gc.collect()
   
        print("\tDone!")

    return exp



if( __name__ == '__main__' ):

    print('using CUDA GPU') if torch.cuda.is_available()  else print('using CPU')

    # 1 - Load the experiment config
    # cfg = loadConfiguration()
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
        '--seed',
        type=int,
        default=97,  # Default config number (corresponds to config2.py)
        help='Seed of the experiment'
    )
    

    parser.add_argument(
        '--arch',
        type=str,
        default='latefusion_lstm',
        choices=['latefusion_lstm', 'DQN', 'mlp_latefusion']
    )

    # Parse the arguments
    args = parser.parse_args()
    
    # Construct the config file name based on the argument (e.g., config1, config2, etc.)
    config_name = f"config{args.config}"  # This will create strings like 'config1', 'config2', etc.
    
    # Load the config module dynamically
    cfg = loadConfiguration(config_name)
    
    cfg.SEED = args.seed 
        
    if args.arch != 'latefusion_lstm':
        cfg.DQN = args.arch

    # Creation save path 
    save_path=creationSavePath(cfg, args)
    cfg.save_path = save_path
    print(save_path)
   
    print('SEED: ', cfg.SEED)

    # Now you can use the configuration settings from the specified config file
    print(f"Using configuration: {config_name}.py")
    
    
    torch.manual_seed(cfg.SEED)
    random.seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    torch.cuda.manual_seed_all(cfg.SEED)

    # # Configurar la reproducibilidad de operaciones determinísticas si usas CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    # 2 - Run the training/validation process
    exp = runExperiment(cfg)




