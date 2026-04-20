
"""
Simulated Environment for Personalized Human-Robot Interaction (HRI)

This module implements a simulation environment used to train and evaluate
robot decision-making policies from egocentric human activity data.

The environment is based on replaying recorded human interaction episodes,
where each episode corresponds to a real-world task (e.g., preparing food).
At each time step, the robot observes a multimodal state representation and
decides whether and how to assist the human.

Key components:
- Episode: represents a single recorded interaction sequence
- AtomicAction: represents a human action segment with optional robot need
- SimulatedEnv: main environment implementing the interaction dynamics
- ActionTracker: utility to track temporal action intervals

Main features:
- Multimodal state (perception features + object state + ASR signals)
- Simulation of robot execution delays and stochasticity
- Reward function combining time delay and energy cost
- Support for anticipatory and reactive robot behaviors

This environment is used for simulation-based pretraining and evaluation.
It does NOT include real robot execution or human-subject experiment infrastructure.
"""

import copy
import os
import glob
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from itertools import cycle
from utils import readTestTxt
from dataloader_profile_person import DataLoaderProfilePerson
from scipy.interpolate import interp1d
import torch 
from stochastic_state_generator_meta_usuario import StochasticEvolution

def resample_vector(vector, new_length):
     """
    Resample a 1D vector to a new length using linear interpolation.

    This helper is used to align temporal signals when feature and video
    sampling rates differ.
    """
    # Original indices
    original_indices = np.arange(len(vector))
    
    # New indices for resampling
    new_indices = np.linspace(0, len(vector) - 1, new_length)
    
    # Create a linear interpolation function
    interpolation_func = interp1d(original_indices, vector, kind='linear')
    
    # Apply the interpolation to the new indices
    resampled_vector = interpolation_func(new_indices)
    
    return resampled_vector

class ActionTracker:
    def __init__(self):
        # Dictionary used to store intervals by label
        self.intervals_by_label = {}
        self.current_label = None
        self.start_frame = None

    def update(self, frame, label):
        """
        Update the interval dictionary only when the label changes or when an interval ends.

        Parameters:
        - frame: current frame index (int).
        - label: current action label (str/int).
        """
        if label != self.current_label:
            # If there was a previous label, close its interval
            if self.current_label is not None:
                if self.current_label not in self.intervals_by_label:
                    self.intervals_by_label[self.current_label] = []
                # Store only the interval (start, end)
                self.intervals_by_label[self.current_label].append((self.start_frame, frame - 1))
            
            # Start a new interval
            self.start_frame = frame
            self.current_label = label

    def finalize(self, frame):
        """
        Finalize action tracking when the episode ends.
        Close the last interval if needed.
        """
        if self.current_label is not None:
            # Make sure a list exists for the current label
            if self.current_label not in self.intervals_by_label:
                self.intervals_by_label[self.current_label] = []
            # Store the final interval from start_frame to the last frame
            self.intervals_by_label[self.current_label].append((self.start_frame, frame))
            # Reset current tracking state
            self.current_label = None
            self.start_frame = None
            
    def get_intervals(self):
        """
        Return the dictionary of intervals grouped by label.
        """
        return self.intervals_by_label
class AtomicAction():
    """
    Represent a single annotated human action segment.

    Each atomic action stores its temporal boundaries, semantic label, involved
    objects, and whether robot assistance is required.
    """

    def __init__(self, index, actionLabel, objectsLabels, frameInit, frameEnd, requiresRobot, requiredObj, cfg):

        self.cfg = cfg 
        self.index = index
        self.actionLabel = actionLabel
        self.objectsLabels = objectsLabels
        self.frameInit = frameInit
        self.frameEnd = frameEnd
        self.duration = frameEnd - frameInit
        self.requiresRobot = requiresRobot
        self.requiredObj  = requiredObj
        
        return
    
    def __str__(self):
    
        print("atomic action number = %d" %self.index)
        print("Action = %s" %(self.cfg.ATOMIC_ACTIONS_MEANINGS[self.actionLabel]))
        print("Objects involved")
        for object in self.objectsLabels:
            print("\t%s" %(self.cfg.OBJECTS_MEANINGS[object[0]]))

        print("frame init = %d" %(self.frameInit))
        print("frame end = %d" %(self.frameEnd))
        print("requires robot = %s" %(str(self.requiresRobot)))  
        print("required object = %s" %(self.requiredObj)) 

        return ""
    
class Episode():
    """
    Represent one recorded interaction episode.

    An episode contains precomputed frame-level features, action annotations,
    task-level timing information, and metrics accumulated during simulation.
    """

    def __init__(self, folderPath, labelsPath, actions2Remove, robotNeededActions,
                 videoFPS, featFPS, selFeat, cfg):
    
        self.cfg = cfg
        self.folderPath = folderPath
        self.folderName = folderPath[folderPath.rfind('/')+1::]
        self.labelsPath = labelsPath
        self.actions2Remove = actions2Remove
        self.robotNeededActions = robotNeededActions
        self.videoFPS = videoFPS
        self.featFPS = featFPS
        self.selFeat = range(selFeat)

        self.atomicActions = []
        self.frameFeatures = None 
        self.lastFrame = None
        self.reactiveDelay = 0
        self.reactiveTime = 0
        self.reactiveReward = 0
        self.reactiveTimeReward = 0
        self.reactiveEnergyReward = 0

        self.minDelay = 0
        self.minTime = 0
        self.minRobotTime = 0
        self.minReward = 0
        self.minTimeReward = 0
        self.minEnergyReward = 0

        self.interactionTime = 0
        self.healthyHuman = 0
        # attributes that must be reset each time the episode starts
        self.currentFrame = 0
        self.delay = 0
        self.terminated = True
        self.truncated = True
        self.actionsTaken = {}
        self.actionsASR = {}
        self.frameInitActionHumanCannotDo = []
        
        # load the episode info
        self.loadEpisode()
        self.visualizeInteractionPerson = []

        return
    
    def reset(self):
        self.currentFrame = 0
        self.delay = 0
        self.terminated = False
        self.truncated = False
        self.actionsTaken = {}
        self.actionsASR ={}
        return
    
    def clearActionsASR(self):
        self.actionsASR ={}
        
    def advanceTo(self, f):

        if f > self.lastFrame:
            self.currentFrame = int(self.lastFrame)
            self.terminated = True
        else:
            self.currentFrame = int(f)

        return
    
    def skipFrames(self, numFrames):
        
        self.advanceTo(self.currentFrame + numFrames)
        return
    
    def findNextRobotAction(self):

    # This function returns the next atomic action that requires robot intervention.
    # If no further robot intervention is needed for the rest of the recipe,
    # it returns a default atomic action.

        ac = AtomicAction(0,0,0,np.inf, np.inf, False, None, self.cfg)

        for ac_i in self.atomicActions:
            if ac_i.frameInit >= self.currentFrame and ac_i.requiresRobot:
                return ac_i
        
        return ac
    
    def findNextAtomicAction(self, ac):

        for ac_i in self.atomicActions:
            if ac_i.index > ac.index:
                return ac_i
        
        # there is no next atomic action
        return None
    
    def isObjectNecessary(self, objName):

        for ac_i in self.atomicActions:
            if ac_i.requiresRobot and ac_i.requiredObj == objName:
                return True
        
        return False
    
    def findNextRewardFrame(self):

        # This function returns the frame where we have the next reward time 
        # instant. All starts and ends of atomic actions are reward instants
        for ac_i in self.atomicActions:
            
            if self.currentFrame < ac_i.frameInit:
                return ac_i.frameInit
            
            if self.currentFrame < ac_i.frameEnd:
                return ac_i.frameEnd
            
        return self.lastFrame
    
    def fps2time(self, frameNum):
        return frameNum*1.0 / self.videoFPS
    
    def time2fps(self, time):
        return int(time * self.videoFPS)

    def getTotalLength(self, mode='frames'):

        assert mode=='frames' or mode == 'time'

        totalFrames = self.frameFeatures.shape[0]
        return totalFrames if mode == 'frames' else self.fps2time(totalFrames)

    def getOptimalLength(self, mode='frames'):

        totalFrames = self.getTotalLength(mode='frames')
        
        for ac in self.atomicActions:
            if ac.requiresRobot:
                totalFrames -= ac.duration

        return totalFrames if mode == 'frames' else self.fps2time(totalFrames)


    def readLabels(self):
        
        # 3 - Read labels
        if self.labelsPath != '': # There are some videos without annotations
            pandasDF = np.load(self.labelsPath, allow_pickle=True)
            
            if pandasDF.empty == False: # Some file annotations could be empty
                # 4 - Fill atomic actions
                offset = 0
                # print(pandasDF)
                rows_to_drop = []
                for index, row in pandasDF.iterrows():
        
                    # read pandas row
                    frameInit    = row.frame_init - offset
                    frameEnd     = row.frame_end - offset
        
                    duration       = frameEnd - frameInit
                    actionToRemove = row.label in self.actions2Remove

                    # If action is to be removed
                    if actionToRemove:
                        offset += duration  # Update the offset for subsequent actions
                        rows_to_drop.append(index)  # Mark this row for removal
                    else:
                        # Update the row with adjusted frame values
                        pandasDF.at[index, 'frame_init'] = frameInit
                        pandasDF.at[index, 'frame_end'] = frameEnd
    
                # Remove actions that cannot be done
                pandasDF.drop(rows_to_drop, inplace=True)

                return pandasDF
                
        
    def generateTimeRemaining(self):
        replicateFactor = self.videoFPS / self.featFPS
        
        data = np.load(os.path.join(self.folderPath,'remaining_frames'), allow_pickle=True)[0,...]
        initAction=np.where((data - np.concatenate([[0],data[:-1]]))>0)[0]
        initAction = np.concatenate(initAction,len(data))
        durations = initAction - np.concatenate([[0],initAction[:-1]]) -1
        result = []
        for num in durations:
            # Generate a series from the current number down to 0
            series = list(range(num, -1, -1))
            # Append the series to the result list
            result.extend(series)
        # Convert the result list to a NumPy array
        result = np.array(result)
        # Interpolate to match the target length
        new_length = int(replicateFactor * len(result))
        addSamples = replicateFactor*(len(data)-len(result))
        result = resample_vector(result, new_length)
        # Add final samples
        result = np.concatenate((result,1000*np.ones(addSamples,)))
        
        return result
    
    def loadEpisode(self):
        """
        Load frame-level features and annotations for a single episode.

        Workflow:
        1. Load the precomputed per-frame feature vectors.
        2. Replicate features to match the target video frame rate.
        3. Parse annotations into atomic actions.
        4. Remove unsupported actions and adjust the timeline accordingly.
        """

        # 1 - Read frame features
        framePathsList = sorted(glob.glob(os.path.join(self.folderPath, 'frame_*')))
        self.frameFeatures = np.zeros( (len(framePathsList), len(self.selFeat)), dtype=np.float32 )
        self.healthyHuman = 0

        for i,framePath in enumerate(framePathsList):
            # Load the per-frame feature file

            fFeat = np.load(framePath, allow_pickle=True)['data']

            if self.cfg.ANNOTATIONS_DIM < 110:
                self.frameFeatures[i,:] = fFeat[32: 32+self.cfg.ANNOTATIONS_DIM] #without next action prediction 
            else:
                self.frameFeatures[i,:] = fFeat[self.selFeat]
            
                
        # 2 - Replicate the features to account for the difference between 
        # the featFPS and the videoFPS
        replicateFactor = self.videoFPS / self.featFPS
        self.frameFeatures = np.repeat(self.frameFeatures, replicateFactor, axis=0)
       
        self.healthyHuman += self.frameFeatures.shape[0]
        
        # 3 - Read labels
        if self.labelsPath != '': # There are some videos without annotations
            pandasDF = np.load(self.labelsPath, allow_pickle=True)
            
            if pandasDF.empty == False: # Some file annotations could be empty
                
                # 4 - Fill atomic actions
                keep = np.full(self.frameFeatures.shape[0], True, dtype=bool)
                offset, numAc = 0,0
                # print(pandasDF)
                for index, row in pandasDF.iterrows():
        
                    # read pandas row
                    actionIndex  = row.label
                    objectsIndex = row.label_obj
                    frameInit    = row.frame_init
                    frameEnd     = row.frame_end
        
                    duration       = frameEnd - frameInit
                    requiresRobot  = row.label in self.robotNeededActions
                    actionToRemove = row.label in self.actions2Remove
                    
                    if actionToRemove:
                        self.frameInitActionHumanCannotDo.append({self.cfg.HUMAN_ACTIONS_2_ROBOT_ACTIONS[row.label]:row.frame_init})
                    if actionToRemove:
                        offset += duration
                        keep[frameInit:frameEnd] = False
                        continue
        
                    requiredObj =self.cfg.HUMAN_ACTIONS_2_OBJECTS[actionIndex] if requiresRobot else None
        
                    # create the atomic action and add it to the episode
                    self.atomicActions.append(AtomicAction(numAc, actionIndex, objectsIndex, 
                        frameInit-offset, frameEnd-offset, requiresRobot, requiredObj, self.cfg))
                    numAc += 1
        
                # Finally, remove the frames that belong to self.actions2Remove
                self.frameFeatures = self.frameFeatures[keep]
                    
            else:
                self.labelsPath = ''
                print('No annotations in label file')
        self.lastFrame = self.frameFeatures.shape[0]-1
        
        return

    def __str__(self):

        print("EPISODE INFO")
        print("\tFolder path %s" %self.folderPath)
        print("\tHuman actions that needs robot: ", self.robotNeededActions)
        print("\tCurrent frame in the episode %d" %(self.currentFrame) )
        print("\nAtomic actions: ")

        for ac in self.atomicActions:
            print(ac)

        return ""
    

    def storeActionsTaken(self, epoch, epi, st):
       
        self.actionsTaken = {
        k: {'action': v['action'], 'duration': v['duration']}
        for t in st.data[epoch][epi].keys()
        for k, v in st.data[epoch][epi][t]['actions_taken'].items()
        }

    # return
        return
    def storeActionsASR(self, epoch, epi, st):
        self.actionsASR = {
        k: {'action': v['action'], 'duration': v['duration']}
        for t in st.data[epoch][epi].keys()
        for k, v in st.data[epoch][epi][t]['actions_ASR'].items()
        }
        
    def visualizeAndSave(self, save_file,epoch,mode='time'):

        # Create a figure and axis
        fig, ax = plt.subplots()

        # Set the y-axis limit and labels
        ax.set_ylim(0.11, 0.31)
        ax.set_yticks([])

        xticks = []
        xticklabels = []

        # Plot the timeline
        for ac in self.atomicActions:

            if mode == 'time':
                duration = round(self.fps2time(ac.duration))
                startTime = round(self.fps2time(ac.frameInit))
            else:
                duration = round(ac.duration)
                startTime = round(ac.frameInit)

            name = "a" + str(ac.index)
            
            c = 'r' if ac.requiresRobot else 'b'
            f = (1.0,0.8,0.8) if ac.requiresRobot else (1.0,1.0,1.0)

            # Create a rectangle for the action
            rect = patches.Rectangle((startTime, 0.1), duration, 0.3, linewidth=1, edgecolor=c, facecolor=f)
            
            # Add the action index and action label
            ax.text(startTime + duration / 2, 0.20, name, ha='center', va='center', fontsize=10, color=c)
            ax.text(startTime + duration / 2, 0.15, ac.actionLabel, ha='center', va='center', fontsize=10, color=c)

            ax.add_patch(rect)

            # Store the x-axis tick positions and labels
            xticks.extend([startTime, startTime + duration])
            xticklabels.extend([str(startTime), str(startTime + duration)])

        for k,v in self.actionsTaken.items():
            ax.text(k, 0.30, v, ha='center', va='center', fontsize=14, color='g')

        # Set the x-axis limit
        if mode == 'time':
            ax.set_xlim(0, self.getTotalLength(mode='time'))
        else:
            ax.set_xlim(0, self.getTotalLength(mode='frames'))

        # Configure the figure size
        fig.set_size_inches(13, 1)

        # Set x-axis ticks and labels
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels, rotation=0, fontsize=8)
        ax.tick_params(axis='x', which='both', bottom=False, top=False)

        for i,label in enumerate(ax.get_xticklabels()):
            if i % 2 == 0:
                label.set_y(-0.3)

        # Save the timeline as a PNG image
        ax.set_title(self.folderName)
        imName = save_file + '/'+self.folderName + '_'+str(epoch)+'.jpg'
        plt.savefig(imName, bbox_inches='tight', dpi=300)
        plt.close()
        return
    
    def visualizeAndSave2(self, save_file, epoch, mode='time'):
    
        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(25, 4), gridspec_kw={'hspace': 0.9})
        ax1.clear()
        ax2.clear()
        # Set the y-axis limit and labels for person's actions
        ax1.set_ylim(0.11, 0.31)
        ax1.set_yticks([])
        ax1.set_title('Person Actions')
    
        # Set the y-axis limit and labels for robot's actions
        ax2.set_ylim(0.11, 0.31)
        ax2.set_yticks([])
        ax2.set_title('Robot Actions')
    
        xticksperson = []
        xticksrobot = []
        xticklabelsperson = []
        xticklabelsrobot = []
    
        # Plot the timeline for the person's actions
        for ac in self.visualizeInteractionPerson:
    
            if mode == 'time':
                duration = round(self.fps2time(ac.duration))
                startTime = round(self.fps2time(ac.frameInit))
            else:
                duration = round(ac.duration)
                startTime = round(ac.frameInit)
    
            name = "a" + str(ac.index)
            
            c = 'b' if ac.requiresRobot else 'k'
            f = (0.8, 0.8, 1.0) if ac.requiresRobot else (1.0,1.0,1.0)

            if ac.actionLabel == 99:
               
                name = 'wait'
                c = 'r'
                f = (1.0, 0.8, 0.8)
            else:
                name =self.cfg.ATOMIC_ACTIONS_MEANINGS[ac.actionLabel]

            # Create a rectangle for the action
            rect = patches.Rectangle((startTime, 0.1), duration, 0.3, linewidth=1, edgecolor=c, facecolor=f)
    
            # Add the action index and action label
            ax1.text(startTime + duration / 2, 0.20, name, ha='center', va='center', fontsize=10, color=c)
    
            ax1.add_patch(rect)
    
            # Store the x-axis tick positions and labels for person's actions
            xticksperson.extend([startTime, startTime + duration])
            xticklabelsperson.extend([str(startTime), str(startTime + duration)])

        # Store the end time of the last person's action
    
        # Plot the timeline for the robot's actions
        for start_frame, action_data in self.actionsTaken.items():
            # Get the start frame and duration of the robot's action
            
            action = action_data['action']
            duration = action_data['duration']
            if mode == 'time':
                duration = round(self.fps2time(duration))
                start_time = round(self.fps2time(start_frame))
                
            else:
                start_time = round(start_frame)
                duration = round(duration)
    
             
            c = 'g'  # Color for robot's actions
            f = (0.8, 1.0, 0.8)
    
            # Create a rectangle for the action

            rect = patches.Rectangle((start_time, 0.1), duration, 0.3, linewidth=1, edgecolor=c, facecolor=f)
    
            # Add the action index and action label for robot's actions
            ax2.text(start_time + duration/ 2, 0.2,self.cfg.ROBOT_ACTIONS_MEANINGS_vis[action], ha='center', va='center', fontsize=10, color=c)
            ax2.add_patch(rect)
    
            # Store the x-axis tick positions and labels for robot's actions
            xticksrobot.extend([start_time, start_time+duration])
            # xticklabelsrobot.append('')  # Empty label for person's actions
            xticklabelsrobot.extend([str(start_time), str(start_time+duration)])  # Label for robot's actions

        # Set x-axis ticks and labels for both subplots
        ax1.set_xticks(xticksperson)
        ax2.set_xticks(xticksrobot)
        
        ax1.set_xticklabels(xticklabelsperson, rotation=45, fontsize=8)  # Set x-axis labels for person's actions
        ax2.set_xticklabels(xticklabelsrobot, rotation=45, fontsize=8)  # Set x-axis labels for robot's actions
    
        ax1.tick_params(axis='x', which='both', bottom=False, top=False)
        ax2.tick_params(axis='x', which='both', bottom=False, top=False)
    
        for i, label in enumerate(ax1.get_xticklabels()):
            if i % 2 == 0:
                label.set_y(-0.3)
    
        for i, label in enumerate(ax2.get_xticklabels()):
            if i % 2 == 0:
                label.set_y(-0.3)
    
        # Create a suptitle with the video name
        fig.suptitle(self.folderName, fontsize=14)
    
        # if not os.path.exists(save_path+'/videos'): os.makedirs(save_path+'/videos')
        if not os.path.exists(save_file+'/videos/'+ self.folderName +'/'+str(epoch)): os.makedirs(save_file+'/videos/'+self.folderName+'/'+str(epoch))
        imName = save_file+'/videos/'+ self.folderName +'/'+str(epoch) +  '/interaction_'+str(epoch)
        plt.savefig(imName, bbox_inches='tight', dpi=300)
        plt.close()
        # plt.show()
        return
    

   
class SimulatedEnv():
    """
    Replay-based simulation environment for anticipatory robot assistance.

    The environment models a human performing a task while a robot assists by
    bringing objects when needed. The robot must decide both what action to take
    and when to take it.

    State inputs are built from precomputed perception outputs rather than raw
    images, which makes the environment lightweight and reproducible.
    """

    def __init__(self, mode, testFile, configuration):
        super(SimulatedEnv, self).__init__()

        """
        Mandatory attributes
            action_space: the Space object corresponding to valid actions
            observation_space: the space object corresponding to valid observations
            metadata: a dict containing the rendering modes, fps, etc.
            render_mode: the render mode of the env determined at initialization
            reward_range: a tuple with the minimum and maximum possible rewards
        """

        # 1 - Attributes
        self.cfg = configuration
        # Keep the same video sampling sequence for each patient to ensure fair evaluation
        self.rng = np.random.default_rng(20)
        self.mode = mode
        self.testfile = testFile
        self.allEpisodes = []
        self.numEpisodes = None
        self.episode = None  
        self.trainWeights, self.testWeights = {},{}
        self.numVideosSplit = []
        self.oit = copy.deepcopy(self.cfg.OBJECTS_INIT_STATE)
        self.asr = np.zeros((self.cfg.NUM_ROBOT_ACTIONS-1,),dtype=np.float32)
        self.robot_execution_times = np.array(self.cfg.HUMAN_AVERAGE_DURATIONS[:-1])
        
        self.trainFolders = None
        self.testFolders = None
        
        # 2 - Set action and observation spaces
        if (self.cfg.NOT_PRED and self.cfg.NOT_REC):
            stateDims = self.cfg.ANNOTATIONS_DIM-64 + len(self.cfg.OBJECTS_INIT_STATE) 
      
        elif self.cfg.NOT_PRED or self.cfg.NOT_REC:
            stateDims = self.cfg.ANNOTATIONS_DIM-32 + len(self.cfg.OBJECTS_INIT_STATE) 
       
     
        else:
            stateDims = self.cfg.ANNOTATIONS_DIM + len(self.cfg.OBJECTS_INIT_STATE) 
            
        if self.cfg.NOT_ASR == False:
            stateDims += self.cfg.NUM_ROBOT_ACTIONS-1
     
        self.stateDims = stateDims
        self.action_space = np.arange(self.cfg.NUM_ROBOT_ACTIONS)
        
        self.dataFramesVideos = {}
        self.dictAnnotations = {}
        
        # Compute video weights if they are required
        self.evolution = None
        self.computeVideoWeights()
            
        # 3 - Read episodes
        self.loadEpisodes()
        self.loadAnnotations()
        print('- episodes loaded')
        
        self.computeEpisodesReactiveTime()
        print('- reactive time computed')
        self.computeEpisodesMinDelay()
        print('- min delay computed')
        
        self.CA_intime = 0
        self.CA_late = 0
        self.IA_intime = 0
        self.IA_late = 0
        self.CI = 0
        self.II = 0
        self.IMPOSA = 0
        
        self.anticipationCorrectRobotSinceBeggining = []
        
        self.anticipationCAandASR = []
        self.durationActionRobot = 0
        
        self.trackerActPred = ActionTracker()
        self.trackerActRec = ActionTracker()
        self.trackerVideo = ActionTracker()

        return
    
    def get_percentage_video_done(self):
        frame = self.episode.currentFrame
        if self.episode.currentFrame > self.episode.lastFrame:
            frame = int(self.episode.lastFrame)
            
        return (frame/ self.episode.lastFrame)*100
    
    def get_robot_execution_times(self):
        return self.robot_execution_times
    
    def set_robot_execution_times(self, robot_execution_times):
        self.robot_execution_times[...] = robot_execution_times[...]
        
    def get_label_for_frame(self,annotations, frame):
        
        # Iterate over annotations and check whether the frame falls within the interval
        last_row = annotations[-1]
        last_frame_values = list(last_row.values())[0][-1]
        find = 0
        for annotation in annotations:
            
            for label, (frame_init, frame_end) in annotation.items():
                if frame_init <= frame <= frame_end:
                    find = 1
                    return label,last_frame_values  # Return the label if the frame is inside the interval
        # No labels for that frame
        if find == 0:
            return None,last_frame_values

    def loadAnnotations(self):
        """ 
        Save the annotations of the videos in a dictionary to easily access them when needed
        """
        
        for i in range(self.numEpisodes):
            self.episode = next(self.allEpisodes)   
            self.episode.reset()
            
            if self.episode.folderName in self.dictAnnotations:
                break
            pandasDF = self.episode.readLabels()
            self.dataFramesVideos[self.episode.folderName] = pandasDF
            
            if pandasDF is not None:
                for _, row in pandasDF.iterrows():
                    label = row['label']
                    frame_init = row['frame_init']
                    frame_end = row['frame_end']
                    
                    if self.episode.folderName not in self.dictAnnotations:
                        self.dictAnnotations[self.episode.folderName] = []
                        
                    self.dictAnnotations[self.episode.folderName].append({label: [frame_init, frame_end]})
            else:
                self.dictAnnotations[self.episode.folderName] = None

            
    def computeEpisodesReactiveTime(self):

        """
        This function computes the times necessary for a recipe to be completed
        under a reactive robot (a robot that only acts via ASR)
        """
        for i in range(self.numEpisodes):
            #Reset the state and advance to the next episode
            self.reset()

            reactive_reward = 0
            reactive_time_reward = 0
            reactive_energy_reward = 0

            while self.episode.terminated is False:
                _, reward, _, _, info_dict, _ = self.step(5)
                reactive_reward += reward
                reactive_time_reward += info_dict['timeReward']
                reactive_energy_reward += info_dict['energyReward']

            self.episode.reactiveDelay = self.episode.delay
            self.episode.reactiveTime = self.episode.delay + self.episode.lastFrame
            self.episode.reactiveReward = reactive_reward
            self.episode.reactiveTimeReward = reactive_time_reward
            self.episode.reactiveEnergyReward = reactive_energy_reward

        return

    def computeEpisodesMinDelay(self):

        """
        This function computes for each episode the minimum-delay reference
        policy, together with its associated rewards. Note that this might change
        on each iteration given that the times taken by the robot are stochastic.
        """
        for i in range(self.numEpisodes):
            self.reset()
            requiredActions = []

            # find the necessary actions to be taken
            for ac_i in self.episode.atomicActions:
                if ac_i.requiresRobot:
                    requiredActions.append(self.cfg.ROBOT_OBJECTS_2_ACTIONS[ac_i.requiredObj])

            # Remove duplicates while preserving order
            requiredActions = list(dict.fromkeys(requiredActions))

            # now, take the actions
            # Accumulate minimum useful robot execution time
            min_robot_time = 0
            min_reward = 0
            min_time_reward = 0
            min_energy_reward = 0

            for action in requiredActions:
                # deterministic estimate of robot execution time
                action_time = int(self.cfg.ROBOT_TIME_BETA * self.cfg.HUMAN_AVERAGE_DURATIONS[action])
                min_robot_time += action_time
                #self.step(action)

                _, reward, _, _, info_dict, _ = self.step(action)
                min_reward += reward
                min_time_reward += info_dict['timeReward']
                min_energy_reward += info_dict['energyReward']

            # do nothing for the rest of the episode
            while self.episode.terminated is False:
                _, reward, _, _, info_dict, _ = self.step(5)
                min_reward += reward
                min_time_reward += info_dict['timeReward']
                min_energy_reward += info_dict['energyReward']

            self.episode.minDelay = self.episode.delay
            self.episode.minTime = self.episode.delay + self.episode.lastFrame
            self.episode.minRobotTime = min_robot_time
            self.episode.minReward = min_reward
            self.episode.minTimeReward = min_time_reward
            self.episode.minEnergyReward = min_energy_reward

        return
    def computeStatesPerson(self):
        
        if self.mode == "train":

            self.evolution = StochasticEvolution(user_path = self.cfg.PROFILE_PATH,
                                      total_epochs=self.cfg.EPOCHS,
                                      seed=self.cfg.SEED,
                                      states_randomness=0.0,                # r=0 for deterministic state transitions
                                      beta=self.cfg.BETA_SG, save_dir=self.cfg.save_path, min_duration=self.cfg.DURATION_STATES_SG)       #
            torch.save(self.evolution, os.path.join(self.cfg.save_path, "evolution.pt"))
        else:
            # If not in train mode, load the saved file
            evolution_path = os.path.join(self.cfg.save_path, "evolution.pt")
            self.evolution = torch.load(evolution_path, weights_only=False)

        print(f'durations {self.mode}: {self.evolution.durations}')
    

    #Function the computes weights associated to the videos in the dataset
    def computeVideoWeights(self):
        # read episode folders
        dbFolder = os.path.join(self.cfg.ANNOTATIONS_FOLDER,self.cfg.DATASET_NAME)
        if isinstance(self.testfile, list):
            self.testfile = self.testfile[0]
        testPath = os.path.join(self.cfg.ANNOTATIONS_FOLDER, self.testfile)

        allFolders  = glob.glob(dbFolder+"/*")
        self.testFolders = [os.path.join(dbFolder, testFolder) for testFolder in readTestTxt(testPath)]
        self.testFolders = sorted(self.testFolders)
        
        self.trainFolders = list( set(allFolders) - set(self.testFolders) )
        self.trainFolders = sorted(self.trainFolders)
        # keep train or test folders
        folders = self.testFolders if self.mode == 'test' else self.trainFolders
        folders = sorted(folders) # sort them or shuffle them
        
        numVideosTrain = len(self.trainFolders)
        self.trainWeights = dict([(key, 1.0/numVideosTrain ) for key in self.trainFolders])
        numVideosTest = len(self.testFolders)
        self.testWeights = dict([(key, 1.0/numVideosTest ) for key in self.testFolders])
            
        return   
    
    def updateVideoWeights(self, epoch):
        profile = self.evolution.get_profile(epoch=epoch)
        dataloader = DataLoaderProfilePerson( self.trainFolders,  self.testFolders)
        dataloader.set_patient_probs(profile)
        self.trainWeights, self.testWeights, self.numVideosSplit = dataloader.get_weights_videos()
        
    """Method that samples videos using the computed probabilities"""
    def sample_videos(self,num_videos):

        probs=np.array(list(self.trainWeights.values()))
        videos=list(self.trainWeights.keys())
        videos = self.rng.choice(videos, size=num_videos, replace=True, p=probs)
        return videos
    
    def loadEpisodes(self):

        # read episode folders
        dbFolder = os.path.join(self.cfg.ANNOTATIONS_FOLDER,self.cfg.DATASET_NAME)
        if isinstance(self.testfile, list):
            self.testfile = self.testfile[0]
        testPath = os.path.join(self.cfg.ANNOTATIONS_FOLDER, self.testfile)

        allFolders  = glob.glob(dbFolder+"/*")
        testFolders = [os.path.join(dbFolder, testFolder) for testFolder in readTestTxt(testPath)]
        trainFolders = list( set(allFolders) - set(testFolders) )
       
        folders = testFolders if self.mode == 'test' else trainFolders
            
        # load the episodes
        e = []
        for f in folders:
            self.frameInitActionHumanCannotDo = []
            if os.path.exists(os.path.join(f,self.cfg.LABELS_FILE_NAME)):
                labelsPath = os.path.join(f,self.cfg.LABELS_FILE_NAME)
                e.append( Episode(f, labelsPath,self.cfg.ACTIONS_TO_REMOVE_FROM_RECIPES, 
                   self.cfg.ROBOT_NEEDED_FOR_ACTIONS,self.cfg.VIDEO_FPS,self.cfg.ANNOTATIONS_FPS, 
                   self.cfg.ANNOTATIONS_DIM, self.cfg))
            else:
                e.append( Episode(f, '',self.cfg.ACTIONS_TO_REMOVE_FROM_RECIPES, 
                   self.cfg.ROBOT_NEEDED_FOR_ACTIONS,self.cfg.VIDEO_FPS,self.cfg.ANNOTATIONS_FPS, 
                   self.cfg.ANNOTATIONS_DIM, self.cfg))

        self.numEpisodes = len(e)
        self.allEpisodes = cycle(e)

        return
    
    def computeNormFactors(self):
        mom1=np.zeros((4,))
        mom2=np.zeros((4,))
        for i in range(self.numEpisodes):
            #Reset the state and advance to the next episode
            self.reset()
            #AR
            mom1[0]+=(self.episode.frameFeatures[:,0:32]).mean()
            mom2[0]+=(self.episode.frameFeatures[:,0:32]**2).mean()
            #AP
            mom1[1]+=(self.episode.frameFeatures[:,32:64]).mean()
            mom2[1]+=(self.episode.frameFeatures[:,32:64]**2).mean()
            #VWM
            mom1[2]+=(self.episode.frameFeatures[:,64:110]).mean()
            mom2[2]+=(self.episode.frameFeatures[:,64:110]**2).mean()
        mom1=mom1/self.numEpisodes
        mom2=mom2/self.numEpisodes
        means=mom1
        stds=np.sqrt(mom2-mom1**2)
        means[3]=np.array(list(self.cfg.OBJECTS_INIT_STATE.values())).mean()
        stds[3]=np.array(list(self.cfg.OBJECTS_INIT_STATE.values())).std()
        
        print(means)
        print(stds)
        return means,stds
        
        
    def isImpossibleAction(self, action):

        """

        This function checks whether the action the robot wants to take
        is possible or not (e.g., because the object is not available).
        Returns True if the action is impossible to execute.

        """
        objTaken =self.cfg.ROBOT_ACTIONS_2_OBJECTS[action]
        
        if objTaken == 'do nothing': 
            return False
        else:
            return self.oit[objTaken] == 1    


                
    def buildStateHidden(self, frame):
        """
        Build the state representation at a given frame without advancing the environment.

        This method is mainly useful for offline analysis or dataset generation,
        where a state must be extracted from a specific frame index.
        """

        feat = self.episode.frameFeatures[frame]
        if (self.cfg.NOT_PRED and self.cfg.NOT_REC):
            feat = feat[self.cfg.FIDX[1]:]
        elif self.cfg.NOT_PRED:
                feat = feat[self.cfg.FIDX[0]:]
        elif self.cfg.NOT_REC:
            feat = np.concatenate((feat[:self.cfg.FIDX[0]],feat[self.cfg.FIDX[1]:]), axis = 0)
            
        oit = np.array(list(self.oit.values()), dtype=np.float32)
        asr = copy.deepcopy(self.asr)
        
        state = np.concatenate((feat, oit), axis=0)
        if self.cfg.NOT_ASR == False:
            state = np.concatenate((feat, oit,asr), axis=0)
            
        if self.cfg.NORM_FEATURES:
            
            if (self.cfg.NOT_PRED and self.cfg.NOT_REC):
                state[:self.cfg.FIDX[2]-64]=(state[0:self.cfg.FIDX[2]-64]-self.cfg.FMEANS[2])/self.cfg.FSTDS[2]
                state[self.cfg.FIDX[2]-64:self.cfg.FIDX[3]-64]=(state[self.cfg.FIDX[2]-64:self.cfg.FIDX[3]-64]-self.cfg.FMEANS[3])/self.cfg.FSTDS[3]
                if self.cfg.NOT_ASR == False:
                    state[self.cfg.FIDX[3]-64:self.cfg.FIDX[4]-64] = (state[self.cfg.FIDX[3]-64:self.cfg.FIDX[4]-64])
            
            elif self.cfg.NOT_PRED:
                    state[:self.cfg.FIDX[0]]=(state[0:self.cfg.FIDX[1]-32]-self.cfg.FMEANS[1])/self.cfg.FSTDS[1]
                    state[self.cfg.FIDX[1]-32:self.cfg.FIDX[2]-32]=(state[self.cfg.FIDX[1]-32:self.cfg.FIDX[2]-32]-self.cfg.FMEANS[2])/self.cfg.FSTDS[2]
                    state[self.cfg.FIDX[2]-32:self.cfg.FIDX[3]-32]=(state[self.cfg.FIDX[2]-32:self.cfg.FIDX[3]-32]-self.cfg.FMEANS[3])/self.cfg.FSTDS[3]
                    if self.cfg.NOT_ASR == False:
                        state[self.cfg.FIDX[3]-32:self.cfg.FIDX[4]-32] = (state[self.cfg.FIDX[3]-32:self.cfg.FIDX[4]-32])
            elif self.cfg.NOT_REC:
                state[:self.cfg.FIDX[0]]=(state[0:self.cfg.FIDX[0]]-self.cfg.FMEANS[0])/self.cfg.FSTDS[0]
                state[self.cfg.FIDX[0]:self.cfg.FIDX[1]]=(state[self.cfg.FIDX[0]:self.cfg.FIDX[1]]-self.cfg.FMEANS[1])/self.cfg.FSTDS[1]
                state[self.cfg.FIDX[2]-32:self.cfg.FIDX[3]-32]=(state[self.cfg.FIDX[2]-32:self.cfg.FIDX[3]-32]-self.cfg.FMEANS[3])/self.cfg.FSTDS[3]
                if self.cfg.NOT_ASR == False:
                    state[self.cfg.FIDX[3]-32:self.cfg.FIDX[4]-32] = (state[self.cfg.FIDX[3]-32:self.cfg.FIDX[4]-32])
           
            else:
                state[:self.cfg.FIDX[0]]=(state[0:self.cfg.FIDX[0]]-self.cfg.FMEANS[0])/self.cfg.FSTDS[0]
                state[self.cfg.FIDX[0]:self.cfg.FIDX[1]]=(state[self.cfg.FIDX[0]:self.cfg.FIDX[1]]-self.cfg.FMEANS[1])/self.cfg.FSTDS[1]
                state[self.cfg.FIDX[1]:self.cfg.FIDX[2]]=(state[self.cfg.FIDX[1]:self.cfg.FIDX[2]]-self.cfg.FMEANS[2])/self.cfg.FSTDS[2]
                state[self.cfg.FIDX[2]:self.cfg.FIDX[3]] = (state[self.cfg.FIDX[2]:self.cfg.FIDX[3]]-self.cfg.FMEANS[3])/self.cfg.FSTDS[3]
                if self.cfg.NOT_ASR == False:
                    state[self.cfg.FIDX[3]:self.cfg.FIDX[4]] = (state[self.cfg.FIDX[3]:self.cfg.FIDX[4]])
        

        return state

        
        
    def buildState(self, randomDelay=30):
        """
        Build the current environment state and optionally advance by a small random delay.

        The delay is used during training to avoid always sampling the exact same
        decision frames, while ensuring that the simulation does not skip over an
        upcoming human-need event.
        """

        # the random delay is there to avoid always taking the same
        # frames, either the 0 frame or the same exact reward instants.
        # Pass 0 if you want to avoid this random delay
        # random.seed(10)
        
        if self.mode == 'train':
            delay = random.randint(0, randomDelay)
        else:
            delay = 0

        # We need to check that the delay does not advance time
        # beyond an ASR event
        futureTime = self.episode.currentFrame + delay
        ac = self.episode.findNextRobotAction()
        maxTime = min(futureTime, ac.frameInit)
        
        self.updateInputsAction(self.episode.currentFrame, maxTime)
        self.episode.advanceTo(maxTime)
        
        #self.episode.skipFrames(delay)
        if self.cfg.ENV_VERBOSE: print("\t\tRandom delay = %d" %delay)
        
        feat = self.episode.frameFeatures[self.episode.currentFrame]
        oit = np.array(list(self.oit.values()), dtype=np.float32)
        asr = copy.deepcopy(self.asr)        

        if (self.cfg.NOT_PRED and self.cfg.NOT_REC):
            feat = feat[self.cfg.FIDX[1]:]
        elif self.cfg.NOT_PRED:
            feat = feat[self.cfg.FIDX[0]:]
        elif self.cfg.NOT_REC:
            feat = np.concatenate((feat[:self.cfg.FIDX[0]],feat[self.cfg.FIDX[1]:]), axis = 0)
         
        
        state = np.concatenate((feat, oit), axis=0)
        if self.cfg.NOT_ASR == False:
            state = np.concatenate((feat, oit,asr), axis=0)
            
        if self.cfg.NORM_FEATURES:
            
            if self.cfg.NOT_PRED and self.cfg.NOT_REC:
                state[:self.cfg.FIDX[2]-64]=(state[0:self.cfg.FIDX[2]-64]-self.cfg.FMEANS[2])/self.cfg.FSTDS[2]
                state[self.cfg.FIDX[2]-64:self.cfg.FIDX[3]-64]=(state[self.cfg.FIDX[2]-64:self.cfg.FIDX[3]-64]-self.cfg.FMEANS[3])/self.cfg.FSTDS[3]
                if self.cfg.NOT_ASR == False:
                    state[self.cfg.FIDX[3]-64:self.cfg.FIDX[4]-64] = (state[self.cfg.FIDX[3]-64:self.cfg.FIDX[4]-64])
           
            elif self.cfg.NOT_PRED:
               
                    state[:self.cfg.FIDX[0]]=(state[0:self.cfg.FIDX[1]-32]-self.cfg.FMEANS[1])/self.cfg.FSTDS[1]
                    state[self.cfg.FIDX[1]-32:self.cfg.FIDX[2]-32]=(state[self.cfg.FIDX[1]-32:self.cfg.FIDX[2]-32]-self.cfg.FMEANS[2])/self.cfg.FSTDS[2]
                    state[self.cfg.FIDX[2]-32:self.cfg.FIDX[3]-32]=(state[self.cfg.FIDX[2]-32:self.cfg.FIDX[3]-32]-self.cfg.FMEANS[3])/self.cfg.FSTDS[3]
                    if self.cfg.NOT_ASR == False:
                        state[self.cfg.FIDX[3]-32:self.cfg.FIDX[4]-32] = (state[self.cfg.FIDX[3]-32:self.cfg.FIDX[4]-32])
            elif self.cfg.NOT_REC:
                state[:self.cfg.FIDX[0]]=(state[0:self.cfg.FIDX[0]]-self.cfg.FMEANS[0])/self.cfg.FSTDS[0]
                state[self.cfg.FIDX[0]:self.cfg.FIDX[1]]=(state[self.cfg.FIDX[0]:self.cfg.FIDX[1]]-self.cfg.FMEANS[1])/self.cfg.FSTDS[1]
                state[self.cfg.FIDX[2]-32:self.cfg.FIDX[3]-32]=(state[self.cfg.FIDX[2]-32:self.cfg.FIDX[3]-32]-self.cfg.FMEANS[3])/self.cfg.FSTDS[3]
                if self.cfg.NOT_ASR == False:
                    state[self.cfg.FIDX[3]-32:self.cfg.FIDX[4]-32] = (state[self.cfg.FIDX[3]-32:self.cfg.FIDX[4]-32])

            else:
                state[:self.cfg.FIDX[0]]=(state[0:self.cfg.FIDX[0]]-self.cfg.FMEANS[0])/self.cfg.FSTDS[0]
                state[self.cfg.FIDX[0]:self.cfg.FIDX[1]]=(state[self.cfg.FIDX[0]:self.cfg.FIDX[1]]-self.cfg.FMEANS[1])/self.cfg.FSTDS[1]
                state[self.cfg.FIDX[1]:self.cfg.FIDX[2]]=(state[self.cfg.FIDX[1]:self.cfg.FIDX[2]]-self.cfg.FMEANS[2])/self.cfg.FSTDS[2]
                state[self.cfg.FIDX[2]:self.cfg.FIDX[3]] = (state[self.cfg.FIDX[2]:self.cfg.FIDX[3]]-self.cfg.FMEANS[3])/self.cfg.FSTDS[3]
                if self.cfg.NOT_ASR == False:
                    state[self.cfg.FIDX[3]:self.cfg.FIDX[4]] = (state[self.cfg.FIDX[3]:self.cfg.FIDX[4]])

       
        return state
    
    def reset(self, options={}):

        """
        This method is going to move to the next episode (or the first episode if this is the
        first time it is called). Returns:
            observation: the initial state (analogous to the observation returned by step)
            info:        a dict with auxiliary info complementing observation (analogous
                         to the one returned by step)
        """
                
        # 1 - Select and reset the next episode
        #Advance one episode
        self.episode = next(self.allEpisodes)   
        self.episode.reset()
        
        # 2 - reset objects in table
        self.oit = copy.deepcopy(self.cfg.OBJECTS_INIT_STATE)
        self.asr = np.zeros((self.cfg.NUM_ROBOT_ACTIONS-1,),dtype=np.float32)

        self.trackerActPred = ActionTracker()
        self.trackerActRec = ActionTracker()
        self.trackerVideo = ActionTracker()

        # 3 - build the next state and info
        state = self.buildState()

        if self.cfg.ENV_VERBOSE:
            self.episode.visualizeAndSave(mode='frames')
            print("Recipe: %s" %(self.episode.folderPath))
            
        self.CA_intime = 0
        self.CA_late = 0
        self.IA_intime = 0
        self.IA_late = 0
        self.UA_intime = 0
        self.UA_late = 0
        self.CI = 0
        self.II = 0
        self.IMPOSA = 0
        self.anticipationCorrectRobotSinceBeggining = []

        self.anticipationCAandASR = []
        
        return state, {}
    
    def step(self, action):
        """
        Execute one environment step for a given robot action.

        Depending on the chosen action and the upcoming human needs, the robot may
        act proactively, remain idle, or trigger a reactive ASR-based correction.

        Returns:
            observation: next state representation.
            reward: scalar reward for the selected action.
            terminated: whether the episode has ended naturally.
            truncated: whether the episode ended due to an external stop condition.
            info: dictionary with diagnostic reward terms and timing metrics.
            actionASR: reactive action triggered by ASR, if any.
        """
        
        # The environment must be reset before starting a new episode
        assert self.episode.terminated == False, "The episode must be reset before calling step the first time"
        assert 0 <= action < self.cfg.NUM_ROBOT_ACTIONS, "Action must be between 0 and 5"
        # self.updateInputs()
        if self.cfg.ENV_VERBOSE:
            print("Robot taking action %d: %s" %(action,self.cfg.ROBOT_ACTIONS_MEANINGS[action]))

        if self.cfg.ROBOT_ACTIONS_MEANINGS[action] == 'do nothing':
            Rt, Re, R, nextState, actionASR = self.stepIdle(action)
            
        else:
            Rt, Re, R, nextState, actionASR = self.stepAction(action)

        if actionASR is not None:
            self.asr[actionASR]=1
            
        # create the info dict and return the rewards and nextState
        if self.cfg.ENV_VERBOSE:
            print("Next state is: %d" %(self.episode.currentFrame))
            print("Rt = %.4f, Re = %.4f, R = %.4f" %(Rt, Re, R))
            print("total recipe delay (in frames): %d" %(self.episode.delay))

        self.episode.interactionTime = self.episode.lastFrame + self.episode.delay
        if self.episode.terminated or self.episode.truncated:
            self.episode.interactionTime = self.episode.lastFrame + self.episode.delay

        infoDict = {'timeReward': Rt, 'energyReward': Re, 'interactionTime': self.episode.interactionTime, 'minTime': self.episode.minTime, 'reactiveTime':self.episode.reactiveTime}
        
        return nextState, R, self.episode.terminated, self.episode.truncated, infoDict, actionASR
    
    def stepIdle(self, action):
        
        # if the robot decides to stay idle, we need to check if we are going
        # to need something from the robot before the next decision time step 
        ac = self.episode.findNextRobotAction()
        frames2NeedRobot = ac.frameInit - self.episode.currentFrame
        self.durationActionRobot =self.cfg.DECISION_RATE
        if self.cfg.ENV_VERBOSE:
            print("\tAction taken in frame: %d" %(self.episode.currentFrame))
            print("\tStill %s frames to next robot action" %(str(frames2NeedRobot)))

        # for staying idle the reward is zero unless a human needs to wait
        timeReward, energyReward, reward = 0,0,0
        actionASR = None

        # in this case, there will be a human needsRobot atomic action event
        # before the robot has a new chance to make a decision
        if frames2NeedRobot <=self.cfg.DECISION_RATE:
            
            # if necessary object is NOT on the table
            if self.oit[ac.requiredObj] != 1:
                self.II += 1
                if self.cfg.ENV_VERBOSE: 
                    print("\tNecessary object not on the table: ASR")
            
                timeReward, energyReward, reward = self.computeReward(
                        ac.frameInit, ac.frameInit, ac.requiredObj, ac.requiredObj, 
                        ASR = True)
                
                actionASR = self.cfg.ROBOT_OBJECTS_2_ACTIONS[ac.requiredObj]
                
                self.oit[ac.requiredObj] = 1
                self.anticipationCAandASR.append(0)
                
                self.updateInputsAction(self.episode.currentFrame, ac.frameInit)
                # advance to start of need robot ASR
                self.episode.advanceTo(ac.frameInit)
                
                
                
            else:
                self.CI += 1
                # the necessary object was on the table, thus, advance frames normally
                self.updateInputsAction(self.episode.currentFrame, self.cfg.DECISION_RATE+self.episode.currentFrame)
                self.episode.skipFrames(self.cfg.DECISION_RATE)

        else:

            if self.cfg.ENV_VERBOSE: 
                print("\tStill time til next robot action")
            self.CI += 1
            self.episode.skipFrames(self.cfg.DECISION_RATE)
            self.updateInputsAction(self.episode.currentFrame, self.cfg.DECISION_RATE+self.episode.currentFrame)
            
        # compute nextState
        nextState = self.buildState(randomDelay=0)
        return timeReward, energyReward, reward, nextState, actionASR
    
    def stepAction(self, action):
        
        """
        This method is called when the robot decides to take an action different
        than staying idle. The logic of the rewards and next states is different
        depending on whether the robot action will finish before or after the 
        next time the human needs something from the robot
        """
        
        framesNeededRobot = self.computeRobotTime(action)
    
        prev_frame = copy.deepcopy(self.episode.currentFrame)
        
        robotActionEnds = self.episode.currentFrame + framesNeededRobot
        self.durationActionRobot = framesNeededRobot
        ac = self.episode.findNextRobotAction()

        actionASR = None
        
        # robot action ends before the next frame where the human needs the robot
        # or robot took action after last need robot atomic action event (ac.frameInit = inf)
        if robotActionEnds < ac.frameInit:
            Rt, Re, R, nextState, actionASR = self.stepActionInTime(
                action, framesNeededRobot, robotActionEnds, ac)
            
            if Rt == 0 and Re == 0:

                self.CA_intime += 1
            elif Re == 0 and Rt !=0:
                self.CA_late += 1
            elif Re != 0 and Rt ==0:
                self.IA_intime += 1
            else:
                # print('ia late, asr: ', str(actionASR))
                self.IA_late += 1
 
            
        # robot action ends AFTER the next needRoot atomic action
        else:
            Rt, Re, R, nextState, actionASR = self.stepActionLate(
                action, framesNeededRobot, robotActionEnds, ac)

            if Rt == 0 and Re ==0:
                self.CA_intime += 1
            elif Re == 0 and Rt != 0:
                self.CA_late += 1
            if Re != 0 and Rt == 0:
                self.IA_intime += 1
            elif Re!=0 and Rt!=0:
                self.IA_late += 1
                
        for ff in self.episode.frameInitActionHumanCannotDo:

            if  int(list(ff.keys())[0]) == action:
                
                if (int(list(ff.values())[0]) - prev_frame) > 0:
                    self.anticipationCAandASR.append(int(list(ff.values())[0]) - prev_frame)
                    self.anticipationCorrectRobotSinceBeggining.append(int(list(ff.values())[0]) - prev_frame)
                else:
                    if actionASR == None:
                        self.anticipationCAandASR.append(0)
                        self.anticipationCorrectRobotSinceBeggining.append(0)

        if actionASR != None:
            self.anticipationCAandASR.append(0)

        return Rt, Re, R, nextState, actionASR
   
    def stepActionInTime(self, action, framesNeededRobot, robotActionEnds, ac):

        """
        This method is called when the robot takes an action that does not cause
        the human to wait. Maybe because it ends before the next human-needs-asistance 
        atomic action or because there are no more human-need-assistance moments. 
        The action might be useful for the recipe, in which case it gets no penalty, 
        or incorrect, in which case it does get penalty
        """

        objTaken =self.cfg.ROBOT_ACTIONS_2_OBJECTS[action]
        actionFrame = self.episode.currentFrame
        self.episode.advanceTo(robotActionEnds)
        self.updateInputsAction(actionFrame, robotActionEnds)
        actionASR = None
        
        if self.cfg.ENV_VERBOSE:
            print("\tRobot needs %d frames to complete action" %(framesNeededRobot))
            print("\tRobot action will end at frame: %d" %(robotActionEnds))
            print("\tRobot action will end before next human-need-robot action")
            print("\tRobot is taking the %s " %(objTaken))
            print("\tIs the object necessary for the recipe?: %s" %(str(
                self.episode.isObjectNecessary(objTaken)  )))
            print("\tIs this action possible?: %s" %(self.oit[objTaken] == 0))

        # compute reward
        if self.episode.isObjectNecessary(objTaken) and self.oit[objTaken] == 0:
            
            timeReward, energyReward, reward = 0,0,0
        
        else:
            if self.episode.currentFrame == ac.frameInit:
               
                # if requiredObj is not on the table, asr is necessary
                asr = (self.oit[ac.requiredObj] == 0)
                timeReward, energyReward, reward = self.computeReward(
                    actionFrame, ac.frameInit, objTaken, ac.requiredObj, ASR=asr,
                    framesNeeded=framesNeededRobot)
                self.oit[ac.requiredObj] = 1
                if asr: actionASR = self.cfg.ROBOT_OBJECTS_2_ACTIONS[ac.requiredObj]
            else:
                timeReward, energyReward, reward = self.computeReward(
                    actionFrame, ac.frameInit, objTaken, 5, ASR=False,
                    framesNeeded=framesNeededRobot)

        self.oit[objTaken] = 1
        nextState = self.buildState()

        return timeReward, energyReward, reward, nextState, actionASR
    
    def stepActionLate(self, action, framesNeededRobot, robotActionEnds, ac):

        """

        This method is called when the robot takes an action that ends after the
        human needs asistance. Human might need to wait or not, depending on previous
        robot actions (current objects in table)
        """
        objTaken =self.cfg.ROBOT_ACTIONS_2_OBJECTS[action]
        actionFrame = self.episode.currentFrame
        actionASR = None
        # find the next needRobot atomic action where the human needs to wait
        acWaiting = None
        for ac_i in self.episode.atomicActions:
            if (ac_i.requiresRobot and
                ac_i.frameInit < robotActionEnds and 
                ac_i.frameEnd > self.episode.currentFrame):

                if self.oit[ac_i.requiredObj] == 0:
                    acWaiting = ac_i
                    break
                
        if self.cfg.ENV_VERBOSE:
            print("\tRobot needs %d frames to complete action" %(framesNeededRobot))
            print("\tRobot action will end at frame: %d" %(robotActionEnds))
            print("\tRobot action will end after next human-need-robot action")
            print("\tRobot is taking the %s " %(objTaken))
            print("\tIs the object necessary for the recipe?: %s" %(str(
                self.episode.isObjectNecessary(objTaken)  )))
            print("\tIs this action possible?: %s" %(self.oit[objTaken] == 0))
            print("\tDoes the human need to wait?: %s" %(str(acWaiting is not None)))


        # if the human does not need to wait
        if acWaiting is None:
            
            # move to robotActionEnds frame
            self.episode.advanceTo(robotActionEnds)
            self.updateInputsAction(actionFrame, robotActionEnds)
            # move to ntri after robot finishes

            # if objTaken is correct for the recipe and is not on the table already
            if self.episode.isObjectNecessary(objTaken) and self.oit[objTaken] == 0: 
                timeReward, energyReward, reward = 0,0,0
            
            # otherwise, just the lambda*energy penalty
            else:
                timeReward, energyReward, reward = self.computeReward(
                    actionFrame, -1, objTaken, 5, ASR=False,
                    framesNeeded=framesNeededRobot)          

        # the human needs to wait because the requiredObj is not on the table
        else: 
           
            # compute reward depending on whether the object was correct or not
            asr = False if objTaken == acWaiting.requiredObj else True
            if self.cfg.ENV_VERBOSE: print("ASR?: %s" %(str(asr)))

            
            timeReward, energyReward, reward = self.computeReward(
                actionFrame, acWaiting.frameInit, objTaken, acWaiting.requiredObj, 
                ASR=asr, framesNeeded=framesNeededRobot)
            
            if asr:
                actionASR = self.cfg.ROBOT_OBJECTS_2_ACTIONS[ac.requiredObj]
                self.oit[acWaiting.requiredObj] = 1

            # move robot to end of needs to wait ac
            self.episode.advanceTo(acWaiting.frameInit)
            self.updateInputsAction(actionFrame, acWaiting.frameInit)
        # update iot and compute next state
        self.oit[objTaken] = 1
        nextState = self.buildState()

        return timeReward, energyReward, reward, nextState, actionASR

    def computeReward(self, actionFrame, waitingFrame, objTook, objNeeded, ASR=False, framesNeeded=None):
        """
        Compute the reward associated with a robot action.

        The reward combines:
        - time penalty: how long the human has to wait
        - energy penalty: cost of incorrect, unnecessary, or impossible actions

        If ASR is triggered, the method also accounts for the corrective robot action.
        """

        timeReward, energyReward, reward = 0,0,0
        lambda_ =self.cfg.FACTOR_ENERGY_PENALTY

        # 1 - compute the time it takes the robot to do the first action or use the one passed
        if framesNeeded is None:
            framesNeeded = self.computeRobotTime(self.cfg.ROBOT_OBJECTS_2_ACTIONS[objTook])
        #Asign the robot time
        robotTime = framesNeeded 

        # 2 - Compute the time reward if the human is waiting and assign
        humanWaits = actionFrame + framesNeeded - waitingFrame
        timeReward = 0 if humanWaits < 0 else humanWaits

        #3.a If the robot has brought the right object
        if objTook == objNeeded:
            if ASR:
                #add the ASR penalty if neccessary
                timeReward +=self.cfg.ASR_PENALTY
                robotTime += self.cfg.ASR_PENALTY
                self.episode.actionsASR[self.episode.currentFrame+self.episode.delay] = {'action':self.cfg.ROBOT_OBJECTS_2_ACTIONS[objNeeded], 'duration': robotTime}    
        #3.a If the robot has brought the wrong object
        else:

            #If it is an impossible object
            if self.oit[objTook] == 1:
                energyReward +=self.cfg.IMPOSSIBLE_ACTION_PENALTY*robotTime
                self.IMPOSA += 1
            else:
                # compute the energy penalty using the object it brought
                energyReward +=robotTime
            
            # Then, if there is an ASR,is because the robot needs to perform a second action
            if ASR:
                robotTimeExtra = self.computeRobotTime(self.cfg.ROBOT_OBJECTS_2_ACTIONS[objNeeded]) + self.cfg.ASR_PENALTY
                #Log the second action
                self.episode.actionsASR[self.episode.currentFrame+self.episode.delay+robotTime] = {'action':self.cfg.ROBOT_OBJECTS_2_ACTIONS[objNeeded], 'duration': robotTimeExtra}
                #Add this extra time to the reward
                timeReward += robotTimeExtra 
                
        # If waitingFrame == -1, force zero time penalty (no waiting scenario)
        if waitingFrame == -1:
            timeReward = 0

        # 4. Compute final reward Rt + lambda*Re
        reward = timeReward + lambda_*energyReward

        # Accumulate episode delay and return
        self.episode.delay += timeReward
        inset = ''
        if timeReward != 0:
            cont = 0
            for ap in self.episode.visualizeInteractionPerson:
                if ap.frameInit > self.episode.currentFrame:
                    cont += 1
                    ap.frameInit += timeReward
                    ap.frameEnd += timeReward
                    if cont ==1:
                        inset = AtomicAction(len(self.episode.visualizeInteractionPerson)+1, 99, 0, 
                            copy.deepcopy(ap.frameInit) -timeReward, ap.frameInit, False, None, self.cfg)
                        
        if inset != '':
            self.episode.visualizeInteractionPerson.append(inset)
        return -timeReward, -energyReward, -reward

    def computeRobotTime(self, action):
        
        # Estimated time required by the robot to execute the action
        # Modeled following a gaussian distribution as stated in the ROBIO paper
        # In the paper it also accounts for robot mistakes
        
        beta =self.cfg.ROBOT_TIME_BETA
        probFail =self.cfg.ROBOT_PROB_FAILURE
        humanTime =self.cfg.HUMAN_AVERAGE_DURATIONS[action]
        failed = random.random() < probFail
        mu = beta*(humanTime + failed*(0.5*humanTime))
        std = 0.2*mu
        
        if self.mode == 'train':
            time = max(1, int(random.gauss(mu, std)))
        else:
            time = int(mu)
        return time

    def updateOit(self, update):
        
        self.oit = copy.deepcopy(update)
        
    def updateASR(self, update):
        
          self.asr = copy.deepcopy(update)
        
    def updateCurrentFrame (self, frame):
        
        self.episode.currentFrame = frame

    def updateDelay (self, delay):
        self.episode.delay = delay
        
    def updateInputs(self):
        feat = self.episode.frameFeatures[self.episode.currentFrame]
        self.trackerActPred.update(self.episode.currentFrame+self.episode.delay,self.cfg.ATOMIC_ACTIONS_MEANINGS[np.argmax(feat[:32])])
        self.trackerActRec.update(self.episode.currentFrame+self.episode.delay,self.cfg.ATOMIC_ACTIONS_MEANINGS[np.argmax(feat[32:64])])
        self.trackerVideo.update(self.episode.currentFrame+self.episode.delay,self.cfg.ATOMIC_ACTIONS_MEANINGS[self.episode.currentFrame])

    def updateInputsAction(self, currentF, endF):

        if self.episode.terminated == True:
            frameFeatures = min(endF, self.episode.lastFrame)
            feat = self.episode.frameFeatures[frameFeatures]

            action_pred = np.argmax(feat[:32])  # Compute it only once
            action_rec = np.argmax(feat[32:64])  # Compute it only once

            self.trackerActPred.finalize(endF+self.episode.delay)
            self.trackerActRec.finalize(endF+self.episode.delay)
            
            dictAnnotation = self.dictAnnotations[self.episode.folderName]
            
            if dictAnnotation is not None:
                 label, endFrame = self.get_label_for_frame(dictAnnotation, endF)
                 if endF > endFrame:
                     label, _ = self.get_label_for_frame(dictAnnotation, endFrame-1)
                     self.trackerVideo.finalize(endF+self.episode.delay)
                 if label is not None:
                     self.trackerVideo.finalize(endF+self.episode.delay)
        else:
            for frame in range(int(currentF), int(endF)):
                frameFeatures = min(frame, self.episode.lastFrame)
                feat = self.episode.frameFeatures[frameFeatures]
    
                action_pred = np.argmax(feat[:32])  # Compute it only once
                action_rec = np.argmax(feat[32:64])  # Compute it only once
    
                self.trackerActPred.update(frame+self.episode.delay,self.cfg.ATOMIC_ACTIONS_MEANINGS[action_pred])
                self.trackerActRec.update(frame+self.episode.delay,self.cfg.ATOMIC_ACTIONS_MEANINGS[action_rec])
                
                dictAnnotation = self.dictAnnotations[self.episode.folderName]
                
                if dictAnnotation is not None:
                     label, endFrame = self.get_label_for_frame(dictAnnotation, frame)
                     if frame > endFrame:
                         label, _ = self.get_label_for_frame(dictAnnotation, endFrame-1)
                         self.trackerVideo.update(frame+self.episode.delay,self.cfg.ATOMIC_ACTIONS_MEANINGS[label])
                     if label is not None:
                         self.trackerVideo.update(frame+self.episode.delay,self.cfg.ATOMIC_ACTIONS_MEANINGS[label])
           
            
    def computeHumanTime(self, object):

        return self.cfg.ROBOT_AVERAGE_DURATIONS[object]

    def render(self, mode='human'):
        return
    
    def close(self):
        return
    
        
    def debug_print_episode_annotations(self, max_rows: int = 200):
        ep = self.episode.folderName
        print("\n" + "="*100)
        print(f"[DEBUG-ANN] EP={ep} lastFrame={int(self.episode.lastFrame)}")
        print("="*100)
    
        # (A) Pandas labels
        df = self.dataFramesVideos.get(ep, None)
        if df is None:
            print("[DEBUG-ANN] pandasDF: None (no labels file or empty)")
        else:
            cols = [c for c in ["frame_init", "frame_end", "label", "label_obj"] if c in df.columns]
            print(f"[DEBUG-ANN] pandasDF rows={len(df)} cols={cols}")
            print(df[cols].head(max_rows).to_string(index=True))
    
        # (B) Atomic actions used by the simulator
        print("\n[DEBUG-ANN] atomicActions:")
        print("idx | frameInit | frameEnd | dur | label | meaning | requiresRobot | requiredObj")
        for ac in self.episode.atomicActions:
            meaning = self.cfg.ATOMIC_ACTIONS_MEANINGS.get(ac.actionLabel, ac.actionLabel)
            print(f"{ac.index:3d} | {int(ac.frameInit):8d} | {int(ac.frameEnd):7d} | {int(ac.duration):3d} | "
                  f"{ac.actionLabel:5} | {str(meaning)[:20]:20s} | {str(ac.requiresRobot):12s} | {str(ac.requiredObj)}")
    
        # (C) Only robot needs 
        needs = [ac for ac in self.episode.atomicActions if ac.requiresRobot]
        needs.sort(key=lambda a: int(a.frameInit))
        print("\n[DEBUG-ANN] robot needs (requiresRobot=True):")
        print("k | obj | need=frameInit | humanSegEnd=frameEnd | actionId(bring)")
        for k, ac in enumerate(needs):
            obj = ac.requiredObj
            aid = self.cfg.ROBOT_OBJECTS_2_ACTIONS.get(obj, None)
            print(f"{k:2d} | {str(obj):10s} | {int(ac.frameInit):12d} | {int(ac.frameEnd):14d} | {aid}")
        print("="*100 + "\n")
    
        
    
