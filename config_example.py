"""
Example configuration file for the Personalized HRI Sim2Real framework.

This configuration is intended for:
- quick testing
- debugging
- understanding the pipeline

NOTE:
This configuration does NOT reproduce the results reported in the paper.
"""

import torch
import numpy as np


# =========================================================
# GENERAL
# =========================================================
EXPERIMENT_NAME = "example_run"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 10


# =========================================================
# DATASET
# =========================================================
DATASET_NAME = 'Dataset'

# Use a single fold for simplicity
FIVE_FOLD = False
FOLDS = ["example_fold.txt"]

# Path to dataset (relative to repo)
ANNOTATIONS_FOLDER = "data/example/"

# File containing action annotations
LABELS_FILE_NAME = "labels_updated.pkl"

# Temporal configuration
VIDEO_FPS = 30
ANNOTATIONS_FPS = 5
ANNOTATIONS_DIM = 110


# =========================================================
# FEATURE NORMALIZATION
# =========================================================
# Mean and std used to normalize input features
FMEANS = np.array([0.03125, 0.03125, 0.15890419, 0.73913043, 0.25], dtype=np.float32)
FSTDS = np.array([0.16001669, 0.16573451, 1.01803785, 0.43910891, 0.433], dtype=np.float32)

# Feature index boundaries (used to split modalities)
FIDX = np.array([32, 64, 110, 133, 138], dtype=int)


# =========================================================
# MODEL ARCHITECTURE
# =========================================================
HIDDEN_SIZE_LSTM = 512
EMBEDDING_SIZE = 32
MLP_SIZE = 512
NOISE_STD = 0.01

# Enable / disable input modalities
NOT_PRED = False     # disable action prediction input
NOT_REC = False      # disable action recognition input
NOT_ASR_RM = False   # disable ASR in replay memory


# =========================================================
# TRAINING PARAMETERS
# =========================================================
LR = 7e-4
EPOCHS = 25
BATCH_SIZE = 512

ONLY_TEST = False
NUM_OPTIMIZATION_PER_STEP = 2


# =========================================================
# ENVIRONMENT PARAMETERS
# =========================================================
DQN = 'DQN'
UPDATE_HIDDEN = True
DECISION_RATE = 30
NORM_FEATURES = True


# =========================================================
# ROBOT PARAMETERS
# =========================================================
ROBOT_TIME_BETA = 2
ROBOT_PROB_FAILURE = 0


# =========================================================
# REINFORCEMENT LEARNING
# =========================================================

# Exploration
EPS_START = 0.5
EPS_END = 0.0
EPS_DECAY = 100
EPS_TEST = 0
EXPLORATION_MODE = 'EPS_GREEDY'

# Target network
TAU = 0.005
UPDATE_TARGET = 50

# N-step learning
N_STEP = 3

# Reward
GAMMA = 0.9
ASR_PENALTY = 3.5 * VIDEO_FPS  # penalty in frames
FACTOR_ENERGY_PENALTY = 1.0
NORMALIZE_REWARD = True

# Action constraints
IMPOSSIBLE_ACTION_PENALTY = 1.0
NO_IMPOSSIBLE_ACTIONS = True

# Replay memory
MEMORY_SIZE = 50000
MIN_MEMORY_SIZE = 1024


# =========================================================
# MODEL LOADING (OPTIONAL)
# =========================================================
LOAD_MODEL = False
MODEL_PATH = './models/ablation_prob-based'


# =========================================================
# DEBUG / VERBOSE
# =========================================================
TEST_VERBOSE = False


# =========================================================
# ENVIRONMENT DEFINITIONS
# =========================================================

# Robot actions
ROBOT_ACTIONS_MEANINGS = {
    0: 'bring butter',
    1: 'bring jam',
    2: 'bring milk',
    3: 'bring nutella',
    4: 'bring tomato sauce',
    5: 'do nothing',
}

ROBOT_ACTIONS_MEANINGS_vis = {
    0: 'bring butter',
    1: 'bring jam',
    2: 'bring milk',
    3: 'bring nutella',
    4: 'bring tomato sauce',
    5: 'idle',
}

# Mapping actions ↔ objects
ROBOT_ACTIONS_2_OBJECTS = {
    0: 'butter',
    1: 'jam',
    2: 'milk',
    3: 'nutella',
    4: 'tomato sauce',
    5: 'do nothing',
}

ROBOT_OBJECTS_2_ACTIONS = {v: k for k, v in ROBOT_ACTIONS_2_OBJECTS.items()}

# Human timing (in frames)
HUMAN_AVERAGE_DURATIONS = [210, 210, 210, 210, 210, 0]


# =========================================================
# HUMAN ↔ ROBOT INTERACTION
# =========================================================

HUMAN_ACTIONS_2_OBJECTS = {
    1: 'milk',
    16: 'butter',
    17: 'jam',
    18: 'tomato sauce',
    19: 'nutella',
    25: 'milk',
}

NEXT_ACTION_PRED_2_ROBOT_ACTIONS = {
    8: 2,
    12: 0,
    13: 1,
    14: 4,
    15: 3,
}


# =========================================================
# RECIPES
# =========================================================

RECIPIES = {
    0: 'Chocolate milk',
    1: 'Latte',
    2: 'American coffee',
    3: 'Cereals bowl',
    4: 'Nutella toast',
    5: 'Butter toast',
    6: 'Tomato toast',
}

RECIPIES_TO_ROBOT_ACTIONS = {
    0: 2,
    1: 2,
    2: 5,
    3: 2,
    4: 3,
    5: [0, 1],
    6: 4,
}


# =========================================================
# OBJECTS
# =========================================================

OBJECTS_INIT_STATE = {
    'background': 0, 'bowl': 1, 'butter': 0, 'cereals': 1, 'coffee': 1,
    'cup': 1, 'cutting board': 1, 'fork': 1, 'fridge': 1,
    'jam': 0, 'knife': 1, 'microwave': 1, 'milk': 0,
    'nesquik': 1, 'nutella': 0, 'olive oil': 1,
    'plate': 1, 'sliced bread': 1, 'spoon': 1,
    'sugar': 1, 'toaster': 1, 'tomato sauce': 0, 'water': 1
}

OBJECTS_MEANINGS = {
	0: 'background', 1: 'bowl',	2: 'butter', 3: 'cereals', 4: 'coffee',	5: 'cup', 6: 'cutting board', 7: 'fork', 8: 'fridge',
	9: 'jam', 10: 'knife', 11: 'microwave', 12: 'milk', 13: 'nesquik', 14: 'nutella', 15: 'olive oil', 16: 'plate', 17: 'sliced bread',
	18: 'spoon', 19: 'sugar', 20: 'toaster', 21: 'tomato sauce', 22: 'water'
    }
# =========================================================
# FINAL DERIVED VALUES
# =========================================================
ENV_VERBOSE = False
INTERACTIVE_OBJECTS_ROBOT = ['butter', 'jam', 'milk', 'nutella', 'tomato sauce']

NUM_ROBOT_ACTIONS = len(ROBOT_ACTIONS_MEANINGS)
NUM_OBJECTS = len(OBJECTS_MEANINGS)

ROBOT_NEEDED_FOR_ACTIONS = [1, 16, 17, 18, 19, 25]
ACTIONS_TO_REMOVE_FROM_RECIPES = [8, 12, 13, 14, 15]

HUMAN_ACTIONS_2_ROBOT_ACTIONS = {
    8: 2,
    12: 0,
    13: 1,
    14: 4,
    15: 3
}