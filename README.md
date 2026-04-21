# Personalized HRI Sim2Real
Official implementation of "Reducing the Sim2Real Gap in Personalized HRI: Simulation-Based Pretraining and Real-World Adaptation and Validation", to appear in ACM THRI.
---

## Overview

This repository presents a framework for **personalized and anticipatory human-robot interaction (HRI)** using reinforcement learning.

The proposed approach reduces the sim-to-real gap by combining:

* simulation-based pretraining using egocentric human data
* multimodal perception (gaze, speech, egocentric vision)
* real-world fine-tuning for user-specific adaptation

The system enables robots to anticipate user needs and provide proactive assistance in collaborative tasks.

As described in the paper, pretrained agents:

* reduces unnecessary robot actions
* improves responsiveness and timing of interventions
* improves task efficiency by learning when and how to act
* achieve higher user satisfaction in real-world experiments 

**Note:** This repository focuses on the simulation and training components of the framework. The full real-world robotic system and human-subject experiment infrastructure described in the paper are not included.

---

## Method Summary

Our framework consists of three main components:

### 1. Egocentric Simulation Environment

We train agents using a replay-based simulation built from real human demonstrations.

* Based on egocentric video, gaze, IMU, and speech
* Avoids synthetic user modeling
* Enables scalable and safe pretraining

### 2. Reinforcement Learning for Decision Making

Robot behavior is learned using Deep Q-Learning with:

* multimodal state representation (actions, gaze, memory, speech)
* n-step learning for delayed rewards
* replay memory augmentation using ASR signals

The robot learns **when and what assistance to provide**.

### 3. Real-World Personalization

Policies are designed to be adapted online during interaction with real users (not included in this repository).

---

## System Overview

The system integrates perception and decision-making modules:

* Active Object Detection (AOD) → user attention
* Human Action Analysis (HAA) → current & future actions
* ASR → explicit commands
* Robot Decision Making (RDM) → action selection

These modules operate in a closed-loop interaction pipeline 

---

## Repository Structure


```text
train.py         # training script for simulation pretraining
test.py          # evaluation script
sim_env.py       # simulation environment
DQN.py           # Deep Q-Network implementation
memory.py        # replay memory
statTracker.py   # training/validation statistics tracking
utils.py         # utility functions
configs/         # configuration files
data/            # sample data and dataset instructions


```

---

## Installation

```bash
git clone <repo_url>
cd personalized-hri-sim2real
pip install -r requirements.txt
```

---

## Dataset

The dataset consists of **253 egocentric recordings of human activities**, including multimodal data such as video, gaze, IMU, and audio, together with annotated action sequences.

It is organized as a collection of interaction episodes, where each episode corresponds to a single recorded activity sequence.

The dataset will be publicly released upon publication.


### Structure

```text id="u6i8qv"
data/
├── episode_1/
│   ├── frame_0000
│   ├── frame_0001
│   ├── ...
│   └── labels_updated.pkl
├── episode_2/
│   └── ...
```

### Description

* Each episode consists of a sequence of time steps (`frame_XXXX`)
* Each frame represents a **processed representation of the scene**, not raw images
* These representations are obtained from perception modules (e.g., object detection and other feature extractors)
* The `labels_updated.pkl` file contains the action annotations associated with each sequence

### Notes

* Frames are expected to be temporally ordered
* Each frame encodes the state information used by the learning algorithm
* The dataset does not contain raw video, but precomputed features
---
## Usage

### 1. Simulation Pretraining

Train the RL agent in the simulation environment:

```bash
python train.py --config 1
```

### 2. Evaluation

```bash
python test.py --config 1
```

---

## Key Contributions

* An egocentric replay-based simulation environment for realistic and scalable HRI pretraining
* A multimodal reinforcement learning framework for anticipatory and personalized robot assistance
* A sim-to-real training pipeline combining offline pretraining and real-world adaptation
* Empirical validation with both objective performance metrics and real-user studies

---

## Citation

```bibtex
@article{
}
```



