# personalized-hri-sim2real
Official implementation of "Reducing the Sim2Real Gap in Personalized HRI: Simulation-Based Pretraining and Real-World Adaptation and Validation", to appear in ACM THRI.
# Personalized HRI Sim2Real

---

## Overview

This repository presents a framework for **personalized and anticipatory human-robot interaction (HRI)** using reinforcement learning.

The proposed approach reduces the sim-to-real gap by combining:

* simulation-based pretraining using egocentric human data
* multimodal perception (gaze, speech, egocentric vision)
* real-world fine-tuning for user-specific adaptation

The system enables robots to anticipate user needs and provide proactive assistance in collaborative tasks.

As described in the paper, pretrained agents:

* reduce irrelevant actions by ~50%
* improve responsiveness
* achieve higher user satisfaction in real-world experiments 

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

Policies are adapted online during interaction with real users:

* lightweight fine-tuning
* structured exploration strategy
* user-specific behavior adaptation

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
sim/            # simulation environment (egocentric replay)
rl/             # reinforcement learning models (RDM)
perception/     # AOD, HAA, ASR modules
real/           # real-world interaction and adaptation
configs/        # experiment configurations
scripts/        # training and evaluation scripts
data/           # dataset instructions (not included)
results/        # logs, metrics, and outputs
```

---

## Installation

```bash
git clone <repo_url>
cd personalized-hri-sim2real
pip install -r requirements.txt
```

---

## Usage

### 1. Simulation Pretraining

Train the RL agent in the simulation environment:

```bash
python scripts/train_sim.py --config configs/sim.yaml
```

### 2. Real-World Fine-Tuning

Adapt the pretrained model to a specific user:

```bash
python scripts/adapt_real.py --config configs/real.yaml
```

### 3. Evaluation

```bash
python scripts/eval.py --config configs/eval.yaml
```

---

## Dataset

The simulation is based on:

* 253 egocentric videos of human activity
* multimodal recordings (video, gaze, IMU, audio)
* annotated action sequences

The dataset will be released upon publication.

---

## Key Contributions

* Egocentric simulation for realistic HRI pretraining
* RL framework for anticipatory and personalized assistance
* Sim2Real transfer with real-world validation
* Hybrid evaluation combining objective metrics and user studies

---

## Citation

```bibtex
@article{
}
```



