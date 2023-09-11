# Honours Thesis: Discrete Diffusion Q-Learning (D2QL)

**Author:** Damien Hood  
**Supervisor:** Junfeng Wen  
Carleton University  

## Table of Contents
- [Introducing D2QL: Discrete Diffusion Q-Learning](#introducing-d2ql-discrete-diffusion-q-learning)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
    - [Background](#background)
    - [Scope](#scope)
  - [Objectives](#objectives)
  - [Repository Folder Structure](#repository-folder-structure)
  - [Setup](#setup)
  - [Usage](#usage)
  - [License](#license)

---

## Introduction

### Background
Reinforcement Learning (RL) and Generative Models are two seminal fields in Artificial Intelligence (AI). RL focuses on agent-environment interactions to maximize a given reward, whereas Generative Models, especially diffusion models, can generate new data resembling the training data. The intersection  of these two fields is ripe for exploration. Specifically, while the Diffusion-QL algorithm exists for RL in continuous action spaces, no known models efficiently adapt it for discrete action spaces to the best of our knowledge.

**Note:** This work is conducted as part of an Honours Thesis at Carleton University.

### Scope
This thesis aims to fill this gap by developing a variant of Diffusion-QL algorithm for discrete action spaces, termed Discrete Diffusion Q-Learning (D2QL). We evaluate its performance against traditional RL algorithms in various discrete environments.

## Objectives

1. **Develop D2QL**: Adapt the Diffusion-QL algorithm for discrete action spaces by modifying the underlying diffusion model and loss function.
2. **Test D2QL**: Validate the algorithm in discrete environments starting with a gridworld benchmark.
3. **Evaluate Performance**: Conduct experiments to evaluate and compare the D2QL algorithm with existing RL algorithms in these environments.

## Repository Folder Structure

- **`d2ql/`**: 
  - Contains all materials related to the Discrete Diffusion Q Learning algorithm.
  - Includes source code, experiments, and utility scripts.
  
- **`literature-review/`**: 
  - Contains papers, summaries, and any related work that informs the research.

- **`baseline-algorithms/`**: 
  - Stores implementations of baseline algorithms for comparative analysis.

- **`environments/`**: 
  - Contains custom and/or modified reinforcement learning environments used in the research.

- **`data/`**: 
  - A place for experimental data, both raw and processed.

- **`results/`**: 
  - Holds generated results, figures, and tables from experiments.

- **`documentation/`**: 
  - Comprehensive documentation directory.
  - Includes details about the code and drafts of the written thesis.

- **`presentation/`**: 
  - Stores slides, scripts, or any related materials for presentations of the thesis.

- **`assets/`**: 
  - For any images, diagrams, or non-code assets referenced in documentation or thesis.

- **`misc/`**: 
  - A catch-all folder for supplementary materials.
  - Contains meeting notes, research plans, and more.

## Setup

Setup instructions will go here

## Usage

Code examples, API details, etc will be go here

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
