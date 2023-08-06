# Flappy Bird AI using NEAT

This project implements an Artificial Intelligence (AI) that plays the popular game "Flappy Bird" using the NeuroEvolution of Augmenting Topologies (NEAT) algorithm. NEAT is a genetic algorithm that trains the AI by creating generations of agents and simulating them in the game environment. Each agent (bird) is equipped with a Neural Network that decides when to make the bird flap based on its inputs. The birds with the best Neural Networks are selected for the next generation, and slight mutations are applied to create new behaviors.

## How it Works

1. **Neural Network Architecture**: Each bird is controlled by a Neural Network with a set of input, hidden, and output nodes. The input nodes receive information about the bird's position, the distance to the upcoming obstacles, and other relevant game data. The output node determines whether the bird should flap or not.

2. **Genetic Algorithm**: The AI starts with an initial population of birds, each with its unique set of Neural Network weights. These birds are then simulated in the Flappy Bird game environment. The AI evaluates each bird's performance based on its ability to survive and progress in the game.

3. **Selection and Reproduction**: The birds with the best performance (highest scores) are selected to form the basis for the next generation. They undergo crossover and slight mutations to create new birds with different Neural Network architectures. This process promotes the evolution of better-performing birds over successive generations.

4. **Training Loop**: The AI continues to create new generations of birds, with each generation gradually improving in their ability to play the game. The process is repeated until the AI achieves a satisfactory level of performance.

## Getting Started

### Prerequisites

- Python 3.x
- Pygame library
- NEAT library (Python implementation of the NEAT algorithm)

### Installation

1. Clone the repository to your local machine.

```bash
git clone https://github.com/your-username/flappy-bird-ai.git
```

2. Install the required libraries using pip.

```bash
pip install pygame neat-python
```

### Usage

1. Run the `flappy_bird_ai.py` script to start the AI learning process and see the birds play the game.

```bash
python flappy_bird_ai.py
```

2. Observe the AI as it evolves over generations to improve its performance in the game.

3. To modify the parameters of the NEAT algorithm or the game environment, check the configuration files (`config-feedforward.txt` for NEAT parameters and `flappy_bird_config.py` for game-related settings).

