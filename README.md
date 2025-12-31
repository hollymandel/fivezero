# FiveZero

AlphaZero-style self-play training for 5x5 tic tac toe (four to win). During game play, a large number of MCTS
rollouts guided by the current policy network are used to select each step. During training, the policy network
is updated in the direction of the empirical distribution from MCTS, while the value network is updated to predict the game outcome. In this implementation the value network is not used during MCTS (for value backpropagation) because all rollouts are taken to the end of the game. 

Training occurs in `fivezero/train/TrainConvNet.py`. An example trained checkpoint (`example_trained.pth`) is included. This repo is just for education about AlphaZero and has not been optimized for performance. The example model is undertrained (but pretty decent). 

Some observations:
- It is critical to detach the MCTS tree every time you take a step during game play. If you allow
  later game steps to backprop all the way to the root, they will concentrate the visit distribution 
  at the step that happened to be chosen, which is not an appropriate target for the training of the policy network.
- The amount of MCTS rollouts per move apparently needs to be high. For an untrained policy network 2-500   
  rollouts were needed to consistently make the right choice in the obvious set up `test/test_step.py`.
- A non-parallelized implementation of the game play step does not run faster when naively dumped onto a GPU! 

## Installation
- Requirements: Python 3.9+ and a PyTorch-capable environment.
- (Recommended) create a virtualenv: `python -m venv .venv && source .venv/bin/activate`
- Install requirements: `pip install -r requirements.txt`
- Install package: `pip install -e .`

## Play the example model
- From the repo root: `python -m fivezero.cli.play --model-path example_trained.pth --human x`
- Use `--human o` to let the network open
- Use `--verbose True` to inspect policy logits and values. 
