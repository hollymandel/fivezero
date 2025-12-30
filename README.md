# FiveZero

AlphaZero-style self-play training for 5x5 tic tac toe (four to win). A convolutional policy/value network plus Monte Carlo Tree Search produces moves. Training occurs in `fivezero/train/TrainConvNet.py`. An example trained checkpoint (`example_trained.pth`) is included. 

This repo is just for education about AlphaZero and has not been optimized for performance. The example model is undertrained (but still decent). 

## Installation
- Requirements: Python 3.9+ and a PyTorch-capable environment.
- (Recommended) create a virtualenv: `python -m venv .venv && source .venv/bin/activate`
- Install requirements: `pip install -r requirements.txt`
- Install package: `pip install -e .`

## Play the example model
- From the repo root: `python -m fivezero.cli.play --model-path example_trained.pth --human x`
- Use `--human o` to let the network open
