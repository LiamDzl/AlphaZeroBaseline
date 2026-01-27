import torch
from connect_4 import graphic
from neural_network import policy

torch.serialization.add_safe_globals([policy])
candidate = torch.load("parameters.pt", weights_only=False)
exploration_constant = 0.45
noise = 0

state = torch.tensor([[0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, -1, 0, 0, 0],
                      [-1, 0, 1, 1, 1, 0, -1]])

candidate.evaluate(name="parameters.pt", state=state, exploration_constant=exploration_constant, noise=noise)

state = torch.tensor([[0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0, 0],
                      [1, 0, -1, -1, -1, 0, 1]])

candidate.evaluate(name="parameters.pt", state=state, exploration_constant=exploration_constant, noise=noise)