import torch
from neural_network import policy, Constant_Network
from functions import generate_400
from connect_4 import mask, graphic, winner, Grid
from tree_module import PUCT, Node, MCTS
from evaluation_functions import one_v_one

print("")
games_played = 0

# Hyperparameters
mcts_depth = 250
exploration_constant = 3.5

torch.serialization.add_safe_globals([policy])

# Agent 1 ----+
agent_1 = torch.load("parameters.pt", weights_only=False)

# Agent 2 ----+
agent_2 = torch.load("random_parameters.pt", weights_only=False)
#agent_2 = Constant_Network()

tally = [0, 0, 0]

for initial_state in generate_400():
    print(tally)
    print("")
    result = one_v_one(initial_state=initial_state,
                       model_1=agent_1,
                       model_2=agent_2,
                       mcts_depth=mcts_depth,
                       exploration_constant=exploration_constant)

    if result == 1:
        tally[0] += 1
    elif result == -1:
        tally[1] += 1
    else:
        tally[2] += 1
