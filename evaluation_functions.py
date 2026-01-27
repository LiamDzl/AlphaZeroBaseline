import torch
from tree_module import MCTS
from connect_4 import graphic, Grid, compute_player, winner, mask
from functions import generate_400

def one_v_one(initial_state, model_1, model_2, mcts_depth, exploration_constant):
    tree_1 =  MCTS(model=model_1, iterations=mcts_depth)
    tree_2 =  MCTS(model=model_2, iterations=mcts_depth)

    model_1_colour = compute_player(initial_state)
    model_2_colour = -1 * model_1_colour 

    if model_1_colour == 1:
        model_1_emoji = "🔴"
        model_2_emoji = "🟡" 
    else:
        model_1_emoji = "🟡"
        model_2_emoji = "🔴"

    # Endgame Conditions (Full Board)
    end_filter = torch.tensor([False, False, False, False, False, False, False])
    filter = torch.tensor([True, True, True, True, True, True, True])

    player = 1
    game_state = initial_state

    while winner(game_state) == 0 and not torch.equal(filter, end_filter):

        if player == 1:
            network = model_1
            tree = tree_1
            to_move = "Agent 1"
            emoji = model_1_emoji

        else:
            network = model_2
            tree = tree_2
            to_move = "Agent 2"
            emoji = model_2_emoji

        print(f"{emoji} {to_move}'s input:\n")
        graphic(game_state)
        print("\n")

        mcts_distribution = tree.run(state=game_state,
                                    exploration_constant=exploration_constant,
                                    epsilon=0,
                                    display=False)
        
        chosen_move = torch.argmax(mcts_distribution, dim=1)
        grid = Grid(state=game_state)
        grid.action(chosen_move)

        # New State
        game_state = grid.state
        player = player * -1

        # Check if Board Full
        filter = mask(game_state)

    print("")
    graphic(game_state)
    print("")

    if winner(game_state) != 0:
        victor = -1 * compute_player(game_state) # Winning Colour
        if victor == model_1_colour:
            return 1
        else:
            return -1
    
    else:
        return 0



def matchoff(model_1, model_2, hyperparameters):
    mcts_depth, exploration_constant, epsilon, gamma, epochs, nabla, noise = hyperparameters
    win_rate = 0

    return win_rate