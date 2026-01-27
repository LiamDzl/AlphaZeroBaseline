import torch

from neural_network import policy
from connect_4 import mask, graphic, winner, Grid
from tree_module import PUCT, Node, MCTS
from functions import colour, generate_57

games_played = 0

# Hyperparameters
mcts_depth = 300
exploration_constant = 0.85
loss_value_constant = 1
loss_dist_constant = 1
epochs = 5
nabla = 0.00075
noise = 0

#### Networks Competing ### 

### Agent 1, Always Moves First ###
torch.serialization.add_safe_globals([policy])
agent_1 = torch.load("random_parameters.pt", weights_only=False)
tree_1 = MCTS(model=agent_1, iterations=50)
agent_1_name = "random_parameters.pt"

### Agent 2, Always Moves Second ###
agent_2 = torch.load("awesome_parameters.pt", weights_only=False)
tree_2 = MCTS(model=agent_2, iterations=50)
agent_2_name = "awesome_parameters.pt"

##### ##### ##### ##### ##### ##### ##### ##### ##### #####

end_filter = torch.tensor([False, False, False, False, False, False, False])
filter = torch.tensor([True, True, True, True, True, True, True])
starting_positions = generate_57()

agent_1_wins = 0
agent_2_wins = 0

##### ##### ##### ##### ##### ##### ##### ##### ##### #####
# First Game - Agent 1 Starts as Red

game_state = starting_positions[0]
player = 1

print(f"\n# # # # # # GAME 0 # # # # # #\n")
print(f"### Stats: {agent_1_name} Wins: {agent_1_wins}. {agent_2_name} Wins: {agent_2_wins}.\n")

while winner(game_state) == 0 and not torch.equal(filter, end_filter):

    # Load Correct Network to Move
    # Agent 1
    if player == 1:
        network = agent_1
        tree = tree_1
        to_move = agent_1_name

    # Agent 2
    elif player == -1:
        network = agent_2
        tree = tree_2
        to_move = agent_2_name

    print(f"\n### {to_move}'s Move ###\n")
    graphic(game_state)
    print("\n")

    mcts_distribution = tree.run(state=game_state,
                                 exploration_constant=exploration_constant,
                                 display=False)
    
    
    print(f"### {to_move}'s Decision Tensor: {mcts_distribution}\n")

    print("---------------------------------------------------------------------------------------------")

    chosen_move = torch.argmax(mcts_distribution, dim=1)
    grid = Grid(state=game_state)
    grid.action(chosen_move)

    # New state
    game_state = grid.state
    player = player * -1

    # As to end when board full
    filter = mask(game_state)

if winner(game_state) == 0:
    print(f"\n### Final Drawn Position ###\n")
    z = 0

if winner(game_state) == 1:
    print(f"\n### Final Position: Red Win ###\n")
    z = 1
    agent_1_wins += 1

if winner(game_state) == -1:
    print(f"\n### Final Position: Yellow Win ###\n")
    z = -1
    agent_2_wins += 1

graphic(game_state)
games_played += 1
print("\n")

##### ##### ##### ##### ##### ##### ##### ##### ##### #####
# Next 7 Games - Agent 1 Starts as Yellow

for i in starting_positions[1:8]:

    end_filter = torch.tensor([False, False, False, False, False, False, False])
    filter = torch.tensor([True, True, True, True, True, True, True])

    print(f"\n# # # # # # GAME {games_played} # # # # # #\n")
    print(f"### Stats: {agent_1_name} Wins: {agent_1_wins}. {agent_2_name} Wins: {agent_2_wins}.\n")
    game_state = i
    player = -1

    while winner(game_state) == 0 and not torch.equal(filter, end_filter):

        # Load Correct Network to Move
        # Agent 1
        if player == -1:
            network = agent_1
            tree = tree_1
            to_move = agent_1_name

        # Agent 2
        elif player == 1:
            network = agent_2
            tree = tree_2
            to_move = agent_2_name

        print(f"\n### {to_move}'s Move ###\n")
        graphic(game_state)
        print("\n")

        mcts_distribution = tree.run(state=game_state,
                                 exploration_constant=exploration_constant,
                                 display=False)
        
        
        print(f"### {to_move}'s Decision Tensor: {mcts_distribution}\n")

        print("---------------------------------------------------------------------------------------------")

        chosen_move = torch.argmax(mcts_distribution, dim=1)
        grid = Grid(state=game_state)
        grid.action(chosen_move)

        # New state
        game_state = grid.state
        player = player * -1

        # As to end when board full
        filter = mask(game_state)

    if winner(game_state) == 0:
        print(f"\n### Final Drawn Position ###\n")
        z = 0

    if winner(game_state) == 1:
        print(f"\n### Final Position: Red Win ###\n")
        z = 1
        agent_2_wins += 1

    if winner(game_state) == -1:
        print(f"\n### Final Position: Yellow Win ###\n")
        z = -1
        agent_1_wins += 1

    graphic(game_state)
    games_played += 1
    print("\n")

##### ##### ##### ##### ##### ##### ##### ##### ##### #####
# Next 49 Games - Agent 1 Starts as Red

for i in starting_positions[8:]:

    end_filter = torch.tensor([False, False, False, False, False, False, False])
    filter = torch.tensor([True, True, True, True, True, True, True]) 

    print(f"\n# # # # # # GAME {games_played} # # # # # #\n")
    print(f"### Stats: {agent_1_name} Wins: {agent_1_wins}. {agent_2_name} Wins: {agent_2_wins}.\n")
    game_state = i
    player = 1

    while winner(game_state) == 0 and not torch.equal(filter, end_filter):

        # Load Correct Network to Move
        # Agent 1
        if player == 1:
            network = agent_1
            tree = tree_1
            to_move = agent_1_name

        # Agent 2
        elif player == -1:
            network = agent_2
            tree = tree_2
            to_move = agent_2_name

        print(f"\n### {to_move}'s Move ###\n")
        graphic(game_state)
        print("\n")

        mcts_distribution = tree.run(state=game_state,
                                 exploration_constant=exploration_constant,
                                 display=False)
        
        
        print(f"### {to_move}'s Decision Tensor: {mcts_distribution}\n")

        print("---------------------------------------------------------------------------------------------")

        chosen_move = torch.argmax(mcts_distribution, dim=1)
        grid = Grid(state=game_state)
        grid.action(chosen_move)

        # New state
        game_state = grid.state
        player = player * -1

        # As to end when board full
        filter = mask(game_state)

    if winner(game_state) == 0:
        print(f"\n### Final Drawn Position ###\n")
        z = 0

    if winner(game_state) == 1:
        print(f"\n### Final Position: Red Win ###\n")
        z = 1
        agent_1_wins += 1

    if winner(game_state) == -1:
        print(f"\n### Final Position: Yellow Win ###\n")
        z = -1
        agent_2_wins += 1

    graphic(game_state)
    games_played += 1
    print("\n")


##### ##### ##### ##### ##### ##### ##### ##### ##### #####
# Now Switch Players
##### ##### ##### ##### ##### ##### ##### ##### ##### #####

# First Game - Agent 2 Starts as Red
game_state = starting_positions[0]
player = 1

end_filter = torch.tensor([False, False, False, False, False, False, False])
filter = torch.tensor([True, True, True, True, True, True, True])

print(f"\n# # # # # # GAME 57 # # # # # #\n")
print(f"### Stats: {agent_1_name} Wins: {agent_1_wins}. {agent_2_name} Wins: {agent_2_wins}.\n")

while winner(game_state) == 0 and not torch.equal(filter, end_filter):

    # Load Correct Network to Move
    # Agent 2
    if player == 1:
        network = agent_2
        tree = tree_2
        to_move = agent_2_name

    # Agent 1
    elif player == -1:
        network = agent_1
        tree = tree_1
        to_move = agent_1_name

    print(f"\n### {to_move}'s Move ###\n")
    graphic(game_state)
    print("\n")

    mcts_distribution = tree.run(state=game_state,
                                 exploration_constant=exploration_constant,
                                 display=False)
    
    print(f"### {to_move}'s Decision Tensor: {mcts_distribution}\n")

    print("---------------------------------------------------------------------------------------------")

    chosen_move = torch.argmax(mcts_distribution, dim=1)
    grid = Grid(state=game_state)
    grid.action(chosen_move)

    # New state
    game_state = grid.state
    player = player * -1

    # As to end when board full
    filter = mask(game_state)

if winner(game_state) == 0:
    print(f"\n### Final Drawn Position ###\n")
    z = 0

if winner(game_state) == 1:
    print(f"\n### Final Position: Red Win ###\n")
    z = 1
    agent_2_wins += 1

if winner(game_state) == -1:
    print(f"\n### Final Position: Yellow Win ###\n")
    z = -1
    agent_1_wins += 1

graphic(game_state)
games_played += 1
print("\n")

##### ##### ##### ##### ##### ##### ##### ##### ##### #####
# Next 7 Games - Agent 2 Starts as Yellow

for i in starting_positions[1:8]:

    end_filter = torch.tensor([False, False, False, False, False, False, False])
    filter = torch.tensor([True, True, True, True, True, True, True])

    print(f"\n# # # # # # GAME {games_played} # # # # # #\n")
    print(f"### Stats: {agent_1_name} Wins: {agent_1_wins}. {agent_2_name} Wins: {agent_2_wins}.\n")
    game_state = i
    player = -1

    while winner(game_state) == 0 and not torch.equal(filter, end_filter):

        # Load Correct Network to Move
        # Agent 2
        if player == -1:
            network = agent_2
            tree = tree_2
            to_move = agent_2_name

        # Agent 1
        elif player == 1:
            network = agent_1
            tree = tree_1
            to_move = agent_1_name

        print(f"\n### {to_move}'s Move ###\n")
        graphic(game_state)
        print("\n")

        mcts_distribution = tree.run(state=game_state,
                                 exploration_constant=exploration_constant,
                                 display=False)
        
        print(f"### {to_move}'s Decision Tensor: {mcts_distribution}\n")

        print("---------------------------------------------------------------------------------------------")

        chosen_move = torch.argmax(mcts_distribution, dim=1)
        grid = Grid(state=game_state)
        grid.action(chosen_move)

        # New state
        game_state = grid.state
        player = player * -1

        # As to end when board full
        filter = mask(game_state)

    if winner(game_state) == 0:
        print(f"\n### Final Drawn Position ###\n")
        z = 0

    if winner(game_state) == 1:
        print(f"\n### Final Position: Red Win ###\n")
        z = 1
        agent_1_wins += 1

    if winner(game_state) == -1:
        print(f"\n### Final Position: Yellow Win ###\n")
        z = -1
        agent_2_wins += 1

    graphic(game_state)
    games_played += 1
    print("\n")

##### ##### ##### ##### ##### ##### ##### ##### ##### #####
# Next 49 Games - Agent 2 Starts as Red

for i in starting_positions[8:]:

    end_filter = torch.tensor([False, False, False, False, False, False, False])
    filter = torch.tensor([True, True, True, True, True, True, True])

    print(f"\n# # # # # # GAME {games_played} # # # # # #\n")
    print(f"### Stats: {agent_1_name} Wins: {agent_1_wins}. {agent_2_name} Wins: {agent_2_wins}.\n")
    game_state = i
    player = 1

    while winner(game_state) == 0 and not torch.equal(filter, end_filter):

        # Load Correct Network to Move
        # Agent 2
        if player == 2:
            network = agent_2
            tree = tree_2
            to_move = agent_2_name

        # Agent 1
        elif player == -1:
            network = agent_1
            tree = tree_1
            to_move = agent_1_name

        print(f"\n### {to_move}'s Move ###\n")
        graphic(game_state)
        print("\n")

        mcts_distribution = tree.run(state=game_state,
                                 exploration_constant=exploration_constant,
                                 display=False)
        
        
        print(f"### {to_move}'s Decision Tensor: {mcts_distribution}\n")

        print("---------------------------------------------------------------------------------------------")

        chosen_move = torch.argmax(mcts_distribution, dim=1)
        grid = Grid(state=game_state)
        grid.action(chosen_move)

        # New state
        game_state = grid.state
        player = player * -1

        # As to end when board full
        filter = mask(game_state)

    if winner(game_state) == 0:
        print(f"\n### Final Drawn Position ###\n")
        z = 0

    if winner(game_state) == 1:
        print(f"\n### Final Position: Red Win ###\n")
        z = 1
        agent_2_wins += 1

    if winner(game_state) == -1:
        print(f"\n### Final Position: Yellow Win ###\n")
        z = -1
        agent_1_wins += 1

    graphic(game_state)
    games_played += 1
    print("\n")


##### ##### ##### ##### ##### ##### ##### ##### ##### #####
# Final Stats
##### ##### ##### ##### ##### ##### ##### ##### ##### #####

print(f"\n### Final Stats: {agent_1_name} Wins: {agent_1_wins}. {agent_2_name}: {agent_2_wins}. Draws: {114 - agent_1_wins - agent_2_wins} ###\n")



