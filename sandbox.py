import torch
from neural_network import policy, Constant_Network
from connect_4 import Grid, winner, graphic, compute_player
from tree_module import MCTS
from functions import alphazero_display, softmax_temp, expand_to_84

torch.serialization.add_safe_globals([policy])
policy_network = torch.load("Agents/trained.pt", weights_only=False)
policy_name = "trained.pt"

initial = torch.zeros(6,7)
environment = Grid(state=initial)

column = ""
move = 0
exploration_constant = 3.5
epsilon = 0
search_depth = 250
noise = 0

text = r"""     _    _       _           _____              
   / \  | |_ __ | |__   __ _|__  /___ _ __ ___  
  / _ \ | | '_ \| '_ \ / _` | / // _ \ '__/ _ \ 
 / ___ \| | |_) | | | | (_| |/ /|  __/ | | (_) |
/_/__ \_\_| .__/|_| |_|\__,_/____\___|_|  \___/ 
/ ___|  __|_|_ __   __| | |__   _____  __       
\___ \ / _` | '_ \ / _` | '_ \ / _ \ \/ /       
 ___) | (_| | | | | (_| | |_) | (_) >  <        
|____/ \__,_|_| |_|\__,_|_.__/ \___/_/\_\        """

print(text)
print("\n")

graphic(environment.state)
print("")

# Red always starts
environment.player = 1
emoji = "🔴"

while column != "end":
    proceed = False
    inrange = False
    
    while not proceed:
        column = input(f"{emoji} Select Column (1-7) : ")
        print(f"\n+ --------------- ( Move {move} )\n")

        if column == "end":
            break

        try:
            colNum = int(column)
        except:
            print("\n❌ Error: Not in Range\n")
            continue

        if 1 <= colNum <= 7:
            if environment.state[0, colNum - 1] != 0:
                print("\n❌ Error: Column Full!\n")
            else:
                proceed = True
        else:
            print("\n❌ Error: Not in Range\n")

    if column == "end":
        break

    environment.action(column=int(column)-1)
    move += 1

    graphic(environment.state)
    print("")

    # Check win
    if winner(environment.state) == 1:
        player = compute_player(environment.state)
        if player == 1:
            print("🟡🟡🟡🟡 Yellow Wins! 🟡🟡🟡🟡\n")
        else:
            print("🔴🔴🔴🔴 Red Wins! 🔴🔴🔴🔴\n")
        break

    tree_search = MCTS(model=policy_network,
                       iterations=search_depth,
                       root_policy=None)

    distribution = tree_search.run(state=environment.state,
                                   exploration_constant=exploration_constant,
                                   epsilon=epsilon,
                                   display=True)

    state = environment.state.reshape(42)
    output = policy_network.forward(expand_to_84(state))

    neural_distribution = output[:7]
    value = output[7]

    root_node = tree_search.explored_nodes[0]
    root_value = root_node.value_sum / root_node.visit_count

    choose_from = softmax_temp(distribution, temp=noise)[0]
    agent_move = torch.multinomial(choose_from, num_samples=1)

    alphazero_display(policy_name=policy_name,
                      state=environment.state,
                      tree_dist=distribution,
                      tree_value=root_value,
                      neural_dist=neural_distribution,
                      neural_value=value,
                      agent_move=agent_move,
                      noise=noise)

    emoji = "🔴" if environment.player == 1 else "🟡"