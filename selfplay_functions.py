import torch
import random
from neural_network import policy
from connect_4 import mask, graphic, winner, Grid, compute_player
from tree_module import MCTS
from functions import colour, generate_400, softmax_temp, expand_to_84
from multiprocessing import Pool

def selfplay(policy_network, initial_state, mcts_depth, exploration_constant, epsilon, gamma, noise, game_set, game_sets, root_policy):
    X = [] # Recorded States (Inputs)
    Y = [] # Recorded Distribution/Value Outputs

    game_state = initial_state
    initial_player = compute_player(initial_state)
    player = compute_player(game_state)
    end_filter = torch.tensor([False, False, False, False, False, False, False])
    filter = torch.tensor([True, True, True, True, True, True, True])
    game_length = 0

    recorded_states = []
    recorded_mcts_dists = []

    while winner(game_state) == 0 and not torch.equal(filter, end_filter):
    
        print(f"\n# {colour(player)}'s Move\n")
        graphic(game_state)
        print("\n")

        tree = MCTS(model=policy_network, iterations=mcts_depth, root_policy=root_policy)
        mcts_distribution = tree.run(state=game_state,
                                          exploration_constant=exploration_constant,
                                          epsilon=epsilon,
                                          display=False)
        
        root = tree.explored_nodes[0]
                                          
        # Data collection
        x = game_state.reshape(42)
        x = expand_to_84(x)
        recorded_states.append(x)
        y = mcts_distribution.reshape(7)
        recorded_mcts_dists.append(y)

        # Choose Move
        choose_from = softmax_temp(mcts_distribution, temp=noise)[0]
        chosen_move = torch.multinomial(choose_from, num_samples=1)
        grid = Grid(state=game_state)
        grid.action(chosen_move)

        print(f"# Decision:")
        print(mcts_distribution)
        print("")
        print(f"# 🌿 Tree: {softmax_temp(mcts_distribution, temp=noise)}")
        print("")
        print(f"# 🕹️  Set: {game_set} / {game_sets}")

        # New state
        game_state = grid.state
        player *= -1
        game_length += 1

        # As to end when board full
        filter = mask(game_state)

    if winner(game_state) == 0:
        z = 0

    if winner(game_state) == 1:
        z = compute_player(game_state) * -1 # True Winner, +1 Red, -1 Yellow

    # Concatenate z values to distribution vectors - as to make vectors of length 8 from 7
    dist_value_outputs = []
    z = z * initial_player # if starting player and result match, then reward the initial state... (+1)
    for index, y in enumerate(recorded_mcts_dists):
        discounted = z * (gamma ** (game_length - index - 1)) # Discount Reward
        dist_value_outputs.append(torch.cat((y, torch.tensor([discounted])), dim=0))
        z *= -1

    graphic(game_state)
    print("\n")

    # Append game results into global dataset, X and Y
    for recorded_state in recorded_states:
        X.append(recorded_state)

    for dist_value_output in dist_value_outputs:
        Y.append(dist_value_output)

    return X, Y # Training Data and MCTS Values

# CPU Parallelisation

def worker(args):
    policy_network, initial_state, mcts_depth, exploration_constant, epsilon, gamma, noise, game_set, game_sets, root_policy = args
    X, Y = selfplay(policy_network=policy_network,
                        initial_state=initial_state,
                        mcts_depth=mcts_depth,
                        exploration_constant=exploration_constant,
                        epsilon=epsilon,
                        gamma=gamma,
                        noise=noise,
                        game_set=game_set,
                        game_sets=game_sets,
                        root_policy=root_policy)

    return X, Y

# Generate First 400 States in Connect 4
initial_states = generate_400()
    
def parallel_selfplay(game_set, game_sets, policy_network, mcts_depth, exploration_constant, epsilon, gamma, noise, cpu_cores, root_policy=None):
    random.shuffle(initial_states)
    args_list = [(policy_network, state, mcts_depth, exploration_constant, epsilon, gamma, noise, game_set, game_sets, root_policy) for state in initial_states[:cpu_cores]]

    with Pool(cpu_cores) as p:
        dataset = p.map(worker, args_list) # Store 8 Game Results, as pairs (X_1, Y_1, ... X_8, Y_8)

    # Compute Dataset Size:
    dataset_size = 0
    for X, Y in dataset:
        dataset_size += len(X)

    # Compute X, Y Tensors:
    X_total = torch.zeros(dataset_size, 84)
    Y_total = torch.zeros(dataset_size, 8)

    index = 0
    for X, Y in dataset:
        for input, output in zip(X, Y):
            X_total[index] = input
            Y_total[index] = output
            index += 1

    print(f"\Set Complete. New Data Points: {dataset_size}")
    return X_total, Y_total

def alphazero_selfplay(model_path, cpu_cores=10, game_sets=10, epochs=10, nabla=0.00002):
    torch.serialization.add_safe_globals([policy])
    policy_network = torch.load(model_path, weights_only=False)
    policy_network.name = "model"

    # Hyperparameters
    mcts_depth = 250
    exploration_constant = 3.5
    epsilon = 0.25
    gamma = 1
    epochs = 10
    nabla = 0.00002
    noise = 0

    for game_set in range(game_sets):
        X, Y = parallel_selfplay(game_set=game_set,
                                game_sets=game_sets,
                                policy_network=policy_network,
                                mcts_depth=mcts_depth,
                                exploration_constant=exploration_constant,
                                epsilon=epsilon,
                                gamma=gamma,
                                cpu_cores=cpu_cores,
                                noise=noise,
                                root_policy=None)

        X_original = X.clone()
        Y_original = Y.clone()

        # Augment Data - Use Symmetry of Connect 4
        X_mirror = X.clone()
        Y_mirror = Y.clone()

        for index, state in enumerate(X_original):
            shaped_me = state[:42].reshape(6,7)
            shaped_them = state[42:].reshape(6,7)
            shaped_mirror_me = torch.flip(shaped_me, dims=[1])
            first = shaped_mirror_me.reshape(42)
            shaped_mirror_them = torch.flip(shaped_them, dims=[1])
            second = shaped_mirror_them.reshape(42)
            mirror_state = torch.cat([first, second], dim=0)
            X_mirror[index] = mirror_state
            y = Y_original[index]
            y_dist = y[:7]
            y_value = y[7]
            y_mirror = torch.cat((torch.flip(y_dist, dims=[0]), torch.tensor([y_value])), dim=0)
            Y_mirror[index] = y_mirror

        X_symm = torch.cat((X_original, X_mirror), dim=0)
        Y_symm = torch.cat((Y_original, Y_mirror), dim=0)

        policy_network.train(training_inputs=X_symm,
                            training_outputs=Y_symm,
                            epochs=epochs,
                            nabla=nabla)

        torch.save(policy_network, model_path)

# -----------------+ Memory Forging

def forge_memories(memory_structure, policy_network, initial_state):
    map = memory_structure["map"]
    starting_id = len(memory_structure["memories"]) + 1

    # To Remember:
    encodings = []
    MCTS_dists = []
    MCTS_values = [] # History of Value Swings
    child_nodes = []

    game_state = initial_state
    initial_player = compute_player(initial_state)
    player = compute_player(game_state)
    end_filter = torch.tensor([False, False, False, False, False, False, False])
    filter = torch.tensor([True, True, True, True, True, True, True])
    game_length = 0

    print("")

    while winner(game_state) == 0 and not torch.equal(filter, end_filter):
    
        graphic(game_state)
        print("\n")

        tree = MCTS(model=policy_network, iterations=250, root_policy=None)
        mcts_distribution = tree.run(state=game_state,
                                          exploration_constant=3.5,
                                          epsilon=0.25,
                                          display=False)
        
        root = tree.explored_nodes[0]
        child_nodes.append(root.children)
        MCTS_values.append(-root.value())
                                          
        # Data collection
        x = game_state.reshape(42)
        x = expand_to_84(x)
        encodings.append(x)
        y = mcts_distribution.reshape(7)
        MCTS_dists.append(y)

        # Choose Move
        choose_from = softmax_temp(mcts_distribution, temp=0)[0]
        chosen_move = torch.multinomial(choose_from, num_samples=1)
        grid = Grid(state=game_state)
        grid.action(chosen_move)

        # New state
        game_state = grid.state
        player *= -1
        game_length += 1

        # As to end when board full
        filter = mask(game_state)

    memories = []

    # Form list of memories
    for index, _ in enumerate(encodings):

        children = child_nodes[index] # List
        for child_index, child in enumerate(children):
            if child is not None and child.state is not None:
                child_memory = {
                    "child_id": f"M{starting_id + index}C{child_index+1}",
                    "encoding" : expand_to_84(child.state).flatten().tolist(),
                    "key" : torch.mv(torch.tensor(map), expand_to_84(child.state).flatten()).tolist(),
                    "value" : -child.value(),
                    "visits" : child.visit_count
                }
            else:
                child_memory = None

            children[child_index] = child_memory

        memory = {
            "id": f"M{starting_id + index}",
            "encoding" : encodings[index].tolist(),
            "key" : torch.mv(torch.tensor(map), encodings[index]).tolist(),
            "value" : MCTS_values[index],
            "children" : children
        }

        memories.append(memory)

    return memories