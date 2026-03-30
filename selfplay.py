import torch
from neural_network import policy
from functions import generate_random
from selfplay_functions import parallel_selfplay
from connect_4 import graphic

torch.serialization.add_safe_globals([policy])
policy_network = torch.load("trained.pt", weights_only=False)
policy_network.name = "model"

# X games
cpu_cores = 8
game_sets = 10

# Hyperparameters
mcts_depth = 250
exploration_constant = 3.5
epsilon = 0.25
gamma = 0.96
epochs = 10
nabla = 0.00002
noise = 0

if __name__ == "__main__":
    for game_set in range(game_sets):

        X, Y = parallel_selfplay(game_set=game_set,
                                game_sets=game_sets,
                                policy_network=policy_network,
                                mcts_depth=mcts_depth,
                                exploration_constant=exploration_constant,
                                epsilon=epsilon,
                                gamma = gamma,
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

        torch.save(policy_network, f"trained.pt")

# --------------+
