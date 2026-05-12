import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tree_module import MCTS
from functions import alphazero_display, expand_to_84
from connect_4 import mask, graphic

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
class policy(nn.Module):
    def __init__(self, name, trunk_structure, policy_structure, value_structure):
        super().__init__()

        self.name = name

        # Conv. Layers
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=12, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, padding=1)

        trunk_layers = []
        policy_layers = []
        value_layers = []

        # Main Trunk
        for i in range(len(trunk_structure) - 1):
            trunk_layers.append(nn.Linear(trunk_structure[i], trunk_structure[i + 1]))

        self.trunk_structure = trunk_structure
        self.trunk_layers = nn.ModuleList(trunk_layers)

        # Policy Arm
        for i in range(len(policy_structure) - 1):
            policy_layers.append(nn.Linear(policy_structure[i], policy_structure[i + 1]))

        self.policy_structure = policy_structure
        self.policy_layers = nn.ModuleList(policy_layers)

        # Value Arm
        for i in range(len(value_structure) - 1):
            value_layers.append(nn.Linear(value_structure[i], value_structure[i + 1]))

        self.value_structure = value_structure
        self.value_layers = nn.ModuleList(value_layers)


    def forward(self, x):
        # Save for Masking
        state_me = x[:42].reshape(6,7) 
        state_them = x[42:].reshape(6,7)
    
        # Forward Conv.
        x = x.reshape(1,2,6,7)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = x.reshape(504)
  
        # Forward Main
        for i in self.trunk_layers:
            x = F.relu(i(x)) 

        x_policy = x.clone()
        x_value = x.clone()

        # Forward Policy
        for i in self.policy_layers[:-1]:
            x_policy = F.relu(i(x_policy)) 

        latent_actions = x_policy.clone()
        x_policy = self.policy_layers[-1](x_policy)

        # Forward Value
        for i in self.value_layers[:-1]:
            x_value = F.relu(i(x_value))

        latent_value = x_value.clone()
        x_value = self.value_layers[-1](x_value)
        x_value = 2 * torch.sigmoid(x_value) - 1 # Force Value in [-1, 1]

        # Mask Full Columns
        filter_me = mask(state_me)
        filter_them = mask(state_them)
        filter = (filter_me & filter_them)

        to_softmax = x_policy[filter]
        reduced_distribution = F.softmax(to_softmax, dim=0)
        distribution = torch.zeros(7)
        distribution[filter] = reduced_distribution
        output = torch.cat([distribution, x_value.unsqueeze(0)[0]], dim=0)
        output = output.reshape(8)

        return output
    
    def train(self, training_inputs, training_outputs, epochs, nabla, batch_size=32):
        training_size = training_inputs.shape[0]
        mse = nn.MSELoss()
        optimiser = optim.Adam(self.parameters(), lr=nabla)

        for epoch in range(epochs):
            permutation = torch.randperm(training_size)
            training_inputs = training_inputs[permutation]
            training_outputs = training_outputs[permutation]

            for i in range(0, training_size, batch_size):
                batch_inputs = training_inputs[i: i + batch_size]
                batch_outputs = training_outputs[i: i + batch_size]

                optimiser.zero_grad()
                loss_scalar = 0

                for x, y in zip(batch_inputs, batch_outputs):
                    nn_output = self.forward(x)

                    value_loss = mse(nn_output[7], y[7])
                    policy_loss = -torch.dot(y[0:7], torch.log(nn_output[0:7] + 1e-8))

                    L2 = 0
                    
                    for param in self.parameters():
                        L2 += torch.sum(param ** 2)

                    loss_scalar += value_loss + policy_loss + 1e-4 * L2

                loss_scalar.backward()
                optimiser.step()

        return "Successfully Trained."
    
    def evaluate(self, name, state, exploration_constant, noise):
        print("\n")
        graphic(state)
        print("\n")
        state = state.reshape(42)
        state_84 = expand_to_84(state)
        state_84 = state_84.float()
        print(state_84)
        output = self.forward(state_84)
        neural_distribution = output[:7]
        value = output[7]

        tree_search = MCTS(model=self, iterations=250)
        state = state.reshape(6, 7)
        dist = tree_search.run(state=state,
                               exploration_constant=exploration_constant,
                               epsilon=0,
                               display=False)
                               
        root_node = tree_search.explored_nodes[0]
        root_value = root_node.value_sum / root_node.visit_count

        alphazero_display(policy_name=name,
                          state=state,
                          tree_dist=dist,
                          tree_value=root_value,
                          neural_dist=neural_distribution,
                          neural_value=value,
                          agent_move=None,
                          noise=noise)