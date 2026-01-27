import torch
from neural_network import policy

parameters = policy(trunk_structure=[84, 168, 168, 168],
                    policy_structure=[168, 84, 7],
                    value_structure=[168, 84, 1])

torch.save(parameters, "conv_parameters.pt")
