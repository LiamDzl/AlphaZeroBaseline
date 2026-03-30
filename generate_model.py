import torch
from neural_network import policy

parameters = policy(trunk_structure=[252, 168, 84],
                    policy_structure=[84, 84, 7],
                    value_structure=[84, 84, 1],
                    name="random")

torch.save(parameters, f"Agents/random.pt")