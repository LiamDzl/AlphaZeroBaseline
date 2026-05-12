import torch
from neural_network import policy

parameters = policy(trunk_structure=[504],
                    policy_structure=[504, 252, 126, 7],
                    value_structure=[504, 252, 126, 1],
                    name="random")

torch.save(parameters, f"random.pt")