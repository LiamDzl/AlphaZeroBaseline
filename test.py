import torch
from connect_4 import graphic
from functions import softmax_temp, expand_to_84
from neural_network import policy

torch.serialization.add_safe_globals([policy])
policy_network = torch.load("parameters.pt", weights_only=False)

x = torch.randint(-1, 2, (42,))
graphic(x.reshape(6,7))
x = expand_to_84(x)
x = x.float()
print(policy_network.forward(x))