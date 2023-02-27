# official example
import torch
import torch.nn as nn
loss = nn.CrossEntropyLoss()
input = torch.randn(3, 5, requires_grad=True)
target = torch.empty(3, dtype=torch.long).random_(5)

print(input)
print(target)
# if you will replace the dtype=torch.float, you will get error

output = loss(input, target)
output.backward()