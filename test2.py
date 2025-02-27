import torch
from torch.distributions import Categorical

a = torch.tensor([1., 2., 3.,])
cat = Categorical(logits=a)

x = [0, 0, 0]
for _ in range(100000):
    sample = cat.sample().item()

    x[sample] += 1

print(x)
result1 = torch.nn.functional.softmax(torch.tensor(a).float(), dim=-1)
result2 = torch.nn.functional.softmax(torch.tensor(x).float(), dim=-1)

print(result1)
print(result2)