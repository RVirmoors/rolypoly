import torch
from typing import List, Tuple

@torch.jit.script
class Adam(object):
    def __init__(self, params: List[torch.Tensor], lr: float = 1e-3, betas: Tuple[float, float] = (0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0):
        self.lr: float = lr
        self.betas: Tuple[float, float] = betas
        self.eps: float = eps
        self.weight_decay: float = weight_decay
        self.steps: int = 0
        self.m: List[torch.Tensor] = [torch.zeros_like(p) for p in params]
        self.v: List[torch.Tensor] = self.m.copy()

    def zero_grad(self):
        for m, v in zip(self.m, self.v):
            m.zero_()
            v.zero_()

    def step(self, params: List[torch.Tensor], grads: List[torch.Tensor]):
        self.steps += 1
        for i, (p, g) in enumerate(zip(params, grads)):
            if g is None:
                continue
            if self.weight_decay != 0:
                g = g + self.weight_decay * p
            self.m[i] = self.betas[0] * self.m[i] + (1 - self.betas[0]) * g
            self.v[i] = self.betas[1] * self.v[i] + (1 - self.betas[1]) * (g ** 2)
            m_hat = self.m[i] / (1 - self.betas[0] ** self.steps)
            v_hat = self.v[i] / (1 - self.betas[1] ** self.steps)
            new_p = p.detach().clone()
            new_p -= self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)
            p.data.copy_(new_p)


# Define a simple network
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(10, 5)
        self.fc2 = torch.nn.Linear(5, 2)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

@torch.jit.export
def printParams(net: torch.nn.Module):
    print(net.parameters())

net = Net()
printParams(net)

optimizer = Adam(net.parameters())

# Define the loss function
loss_fn = torch.nn.CrossEntropyLoss()

# Generate some random data
x = torch.randn(4, 10)
y = torch.tensor([0, 1, 0, 1])

scripted_net = torch.jit.script(net)
print(type(scripted_net))
printParams(scripted_net)
# printParams(net)

for epoch in range(1000):
    # Forward pass
    y_pred = net(x)
    loss = loss_fn(y_pred, y)
    if epoch % 30 == 0:
        print("Epoch", epoch, "loss", loss.item())

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step(net.parameters(), [p.grad for p in net.parameters()])


# Check the updated weights
for name, param in net.named_parameters():
    print(name, param)
