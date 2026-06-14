import torch
import torch.nn as nn
import torch.nn.functional as F

#Tensors
a = torch.tensor([1.0, 2.0, 3.0])      # straight from a Python list
b = torch.zeros(2, 3)                   # a 2x3 grid of zeros
c = torch.ones(2, 3)                    # a 2x3 grid of ones
d = torch.rand(2, 3)                    # 2x3 of random numbers in [0, 1)

print(a.shape)
print(a.dtype)
print(c*d)

#Concept of autograd
w = torch.tensor(4.0, requires_grad=True)   # a knob we want to tune
x = torch.tensor(3.0)                        # fixed input, no requires_grad

y = w * x                                    # PyTorch records: "y came from w times x"
y.backward()                                 # how does y change as w changes?

print(w.grad)    # tensor(3.)  -> because y = w*x, so dy/dw = x = 3
print(x.grad)    # None        -> x wasn't a knob, so nothing was tracked for it
class BiggerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)   # input layer  -> hidden 1
        self.fc2 = nn.Linear(256, 128)   # hidden 1     -> hidden 2
        self.fc3 = nn.Linear(128, 10)    # hidden 2     -> output (10 classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))          # layer 1, then "bend" it
        x = F.relu(self.fc2(x))          # layer 2, then "bend" it
        x = F.softmax(self.fc3(x))                  # final scores (no bend on the last one)
        return x

model = BiggerNet()
X = torch.rand(784)
print(model(X))


# --- the model (nn.Module): one knob-holding layer ---
model = nn.Linear(1, 1)

# --- the data: each y is exactly 2*x, the rule we want it to discover ---
X = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
Y = torch.tensor([[2.0], [4.0], [6.0], [8.0]])

# --- loss (how wrong?) + optimizer (how to fix?) ---
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# --- the training loop: the five-step heartbeat, repeated ---
for epoch in range(2000):
    optimizer.zero_grad()         # 1. clear last step's slopes
    preds = model(X)              # 2. forward pass: predict
    loss = criterion(preds, Y)    # 3. score the wrongness (one number)
    loss.backward()               # 4. autograd: dump slopes into every .grad
    optimizer.step()              # 5. optimizer: spend the slopes, update knobs

    if epoch % 40 == 0:
        print(f"epoch {epoch:3d}   loss {loss.item():.4f}")

print("\nlearned weight:", round(model.weight.item(), 3))   # should be ~2.0
print("learned bias:  ", round(model.bias.item(), 3))       # should be ~0.0
print("predict x=5:   ", round(model(torch.tensor([[5.0]])).item(), 3))  # ~10.0