import torch

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