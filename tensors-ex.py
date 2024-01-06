import numpy as np
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

# task 1
torch.manual_seed(42)
x = torch.rand(6, 5)
y = torch.rand(1, 5)
result = torch.mm(x, torch.t(y))
print(result)

# task 2
torch.manual_seed(7)
features = torch.randn((1, 5))
weights = torch.randn_like(features)
bias = torch.randn((1, 1))


def model(features, weights, bias):
    return features * weights + bias


# task 3
print(torch.min(result))
print(torch.max(result))
print(torch.argmin(result))
print(torch.argmax(result))

# task 4
my_tensor = torch.randn((1, 1, 1, 25))
reshaped_tensor = torch.squeeze(my_tensor)
print(my_tensor, my_tensor.shape)
print(reshaped_tensor, reshaped_tensor.shape)

# task 5
tensor_1 = torch.range(1, 5)
reshaped_tensor_2 = torch.reshape(tensor_1, [1, 5])
print(tensor_1, tensor_1.shape)
print(reshaped_tensor_2, reshaped_tensor_2.shape)

# task 6
tensor_3 = torch.tensor([[1, 2, 3], [4, 5, 6]], device=device)
tensor_4 = torch.tensor([[7, 8, 9], [10, 11, 12]], device=device)
print(tensor_3 + tensor_4)

# task 7
tensor_5 = torch.rand((4, 4))
first_row = tensor_5[0, :]
last_column = tensor_5[:, -1]
print(first_row, "\n", last_column)

# task 8
tensor_6 = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
tensor_7 = torch.tensor([1, 1, 1])
reshaped_tensor_3 = tensor_6 + tensor_7
print(reshaped_tensor_3)

# task 9
tensor_9 = torch.rand([3, 4])
tensor_9_sum = torch.sum(tensor_9)
print("tensor_9 elements sum is ", tensor_9_sum)

# task 10
tensor_10 = torch.rand([3, 4])
tensor_10_mean = torch.mean(tensor_10, dim=1)
print(tensor_10_mean, tensor_10_mean.shape)
