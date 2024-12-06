# Tutorial: PyTorch Basics - Tensor Operations

import torch

# 1. Create a 1D tensor with values [1, 2, 3]
tensor_1d = torch.tensor([1, 2, 3])
print("Task 1: Create a 1D tensor with values [1, 2, 3]")
print(tensor_1d)

# 2. Create a 2D tensor with shape (2, 3) filled with zeros
tensor_2d_zeros = torch.zeros((2, 3))
print("\nTask 2: Create a 2D tensor of zeros with shape (2, 3)")
print(tensor_2d_zeros)

# 3. Create a random tensor of shape (2, 2)
tensor_random = torch.rand((2, 2))
print("\nTask 3: Create a random tensor of shape (2, 2)")
print(tensor_random)

# 4. Reshape a 1D tensor of 6 elements into a 2x3 matrix
tensor_1d_6 = torch.tensor([1, 2, 3, 4, 5, 6])
reshaped_tensor = tensor_1d_6.view(3, 2)
print("\nTask 4: Reshape a 1D tensor of 6 elements into a 2x3 matrix")
print(reshaped_tensor)

# 5. Indexing: Extract the second row of a tensor
tensor_2d = torch.tensor([[1, 2, 3], [4, 5, 6]])
second_row = tensor_2d[1, :]
print("\nTask 5: Extract the second row of the tensor")
print(second_row)

# 6. Slicing: Extract the first two columns of the tensor
first_two_columns = tensor_2d[1:, -1]
print("\nTask 6: Extract the first two columns of the tensor")
print(first_two_columns)

# 7. Element-wise operations: Add a scalar value to a tensor
tensor_add_scalar = tensor_2d + 3
print("\nTask 7: Add a scalar value (3) to each element of the tensor")
print(tensor_add_scalar)

# 8. Broadcasting: Add a 1D tensor to a 2D tensor (broadcasting example)
tensor_1d_broadcast = torch.tensor([1, 2, 3])
tensor_broadcasted = tensor_2d + tensor_1d_broadcast
print("\nTask 8: Broadcasting - Add a 1D tensor to a 2D tensor")
print(tensor_1d_broadcast)
print(tensor_2d)
print(tensor_broadcasted)

# 9. Matrix multiplication: Multiply two matrices (2x3) and (3x2)
tensor_2x3 = torch.tensor([[1, 2, 3], [4, 5, 6]])
tensor_3x2 = torch.tensor([[7, 8], [9, 10], [11, 12]])
result_matrix = torch.matmul(tensor_2x3, tensor_3x2)
print("\nTask 9: Matrix multiplication of two matrices (2x3) and (3x2)")
print(result_matrix)

# 10. Broadcasting with a scalar: Multiply a 2D tensor with a scalar
tensor_scalar_multiply = tensor_2d * 2
print("\nTask 10: Multiply a 2D tensor with a scalar")
print(tensor_scalar_multiply)

# 11. Transposing a tensor (2x3 -> 3x2)
transposed_tensor = tensor_2d.T
print("\nTask 11: Transpose a 2D tensor")
print(transposed_tensor)

# 12. Compute the sum of all elements in a tensor
tensor_sum = tensor_2d.sum()
tensor_3x3 = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
tensor_3_3_diag = tensor_3x3[1, 2]
print("\nTask 12: Diag of all elements in the tensor")
print(tensor_3_3_diag)

# 13. Reshaping a tensor using .reshape
reshaped_tensor_2 = tensor_2d.reshape(6, 1)
print("\nTask 13: Reshape tensor using .reshape (3, 2)")
print(tensor_2d)
print(reshaped_tensor_2)
print(tensor_2d.T)

# 14. Flatten a tensor (2D to 1D)
flattened_tensor = tensor_2d.flatten()
print("\nTask 14: Flatten a tensor (2D to 1D)")
print(flattened_tensor)


tensor2 = torch.tensor([[1, 0, 0], [1, 1, 0], [1, 1, 1]])
tensor3 = torch.tensor([1, 2, 3])
print(tensor2@tensor3)
