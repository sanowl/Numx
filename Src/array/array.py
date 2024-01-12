import numpy as np

class Array:
    def __init__(self, shape, dtype=float, buffer=None):
        self.shape = shape
        self.dtype = dtype
        self.data = np.zeros(shape, dtype=dtype) if buffer is None else np.array(buffer, dtype=dtype)

    def __str__(self):
        return str(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def __setitem__(self, index, value):
        self.data[index] = value

    def _ensure_array_or_scalar(self, other):
        if isinstance(other, Array):
            return other.data
        elif np.isscalar(other):
            return other
        else:
            raise ValueError("The operand must be an Array or a scalar")

    def __add__(self, other):
        other = self._ensure_array_or_scalar(other)
        return Array(self.shape, dtype=self.dtype, buffer=self.data + other)

    def __sub__(self, other):
        other = self._ensure_array_or_scalar(other)
        return Array(self.shape, dtype=self.dtype, buffer=self.data - other)

    def __mul__(self, other):
        other = self._ensure_array_or_scalar(other)
        return Array(self.shape, dtype=self.dtype, buffer=self.data * other)

    def __truediv__(self, other):
        other = self._ensure_array_or_scalar(other)
        return Array(self.shape, dtype=self.dtype, buffer=self.data / other)

    def matmul(self, other):
        if not isinstance(other, Array):
            raise ValueError("Matrix multiplication is only supported with another Array")
        if self.shape[-1] != other.shape[0]:
            raise ValueError("Shapes are not aligned for matrix multiplication")
        result = np.matmul(self.data, other.data)
        return Array(result.shape, dtype=self.dtype, buffer=result)

    def transpose(self):
        return Array(self.data.T.shape, dtype=self.dtype, buffer=self.data.T)

    # Additional methods like inverse, determinant, etc., can be added here.

# Example usage
if __name__ == "__main__":
    arr1 = Array((2, 2), buffer=[[1, 2], [3, 4]])
    arr2 = Array((2, 2), buffer=[[5, 6], [7, 8]])

    print("Array 1:")
    print(arr1)

    print("Array 2:")
    print(arr2)

    print("Matrix Multiplication:")
    print(arr1.matmul(arr2))

    print("Transpose of Array 1:")
    print(arr1.transpose())
