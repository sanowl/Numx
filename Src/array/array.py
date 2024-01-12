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

    def _apply_operation(self, other, operation):
        other = self._ensure_array_or_scalar(other)
        result_data = operation(self.data, other)
        return Array(result_data.shape, dtype=result_data.dtype, buffer=result_data)

    def __add__(self, other):
        return self._apply_operation(other, np.add)

    def __sub__(self, other):
        return self._apply_operation(other, np.subtract)

    def __mul__(self, other):
        return self._apply_operation(other, np.multiply)

    def __truediv__(self, other):
        return self._apply_operation(other, np.divide)

    def matmul(self, other):
        if not isinstance(other, Array):
            raise ValueError("Matrix multiplication is only supported with another Array")
        result = np.matmul(self.data, other.data)
        return Array(result.shape, dtype=result.dtype, buffer=result)

    def transpose(self):
        return Array(self.data.T.shape, dtype=self.dtype, buffer=self.data.T)

    def dot(self, other):
        result = np.dot(self.data, other.data)
        return Array(result.shape, dtype=result.dtype, buffer=result)

    def inverse(self):
        if self.data.shape[0] != self.data.shape[1]:
            raise ValueError("Inverse only applicable to square matrices")
        result = np.linalg.inv(self.data)
        return Array(result.shape, dtype=result.dtype, buffer=result)

    def determinant(self):
        if self.data.shape[0] != self.data.shape[1]:
            raise ValueError("Determinant only applicable to square matrices")
        return np.linalg.det(self.data)

    # Additional methods like eigenvalues, eigenvectors, etc., can be added here.

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

    print("Inverse of Array 1:")
    print(arr1.inverse())

    print("Determinant of Array 1:")
    print(arr1.determinant())
