import numpy as np

class Array:
    def __init__(self, shape, dtype=float, buffer=None):
        """Initialize an Array with a given shape.

        Args:
        shape (tuple): The shape of the array.
        dtype (type, optional): The data type of the array. Defaults to float.
        buffer (np.ndarray, optional): A buffer to initialize the array. Defaults to None.
        """
        self.shape = shape
        self.dtype = dtype
        self.data = np.zeros(shape, dtype=dtype) if buffer is None else np.array(buffer, dtype=dtype)

    def __str__(self):
        """String representation of the array for printing."""
        return str(self.data)

    def __getitem__(self, index):
        """Get an item from the array."""
        return self.data[index]

    def __setitem__(self, index, value):
        """Set an item in the array."""
        self.data[index] = value

    def __add__(self, other):
        """Add two arrays element-wise."""
        return Array(self.shape, dtype=self.dtype, buffer=self.data + other.data)

    def __sub__(self, other):
        """Subtract two arrays element-wise."""
        return Array(self.shape, dtype=self.dtype, buffer=self.data - other.data)

    def __mul__(self, other):
        """Multiply two arrays element-wise."""
        return Array(self.shape, dtype=self.dtype, buffer=self.data * other.data)

    def __truediv__(self, other):
        """Divide two arrays element-wise."""
        return Array(self.shape, dtype=self.dtype, buffer=self.data / other.data)

    # Advanced methods could include matrix multiplication, inverse, etc.

# Example usage
if __name__ == "__main__":
    arr1 = Array((2, 2), buffer=[[1, 2], [3, 4]])
    arr2 = Array((2, 2), buffer=[[5, 6], [7, 8]])

    print("Array 1:")
    print(arr1)

    print("Array 2:")
    print(arr2)

    print("Added Arrays:")
    print(arr1 + arr2)
