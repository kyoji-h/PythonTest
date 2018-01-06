import numpy as np
import matplotlib.pyplot as plt

arr1 = np.arange(0, 10, 0.1)
print(arr1)

y = arr1 * 2 + 3
print(y)

plt.plot(arr1, y)
plt.show()
