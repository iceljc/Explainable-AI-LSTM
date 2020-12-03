import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

x = np.array([[]])

# plt.imshow(x, cmap='hot', interpolation='nearest')
# plt.colorbar()

fig,ax = plt.subplots(1, 1, figsize=(5,2))
sns.heatmap(x, linewidth=1.5, cmap='gray_r', linecolor='black', 
	cbar=False, square=True, xticklabels=False, yticklabels=False)

plt.savefig('temp.png')
plt.show()