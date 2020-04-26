from tifffile import imread
import matplotlib.pyplot as plt
import numpy as np

RI = 'USGS_13_n42w072.tif'
CA = 'USGS_13_n38w122.tif'

data = 'data/' + RI

tif = imread(data)

print(tif[10**3, 10**3])
print(tif[10**4, 10**4])

tif[tif < -10e20] = 0


plt.figure()
# plt.hist(tif.flatten(), bins=np.arange(75, 100),  log=True)
plt.imshow(tif)
plt.show()



