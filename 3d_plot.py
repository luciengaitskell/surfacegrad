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
tif = tif [:500, :500]


xx, yy = np.meshgrid(np.arange(tif.shape[1]),np.arange(tif.shape[0]))

plt.figure()
ax = plt.axes(projection='3d')
ax.contour(xx, yy, tif, levels=100, cmap='binary')
# plt.hist(tif.flatten(), bins=np.arange(75, 100),  log=True)
#plt.imshow(tif)
plt.show()



