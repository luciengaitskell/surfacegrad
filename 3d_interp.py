from tifffile import imread
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate

SCALE_FROM_Y = True
TWODIM = True


RI = 'USGS_13_n42w072.tif'
CA = 'USGS_13_n38w122.tif'

data = 'data/' + RI

orig_tiff = imread(data)

lat = 42
lon = 72


print(orig_tiff[10**3, 10**3])
print(orig_tiff[10**4, 10**4])

orig_tiff[orig_tiff < -10e20] = 0
tif = orig_tiff[:100, :100]

x, y = np.meshgrid(np.arange(tif.shape[1]), np.arange(tif.shape[0]))

x = x.astype('float64')
y = y.astype('float64')

x = x/orig_tiff.shape[1] + lon
y = y/orig_tiff.shape[0] + lat

"""
R = 6370997.2  # meters

y_rad = np.radians(y)
x_rad = np.radians(x)

y_m_orig = R * (y_rad - y_rad[0, 0])

if SCALE_FROM_Y:
    scale = 1 / y_m_orig[1, 0]
else:
    scale = 1e-1

y_m_orig *= scale
x_m_orig = R * np.cos(y_rad) * (x_rad - x_rad[0, 0]) * scale
"""


y_m_orig = (y - y[0, 0]) * 60 * 1852  # y * (nmi / deg) * (m / nmi)
x_m_orig = ((x - x[0, 0]) * 60 * 1852  # x * (nmi / deg) * (m / nmi)
            * np.cos(np.radians(y)))

print("Pre interp")

if TWODIM:
    x_map, y_map = np.meshgrid(np.arange(0, np.floor(np.max(x_m_orig))), np.arange(0, np.floor(np.max(y_m_orig))))
    if True:  # https://stackoverflow.com/questions/37872171/how-can-i-perform-two-dimensional-interpolation-using-scipy
        z_griddata = interpolate.griddata(np.array([x_m_orig.ravel(), y_m_orig.ravel()]).T, tif.ravel(),
                                          (x_map, y_map), method='nearest')  # default method is linear

    else:
        f = interpolate.interp2d(x_m_orig, y_m_orig, tif, kind='linear')
else:
    def multiInterp2(x, xp, fp):
        i = np.arange(x.size)
        j = np.searchsorted(xp, x) - 1
        d = (x - xp[j]) / (xp[j + 1] - xp[j])
        return (1 - d) * fp[i, j] + fp[i, j + 1] * d
    #https://stackoverflow.com/questions/43772218/fastest-way-to-use-numpy-interp-on-a-2-d-array
    x_map, y_map = np.meshgrid(np.arange(0, np.floor(np.max(x_m_orig))), np.arange(0, np.floor(np.max(y_m_orig))))
    np.interp()

print("Post interp")
plt.figure()
ax = plt.axes(projection='3d')

#ax.contour(x, y, tif, levels=10, cmap='binary')

#ax.scatter(x_map, y_map, z_griddata, marker="o")

ax.contour(x_map, y_map, z_griddata, levels=30, cmap='binary')
# plt.hist(tif.flatten(), bins=np.arange(75, 100),  log=True)
#plt.imshow(tif)
plt.show()



