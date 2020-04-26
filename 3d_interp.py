from tifffile import imread
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate

SCALE_FROM_Y = True
TWODIM = False


RI = 'USGS_13_n42w072.tif'
CA = 'USGS_13_n38w122.tif'

data = 'data/' + RI

orig_tiff = imread(data)

lat = 42
lon = 72


print(orig_tiff[10**3, 10**3])
print(orig_tiff[10**4, 10**4])

orig_tiff[orig_tiff < -10e20] = 0
tif = orig_tiff[:8000, :8000]


# TODO: THIS IS PERFORMED TWICE
x, y = np.meshgrid(np.arange(tif.shape[1]), np.arange(tif.shape[0]))

x = x.astype('float64')
y = y.astype('float64')


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



assert orig_tiff.shape[0] == orig_tiff.shape[1]
SCALE = 60 * 1852 / orig_tiff.shape[0]
y_m_orig = y * SCALE  # y * (nmi / deg) * (m / nmi)
x_m_orig = (x * SCALE  # x * (nmi / deg) * (m / nmi)
            * np.cos(np.radians(y/orig_tiff.shape[0] + lat)))

print("Pre interp")

if TWODIM:
    x_map, y_map = np.meshgrid(np.arange(0, np.floor(np.max(x_m_orig))), np.arange(0, np.floor(np.max(y_m_orig))))
    if True:  # https://stackoverflow.com/questions/37872171/how-can-i-perform-two-dimensional-interpolation-using-scipy
        out_z = interpolate.griddata(np.array([x_m_orig.ravel(), y_m_orig.ravel()]).T, tif.ravel(),
                                          (x_map, y_map), method='nearest')  # default method is linear

    else:
        f = interpolate.interp2d(x_m_orig, y_m_orig, tif, kind='linear')
else:
    num_x = np.floor(np.max(x_m_orig) / SCALE)
    x_out = np.arange(0, num_x)  # NEED TO MAKE GRID INSTEAD

    x_map, y_map = np.meshgrid(x_out, np.arange(tif.shape[0]))
    '''
    def multiInterp2(x, xp, fp):
        i = np.arange(x.size)
        j = np.searchsorted(xp, x) - 1
        d = (x - xp[j]) / (xp[j + 1] - xp[j])
        return (1 - d) * fp[i, j] + fp[i, j + 1] * d'''
    #https://stackoverflow.com/questions/43772218/fastest-way-to-use-numpy-interp-on-a-2-d-array

    out_z = np.array([np.interp(x_out*SCALE, x_m_orig[i], tif[i]) for i in range(x_m_orig.shape[0])])

print("Post interp")
plt.figure()
ax = plt.axes(projection='3d')
plt.gca().invert_xaxis()
ax.view_init(elev=90., azim=90.)
#ax.contour(x, y, tif, levels=10, cmap='binary')

#ax.scatter(x_map, y_map, z_griddata, marker="o")

ax.contour(x_map*SCALE, y_map*SCALE, out_z, levels=50, cmap='binary')
# plt.hist(tif.flatten(), bins=np.arange(75, 100),  log=True)
#plt.imshow(tif)
plt.show()



