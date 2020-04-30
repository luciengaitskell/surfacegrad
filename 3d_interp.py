from tifffile import imread
from vispy import app, scene, color
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
tif = orig_tiff[:10000, :10000]


x, y = np.arange(tif.shape[1]), np.arange(tif.shape[0])

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

    x_map, y_map = np.meshgrid(x_out, y)
    '''
    def multiInterp2(x, xp, fp):
        i = np.arange(x.size)
        j = np.searchsorted(xp, x) - 1
        d = (x - xp[j]) / (xp[j + 1] - xp[j])
        return (1 - d) * fp[i, j] + fp[i, j + 1] * d'''
    #https://stackoverflow.com/questions/43772218/fastest-way-to-use-numpy-interp-on-a-2-d-array

    out_z = np.array([np.interp(x_out*SCALE, x_m_orig, tif[i]) for i in range(x_m_orig.shape[0])])

print("Post interp")

canvas = scene.SceneCanvas(keys='interactive', bgcolor='w')
view = canvas.central_widget.add_view()
view.camera = scene.TurntableCamera(up='z', fov=60)

print("Pre gradient")


def calc_gradient(A, axis):
    # Get first and last row / col and expand dimensions to match original
    if axis == 0:  # x
        prep = np.expand_dims(A[0, :], axis=0)  # [a,b,c,...] -> [[a,b,c,...]]
        app = np.expand_dims(A[-1, :], axis=0)
    elif axis == 1:  # y
        prep = np.expand_dims(A[:, 0], axis=1)  # [a,b,c,...] -> [[a],[b],[c],...]
        app = np.expand_dims(A[:, -1], axis=1)
    else:
        raise ValueError

    # Take difference with extra first and last row / col
    D = np.diff(A, axis=axis, prepend=prep, append=app)  # Take difference between adjacent elements

    # Take sum of adjacent rows / col
    if axis == 0:  # x
        B = D[:-1] + D[1:]
    elif axis == 1:  # y
        B = D[:, :-1] + D[:, 1:]

    return B / 2  # Complete formula

gradient = (calc_gradient(out_z, 0), calc_gradient(out_z, 1))
#gradient = np.gradient(out_z)
gradient_mag = np.sqrt(np.square(gradient[0]) * np.square(gradient[1]))
gradient_dir = np.arctan2(*gradient)
#gradient = np.log(gradient)

print("Post gradient")

downsample = 10
z = out_z[::downsample, ::downsample]

#color_source = z
gradient_dir = gradient_dir[::downsample, ::downsample]
gradient_mag = gradient_mag[::downsample, ::downsample]

color_source = gradient_dir
#color_source[color_source > 0.4] = 0.4

# https://github.com/vispy/vispy/issues/1006#issuecomment-250983610
c = color.get_colormap('hsv').map(color_source/np.abs(np.max(color_source))).reshape(color_source.shape + (-1,))
c[gradient_mag > 0.4, 3] = 0

c = c.flatten().tolist()
c=list(map(lambda x,y,z,w:(x,y,z,w), c[0::4],c[1::4],c[2::4],c[3::4]))

p1 = scene.visuals.SurfacePlot(x=y[::downsample], y=x_out[::downsample], z=z)

p1.mesh_data.set_vertex_colors(c)
view.add(p1)

plt.figure()
plt.hist(gradient_mag.ravel(), bins=np.arange(0,1.5,0.01))
plt.show()


axis = scene.visuals.XYZAxis(parent=view.scene)

if __name__ == '__main__':
    canvas.show()
    app.run()
