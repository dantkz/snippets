import numpy as np
import json
import matplotlib.pyplot as plt

cmap = plt.get_cmap('hsv')

rgba = cmap(np.linspace(0,1,255, endpoint=True))
rgba = np.round(255*rgba).astype(np.int32)
rgb = rgba[:,0:3]

rgb = rgb[np.random.permutation(rgb.shape[0]),:]

color_str = ['#%02x%02x%02x' % (rgb[i,0], rgb[i,1], rgb[i,2]) for i in range(rgb.shape[0])]
with open('colormap.csv', 'w') as f:
    for c in color_str:
        f.write(c+'\n')
