import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, draw, pause
from functions import *
from IPython.display import clear_output
from matplotlib import animation
# from IPython import get_ipython
# get_ipython().run_line_magic('matplotlib', 'qt')
import time

folder = r'C:\Users\mlope\OneDrive - Universidad EAFIT\EAFIT\DLHM-data\11122021'

RGB, lambd = color('G')
holo = Image.open(folder + r'\h2_r4.tif')
holo = square(rgb2x(holo, RGB), -1)
ref = Image.open(folder + r'\r4.tif')
ref = square(rgb2x(ref, RGB), -1)

# an_s = as_reconstruct(holo, ref, 3.6e-6, lambd, 2.5e-3)
# plt.figure(1)
# plt.imshow(inten(an_s), cmap='gray')
# plt.show(block=True)

fb = fb_reconstruct(holo, ref, 10e-3, 15e-3, 3.6e-3, lambd)
plt.figure(1)
plt.imshow(inten(fb), cmap='gray')
plt.show(block=True)