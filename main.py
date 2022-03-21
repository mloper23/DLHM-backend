import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, draw, pause
from functions import *
from IPython.display import clear_output
from matplotlib import animation

import time

folder = r'C:\Users\mlope\OneDrive - Universidad EAFIT\EAFIT\Microscopio portable\06232021'

RGB,lambd = color('B')
holo = Image.open(folder + r'\boca1.jpg')
holo = square(rgb2x(holo, RGB), -1)
ref = Image.open(folder + r'\ref2.jpg')
ref = square(rgb2x(ref, RGB), -1)
z = np.linspace(63e-3, 65e-3, num=1)

for i in z:
    uz = as_reconstruct(holo, ref, 1e-6, lambd, i)
    plt.figure(1)
    plt.imshow(np.abs(uz)**2, cmap='gray')
    plt.show(block=True)
    # plt.pause(1.5)
    # plt.close('all')
