import matplotlib.pyplot as plt
from functions import *
import time
from matplotlib.pyplot import figure, draw, pause

tic = time.perf_counter()

folder = r'C:\Users\mlope\OneDrive - Universidad EAFIT\EAFIT\Microscopio portable\06232021'
laser = 'B'
reduction = 14
rl = -1
[c, l] = color('B')
p = 1e-6
z1 = 50e-3
z2 = 70e-3
it = 500

holo = Image.open(folder + r'\boca2.jpg')
holo = square(rgb2x(holo, c), rl)
ref = Image.open(folder + r'\ref2.jpg')
ref = square(rgb2x(ref, c), rl)

[d, cl, image] = autofocus(holo, ref, reduction, p, l, z1, z2, it)
plt.figure(1)
plt.imshow(amp(image) ** 2, cmap='gray')
plt.show(block=True)

plt.plot(cl)
plt.show(block=True)
toc = time.perf_counter()

t = toc-tic
print('Running time : ', t)
print('Reconstruction distance : ', d)

a = 1