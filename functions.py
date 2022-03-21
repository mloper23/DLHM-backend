# March 19 2022
# DLHM backend library
# Author: Maria J Lopera, Kreuzer taken from C Trujillo (Apr 2018) Matlab

# Requirements: numpy, PIL, cmath, scipy, Tensorflow 2.6.0
import numpy as np
from PIL import Image
import cmath
from scipy import signal


# def focusnet(holo, ref):
#     model =


# Jurgen Kreuzer's patent to reconstruct inline holograms

def fb_reconstruct(holo, ref, z, L, x, lamvda):
    # Definition of geometrical parameters, definition of contrast hologram
    holoContrast = holo - ref
    [fi, co] = holo.shape
    # dx: real pixel size
    dx = x / fi
    # deltaX: pixel size at reconstruction plane
    deltaX = z * dx / L
    # Cosenus filter creation
    FC = filtcosenoF(100, fi)
    # Reconstruct
    K = kreuzer3F(holoContrast, z, L, lamvda, dx, deltaX, FC)
    return K


def kreuzer3F(CH_m, z, L, lamvda, deltax, deltaX, FC):
    # Squared pixels
    deltaY = deltaX
    # Matrix size
    [row, a] = CH_m.shape
    # Parameters
    k = 2 * np.pi / lamvda
    W = deltax * row
    #  Matrix coordinates
    delta = np.linspace(1, row, num = row)
    [X, Y] = np.meshgrid(delta, delta)
    # Hologram origin coordinates
    xo = -W / 2
    yo = -W / 2
    # Prepared hologram, coordinates origin
    xop = xo * L / np.sqrt(L ** 2 + xo ** 2)
    yop = yo * L / np.sqrt(L ** 2 + yo ** 2)
    # Pixel size for the prepared hologram (squared)
    deltaxp = xop / (-row / 2)
    deltayp = deltaxp
    # Coordinates origin for the reconstruction plane
    Yo = -deltaX * row / 2
    Xo = -deltaX * row / 2

    Xp = (deltax * (X - row / 2) * L / (
        np.sqrt(L ** 2 + (deltax ** 2) * (X - row / 2) ** 2 + (deltax ** 2) * (Y - row / 2) ** 2)))
    Yp = (deltax * (Y - row / 2) * L / (
        np.sqrt(L ** 2 + (deltax ** 2) * (X - row / 2) ** 2 + (deltax ** 2) * (Y - row / 2) ** 2)))
    # Preparation of the hologram
    CHp_m = prepairholoF(CH_m, xop, yop, Xp, Yp)
    # Multiply prepared hologram with propagation phase
    Rp = np.sqrt((L ** 2) - (deltaxp * X + xop) ** 2 - (deltayp * Y + yop) ** 2)
    r = np.sqrt((deltaX ** 2) * ((X - row / 2) ** 2 + (Y - row / 2) ** 2) + z ** 2)
    CHp_m = CHp_m * ((L / Rp) ** 4) * np.exp(-0.5 * 1j * k * (r ** 2 - 2 * z * L) * Rp / (L ** 2))
    # Padding constant value
    pad = int(row / 2)
    # Padding on the cosine rowlter
    FC = np.pad(FC, (int(pad), int(pad)))
    # Convolution operation
    # First transform
    T1 = CHp_m * np.exp((1j * k / (2 * L)) * (
            2 * Xo * X * deltaxp + 2 * Yo * Y * deltayp + X ** 2 * deltaxp * deltaX + Y ** 2 * deltayp * deltaY))
    T1 = np.pad(T1, (int(pad), int(pad)))
    T1 = ft(T1 * FC)
    # Second transform
    T2 = np.exp(-1j * (k / (2 * L)) * ((X - row / 2) ** 2 * deltaxp * deltaX + (Y - row / 2) ** 2 * deltayp * deltaY))
    T2 = np.pad(T2, (int(pad), int(pad)))
    T2 = ft(T2 * FC)
    # Third transform
    K = ift(T2 * T1)
    K = K[pad + 1:pad + row, pad + 1: pad + row]

    return K


def prepairholoF(CH_m, xop, yop, Xp, Yp):
    # User function to prepare the hologram using nearest neihgboor interpolation strategy
    [row, a] = CH_m.shape
    # New coordinates measured in units of the -2*xop/row pixel size
    Xcoord = (Xp - xop) / (-2 * xop / row)
    Ycoord = (Yp - yop) / (-2 * xop / row)
    # Find lowest integer
    iXcoord = np.floor(Xcoord)
    iYcoord = np.floor(Ycoord)
    # Assure there isn't null pixel positions
    iXcoord[iXcoord == 0] = 1
    iYcoord[iYcoord == 0] = 1
    # Calculate the fractionating for interpolation
    x1frac = (iXcoord + 1.0) - Xcoord  # Upper value to integer
    x2frac = 1.0 - x1frac
    y1frac = (iYcoord + 1.0) - Ycoord  # Lower value to integer
    y2frac = 1.0 - y1frac

    x1y1 = x1frac * y1frac  # Corresponding pixel areas for each direction
    x1y2 = x1frac * y2frac
    x2y1 = x2frac * y1frac
    x2y2 = x2frac * y2frac
    # Pre allocate the prepared hologram
    CHp_m = np.zeros([row, row])
    # Prepare hologram (preparation - every pixel remapping)
    for it in range(0, row - 2):
        for jt in range(0, row - 2):
            CHp_m[int(iYcoord[it, jt]), int(iXcoord[it, jt])] = CHp_m[int(iYcoord[it, jt]), int(iXcoord[it, jt])] + (x1y1[it, jt]) * CH_m[it, jt]
            CHp_m[int(iYcoord[it, jt]), int(iXcoord[it, jt]) + 1] = CHp_m[int(iYcoord[it, jt]), int(iXcoord[it, jt]) + 1] + (x2y1[it, jt]) * CH_m[it, jt]
            CHp_m[int(iYcoord[it, jt]) + 1, int(iXcoord[it, jt])] = CHp_m[int(iYcoord[it, jt]) + 1, int(iXcoord[it, jt])] + (x1y2[it, jt]) * CH_m[it, jt]
            CHp_m[int(iYcoord[it, jt]) + 1, int(iXcoord[it, jt]) + 1] = CHp_m[int(iYcoord[it, jt]) + 1, int(iXcoord[it, jt]) + 1] + (x2y2[it, jt]) * CH_m[it, jt]

    return CHp_m


def filtcosenoF(par, fi):
    # Function to create the Cosinus filter
    # Par: Size parameter
    # Fig: Matrix size
    # Coordinates
    delta = np.linspace(-fi / 2 , fi/2, fi)
    [xfc, yfc] = np.meshgrid(delta, delta)
    # Normalize coordinates on the interval [-pi, pi] and create filters in vertical and horizontal directions
    FC1 = np.cos(xfc * (np.pi / par) * 1 / xfc.max()) ** 2
    FC2 = np.cos(yfc * (np.pi / par) * 1 / yfc.max()) ** 2
    # Intersect both directions
    FC = (FC1 > 0) * FC1 * (FC2 > 0) * FC2
    # Re-scale
    FC = FC / FC.max()
    return FC


# Reconstruction with angular spectrum and autofocusing methods implementation
def autofocus(holo, ref, reduction, p, lamvda, z1, z2, it):
    # Function to find the focus distance using angular spectrum and ___ metric
    # Size
    [n, m] = holo.shape
    # Reduce the hologram on certain percentage to improve efficiency
    h = np.resize(holo, (int(n / reduction), int(m / reduction)))
    r = np.resize(ref, (int(n / reduction), int(m / reduction)))
    # h1 = signal.resample(holo, int(n / reduction), axis=0)
    # h = signal.resample(h1, int(m / reduction), axis=1)
    # r1 = signal.resample(ref, int(n / reduction), axis=0)
    # r = signal.resample(r1, int(m / reduction), axis=1)
    # Calculate the AS reconstruction for each plane
    pr = p * reduction
    delta = (z2 - z1) / it
    cl = np.zeros([it, it])
    for i in range(0, it):
        uz = as_reconstruct(h, r, pr, lamvda, z1 + i * delta)
        cl[i] = np.sum(amp(uz))
    # Defining the criteria to find the best focusing plane
    # m = np.argmin(np.diff(cl))
    m = np.argmin(cl)
    d = z1 + m * delta
    image = as_reconstruct(holo, ref, p, lamvda, d)
    # We return the best distance, the metric and the best reconstructed image
    return d, cl, image


def amp(i):
    # Calculates the amplitude of a complex matrix
    a = np.sqrt(np.imag(i) ** 2 + np.real(i) ** 2)
    return a


def inten(i):
    # Calculates the intensity of a complex matrix
    a = amp(i) ** 2
    return a


def phase(i):
    # Calculates the phase of a complex matrix
    a = np.imag(i) / np.real(i)
    return a


def color(x):
    # Useful function to return wavelength and channel when working with RGB cameras
    if x == 'R':
        c = 0
        l = 633e-9
    if x == 'G':
        c = 1
        l = 532e-9
    if x == 'B':
        c = 2
        l = 473e-9
    return c, l


def square(i, x):
    # Useful function to convert a horizontal rectangle image into a squared image.
    # If -1 cuts the left part of the image, if +1 curts the right part
    n, m = i.shape
    if n < m:
        if x == -1:
            si = i[:, 0:n]
        if x == 1:
            si = i[:, m - n:m]
    if n > m:
        if x == -1:
            si = i[0:m, :]
        if x == 1:
            si = i[n - m:n, :]
    return si


def rgb2x(i, x):
    # Useful function to convert a rgb image to a bw one by taking just one chanel x
    ic = np.array(i, dtype=float)
    return ic[:, :, x]


def img_norm(img):
    # Normalization
    img = img / img.max()
    return img

# Angular spectrum reconstruction
def as_reconstruct(holo, ref, dp, lo, z):
    k = (2 * np.pi) / lo
    holo = img_norm(holo)
    ref = img_norm(ref)
    # Contrast hologram
    hr = holo - ref
    uz = angular_spectrum(hr, k, dp, z)
    return uz


def angular_spectrum(u0, k, dp, z):
    n, m = u0.shape
    df = 1 / (n * dp)
    f = np.arange(-n / 2 * df, n / 2 * df, df)
    [xf, yf] = np.meshgrid(f, f)
    e = np.exp((1j * z * np.lib.scimath.sqrt((k ** 2 - 4 * np.pi ** 2 * (xf ** 2 + yf ** 2)))))
    uz = ift(ft(u0) * e)
    return uz


def ft(u):
    ut = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(u)))
    return ut


def ift(u):
    uit = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(u)))
    return uit
