#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 18 20:21:32 2022

@author: bimbim F4JBO
"""

def doc():
    print("\nsyntax : \n\n"
        
        "convert an image to 2-bit 200x200 image \n"
        ">> ./vibe.py -tx -i <input_image_name> <output_image_name> \n\n"

        "convert an image to QPSK sound signal \n"
        ">> ./vibe.py -tx -s <input_image_name> <output_soundFile_name> \n\n"
        
        "decode a QPSK sound signal to image \n"
        ">> ./vibe.py -rx <input_soundFile_name> <image_name> <center_frequency (optionnal)> \n\n"
        
        "help ! \n"
        ">> ./vibe.py -help \n\n")

#%% libraries

import sys

from skimage import io
from skimage.util import crop
from skimage.transform import resize

import numpy as np

from scipy.interpolate import interp1d

import commpy

import soundfile as sf

#%% LOAD A PICTURE


def squareCropCoordinate(image):

    height = image.shape[0]
    width = image.shape[1]

    # is the image already square ?
    if height == width:
        return ((0, 0), (0, 0), (0, 0))

    # Is the image in portrait or landscape mode
    is_portrait = height > width

    if is_portrait:
        deltaX = int((height - width) / 2)
        return ((deltaX, deltaX), (0, 0), (0, 0))
    else:
        deltaY = int((width - height) / 2)
        return ((0, 0), (deltaY, deltaY), (0, 0))


#..............................................................................


def loadPicture(filename):
    image = io.imread(filename)
    image = crop(image, squareCropCoordinate(image))
    image = resize(image, (204, 202, 3))
    return image

#%% COMPRESS THE PICTURE


def palette4bit():
    return np.array([[[50, 50, 110],
                      [172, 70, 70],
                      [80, 182, 80],
                      [255, 255, 255]]])/255

#..............................................................................


def findClosest(value, palette):
    x = palette[0, :, 0]
    y = palette[0, :, 1]
    z = palette[0, :, 2]

    dx = x - value[0]
    dy = y - value[1]
    dz = z - value[2]

    dist = np.sqrt(dx**2 + dy**2 + dz**2)

    return palette[0, np.argmin(dist), :]

#..............................................................................


def dithering(image, palette, bias=2):
    # this function will constrain image color to palette color
    # using dithering (the atkinson algorithm)

    output = np.copy(image)

    for y in np.arange(0, output.shape[1] - 2, 1):
        for x in np.arange(2, output.shape[0] - 2, 1):

            oldPixel = np.copy(output[x, y, :])
            newPixel = findClosest(oldPixel, palette)
            output[x, y, :] = newPixel

            quant_error = (oldPixel - newPixel) / (6 + bias)

            output[x+1, y, :] = output[x+1, y, :] + quant_error
            output[x+2, y, :] = output[x+2, y, :] + quant_error

            output[x-1, y+1, :] = output[x-1, y+1, :] + quant_error
            output[x, y+1, :] = output[x, y+1, :] + quant_error
            output[x+1, y+1, :] = output[x+1, y+1, :] + quant_error

            output[x, y+2, :] = output[x, y+2, :] + quant_error

    # crop the un-converged pixels
    output = output[2:-2, 0:-2, :]

    return output

#%% GENERATE A MESSAGE TO TRANFER


def im2bin(dithered_image):
    im = np.copy(dithered_image)*255
    
    # only keep the red channel since this is sufficient to differenciate
    # the palette's colors
    im = im[:,:,0]
    im = im.flatten()
    
    output = np.zeros(im.size)
    
    # important :
    # 50  = palette[0] = (blue black)
    # 172 = palette[1] = (medium red)
    # 80  = palette[2] = (light green)
    # 255 = palette[3] = (pure white)
    
    color2bit = {50:0, 172:1, 80:2, 255:3}
    
    for i in range(im.size):
        output[i] = color2bit[im[i]]
    
    output = output.astype(np.uint8)
    output = np.unpackbits(output)
    output = np.reshape(output, (-1,8))[:, -2:]
    output = output.flatten()
    
    return output

#..............................................................................

def framing(binaries):
    # freq and time synchronisation
    sync_seq = (np.linspace(0,1000,1501)%2).astype(np.uint8)
    
    # frame synchronisation
    barker_seq = np.array([1,1,1,1,1,0,0,1,1,0,1,0,1]).astype(np.uint8)
    return np.concatenate((sync_seq, barker_seq, binaries))

#%% MODULATE THE MESSAGE ONTO SOUND


def bin2pulse(message):
    output = np.zeros(int(len(message)/2)).astype(np.complex128)
    
    for i in range(len(output)):
        if message[i*2] == 0 and message[i*2 + 1] == 0:
            output[i] = -1-1j
        if message[i*2] == 0 and message[i*2 + 1] == 1:
            output[i] = -1+1j
        if message[i*2] == 1 and message[i*2 + 1] == 0:
            output[i] = +1-1j
        if message[i*2] == 1 and message[i*2 + 1] == 1:
            output[i] = +1+1j
    
    return output

#..............................................................................


def pulseShaping(pulses, sps, nb_taps):
    h = commpy.filters.rrcosfilter(nb_taps, 0.35, sps*(1/48e3), 48e3)
    h *= np.hamming(nb_taps)
    
    output = np.zeros(len(pulses)*sps).astype(np.complex128)
    for i in range(len(pulses)):
        output[i*sps] = pulses[i]
    
    r_output = np.convolve(np.real(output), h[1], 'full')
    i_output = np.convolve(np.imag(output), h[1], 'full')
    
    output = r_output + i_output*1j
    
    return output
    
#..............................................................................


def pulse2signal(pulses, Fs, freq):
    t = np.linspace(0,len(pulses)/Fs, len(pulses))
    sig = np.exp(1j*2*np.pi*freq*t) * pulses
    sig /= np.max(np.abs(sig))
    return np.real(sig)

#%% DEMODULATE THE SOUND

def demodulate(filename, fTuning, nb_taps, sps, subSteps=0):
    data, samplerate = sf.read(filename)
    
    
    # float to complex
    data = data.astype(np.complex128)
    
    # multiply by sin
    t = np.linspace(0,len(data)/samplerate, len(data))
    data *= np.exp(-1j*2*np.pi*fTuning*t)
    
    
    # matched filter
    h = commpy.filters.rrcosfilter(nb_taps, 0.35, sps*(1/48e3), 48e3)
    h *= np.hamming(nb_taps)
    data = np.convolve(np.real(data), h[1], 'full') + \
        1j*np.convolve(np.imag(data), h[1], 'full')
        
    # resample
    if subSteps > 0:
        data = resample(data, subSteps)
    
    # polyphase clock sync
    data = polyphaseClockSync(data, sps*(subSteps+1))
    
    # normalize
    data /= np.mean(np.abs(data[1000:]))
    
    # costa loop
    data = costaLoop(data, sps*(subSteps+1))
    
    return data

#..............................................................................

def resample(data, subSteps):
    x = np.linspace(0, len(data), len(data))
    f = interp1d(x, data, kind='cubic')
    x_new = np.linspace(0, len(data), len(data)*(subSteps+1))
    return f(x_new)
    

#..............................................................................


def polyphaseClockSync(data, sps):
    mu = 0 # initial estimate of phase of sample
    out = np.zeros(len(data) + 10, dtype=np.complex128)
    
    # stores values, each iteration we need the previous 2 values plus current value
    out_rail = np.zeros(len(data) + 10, dtype=np.complex128)
    
    i_in = 0 # input data index
    i_out = 2 # output index (let first two outputs be 0)
    
    while i_out < len(data) and i_in+16 < len(data):
        out[i_out] = data[i_in + int(mu)] # grab what we think is the "best" sample
        out_rail[i_out] = int(np.real(out[i_out]) > 0) + 1j*int(np.imag(out[i_out]) > 0)
        
        x = (out_rail[i_out] - out_rail[i_out-2]) * np.conj(out[i_out-1])
        y = (out[i_out] - out[i_out-2]) * np.conj(out_rail[i_out-1])
        
        mm_val = np.real(y - x)
        mu += sps + 0.3*mm_val
        
        # round down to nearest int since we are using it as an index
        i_in += int(np.floor(mu))
        mu = mu - np.floor(mu) # remove the integer part of mu  
        i_out += 1 # increment output index
    
    # remove the first two, and anything after i_out (that was never filled out)
    out = out[2:i_out]
    data = out
    
    return data

#..............................................................................


def costaLoop(data, samplerate):
    N = len(data)
    phase = 0
    freq = 0
    
    # These next two params is what to adjust,
    # to make the feedback loop faster or slower (which impacts stability)
    alpha = 0.132
    beta = 0.00932
    
    out = np.zeros(N, dtype=np.complex128)
    for i in range(N):
        # adjust the input sample by the inverse of the estimated phase offset
        out[i] = data[i] * np.exp(-1j*phase)
        
        # order 4 error equation
        if out[i].real > 0:
            a = 1.0
        else :
            a = -1.0
        if out[i].imag > 0:
            b = 1.0
        else :
            b = -1.0
            
        error = a * out[i].imag - b * out[i].real 
    
        # Advance the loop (recalc phase and freq offset)
        freq += (beta * error)
        phase += freq + (alpha * error)
    
    return out


#%% DECODE THE SIGNAL

def constDecode(sample):
    const = np.array([-1-1j, -1+1j, 1+1j, 1-1j])*(np.sqrt(2)/2)
    symbols = np.array([[0,0],[0,1],[1,1],[1,0]])
    dist = abs(const - sample)
    return symbols[np.argmin(dist),:]

def const2bin(dem_signal):
    out = np.zeros(len(dem_signal)*2)
    
    for i in range(len(dem_signal)):
        out[2*i:(2*i)+2] = constDecode(dem_signal[i])
    
    return out

#..............................................................................


def frameStartDetect(received_bin):
    barker_seq = np.array([1,1,1,1,1,0,0,1,1,0,1,0,1])
    
    out = np.zeros(len(received_bin))
    
    for i in range(len(received_bin)-len(barker_seq)):
        out[i] = np.sum(np.abs(received_bin[i:i+len(barker_seq)]-barker_seq))
    
    out = out.astype(int)
    
    i=0
    while out[i] > 0:
        i += 1
    
    return i + len(barker_seq)

#%% DECODE THE PICTURE

def bin2img(received_binaries):
    palette = palette4bit()
    img = np.zeros((200,200,3))
    
    received_binaries = np.reshape(received_binaries, (40000,2))
    received_binaries = np.append(np.zeros((40000,6)),\
                         received_binaries,axis=1).astype(np.uint8)
    color_index = np.packbits(received_binaries,axis=1)
    
    color_index = np.reshape(color_index, (200,200))
    
    for x in range(200):
        for y in range(200):
            img[x,y,:] = palette[0,color_index[x,y],:]
    
    return img


#%% MAIN

if __name__ == "__main__":
    
    # convert an image to 2-bit 200x200 image
    if sys.argv[1] == '-tx':
        if sys.argv[2] == '-i':
            image = loadPicture(sys.argv[3])
            dithered_image = dithering(image, palette4bit())
            dithered_image = (dithered_image*255).astype(np.uint8)
            io.imsave(sys.argv[4],dithered_image)
    
    # convert an image to QPSK sound signal
        elif sys.argv[2] == '-s':
            image = loadPicture(sys.argv[3])
            dithered_image = dithering(image, palette4bit())
            binaries = im2bin(dithered_image)
            message = framing(binaries)
            pulses = bin2pulse(message)
            shaped = pulseShaping(pulses, 32, 1001)
            sig = pulse2signal(shaped,48e3,1800)
            sound_filename = sys.argv[4]
            sf.write(sys.argv[4], sig, 48000)
        
        else:
            doc()
    
    # decode a QPSK sound signal to image
    elif sys.argv[1] == '-rx':
    
        try:
            if sys.argv[4]:
                print('central frequency set to : {}Hz'.format(sys.argv[4]))
                dem_signal = demodulate(sys.argv[2],
                                        float(sys.argv[4]), 1001, 32, 16)
        except IndexError:
            dem_signal = demodulate(sys.argv[2], 1800, 1001, 32, 16)
            
        decoded_binaries = const2bin(dem_signal)
        start_idx = frameStartDetect(decoded_binaries)
        received_binaries = decoded_binaries[start_idx:start_idx+200*200*2]
        img = bin2img(received_binaries)
        img = (img*255).astype(np.uint8)
        io.imsave(sys.argv[3], img)
        
    else :
        doc()
