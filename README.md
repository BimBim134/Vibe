# Vibe
Vibe is a small python executable to send image using sound.

It started as an experiment, and here's my very first python program ! It may be slow, but it works. One could call it a success allready.

![tests](https://i.imgur.com/vJF95Gs.jpg)

## how to use it :

first :
    
    git clone https://github.com/BimBim134/Vibe
    chmod +x vibe.py

then :

    syntax : 

    convert an image to 2-bit 200x200 image 
    >> ./vibe.py -tx -i <input_image_name> <output_image_name> 

    convert an image to QPSK sound signal 
    >> ./vibe.py -tx -s <input_image_name> <output_soundFile_name> 

    decode a QPSK sound signal to image 
    >> ./vibe.py -rx <input_soundFile_name> <image_name> 

    help ! 
    >> ./vibe.py -help 


## Backstory :

I'm a student interested about everything telecom and especially radio communications. I am a certified radio-ham (F4JBO) and as such, intrigued by SSTV and digital modes.

I rapidly wanted to try to find a new way to send picture in a more effective way using more modern technique than SSTV. I explored lots of different modulation technique, ways of compressing image etc...

I finally started making experiment and I converged on what would be the base of Vibe : *2-bit color palette, Atkinson dithering and QPSK modulation*.

![2 bit color palette](https://i.imgur.com/daqP9Z5.jpg)
![QPSK](https://i.imgur.com/MudBpNx.png)

## So how does it works ?

### Transmit mode

in transmit mode, there's two main phases :
* image processing
  * crop to square format
  * resizing to 200x200
  * dithering
* signal generation
  * encode each pixel to binaries array (x00, x01, x10, x11)
  * add frame ([Barker sequence](https://www.wikiwand.com/en/Barker_code) to know when does the data begin)
  * pulses generation and pulse shaping
  * signal generation

#### Image processing

During the image processing phase, the image is first crop in square format and resized to a very small 200x200. anything bigger would have been too long to transmit. And frankly, small image are kinda fun.

Next step is dithering dithering. I choosed the Atkinson algorithm because it tends to be way more robust than any other algorithm. I tried every algorithms while using only 4 color but see for yourself :

here are the references images :
![reference images](https://i.imgur.com/uaXxVlh.png)

here's a few test using different algorithms (don't mind the colors, I was using another color palette at the time)

![test dithering 1](https://i.imgur.com/5Qh2nyi.jpg)
![test dithering 2](https://i.imgur.com/fF7BOfZ.jpg)
![test dithering 3](https://i.imgur.com/B78OthH.jpg)
![test dithering 4](https://i.imgur.com/qqDMxjP.jpg)

As you can see, the Atkinson algorithms is clearly the big winner here, despites the very low resolution. I tweaked the color palette a little bit to have some nicer colors and here's the final result :

![test dithering 4](https://i.imgur.com/TO3WqDg.jpg)

#### Signal generation

I'm not a genius, I just [followed instruction](https://pysdr.org/index.html) on how to generate QPSK signal.

_the following is heavely inspired from everything on [pySdr](https://pysdr.org/index.html)_

#### Binaries
first, you have to encore your image to a succesion of 0 and 1. As I have only 4 color, I only need 2-bit to encode each pixel. So for a 200x200 image I only need 80000 bits (--> 10ko).

* blue  : x00
* red   : x01
* green : x10
* white : x11

#### Pulse generation
then, you have two convert your bits to pulses. In this example, 0 is -1 and 1 is 1. Each pulse is separated by 32 sample.

![pulses](https://pysdr.org/_images/pulse_shaping_python1.png)

since I'm using QPSK, this is a bit more complex :

![QPSK](https://i.imgur.com/UkgD7UZ.jpg)

each pulse is on a complex plane and works as follow :
* x00 : -1-1j
* x01 : -1+1j
* x10 : +1+1j
* x11 : +1-1j

#### Pulses shaping
once you have your complex pulses, you need to convolve everything with a _Root Raised-Cosine Filter_. This is a type of filter that reduce the Inter-Symbols Interference.

![RRC](https://pysdr.org/_images/rrc_rolloff.svg)

After the convolution, your pulses whould look something like this :

![pulses shaped](https://pysdr.org/_images/pulse_shaping_python3.svg)

#### Signal generation

Here's the magical part : you juste have to multiply your pulse by exp(-2j * frequency * t) !

What is does is __upconvert__ the baseband band signal we generated before to a more audible frequency. Just get rid of the complex part and you have a QPSK signal at 1800 Hz ready to be transmitted.

#### There you have it, a 200x200 picture in 2-bit transmitted using QPSK !

the program then save you a wonderfull sound file you can now use as you want!

### Receive mode

![hum...](https://media2.giphy.com/media/LRVnPYqM8DLag/giphy.gif?cid=ecf05e476ks0vgw8xl87cp50a3ell8rd2nlhat1tnaljr9pi&rid=giphy.gif&ct=g)

This was harder.

I first tried on GNU-radio to understand what I was doing while limiting the amount of coding needed. Here's the GNU-radio project that seems to give me excellent results.

![gnu-radio](https://i.imgur.com/hYsnreg.png)

Once here, I just tried to code everything in python.

![decoding](https://pysdr.org/_images/sync-diagram.svg)

#### Going backward (coarse Freq Sync)

first you have to bring back the signal to baseband by multiplying everything by exp(-2j * frequency * t) once again.

#### Matched filter

you then have to filter everything that is not the signal using the exact same RRC filter used in transmit mode.

#### Polyphase Clock Sync (time synchronization)

This section of my code is largely based on what you can find on [pysdr](https://pysdr.org/content/sync.html#time-synchronization). I used the Mueller and Muller clock recovery technique.

![time sync](https://pysdr.org/_images/time-sync-constellation.svg)
![time sync 2](https://pysdr.org/_images/time-sync-constellation-animated.gif)

#### Costas Loop (fine frequency and phase synchronization)

Once again, largely based on on [pysdr](https://pysdr.org/content/sync.html#fine-frequency-synchronization). This part will try to ajust for slight offset in phase and in frequency. This is the part that I'm less confident works really like it should.

![fine frequency sync](https://pysdr.org/_images/costas_animation.gif)

#### Frame detection

At this point, you should have your pulses back. You can now convert those back to binaries. You'll have a lot more binaries than what was really transmitted because each transmission begin with a synchronisation sequence and the Barker sequence.

This step looks in the binaries where the data really begin by searching the Barker sequence. Once found, the program just take the next 80000 bits !

#### convert bits back to a pretty image !

By using the same color palette than before, it's easy to convert bits to an image :)

# and voila !

In the future, I would like to developp an android app in order to be REALLY simple to use with avery radio by placing the phone near the microphone in transmit mode, and near the speaker in receive mode.
