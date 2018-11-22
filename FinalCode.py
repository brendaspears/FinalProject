import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pyaudio
import struct
from scipy.fftpack import fft
import time


class AudioStream(object):
    def __init__(self):

        # stream constants
        self.CHUNK = 1024 * 2               # Samples per Frame
        self.FORMAT = pyaudio.paInt16       # Audio Format
        self.CHANNELS = 1                   # Channel for Microphone
        self.RATE = 44100                   # Samples per Second
        self.pause = False

        # stream object to get data from microphone
        self.p = pyaudio.PyAudio()          # PyAudio class instance
        self.stream = self.p.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            output=True,
            frames_per_buffer=self.CHUNK )
        self.init_plots()
        self.start_plot()

    def init_plots(self):

        # variables for plotting
        samples = np.arange(0, 2 * self.CHUNK, 2)               # Samples (Waveform)
        frequencies = np.linspace(0, self.RATE, self.CHUNK)     # Frequencies (spectrum)

        # create matplotlib figure and axes
        self.fig, (ax1, ax2) = plt.subplots(2, figsize=(15, 7))
        self.fig.canvas.mpl_connect('button_press_event', self.onClick)

        # create a line object with random data
        self.line, = ax1.plot(samples, np.random.rand(self.CHUNK), color = 'aqua', lw=2)

        # create semilogx line for spectrum
        self.line_fft, = ax2.semilogx(
            frequencies, np.random.rand(self.CHUNK), color='aqua', lw=2)

        # format waveform axes
        ax1.set_title('Audio Waveform (Top) & Audio Spectrum (Bottom)')
        ax1.set_xlabel('Samples')
        ax1.set_ylabel('Volume')
        ax1.set_ylim(0, 255)
        ax1.set_xlim(0, 2 * self.CHUNK)
        ax1.set_facecolor('k')
        ax2.set_facecolor('k')
        plt.setp(
            ax1, yticks=[0, 128, 255],
            xticks=[0, self.CHUNK, 2 * self.CHUNK],
        )
        plt.setp(ax2, yticks=[0, 1],)

        # format spectrum axes
        ax2.set_xlim(20, self.RATE / 2)

        self.fig.patch.set_facecolor('rosybrown')

        # show axes
        plt.show(block=False)

    def start_plot(self):

        print('Stream Started')
        frame_count = 0
        start_time = time.time()

        while not self.pause:

            # Binary Data
            data = self.stream.read(self.CHUNK , exception_on_overflow = False)

            # Convert data to integers
            data_int = struct.unpack(str(2 * self.CHUNK) + 'B', data)

            # Create Numpy Array, offset by 128
            data_np = np.array(data_int, dtype='b')[::2] + 128

            self.line.set_ydata(data_np)        # Sets data to the audio waveform graph.

            # compute FFT and update line
            yf = fft(data_int)
            self.line_fft.set_ydata(
                np.abs(yf[0:self.CHUNK]) / (128 * self.CHUNK))

            # update figure canvas
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            frame_count += 1

        else:
            self.fr = frame_count / (time.time() - start_time)
            print('average frame rate = {:.0f} FPS'.format(self.fr))
            self.exit_app()

    def exit_app(self):
        print('Stream Closed')
        self.p.close(self.stream)

    def onClick(self, event):
        self.pause = True


# Calling the Class (main program)
AudioStream()
