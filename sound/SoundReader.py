
# coding: utf-8

# In[1]:
import struct
import matplotlib.pyplot as plt

# Use pydub for AAC/m4a decode
# install pydub: pip install pydub
# install ffmpeg: brew install ffmpeg --with-libvorbis --with-ffplay --with-theora
from pydub import AudioSegment


# In[2]:

def readAAC(path):
    sig = AudioSegment.from_file(path, format="mp4")
    return sig.frame_rate, sig.get_array_of_samples()

def readRawAAC(path):
    sig = AudioSegment.from_file(path, format="aac")
    return sig.frame_rate, sig.get_array_of_samples()
    
def readWave(path):
    sig = AudioSegment.from_file(path, format="wav", channels=1)
    sig = sig.split_to_mono()[0]
    return sig.frame_rate, sig.get_array_of_samples(), sig

def read3GP(path):
    sig = AudioSegment.from_file(path, format="3gp")
    return sig.frame_rate, sig.get_array_of_samples()


# In[ ]:
