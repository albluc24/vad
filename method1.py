import sys
import numpy as np
from utils import find_nonzero
from pydub import AudioSegment as audio
import librosa
from pysptk.sptk import rapt as swipe
audio_file=sys.argv[1]
output_file=sys.argv[2]
stride=0.01
sound=audio.from_file(audio_file)[:10000]
sound.set_frame_rate(16000).export("temp.wav",format="wav")
arr, sr=librosa.load("temp.wav")

framelen=int(stride*sr)
pitch=swipe(arr, sr, framelen)
sp=librosa.feature.mfcc(arr, sr, n_mfcc=13, hop_length=framelen, n_fft=framelen*2).T
breakpoint()
segments=find_nonzero(pitch)
for n in range(len(segments)):
    segments[n][0]=segments[n][0]*stride*1000
    segments[n][1]=segments[n][1]*stride*1000
newsegments=[]
#merge segments that have a gap between them of less than 1000ms
for n in range(len(segments)):
    if len(newsegments)==0: newsegments.append(segments[n]); continue
    if segments[n][0]-newsegments[-1][1]<=1000: newsegments[-1][1]=segments[n][1]
    else: newsegments.append(segments[n])
segments=newsegments
out=audio.empty()
for start, end in segments:
    out+=sound[start:end]
out.export(output_file,format="wav")