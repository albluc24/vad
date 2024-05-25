import sys, torch
from s3prl import hub
import numpy as np
from utils import find
from joblib import dump
from pydub import AudioSegment as audio
from sklearn.mixture import GaussianMixture
#using kmeans instead of GMM
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import librosa
from pysptk.sptk import rapt as swipe
audio_file=sys.argv[1]
output_file=sys.argv[2]
model=hub.modified_cpc()
stride=0.01
sound=audio.from_file(audio_file)[:5000]
sound.set_frame_rate(16000).export("temp.wav", format="wav")
arr, sr=librosa.load("temp.wav")
framelen=int(stride*sr)
pitch=swipe((arr*32767).astype(np.int16), sr, framelen)
breakpoint()
features=model(arr)
#features=np.hstack((sp, pitch.reshape(-1,1)))
scaler = StandardScaler()
#features_scaled = scaler.fit_transform(features)
gmm = GaussianMixture(n_components=2, random_state=42)
#gmm=KMeans(n_clusters=2, random_state=42)
gmm.fit(features)
labels = gmm.predict(features_scaled)
#segments=find_nonzero(pitch)
segments=find(labels, lambda x: x==0)
for n in range(len(segments)):
    segments[n][0]=int(segments[n][0]*stride*1000)
    segments[n][1]=int(segments[n][1]*stride*1000)
out=audio.empty()
for start, end in segments:
    if end>=len(sound): end=len(sound)
    out+=sound[int(start):int(end)]
out.export(output_file,format="wav")