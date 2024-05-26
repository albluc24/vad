import sys, torch
torch.set_grad_enabled(False)
from s3prl import hub
import numpy as np
from utils import find
from joblib import dump
from pydub import AudioSegment as audio
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
#using kmeans instead of GMM
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import librosa
from soundfile import read
from pysptk.sptk import rapt as swipe
audio_file=sys.argv[1]
output_file=sys.argv[2]
model=hub.tera()
stride=0.01
sound=audio.from_file(audio_file)[:60000*5]
sound.set_frame_rate(16000).export("temp.wav", format="wav")
arr, sr=read('temp.wav')
framelen=int(stride*sr)
pitch=swipe((arr*32767).astype(np.int16), sr, framelen)
aud=torch.tensor(arr)
features=model([aud.to(torch.float32)])['last_hidden_state'][0].to(torch.float64).numpy()
pca=PCA(500)#20without scaling 30 without scaling
features=pca.fit_transform(features)
features=np.hstack((features[:LEN(PITCH)], pitch.reshape(-1,1)))
scaler = StandardScaler()
#features_scaled = scaler.fit_transform(features)
features=features_scaled
gmm = GaussianMixture(n_components=2, random_state=42, verbose=1)
#gmm=KMeans(n_clusters=2, random_state=42)
gmm.fit(features)
labels = gmm.predict(features)
silence=int(labels[0])
#segments=find_nonzero(pitch)
segments=find(labels, lambda x: x!=silence)
for n in range(len(segments)):
    segments[n][0]=int((segments[n][0]*stride*1000))
    segments[n][1]=int((segments[n][1]*stride*1000))
#if 2 segments have a gap <=300 ms then merge them
newsegments=[]
for n in range(len(segments)):
    if n==0: newsegments.append(segments[n]); continue
    if segments[n][0]-newsegments[-1][1]<=300: newsegments[-1][1]=segments[n][1]
    else: newsegments.append(segments[n])
segments=newsegments
out=audio.empty()
for start, end in segments:
    if end>=len(sound): end=len(sound)
    if start<0: start=0
    out+=sound[int(start):int(end)]
out.export(output_file,format="wav")