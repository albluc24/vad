import sys
from pydub import AudioSegment as audio
from scipy.io import wavfile
from pysptk.sptk import swipe
audio_file=sys.argv[1]
output_file=sys.argv[2]
sound=audio.from_file(audio_file)
sound.set_frame_rate(16000).export("temp.wav",format="wav")
arr, sr=wavfile.read("temp.wav")
result=swipe(arr, sr, 0.01*sr)
breakpoint()