import numpy as np
def find(arr, condition):
    # Apply the condition to the array and get indices where the condition is true
    condition_indices = np.where(condition(arr))[0]
    # Initialize list to hold start and end indices of continuous blocks
    continuous_blocks = []
    
    if condition_indices.size > 0:
        # Starting index of the first block where condition is true
        start_index = condition_indices[0]

        # Loop over the indices where condition is true to find continuous blocks
        for i in range(1, len(condition_indices)):
            # Check if current index is not consecutive
            if condition_indices[i] != condition_indices[i - 1] + 1:
                # Record the end index of the previous block
                end_index = condition_indices[i - 1]
                # Add the start and end to the list
                continuous_blocks.append([start_index, end_index])
                # Update the start index to current
                start_index = condition_indices[i]
        
        # Add the last block
        continuous_blocks.append([start_index, condition_indices[-1]])
    
    return continuous_blocks


def find_nonzero(arr):
    # Create a list with segment start and end 
    nonzero_indices = np.nonzero(arr)[0]
    # Initialize list to hold start and end indices of nonzero blocks
    nonzero_blocks = []
    
    if nonzero_indices.size > 0:
        # Starting index of the first nonzero block
        start_index = nonzero_indices[0]

        # Loop over the nonzero indices to find continuous blocks
        for i in range(1, len(nonzero_indices)):
            # Check if current index is not consecutive
            if nonzero_indices[i] != nonzero_indices[i - 1] + 1:
                # Record the end index of the previous block
                end_index = nonzero_indices[i - 1]
                # Add the start and end to the list
                nonzero_blocks.append([start_index, end_index])
                # Update the start index to current
                start_index = nonzero_indices[i]
        
        # Add the last block
        nonzero_blocks.append([start_index, nonzero_indices[-1]])
    
    return nonzero_blocks
def wvad(file_path, frame_size=10, padding=100, aggressiveness=3, cleanup_temp=True):
    name=file_path.split('/')[-1]
    if exists(f'{cache_path}/{name}silences.dat'): segments=load(f'{cache_path}/{name}silences.dat')
    else:
        tempfiles=create_temporary_audio(file_path, [16000, 32000])
        sound, sr=vad.read_wave(tempfiles[0])
        vadobj=vad.webrtcvad.Vad(aggressiveness)
        frames=vad.frame_generator(frame_size, sound, sr)
        frames=list(frames)
        segments=[]
        for s,e in vad.vad_collector(sr, frame_size, padding, vadobj, frames):
            s=int(s*1000)
            e=int(e*1000)
            if len(segments)==0: segments.append([s, e]); continue
            if e-s<=1000: segments[-1][1]=e; continue
            segments.append([s, e])
        dump(segments, f'{cache_path}/{name}silences.dat')
    if cleanup_temp:
        cleanup_temporary_audio(file_path)
        try: os.remove('tosplit.wav')
        except: pass
    return segments
