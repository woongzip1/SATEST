import numpy as np
import librosa as lr
import matplotlib.pyplot as plt
import soundfile as sf

"""
SATEST 3
Short-Time Energy
Zero-Crossing Rate
"""
sr = 16000
win_time = 0.02
win_len = int(win_time * sr)

print(win_len)

### Read Audio Files
file_path = "results/sa.wav"

yr,ori_sr = lr.load(file_path,sr=sr)
print(f"Loaded: {file_path}, Shape: {np.array(yr).shape}, Original sr: {ori_sr}")

# # Resample
# yr = lr.resample(yr, orig_sr=ori_sr, target_sr=sr)
# print(f"Loaded: {file_path}, Shape: {np.array(yr).shape}, Original sr: {sr}")

# Time axis
time = np.linspace(0,len(yr),len(yr),endpoint=False)/sr
print(len(time))

###### Hamming Window with 320 samples
hamm_window = lr.filters.get_window(window="hamming", Nx=win_len)
rect_window = lr.filters.get_window(window='rectangular',Nx=win_len)
hop_len = int(win_len * 0.5)

print(win_len)
print(hop_len)

## Short Time Energy Calculation
ste_arr= np.zeros_like(yr)
for n in range(0,len(yr)-win_len+1,hop_len):
    sum=0
    for m in range(max(0,n-win_len+1), n+1):
        sum += (yr[m] * hamm_window[n-m])**2
    ste_arr[n] = sum

## Zero Crossing Rate
zcr_arr= np.zeros_like(yr)
for n in range(0,len(yr)-win_len+1,hop_len):
    sum=0
    for m in range(max(0,n-win_len+1), n+1):
        sum += (np.abs(np.sign(yr[m])-np.sign(yr[m-1]))) * 0.5 * rect_window[n-m]
    zcr_arr[n] = sum

## Plot
fig, axs = plt.subplots(3,1, figsize = (30,5))
axs = axs.flatten()
axs[0].plot(time, yr)
axs[1].plot(time,ste_arr)
axs[2].plot(time,zcr_arr)


title_list = ["Time Plot","STE","ZCR"]
for i,ax in enumerate(axs):
    ax.set(xlabel="Time(s)", ylabel="Amplitude")
    ax.set_title(title_list[i])

plt.show()    