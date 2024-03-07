import numpy as np
from matplotlib import pyplot as plt
import librosa as lr
import scipy
import os
import glob
from scipy.signal import argrelmin

"""
목표: 
주어진 시간 단위를 갖는 rect, hamming window frequency domain에서 생성하기

진행 과정:
1. 각 함수를 얻은 후, 양쪽에 zero를 padding 하여 충분한 길이의 list 생성
2. 충분한 길이를 얻은 list에 바로 fft를 수행
3. fftshift를 수행하여 가운데 값이 0이 되고, 양 끝이 +-pi 가 되도록 축 이동
"""
plt.close('all')

sr = 16000
win_len = [0.005, 0.02, 0.04]

window_size = [int(e*sr) for e in win_len]
print(window_size)

### Get Windows
rect_windows = []
hamm_windows = []

rect_windows_pad = []
hamm_windows_pad = []

for size in window_size:
    rect_window = lr.filters.get_window(window='rectangular', Nx=size)
    rect_windows.append(rect_window)
    
for size in window_size:
    hamm_window = lr.filters.get_window(window='hamming', Nx=size)
    hamm_windows.append(hamm_window)

### Pad Windows
len_sec = 0.1 
dftlen = int(len_sec* sr)

# dftlen 길이를 갖는 time 축 생성
time = np.linspace(-len_sec/2, len_sec/2, dftlen, endpoint=False)
time = time * 1000
print(len(time))


fig, axs = plt.subplots(2,3, figsize=(15,10))
for i, (rect_window, hamm_window) in enumerate(zip(rect_windows, hamm_windows)):
    # list의 양쪽에 zero-pad
    rect_window = np.pad(rect_window, (dftlen-len(rect_window))//2, mode="constant", constant_values=0)
    hamm_window = np.pad(hamm_window, (dftlen-len(hamm_window))//2, mode="constant", constant_values=0)
    
    rect_windows_pad.append(rect_window)
    hamm_windows_pad.append(hamm_window)
    
    axs[0, i].plot(time,rect_window, label='Rectangular')
    axs[1, i].plot(time,hamm_window, label='Hamming', linestyle='--')
    axs[0, i].set_title(f"Rectangular window: {int(win_len[i]*1000)}ms")
    axs[1, i].set_title(f"Hamming window: {int(win_len[i]*1000)}ms")
    

for ax in axs.flat:
    ax.set(xlabel='Time(ms)', ylabel='Amplitude')


#### 길이 1600 의 DFT array 생성하기
freq = np.fft.fftshift(np.fft.fftfreq(dftlen, d=1/sr)) /sr * 2 * np.pi



rect_ffts = [np.fft.fftshift(np.fft.fft(rect_window_pad)) for rect_window_pad in rect_windows_pad]


rect_mag = [np.abs(i) for i in rect_ffts]
rect_mag_log = [np.log(i+1) for i in rect_mag]

hamm_ffts = [np.fft.fftshift(np.fft.fft(hamm_window_pad)) for hamm_window_pad in hamm_windows_pad]

hamm_mag = [np.abs(i) for i in hamm_ffts]
hamm_mag_log = [np.log(i+1) for i in hamm_mag]

fig, axs = plt.subplots(2,3, figsize=(15,10))

for i,(rect, hamm) in enumerate(zip(rect_mag,hamm_mag)):

    axs[0, i].plot(freq,rect, label='Rectangular')
    axs[1, i].plot(freq,hamm, label='Hamming')
    
    axs[0, i].set_title(f"Rectangular window: {int(win_len[i]*1000)}ms")
    axs[1, i].set_title(f"Hamming window: {int(win_len[i]*1000)}ms")
    
    
for ax in axs.flat:
    ax.set(xlabel='Frequency($\pi$)', ylabel='log(|H(w)|)')
    ax.set_xticks([-np.pi, 0, np.pi])
    ax.set_xticklabels(['$-\pi$', '0', '$\pi$'])
    
plt.show()
    
# for rect in rect_mag :
#     maxindex = np.argmax(rect)
#     minindex = argrelmin(rect)
#     print(maxindex)
#     print(rect.shape)
#     print(minindex)

# print("Hamm")
# for hamm in hamm_mag :
#     maxindex = np.argmax(hamm)
#     minindex = argrelmin(hamm)
#     print(maxindex)
#     print(rect.shape)
#     print(minindex)

