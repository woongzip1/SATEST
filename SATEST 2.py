import numpy as np
from matplotlib import pyplot as plt
import librosa as lr
import scipy
import os
import glob

""" Plot Window Functions in Time Domain """
sr = 16000
win_len = [0.005, 0.02, 0.04]

window_size = [int(e*sr) for e in win_len]
print(window_size)

rect_windows = []
hamm_windows = []

for size in window_size:
    rect_window = lr.filters.get_window(window='rectangular', Nx=size)
    rect_windows.append(rect_window)
    
for size in window_size:
    hamm_window = lr.filters.get_window(window='hamming', Nx=size)
    hamm_windows.append(hamm_window)

fig, axs = plt.subplots(2,3, figsize=(15,10))

# axs=axs.flatten()
# for i,window in enumerate(rect_windows):
#     print(i,window)
#     axs[i].plot(window)
#     axs[i].set_title(f"Rectangular window : {int(win_len[i]*1000)}ms")
#     # Axis 단위를 같도록 설정
#     axs[i].set_xlim(0, sr*win_len[-1]+50)
# plt.show()

for i, (rect_window, hamm_window) in enumerate(zip(rect_windows, hamm_windows)):
    ## 여기 이상해
    time = np.linspace(0,len(rect_window),len(rect_window),endpoint=False)/sr * 1000
    # time = lr.times_like(rect_window, sr=sr)
    
    axs[0, i].plot(time,rect_window, label='Rectangular')
    axs[1, i].plot(time,hamm_window, label='Hamming', linestyle='--')
    axs[0, i].set_title(f"Rectangular window: {int(win_len[i]*1000)}ms")
    axs[1, i].set_title(f"Hamming window: {int(win_len[i]*1000)}ms")

# 모든 subplot에 공통된 x축, y축 레이블 추가
for ax in axs.flat:
    ax.set(xlabel='Time(ms)', ylabel='Amplitude')

# subplot들 간의 간격 조정
plt.tight_layout()

# 범례 추가
axs[0, 0].legend()
axs[1, 0].legend()

plt.show()