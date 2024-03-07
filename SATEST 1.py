import numpy as np
from matplotlib import pyplot as plt
import librosa as lr
import scipy
import os
import glob
import soundfile as sf


"""
Customization 
1. folder_path 알맞게 수정하기

추가적으로 해야 할 일
1. Load 한 오디오 파일을 .wav 파일로 저장하기
2. 그래프의 Axis 축 단위 제대로 표기하기
3. 삼/사 의 음소 단위 구분 지역 고민해보기
"""

target_sr = 16000
folder_path = "audio"

# audio 폴더 내의 모든 파일 찾기
mp3_files = glob.glob(os.path.join(folder_path,'*.mp3'))
print(mp3_files)

# subplot
# fig, axs = plt.subplots(2,4, figsize=(15,10))
# axs = axs.flatten()

for i, mp3_file in enumerate(mp3_files):
    
    fig,axs = plt.subplots(figsize=(15,5))

    file_path = mp3_file
    
    # Audio Load
    yr, sr = lr.load(file_path)
    print(f"Loaded: {file_path}, Shape: {np.array(yr).shape}, Original sr: {sr}")

    # Resampling
    yr_resampled = lr.resample(yr, orig_sr=sr, target_sr=target_sr)
    yr_resampled = yr_resampled[12000:-4000]
    print(f"Resampled Shape: {np.array(yr_resampled).shape}")

    # Time
    time = np.linspace(0,len(yr_resampled),len(yr_resampled),endpoint=False) / target_sr
    
    # Plot Graph
    axs.plot(time, yr_resampled)
    axs.set_title(f"Resampled Audio : {os.path.basename(mp3_file)}")
    axs.set_xlabel("Time(s)")
    axs.set_ylabel("Amplitude")
    
    # Write into wav 
    result_folder = "results"
    save_path = os.path.join(result_folder,os.path.basename(mp3_file)[:-4]+".wav")
    
    # print(save_path)
    sf.write(save_path, yr_resampled, target_sr)
    
    # Save Figures
    fig.savefig(os.path.join(result_folder,os.path.basename(mp3_file)[:-4]+".png"))
    
    
    

plt.show()
# print(yr_resampled.dtype)

## .wav 파일로 write 하기 해야 함



