
## Utility Functions to Use

import librosa as lr
import numpy as np


class FrameExtractor:
    def __init__(self, function, win_len, hop_len):
        self.function = function
        self.win_len = win_len
        self.hop_len = hop_len
        self.total_frames = 0
        self.hamm_window = lr.filters.get_window(window="hamming", Nx=self.win_len)
        self.rect_window = lr.filters.get_window(window="rectangular", Nx=self.win_len)
        
    def extract_frames(self, win_type="rectangular"):
        # Make window
        
        frames=[]
        for n in range(0, len(self.function), self.hop_len):
            # 마지막에서 끝을 clip하고 난 후 return 한다
            if n + self.win_len > len(self.function):
                break
            
            region = self.function[n:n + self.win_len]
            if win_type == "hamming":
                region = region * self.hamm_window
            
            self.total_frames += 1
            frames.append(region)
        print(f"From {len(self.function)} samples, total {self.total_frames} frames are generated")
        return frames
    
    # input function의 sample no 를 넣어, 해당하는 제일 작은 frame index를 찾아주는 함수
    # frame index 와 rectangular windowed region 을 반환한다
    def frame_index_finder(self, search_index):
        frameindex =0
        for n in range(0, len(self.function), self.hop_len):
            if (search_index >= n) and (search_index < n+self.win_len):
                region = self.function[n:n + self.win_len]
                region = region * self.rect_window
                break
            frameindex += 1 
            # n == hoplen * index : hoplen * index + winlen
        return frameindex, region
        
    #win, hop len은 class 생성할 때 이미 지정함
    def STFT(self,win_type="rectangular",dft_len=512):
        # For every frames
        frames = self.extract_frames(win_type)
        specgram = np.zeros([dft_len//2 +1, self.total_frames],dtype=complex)
        for frameindex,frame in enumerate(frames):
            # 각 frame 마다 FFT 실행 후 0 - 0.5Fs 추출
            freqbin = (np.fft.fftshift(np.fft.fft(frame,dft_len)))
            # ttt = freqbin[:len(freqbin)//2 +1]
            specgram[:,frameindex] = freqbin[:len(freqbin)//2 +1]
        return specgram

# print('Hi')
