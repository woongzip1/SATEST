
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
        self.hann_window = lr.filters.get_window(window="hann", Nx=self.win_len)


    def extract_frames(self, win_type="rectangular"):
        # Make window
        
        print(win_type)
        frames=[]
        for n in range(0, len(self.function), self.hop_len):
            # 마지막에서 끝을 clip하고 난 후 return 한다
            if n + self.win_len > len(self.function):
                break
            
            region = self.function[n:n + self.win_len]
            if win_type == "hamming":
                region = region * self.hamm_window
            elif win_type == "hann":
                region = region * self.hann_window
            # elif win_type == "rectangular":
            #     region = region * self.rect_window
            
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

## Auto Correlation Sequence with signal length
def auto_corr(signal):
    corr = np.correlate(signal, signal, mode='full')
    corr = corr[len(corr)//2:]
    return corr 

# My Derbin's Algorithm
def derbin(r, p):
    E = np.zeros(p+1)
    a = np.zeros((p+1,p+1))
    
    a[0][0] = 1
    E[0] = r[0]

    for i in range(1,p+1):
        ## sigma
        j=1
        sumj = 0
        while(j <= i-1):
            sumj+=a[i-1][j]*r[i-j]
            j += 1
        
        k_i = (r[i] - sumj) / E[i-1] 
        a[i][i] = k_i

        ## i-order 새로운 coeff 갱신
        for j in range(1,i):
            a[i][j] = a[i-1][j] - k_i * a[i-1][i-j]
            
        E[i] = (1 - k_i**2)*E[i-1]
        coeff = a[p][1:]
    return coeff,E

# Calculate LPC Coefficients in the Frame
def LPC(frame, order=10):
    coeff_arr = np.zeros(order)
    # error
    if len(frame) < order:
        print('frame is longer than order')
        return -1
    # Tx = b
    coeff, err = derbin(auto_corr(frame),p=order)
    return coeff

## LPC with Direct Matrix Inverse
def LPC_inv(frame, order=10):
    coeff_arr = np.zeros(order)
    # error
    if len(frame) < order:
        print('frame is longer than order')
        return -1
    # Tx = b
    ac = auto_corr(frame)[:order]
    mat_T = make_toeplitz(ac)
    vec_b = auto_corr(frame)[1:order+1]
    coeff_arr  = np.dot(np.linalg.inv(mat_T),vec_b)
    return coeff_arr


# Make Toeplitz Matrix using Auto Correlation
def make_toeplitz(ac):
    p = len(ac)
    toeplitz_mat = np.zeros((p,p))
    ac_flip = ac[::-1][:-1]
    
    for i in range(p):
        toeplitz_mat[i,:] = np.concatenate((ac_flip[p-i-1:],ac[:p-i]))
        

    return toeplitz_mat

# Derbin's Algorithm (As a reference)
def ref_derbin(r, order):
    # r : 1-D auto corr array
    a = np.zeros((order+1,order+1))
    # store prediction error for each step
    E = np.zeros(order+1)
    # First coeff
    a[0][0] = 1
    # Initial prediction error : power
    E[0] = r[0]
    
    # iterate from 1 to order p 
    for i in range(1,order+1):
        sum_j = sum(a[i-1][j] * r[i-j] for j in range(1,i))
        k_i = (r[i] - sum_j ) / E[i-1]
        
        # Update coefficeints for current step
        a[i][i] = k_i
        for j in range(1,i):
            a[i][j] = a[i-1][j] - k_i * a[i-1][i-j]
            
        #Update Error
        E[i] = (1-k_i**2) * E[i-1]
        # print("i={}, ki={}".format(i,k_i))
    # Extract final coeff, exclude a0    
    coeff = a[order][1:]
    return coeff,E
