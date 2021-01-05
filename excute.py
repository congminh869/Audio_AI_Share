from pathlib import Path
# code_path=Path('E:\Google Driver')
parameter_file=open("/home/pi/audio/testAI/parameter.txt",'r')
code_path=parameter_file.readline().splitlines()[0]
#code_path=Path(code_path[0])
#code_path=Path('/home/pi/Documents/audioAI/')
#code_path is the path name to the directory containing the ESCA software package. 
#example: D:\06.Code\Quoc\realtime_system\ESCA_realtime_package
is_test_mode = int(parameter_file.readline())
#is_test_mode =1 is test mode (run the test.wav file); =0 is realtime mode
#timesleep = int(parameter_file.readline())
#timesleep is the time (minute) for waiting for the file to appear (the file that was sent from VoiceIP-Terminal)
max_time_run = int(parameter_file.readline())
#max_time_run is the maximum minute that this software run in realtime mode, We set this limits to avoid computer overflow memory
parameter_file.close()
is_current_frame_have_abnormal=0
threshold_predict=450
is_mono=True
seq_len = 5
sr=44100 
nb_mel_bands = 40
nfft = 512
win_len = nfft
hop_len = win_len / 2
import numpy as np
#import scipy.io
#from scipy.io import wavfile
import librosa.core,librosa.filters
import wave
from keras.models import load_model,model_from_json
import os
import glob
import csv
import time
from sklearn import preprocessing
import time
from keras.models import model_from_json
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
#---------------------------------------
nb_ch = 1  if is_mono else 2
# batch_size = 256    # Decrease this if you want to run on smaller GPU's
cnt=0
eps = np.finfo(np.float).eps
def load_audio(filename, mono=True, fs=44100):
    file_base, file_extension = os.path.splitext(filename) #split filename out of path name
    if file_extension == '.wav':
        _audio_file = wave.open(filename)

        # Audio info
        sample_rate = _audio_file.getframerate()
        sample_width = _audio_file.getsampwidth()
        number_of_channels = _audio_file.getnchannels()
        number_of_frames = _audio_file.getnframes()

        # Read raw bytes
        data = _audio_file.readframes(number_of_frames)
        _audio_file.close()

        # Convert bytes based on sample_width
        num_samples, remainder = divmod(len(data), sample_width * number_of_channels)
        if remainder > 0:
            raise ValueError('The length of data is not a multiple of sample size * number of channels.')
        if sample_width > 4:
            raise ValueError('Sample size cannot be bigger than 4 bytes.')

        if sample_width == 3:
            # 24 bit audio
            a = np.empty((num_samples, number_of_channels, 4), dtype=np.uint8)
            raw_bytes = np.fromstring(data, dtype=np.uint8)
            a[:, :, :sample_width] = raw_bytes.reshape(-1, number_of_channels, sample_width)
            a[:, :, sample_width:] = (a[:, :, sample_width - 1:sample_width] >> 7) * 255
            audio_data = a.view('<i4').reshape(a.shape[:-1]).T
        else:
            # 8 bit samples are stored as unsigned ints; others as signed ints.
            dt_char = 'u' if sample_width == 1 else 'i'
            a = np.fromstring(data, dtype='<%s%d' % (dt_char, sample_width))
            audio_data = a.reshape(-1, number_of_channels).T

        if mono:
            # Down-mix audio
            audio_data = np.mean(audio_data, axis=0)

        # Convert int values into float
        audio_data = audio_data / float(2 ** (sample_width * 8 - 1) + 1)

        # Resample
        if fs != sample_rate:
            audio_data = librosa.core.resample(audio_data, sample_rate, fs)
            sample_rate = fs

        return audio_data, sample_rate
    return None, None

def split_in_seqs(data, subdivs):
    if len(data.shape) == 1:
        if data.shape[0] % subdivs:
            data = data[:-(data.shape[0] % subdivs), :]
        data = data.reshape((int(data.shape[0] / subdivs), subdivs, 1))
    elif len(data.shape) == 2:
        if data.shape[0] % subdivs:
            data = data[:-(data.shape[0] % subdivs), :]
        data = data.reshape((int(data.shape[0] / subdivs), subdivs, data.shape[1]))
    elif len(data.shape) == 3:
        if data.shape[0] % subdivs:
            data = data[:-(data.shape[0] % subdivs), :, :]
        data = data.reshape((int(data.shape[0] / subdivs), subdivs, data.shape[1], data.shape[2]))
    return data
def split_multi_channels(data, num_channels):
    in_shape = data.shape
    if len(in_shape) == 3:
        hop = int(in_shape[2] / num_channels)
        tmp = np.zeros((in_shape[0], num_channels, in_shape[1], int(hop)))
        for i in range(num_channels):
            tmp[:, i, :, :] = data[:, :, i * hop:(i + 1) * hop]
    else:
        print("ERROR: The input should be a 3D matrix but it seems to have dimensions ", in_shape)
        exit()
    return tmp
def extract_mbe(_y, _sr, _nfft, _nb_mel): #this function extract mel band energy feature
    # input: _y: time series signal        
    # print("y.shape:{}".format(_y.shape)) # y.shape:(441000,)
    #spec, n_fft = librosa.core.spectrum._spectrogram(y=_y, n_fft=_nfft, hop_length=int(_nfft/2), power=1)
    spec = np.abs(librosa.core.stft(y=_y, n_fft=_nfft, hop_length=int(_nfft/2)))
    # f,t,spec=signal.stft(_y, fs=44100, nperseg=_nfft,noverlap=int(_nfft/2),boundary=None)
    mel_basis = librosa.filters.mel(sr=_sr, n_fft=_nfft, n_mels=_nb_mel)
    # mel_basis.shape:(40, 1025)
    return (np.dot(mel_basis, spec))
def feature_extraction():
    mbe = None
    if is_mono: 
        mbe = extract_mbe(audio_data, sr, nfft, nb_mel_bands).T				
    else:
        for ch in range(audio_data.shape[0]):
            mbe_ch = extract_mbe(audio_data[ch, :], sr, nfft, nb_mel_bands).T
            # print("mbe_ch.shape:{}".format(mbe_ch.T.shape))
            if mbe is None:
                mbe = mbe_ch
            else:
                mbe = np.concatenate((mbe, mbe_ch), 1)
    return mbe

# model_folder=code_path //
def main_process_AE(mbe):
    # load json and create model
    json_file = open(str(code_path) +'/AE.json', 'r')
    loaded_AE_json = json_file.read()
    json_file.close()
    print("1")
    AE = model_from_json(loaded_AE_json)
    # load weights into new model
    AE.load_weights(str(code_path)+'/AE.h5')
    print(mbe.shape,"   mbe.shape")
    #scaler = preprocessing.StandardScaler()
    #mbe = scaler.fit_transform(mbe)

    mbe = split_in_seqs(mbe, seq_len) #Ex:(1879, 40) and seq_len=5- >output: (375, 5, 40)
    print(mbe.shape,"   mbe.shape")
    mbe = split_multi_channels(mbe, nb_ch)
    print(mbe.shape,"   mbe.shape")
    print('AE.layers[0].input_shape[1]', AE.layers[0].input_shape[1])
    mbe = mbe.reshape(-1,AE.layers[0].input_shape[1]) #Ex:(375, 200) 
    #print("pred.shape ", pred.shape) #(375, 200)
    #print("np.subtract(mbe,pred) .shape", np.subtract(mbe,pred).shape)  #(375, 200)
    #print("np.square(np.subtract(mbe,pred)).shape ", np.square(np.subtract(mbe,pred)).shape) #(375, 200)
    pred = AE.predict(mbe)
    mse=np.mean(np.square(np.subtract(mbe,pred)),axis=-1)
    print("12")
    #print("mse.shape ", mse.shape) #(375, )
    return mse		
def graph_output(pred):
    change_2_time = ((seq_len * nfft * 1000/ 2) / sr);

    pred_x_var_norm_time = np.empty([1, int(pred.shape[0]*change_2_time)])
    for i in range(int(pred.shape[0])):
      pred_x_var_norm_time[0, int(i*change_2_time): int((i+1)*change_2_time)] = pred[i]

    range_to_plot = int(1*59800)
    time_start = 0 *60000
    # pred_x_var_norm_time[-1, 0:10] = 0
    x = [datetime.datetime(2019, 12, 19, 0, 0, 0, 0) + datetime.timedelta(milliseconds=i) for i in range(time_start, time_start + range_to_plot)]
    plt.figure(1);
    line2, = plt.plot(x, pred_x_var_norm_time.reshape(-1)[time_start:time_start + range_to_plot], label="mse")
    plt.legend([line2], ["mse"])

    plt.suptitle('MSE of Auto-Encoder Model')
    plt.xlabel('time')
    plt.ylabel('mse')

    plt.gcf().autofmt_xdate()
    myFmt = mdates.DateFormatter('%H:%M:%S')
    plt.gca().xaxis.set_major_formatter(myFmt)
    plt.show()			
			
def save_output(pred):
    name_output_file = file_name +'.csv'
    timestamp=np.linspace(0,10,num=pred.shape[0],endpoint=False)
    timestamp=np.round(timestamp,3)
    #pred=np.round(pred,3)
    pred_threshold=np.multiply(pred>threshold_predict, 1)   # also convert True to 1
    is_current_frame_have_abnormal=np.max(pred_threshold)
    Data = np.concatenate((timestamp.reshape(-1,1),pred.reshape(-1,1),pred_threshold.reshape(-1,1)),axis=1)
    # print("done",Data.shape)
    # with open(code_path+'/data/output_realtime/'+file_name+'.csv', 'w') as csvFile:
    # with open(code_path+'/data/output_realtime/'+file_name+'.csv', 'w') as csvFile:
    # with open(code_path+'\\data\\output_realtime\\'+file_name+'.csv', 'w') as csvFile:
    # with open(code_path+'\\data\\output_realtime\\'+file_name+'.csv', 'w') as csvFile:
    # with open(code_path+'\\data\\output_realtime\\'+file_name+'.csv', 'w') as csvFile:
    with open(str(code_path)+'/data/output_realtime/'+ name_output_file, 'w') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(Data)
    csvFile.close()
    return is_current_frame_have_abnormal
# np.save(code_path+'/data/'+name_output_file, pred)
# new_file_name = 'Room1{}'.format(time.strftime("%Y%m%d%H%M"))# Room1201908160939
# new_file_name=code_path+'/data/data_wav_sin/p2.wav'
# new_file_name='D:\\Anaconda\\share\\jupyter\\lab\\SPARCLab_201912070944.wav'

  
#     D:\Anaconda\share\jupyter\lab\VoipApp\Test_byQ\SPARCLab1
start_time = time.time()
current_file = ''
check = '0'
# check_file = open("/home/pi/Documents/audioAI/data/input_file/check.txt",'r')
# check = check_file.readline()
# check_file.close()
path = '/home/pi/audio/testAI/data/input_file/'
while True:#cnt<max_time_run:
    if (is_test_mode):
        full_new_file_name='/home/pi/Documents/audioAI/data/input_file/Room20201231004218.wav'
        #full_new_file_name=code_path+'/test.wav'
        file_name = 'Room20201231004218.wav'
        audio_data, sr = load_audio(str(full_new_file_name), mono=is_mono, fs=sr)
        mbe=feature_extraction()
        mse=main_process_AE(mbe)
        is_current_frame_have_abnormal=save_output(mse)
        cnt=cnt+1
        print("%d file has been processed" %cnt)
        print("--- %s seconds ---" % (time.time() - start_time))
        print("is_current_frame_have_abnormal = %d " %is_current_frame_have_abnormal)
    else:
        start_time = time.time()
        if(len([f for f in os.listdir(path)if os.path.isfile(os.path.join(path, f))]) > 1):
            read_file = open("/home/pi/audio/testAI/data/input_file/check.txt",'r')
            k_check = read_file.readline()
            read_file.close()
            list_of_files = glob.glob('/home/pi/audio/testAI/data/input_file/*wav') # * means all if need specific format then *.csv
            latest_file = max(list_of_files, key=os.path.getctime)
            if(k_check != check):
                check = k_check
                if(current_file < latest_file):
                    current_file = latest_file
                    file_name=os.path.basename(current_file)
                    full_new_file_name = current_file
                    check_file = open("/home/pi/audio/testAI/data/input_file/check.txt",'w')
                    check_file.write('0')
                    check_file.close()
        # full_new_file_name=code_path+'\\VoipApp\\Test_byQ\\SPARCLab1\\'+file_name+'.wav'
        #full_new_file_name=code_path+'/VoipApp/Test_byQ/SPARCLab1/'+file_name+'.wav'
        #full_new_file_name=code_path/'VoipApp'/'Test_byQ'/'SPARCLab1'/(file_name+'.wav')
    #time.sleep(timesleep)
#     D:\Anaconda\share\jupyter\lab\VoipApp\Test_byQ\SPARCLab1
    ######################## load audio   ################################
                    audio_data, sr = load_audio(str(full_new_file_name), mono=is_mono, fs=sr)
    ######################## feature extraction   ################################
                    mbe=feature_extraction() # mbe.shape (1879,40)
    #normalization in main_process block
    ######################## main process  ################################
                    mse=main_process_AE(mbe)   #(375,)
    ######################## save output  ################################
                    is_current_frame_have_abnormal=save_output(mse)
                    cnt=cnt+1
                    print("%d file has been processed" %cnt)
                    print("--- %s seconds ---" % (time.time() - start_time))
                    print("is_current_frame_have_abnormal = %d " %is_current_frame_have_abnormal)
    #graph_output(mse)
                    if (is_current_frame_have_abnormal):        
                        print("!!! DETECTED ABNORMAL SOUND !!! in file %s" %file_name)
                        
    
print("End program %d")
print("--- %s seconds ---" % (time.time() - start_time))

