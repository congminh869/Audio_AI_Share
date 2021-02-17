import pyaudio
import wave
import datetime
import time

chunk = 4096  # Record in chunks of 4096 samples
sample_format = pyaudio.paInt16  # 16 bits per sample
print(sample_format)
channels = 1
rate = 44100  # Record at 44100 samples per second
seconds = 10
while True:
    x = datetime.datetime.now()
    filename = "Room"+ x.strftime("%Y%m%d%H%M%S") + ".wav"
    
    p = pyaudio.PyAudio() 
    print(filename)
    # help(p)
    print('Recording')

    stream = p.open(format=sample_format,
                    channels=channels,
                    rate=rate,
                    frames_per_buffer=chunk,
                    input_device_index=1,
                    output = False,
                    input=True)

    frames = [] 

    for i in range(0, int(rate / chunk * seconds)):
        data = stream.read(chunk)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    # Terminate the PortAudio interface
    p.terminate()

    print('Finished recording')

    # Save the recorded data as a WAV file
    wf = wave.open('/home/pi/audio/testAI/data/input_file/'+filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(sample_format))
    wf.setframerate(rate)
    wf.writeframes(b''.join(frames))
    r = open("/home/pi/audio/testAI/data/input_file/check.txt",'r')
    k = r.readline()
    w = open("/home/pi/audio/testAI/data/input_file/check.txt",'w')
    if(k=='0'):
        w.write('1')
    else:
        w.write('0')
    wf.close()
