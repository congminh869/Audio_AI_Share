from pathlib import Path
# code_path=Path('E:\Google Driver')
parameter_file=open("/home/pi/audio/testAI/parameter.txt",'r')
code_path=parameter_file.readline().splitlines()[0]
