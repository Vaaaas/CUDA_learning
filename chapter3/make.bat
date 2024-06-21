@REM call vcvars64.bat
@REM cl /Zi /GX /Fe:chapter3\add.exe chapter3\add.cpp
nvcc .\chapter3\add.cu -o .\chapter3\add.exe -arch=sm_80
