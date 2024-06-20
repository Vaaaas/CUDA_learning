@REM call vcvars64.bat
@REM cl /Zi /GX /Fe:chapter2\hello.exe chapter2\*.cpp
nvcc .\chapter2\hello.cu -o .\chapter2\hello.exe
