REM configuration of paths
set VSFORPYTHON="C:\Program Files\Common Files\Microsoft\Visual C++ for Python\9.0"
set SCISOFT=D:\Apps
REM %~dp0

REM add tdm gcc stuff AFTER WINPYTHON!
set PATH=%SCISOFT%\TDM-GCC-32\bin;%SCISOFT%\TDM-GCC-32\mingw32\bin;%PATH%

REM add winpython stuff
CALL %SCISOFT%\WinPython-32bit-2.7.9.5\scripts\env.bat

REM configure path for msvc compilers
REM for a 32 bit installation change this line to
REM CALL %VSFORPYTHON%\vcvarsall.bat
CALL %VSFORPYTHON%\vcvarsall.bat 
REM amd64

REM return a shell
cmd.exe /k