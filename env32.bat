REM configuration of paths
set VSFORPYTHON="C:\Program Files (x86)\Common Files\Microsoft\Visual C++ for Python\9.0"
set SCISOFT=D:\Apps
set SWIGWIN=D:\Apps\swigwin-3.0.10
REM %~dp0

REM add winpython stuff
CALL %SCISOFT%\WinPython-32bit-2.7.10.3\scripts\env.bat

REM add tdm gcc stuff
set PATH=%SCISOFT%\TDM-GCC-32\bin;%SCISOFT%\TDM-GCC-32\mingw32\bin;%SWIGWIN%;%PATH%
REM set PATH=%SWIGWIN%;%PATH%

REM configure path for msvc compilers
REM for a 32 bit installation change this line to
REM CALL %VSFORPYTHON%\vcvarsall.bat
CALL %VSFORPYTHON%\vcvarsall.bat

REM return a shell
cmd.exe /k