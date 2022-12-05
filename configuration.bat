IF EXIST "C:\Users\%USERNAME%\Anaconda3\" goto :f1

IF NOT EXIST "C:\Users\%USERNAME%\Anaconda3\" goto :f2


:f1
PATH=%PATH%;C:\Users\%USERNAME%\Anaconda3;C:\Users\%USERNAME%\Anaconda3\Scripts
call _conda.exe env create -f environment.yml -n coastpred
call activate coastpred
call earthengine authenticate
call deactivate

pause

goto :f3

:f2
cd C:\
for /f "delims=" %%a in ('dir /s /b Uninstall-Anaconda3.exe') do set "name=%%a"
for %%a in ("%name%") do set "p_dir=%%~dpa"
PATH=%PATH%;%p_dir%;%p_dir%\Scripts
call _conda.exe env create -f environment.yml -n coastpred
call activate coastpred
call earthengine authenticate
call deactivate

pause
:f3
