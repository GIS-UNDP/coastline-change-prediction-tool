IF EXIST "C:\Users\%USERNAME%\Anaconda3\" goto :f1

IF NOT EXIST "C:\Users\%USERNAME%\Anaconda3\" goto :f2


:f1
PATH=%PATH%;C:\Users\%USERNAME%\Anaconda3;C:\Users\%USERNAME%\Anaconda3\Scripts;C:\Users\%USERNAME%\Anaconda3\Library;C:\Users\%USERNAME%\Anaconda3\Library\bin
call activate coastpred
jupyter notebook

pause

goto :f3

:f2
cd C:\
for /f "delims=" %%a in ('dir /s /b Uninstall-Anaconda3.exe') do set "name=%%a"
for %%a in ("%name%") do set "p_dir=%%~dpa"
PATH=%PATH%;%p_dir%;%p_dir%\Scripts
call activate coastpred
jupyter notebook


:f3