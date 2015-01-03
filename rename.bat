@echo off&setlocal EnableDelayedExpansion
 set a=1
 for /R %%d in (*.jpg) do (
 if not "%%~nd"=="%~n0" (
 echo copy %%d "!a!.jpg"
 if !a! LSS 10 (ren "%%d" "!a!.jpg") else (ren "%%d" "!a!.jpg")
 set /a a+=1
 )
 )
 echo finished!