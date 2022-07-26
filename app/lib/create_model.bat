echo "Starting Python script"
python build_rfem_model.py %1
echo "Calculating.."

@echo off
:check_file
if exist "%~dp0\output.json" goto wait_and_close
ping localhost -n 1 >NUL
goto :check_file

:wait_and_close
@echo on
echo "Done"
exit