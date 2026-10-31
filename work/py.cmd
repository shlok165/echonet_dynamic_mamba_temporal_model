@echo off
rem Launcher for the project venv on E: (C: is short on space). The pip CUDA wheels used here bundle their
rem own cuDNN/CUDA DLLs, so no PATH fix is needed.
set "ECHO_DATA=E:\echo_project"
set "TORCH_HOME=%ECHO_DATA%\torch-cache"
set "TMP=%ECHO_DATA%\tmp"
set "TEMP=%ECHO_DATA%\tmp"
set "PIP_CACHE_DIR=%ECHO_DATA%\pip-cache"
set PYTHONUTF8=1
rem pip numpy uses OpenBLAS, which reserves per-thread buffers (16 threads here) in every process; with
rem 8+ DataLoader workers this exhausted the commit limit (MemoryError / "OpenBLAS memory allocation failed").
rem Workers do not need threaded BLAS.
set OPENBLAS_NUM_THREADS=1
set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1
"%ECHO_DATA%\venv\Scripts\python.exe" %*
