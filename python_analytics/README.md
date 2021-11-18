* Windows Specific Instructions

The following packages can't be installed in linux, so they aren't included in the pip config.

1. Install python package `tesserocr` with `conda install -c conda-forge tesserocr`
   1. from https://pypi.org/project/tesserocr/
2. Install python package `pywin32` with `pip install pywin32`
   1. from https://pypi.org/project/pywin32/
3. Download HLAE's zip package, make sure unzipped so HLAE's binary is at `C:\Users\Administrator\Documents\hlae_2_123_0\HLAE.exe`
   1. configure HLAE to use the hlae_configs folder for the parent repo as the configs
4. Install OBS, make sure it's at `C:\Program Files\obs-studio\bin\64bit\obs64.exe`
   1. In Hotkeys, set `F1` to start recording in OBS, `F3` to stop recording
5. Install CSGO