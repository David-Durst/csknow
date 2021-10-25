conda activate csknow

SET script_dir=%~dp0
cd %script_dir%csknow-python-analytics\visibility

python recordGameplay.py
