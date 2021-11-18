call conda activate csknow
SET script_dir=%~dp0

cd %script_dir%csknow_python_analytics\visibility

python recordSnippets.py %script_dir%visibilitySignImages C:\Users\Administrator\Videos\ 9_20_21_no_wallhacks_pre_load_i_eat_short_people_for_breakfast_1.cfg %script_dir%snippets.csv

cd %script_dir%
