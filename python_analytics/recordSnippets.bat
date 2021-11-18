call conda activate csknow
SET script_dir=%~dp0

cd %script_dir%csknow_python_analytics\visibility

python recordSnippets.py %script_dir%visibilitySignImages C:\Users\Administrator\Videos\  2351898_128353_g2-vs-ence-m3-dust2_5f2f16a6-292a-11ec-8e27-0a58a9feac02.dem 2351898_128353_g2-vs-ence-m3-dust2_5f2f16a6-292a-11ec-8e27-0a58a9feac02_pre_load_NiKo_1.cfg

cd %script_dir%
