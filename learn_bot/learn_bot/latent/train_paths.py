from pathlib import Path

checkpoints_path = Path(__file__).parent / 'checkpoints'
plot_path = Path(__file__).parent / 'distributions'
runs_path = Path(__file__).parent / 'runs'
train_test_split_file_name = 'all_human_and_manual.pickle'
default_selected_retake_rounds_path = Path(__file__).parent / 'vis' / 'good_retake_round_ids.txt'
