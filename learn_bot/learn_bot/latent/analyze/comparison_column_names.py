from pathlib import Path

predicted_name_col = 'predicted name'
best_fit_ground_truth_name_col = 'best fit ground truth name'
metric_type_col = 'metric type'
predicted_trace_batch_col = 'predicted trace batch'
best_fit_ground_truth_trace_batch_col = 'best fit ground truth trace batch'
predicted_round_id_col = 'predicted round id'
best_fit_ground_truth_round_id_col = 'best fit ground truth round id'
predicted_round_number_col = 'predicted round number'
best_fit_ground_truth_round_number_col = 'best fit ground truth round number'
predicted_first_game_tick_number_col = 'predicted first game tick number'
best_fit_ground_truth_first_game_tick_number_col = 'best fit ground truth first game tick number'
best_match_id_col = 'best match ids'
predicted_start_trace_index_col = 'predicted start trace index'
predicted_end_trace_index_col = 'predicted end trace index'
best_fit_ground_truth_start_trace_index_col = 'best fit ground truth start trace index'
best_fit_ground_truth_end_trace_index_col = 'best fit ground truth end trace index'
dtw_cost_col = 'dtw cost'
delta_time_col = 'delta time'
delta_distance_col = 'delta distance'
agent_mapping_col = 'agent mapping'
start_dtw_matched_indices_col = 'start dtw matched indices'
length_dtw_matched_inidices_col = 'length dtw matched indices'
first_matched_index_col = 'first matched index'
second_matched_index_col = 'second matched index'

similarity_plots_path = Path(__file__).parent / 'similarity_plots'
hand_crafted_bot_vs_hand_crafted_bot_similarity_hdf5_data_path = Path(__file__).parent / '..' / '..' / '..' / '..' / 'analytics' / 'manual_outputs' / 'botTrajectorySimilarity.hdf5'
time_vs_hand_crafted_bot_similarity_hdf5_data_path = Path(__file__).parent / '..' / '..' / '..' / '..' / 'analytics' / 'rollout_outputs' / 'learnedTimeBotTrajectorySimilarity.hdf5'
no_time_vs_hand_crafted_bot_similarity_hdf5_data_path = Path(__file__).parent / '..' / '..' / '..' / '..' / 'analytics' / 'rollout_outputs' / 'learnedNoTimeNoWeightDecayBotTrajectorySimilarity.hdf5'
human_vs_human_similarity_hdf5_data_path = Path(__file__).parent / '..' / '..' / '..' / '..' / 'analytics' / 'csv_outputs' / 'humanTrajectorySimilarity.hdf5'
all_human_vs_all_human_similarity_hdf5_data_path = Path(__file__).parent / '..' / '..' / '..' / '..' / 'analytics' / 'all_train_outputs' / 'humanTrajectorySimilarity.hdf5'
all_human_vs_small_human_similarity_hdf5_data_path = Path(__file__).parent / '..' / '..' / '..' / '..' / 'analytics' / 'all_train_outputs' / 'humanVsSmallHumanTrajectorySimilarity.hdf5'
all_human_vs_human_28_similarity_hdf5_data_path = Path(__file__).parent / '..' / '..' / '..' / '..' / 'analytics' / 'all_train_outputs' / 'humanVsHuman28TrajectorySimilarity.hdf5'
rollout_vs_all_human_similarity_hdf5_data_path = Path(__file__).parent / '..' / '..' / '..' / '..' / 'analytics' / 'rollout_outputs' / 'humanTrajectorySimilarity.hdf5'

bot_good_rounds = [4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55,
                   57, 59, 61, 63, 65, 67, 69, 71, 73, 75, 78, 80, 82, 85, 87, 89, 91, 93, 95, 97, 99, 102, 104, 106, 108,
                   110, 112, 114, 116, 118, 120, 122, 124, 126, 128, 130, 132, 134, 136, 138, 140, 142, 144, 146, 148,
                   150, 154, 156, 158, 160, 162, 164, 166, 168, 170, 172, 174, 176, 178, 180, 182, 184, 186, 188, 190,
                   192, 194, 196, 198, 200, 202, 204, 206, 208, 210, 212, 214, 216, 218, 220, 222, 224, 226, 228, 230,
                   232, 234, 236, 238, 240, 242, 244, 246, 248, 250, 252, 254, 256, 258, 260, 262, 264, 266, 268, 270,
                   272, 274, 276, 278, 280, 282, 284, 286, 288, 290, 292, 294, 296, 298, 300, 302, 304, 306, 308, 310,
                   312, 314, 316, 318, 320, 322, 324, 326, 328, 330, 332, 334, 336, 338, 340, 342, 344, 346, 348, 350,
                   352, 354, 356, 358, 360, 362, 364, 366, 368, 370, 372, 374, 376, 378, 380, 382, 384, 386, 388, 391,
                   393, 396, 398, 400, 402, 404, 406, 408, 410, 412, 414, 416, 418, 420, 422, 424, 426, 428, 430, 432,
                   434, 436, 438, 440, 442, 444, 446, 450, 452, 454, 456, 458, 460, 462, 464, 466, 468, 470, 472, 474,
                   476, 478, 480, 482, 484, 486, 488, 490, 492, 494, 496, 498, 500, 502, 504, 506, 508, 510, 512, 514,
                   516, 518, 520, 522, 524, 526, 528, 530, 532, 534, 537, 539]

small_human_good_rounds = [512, 1, 4, 517, 520, 10, 15, 529, 534, 535, 25, 26, 27, 28, 541, 542, 544, 549, 38, 550, 552, 41, 42, 46, 558, 49, 566,
                           567, 56, 571, 61, 64, 65, 577, 67, 581, 71, 583, 584, 586, 587, 589, 592, 85, 88, 90, 91, 603, 96, 609, 610, 99, 101,
                           613, 615, 110, 111, 113, 626, 116, 629, 118, 122, 123, 124, 638, 127, 129, 641, 131, 133, 134, 135, 136, 137, 647, 139,
                           140, 652, 655, 144, 656, 148, 153, 666, 158, 159, 670, 162, 165, 166, 167, 678, 176, 691, 181, 182, 185, 699, 190, 706,
                           709, 199, 205, 207, 210, 217, 227, 234, 236, 237, 239, 242, 253, 255, 258, 261, 264, 269, 270, 276, 278, 280, 281, 299,
                           302, 303, 306, 308, 313, 321, 324, 326, 331, 335, 337, 345, 346, 347, 349, 355, 358, 365, 373, 375, 377, 380, 384, 394,
                           398, 408, 412, 422, 424, 427, 429, 431, 435, 439, 441, 442, 443, 451, 453, 456, 458, 461, 468, 479, 481, 484, 485, 488,
                           500, 507, 508]

all_human_28_first_filter_good_rounds = [1, 514, 516, 521, 522, 13, 525, 529, 20, 21, 532, 25, 537, 30, 543, 33, 546, 548, 550, 42, 558, 47, 48, 560, 561, 54, 55, 566, 57, 570, 62, 63, 65, 579, 582, 71, 584, 585, 586, 79, 80, 593, 594, 595, 84, 596, 598, 599, 600, 601, 90, 602, 603, 94, 96, 97, 608, 99, 100, 611, 613, 615, 616, 105, 107, 108, 620, 621, 114, 115, 626, 630, 119, 631, 121, 124, 638, 642, 644, 135, 141, 142, 145, 147, 158, 161, 170, 173, 179, 184, 188, 189, 190, 193, 205, 212, 213, 223, 227, 235, 248, 256, 263, 266, 268, 273, 279, 281, 282, 284, 285, 297, 298, 299, 300, 304, 310, 312, 313, 314, 316, 322, 323, 327, 329, 331, 335, 337, 338, 340, 342, 345, 348, 349, 353, 355, 358, 362, 368, 371, 377, 380, 384, 387, 388, 390, 391, 393, 394, 395, 406, 407, 410, 413, 414, 418, 420, 422, 426, 427, 430, 431, 432, 445, 446, 447, 451, 452, 453, 455, 458, 461, 463, 466, 468, 474, 475, 477, 479, 489, 491, 494, 495, 496, 498, 509]
all_human_28_second_filter_save_rounds = [62, 107, 108, 190, 281, 284, 285, 299, 312, 335, 337, 340, 345, 358, 387, 393, 395, 410, 413, 418, 432, 461, 479, 491, 496, 560, 596]
all_human_28_second_filter_good_rounds = [r for r in all_human_28_first_filter_good_rounds
                                          if r not in all_human_28_second_filter_save_rounds]
