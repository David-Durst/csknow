from learn_bot.latent.place_area.pos_abs_from_delta_grid_or_radial import data_ticks_per_second, data_ticks_per_sim_tick

num_seconds_per_loop = 5
num_time_steps = data_ticks_per_second // data_ticks_per_sim_tick * num_seconds_per_loop
