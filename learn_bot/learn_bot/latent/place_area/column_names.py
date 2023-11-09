from math import isqrt

from learn_bot.latent.order.column_names import *
from learn_bot.libs.io_transforms import flatten_list

# rectangular grid const
delta_pos_grid_radius = 130
delta_pos_grid_cell_dim = 20
delta_pos_z_num_cells = 3
delta_pos_grid_num_cells = delta_pos_z_num_cells * \
                           int((delta_pos_grid_radius * 2 * delta_pos_grid_radius * 2) /
                               (delta_pos_grid_cell_dim * delta_pos_grid_cell_dim))
delta_pos_grid_num_cells_per_xy_dim = isqrt(int(delta_pos_grid_num_cells / delta_pos_z_num_cells))
delta_pos_grid_num_xy_cells_per_z_change = delta_pos_grid_num_cells_per_xy_dim * delta_pos_grid_num_cells_per_xy_dim

# radial grid const
class StatureOptions(Enum):
    Standing = 0
    Walking = 1
    Ducking = 2
    NUM_STATURE_OPTIONS = 3

walking_modifier = 0.52
ducking_modifier = 0.34
airwalk_speed = 30.
num_directions = 16
direction_angle_range = 360. / num_directions
num_z_axis_layers = 3
num_radial_bins_per_z_axis = num_directions * StatureOptions.NUM_STATURE_OPTIONS.value
num_radial_bins = 1 + num_z_axis_layers * num_radial_bins_per_z_axis
# half of max speed, can upgrade simulator with per-weapon speeds later
default_speed = 125



float_c4_cols = c4_time_left_percent #c4_ticks_since_plant #[c4_distance_to_a_site_col, c4_distance_to_b_site_col] + c4_pos_cols + c4_ticks_since_plant

test_success_col = 'test success'
round_test_name_col = 'round test name'

hdf5_id_columns = ['id', tick_id_column, game_tick_number_column, round_id_column, round_number_column,
                   game_id_column, test_success_col]

def get_player_distribution_nearest_place(player_index: int, place_index: int, team_str: str) -> str:
    return "distribution nearest place " + str(place_index) + " " + team_str + " " + str(player_index)


def get_player_distribution_nearest_grid_area(player_index: int, area_grid_index: int, team_str: str) -> str:
    return "distribution nearest area grid in place " + str(area_grid_index) + " " + team_str + " " + str(player_index)


class PlayerPlaceAreaColumns:
    player_id: str
    index_on_team: list[str]
    ct_team: str
    alive: str
    distance_to_a_site: str
    distance_to_b_site: str
    pos: list[str]
    prior_pos: list[str]
    vel: list[str]
    decrease_distance_to_c4_5s: list[str]
    decrease_distance_to_c4_10s: list[str]
    decrease_distance_to_c4_20s: list[str]
    cur_place: list[str]
    #prior_place: list[str]
    area_grid_cell_in_place: list[str]
    #prior_area_grid_cell_in_place: list[str]
    distribution_nearest_place: list[str]
    distribution_nearest_grid_area: list[str]
    # only used for trace analysis
    trace_is_bot_player: str

    def __init__(self, team_str: str, player_index: int):
        self.player_id = player_id_column + " " + player_team_str(team_str, player_index)
        self.player_id_uniform_space = player_id_column + " " + player_team_str(team_str, player_index,
                                                                                uniform_space=True)
        self.index_on_team = get_player_index_on_team_column(player_index, team_str)
        self.ct_team = get_player_ctteam_column(player_index, team_str)
        self.alive = get_player_alive_column(player_index, team_str)
        self.distance_to_a_site = "distance to a site " + player_team_str(team_str, player_index)
        self.distance_to_b_site = "distance to b site " + player_team_str(team_str, player_index)
        self.view_angle = [get_player_view_angle_columns(player_index, team_str, dim_str) for dim_str in ["x", "y"]]
        self.pos = [get_player_pos_columns(player_index, team_str, dim_str) for dim_str in ["x", "y", "z"]]
        self.aligned_pos = [get_player_aligned_pos_columns(player_index, team_str, dim_str) for dim_str in ["x", "y", "z"]]
        self.prior_pos = []
        self.vel = [get_player_velocity_columns(player_index, team_str, dim_str) for dim_str in ["x", "y", "z"]]
        self.prior_vel = []
        self.nearest_crosshair_distance_to_enemy = \
            get_player_nearest_crosshair_distance_to_enemy_columns(player_index, team_str)
        self.prior_nearest_crosshair_distance_to_enemy = []
        self.player_hurt_in_last_5s = get_player_hurt_in_last_5s_columns(player_index, team_str)
        self.seconds_after_prior_hit_enemy = get_player_seconds_after_prior_hit_enemy(player_index, team_str)
        self.seconds_until_next_hit_enemy = get_player_seconds_until_next_hit_enemy(player_index, team_str)
        self.player_fire_in_last_5s = get_player_fire_in_last_5s_columns(player_index, team_str)
        self.player_no_fov_enemy_visible_in_last_5s = get_player_no_fov_enemy_visible_in_last_5s_columns(player_index, team_str)
        self.player_fov_enemy_visible_in_last_5s = get_player_fov_enemy_visible_in_last_5s_columns(player_index, team_str)
        self.player_health = get_player_health_columns(player_index, team_str)
        self.player_armor = get_player_armor_columns(player_index, team_str)
        self.decrease_distance_to_c4_5s = f"player decrease distance to c4 over 5s {team_str} {player_index}"
        self.decrease_distance_to_c4_10s = f"player decrease distance to c4 over 10s {team_str} {player_index}"
        self.decrease_distance_to_c4_20s = f"player decrease distance to c4 over 20s {team_str} {player_index}"
        self.cur_place = []
        #self.prior_place = []
        self.area_grid_cell_in_place = []
        #self.prior_area_grid_cell_in_place = []
        self.distribution_nearest_place = []
        self.distribution_nearest_grid_area = []
        self.trace_is_bot_player = get_trace_is_bot_player(player_index, team_str)
        self.delta_pos = []
        self.radial_vel = []
        self.future_radial_vel: List[List] = []
        for place_index in range(num_places):
            self.cur_place \
                .append(get_player_cur_place_columns(player_index, place_index, team_str))
            self.distribution_nearest_place \
                .append(get_player_distribution_nearest_place(player_index, place_index, team_str))
        for area_grid_index in range(area_grid_size):
            self.area_grid_cell_in_place \
                .append(get_player_area_grid_cell_in_place_columns(player_index, area_grid_index, team_str))
            self.distribution_nearest_grid_area \
                .append(get_player_distribution_nearest_grid_area(player_index, area_grid_index, team_str))
        for delta_pos_index in range(delta_pos_grid_num_cells):
            self.delta_pos \
                .append(get_delta_pos_columns(player_index, delta_pos_index, team_str))
        for radial_vel_index in range(num_radial_bins):
            self.radial_vel \
                .append(get_radial_vel_columns(player_index, radial_vel_index, team_str))
        for future_tick in range(num_future_ticks):
            self.future_radial_vel.append([])
            for radial_vel_index in range(num_radial_bins):
                self.future_radial_vel[future_tick] \
                    .append(get_future_radial_vel_columns(player_index, radial_vel_index, team_str, future_tick+1))
        for prior_tick in range(1, num_prior_ticks+1):
            self.prior_nearest_crosshair_distance_to_enemy.append(
                get_player_nearest_crosshair_distance_to_enemy_columns(player_index, team_str, prior_tick))
            for dim_str in ["x", "y", "z"]:
                self.prior_pos.append(get_player_pos_columns(player_index, team_str, dim_str, prior_tick))
                self.prior_vel.append(get_player_velocity_columns(player_index, team_str, dim_str, prior_tick))
            #for place_index in range(num_places):
            #    self.prior_place \
            #        .append(get_player_prior_place_columns(player_index, place_index, team_str, prior_tick))
            #for area_grid_index in range(area_grid_size):
            #    self.prior_area_grid_cell_in_place \
            #        .append(get_player_prior_area_grid_cell_in_place_columns(player_index, area_grid_index, team_str, prior_tick))

    def to_list(self) -> list[str]:
        return [self.player_id, self.distance_to_a_site, self.distance_to_b_site] + \
            flatten_list([self.pos, self.prior_pos, self.vel, self.cur_place, #self.prior_place,
                          self.area_grid_cell_in_place, #self.prior_area_grid_cell_in_place,
                          self.distribution_nearest_place, self.distribution_nearest_grid_area])

    def to_input_float_list(self) -> list[str]:
        #return [self.distance_to_a_site, self.distance_to_b_site] + \
        #    flatten_list([self.pos, self.prior_pos, self.vel])
        #return flatten_list([self.pos, self.prior_pos, self.vel])
        #return flatten_list([self.pos, self.vel, self.prior_pos, self.prior_vel])
        return flatten_list([self.pos, self.prior_pos,
                             [self.nearest_crosshair_distance_to_enemy], self.prior_nearest_crosshair_distance_to_enemy,
                             [self.player_hurt_in_last_5s], [self.player_fire_in_last_5s],
                             [self.player_no_fov_enemy_visible_in_last_5s], [self.player_fov_enemy_visible_in_last_5s],
                             [self.player_health], [self.player_armor],
                             [self.seconds_after_prior_hit_enemy], [self.seconds_until_next_hit_enemy]])
        #return flatten_list([self.pos])
        #return flatten_list([self.aligned_pos])

    def to_input_distribution_cat_list(self) -> list[list[str]]:
        #return [self.cur_place, self.area_grid_cell_in_place, [self.alive]]#, self.index_on_team, [self.ct_team]]
        #return [self.cur_place, self.area_grid_cell_in_place, [self.alive], [self.ct_team]]#, self.index_on_team, [self.ct_team]]
        return [[self.alive], #self.index_on_team,
                [self.ct_team],
                [self.decrease_distance_to_c4_5s], [self.decrease_distance_to_c4_10s],
                [self.decrease_distance_to_c4_20s]]

    def to_output_cat_list(self, place: bool, area: bool, delta: bool, radial: bool) -> list[list[str]]:
        result = []
        if place:
            result.append(self.distribution_nearest_place)
        if area:
            result.append(self.distribution_nearest_grid_area)
        if delta:
            result.append(self.delta_pos)
        if radial:
            result.append(self.radial_vel)
            for future_tick in range(num_future_ticks):
                result.append(self.future_radial_vel[future_tick])
        return result

    def get_vis_only_columns(self) -> list[str]:
        return [self.player_id] + self.vel + self.view_angle


specific_player_place_area_columns: list[PlayerPlaceAreaColumns] = \
    [PlayerPlaceAreaColumns(team_str, player_index) for team_str in team_strs for player_index in range(max_enemies)]
flat_input_float_place_area_columns: list[str] = \
    float_c4_cols + [col for cols in specific_player_place_area_columns for col in cols.to_input_float_list()]
flat_input_cat_place_area_columns: list[str] = [c4_status_col]
flat_input_cat_place_area_num_options: list[int] = [num_c4_status]
flat_input_distribution_cat_place_area_columns: list[list[str]] = \
    flatten_list([cols.to_input_distribution_cat_list() for cols in specific_player_place_area_columns]) + \
    [[c4_plant_a_col, c4_plant_b_col, c4_not_planted_col]]
#[['baiting'], [c4_plant_a_col, c4_plant_b_col, c4_not_planted_col]]
flat_output_cat_place_distribution_columns: list[list[str]] = \
    flatten_list([cols.to_output_cat_list(True, False, False, False) for cols in specific_player_place_area_columns])
flat_output_cat_area_distribution_columns: list[list[str]] = \
    flatten_list([cols.to_output_cat_list(False, True, False, False) for cols in specific_player_place_area_columns])
flat_output_cat_delta_pos_columns: list[list[str]] = \
    flatten_list([cols.to_output_cat_list(False, False, True, False) for cols in specific_player_place_area_columns])
flat_output_cat_radial_vel_columns: list[list[str]] = \
    flatten_list([cols.to_output_cat_list(False, False, False, True) for cols in specific_player_place_area_columns])

place_area_input_column_types = get_simplified_column_types(flat_input_float_place_area_columns,
                                                            [], #flat_input_cat_place_area_columns,
                                                            [], #flat_input_cat_place_area_num_options,
                                                            flat_input_distribution_cat_place_area_columns)
                                                            #flat_input_distribution_cat_place_area_columns)
#output_column_types = get_simplified_column_types([], flat_output_cat_columns, flat_output_num_options,
#                                                  flat_output_cat_distribution_columns)
place_output_column_types = get_simplified_column_types([], [], [], flat_output_cat_place_distribution_columns)
area_output_column_types = get_simplified_column_types([], [], [], flat_output_cat_area_distribution_columns)
delta_pos_output_column_types = get_simplified_column_types([], [], [], flat_output_cat_delta_pos_columns)
radial_vel_output_column_types = get_simplified_column_types([], [], [], flat_output_cat_radial_vel_columns)

vis_only_columns: list[str] = c4_pos_cols + \
                              flatten_list([player_place_area_columns.get_vis_only_columns() for
                                            player_place_area_columns in specific_player_place_area_columns])


def get_base_similarity_column() -> str:
    return 'similarity'


def get_similarity_column(idx: int) -> str:
    return f'similarity {idx}'
