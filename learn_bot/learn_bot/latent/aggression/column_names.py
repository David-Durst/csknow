from learn_bot.latent.engagement.column_names import *

@dataclass
class TeammateColumns:
    teammate_player_id: str
    teammate_world_distance: str
    crosshair_distance_to_teammate: str

    def to_list(self) -> list[str]:
        return [self.teammate_player_id, self.teammate_world_distance, self.crosshair_distance_to_teammate]

    def to_input_float_list(self) -> list[str]:
        return [self.teammate_world_distance, self.crosshair_distance_to_teammate]


base_teammate_columns: TeammateColumns = TeammateColumns(
    "teammate player id",
    "teammate world distance",
    "crosshair distance to teammate",
)

def get_ith_teammate_columns(i: int) -> TeammateColumns:
    result = copy.copy(base_teammate_columns)
    result.teammate_player_id += f" {i}"
    result.teammate_world_distance += f" {i}"
    result.crosshair_distance_to_teammate += f" {i}"
    return result

change_options = ["decrease", "constant", "increase"]

pct_nearest_enemy_change_2s_columns = ["pct nearest enemy change 2s " +
                                       change_option for change_option in change_options]

specific_teammate_columns: list[TeammateColumns] = [get_ith_teammate_columns(i) for i in range(max_enemies)]
flat_input_float_specific_teammate_columns: list[str] = \
    [col for cols in specific_enemy_columns for col in cols.to_input_float_list()]

aggression_input_column_types = get_simplified_column_types(flat_input_float_specific_enemy_columns +
                                                            flat_input_float_specific_teammate_columns,
                                                 flat_input_cat_specific_enemy_columns,
                                                 flat_input_cat_specific_enemy_num_options, [])
#output_column_types = get_simplified_column_types([], flat_output_cat_columns, flat_output_num_options,
#                                                  flat_output_cat_distribution_columns)
aggression_output_column_types = get_simplified_column_types([], [], [],
                                                             [pct_nearest_enemy_change_2s_columns])
