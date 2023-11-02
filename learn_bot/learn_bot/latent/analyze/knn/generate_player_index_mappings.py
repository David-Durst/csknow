import copy
from dataclasses import dataclass, field
from typing import Set, Dict, List


@dataclass(frozen=True)
class PartialMapping:
    used_columns: Set[int] = field(default_factory=set)
    player_to_column: Dict[int, int] = field(default_factory=dict)

    def __str__(self):
        return f"used columns: {str(self.used_columns)}\nplayer to column: {str(self.player_to_column)}"


MappingOptions = List[PartialMapping]


# for one team, compute mapping from index to columns
def compute_player_to_column_mappings_for_one_team(init_mappings: MappingOptions, num_alive: int,
                                                   start_index: int) -> MappingOptions:
    result_mappings = copy.deepcopy(init_mappings)
    # for each player, generate all possible options for where that player can go
    for player_index in range(start_index, start_index + num_alive):
        old_result_mappings: MappingOptions = copy.deepcopy(result_mappings)
        result_mappings = []
        # for entry in each partial mapping, create a new partial mapping with the current player_index
        # in that column_index if the column_index hasn't already been used
        for partial_mapping in old_result_mappings:
            for column_index in range(start_index, start_index + num_alive):
                if column_index in partial_mapping.used_columns:
                    continue
                new_partial_mapping: PartialMapping = copy.deepcopy(partial_mapping)
                new_partial_mapping.used_columns.add(column_index)
                new_partial_mapping.player_to_column[player_index] = column_index
                result_mappings.append(new_partial_mapping)
    return result_mappings


def generate_all_player_to_column_mappings(ct_alive: int, t_alive: int) -> MappingOptions:
    ct_partial_mappings = compute_player_to_column_mappings_for_one_team([PartialMapping()], ct_alive, 0)
    return compute_player_to_column_mappings_for_one_team(ct_partial_mappings, t_alive, ct_alive)


def print_mapping_options(mapping_options: MappingOptions):
    print(f"num options {len(mapping_options)}")
    for mapping_option in mapping_options:
        print(mapping_option.player_to_column)

if __name__ == "__main__":
    print("ct alive 2, t alive 3")
    print_mapping_options(generate_all_player_to_column_mappings(ct_alive=2, t_alive=3))
    print("ct alive 4, t alive 2")
    print_mapping_options(generate_all_player_to_column_mappings(ct_alive=4, t_alive=2))
