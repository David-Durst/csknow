from enum import IntEnum
from dataclasses import dataclass
from typing import List


class RetakeBotType(IntEnum):
    CSKnowLearned = 0
    CSKnowHeuristic = 1
    CSGODefault = 2
    Human = 3


retake_bot_types: List[RetakeBotType] = \
    [RetakeBotType.CSKnowLearned, RetakeBotType.CSKnowLearned, RetakeBotType.CSGODefault, RetakeBotType.Human]

retake_bot_types_to_names = {
    int(RetakeBotType.CSKnowLearned): "CSKnowLearned",
    int(RetakeBotType.CSKnowHeuristic): "CSKnowHeuristic",
    int(RetakeBotType.CSGODefault): "CSGODefault",
    int(RetakeBotType.Human): "Human",
}


@dataclass
class TeamMomentStrs:
    # level 0 - base performance
    win: str
    shots_per_total_players: str
    kills_per_total_players: str

    # level 1 - key metrics
    distance_traveled_per_player: str
    max_distance_from_start: str
    shots_per_kill: str
    average_speed_while_shooting: str

    # level 2 - bug detectors
    num_players_alive_tick_before_explosion: str

    # base statistics
    botType: str
    numPlayers: str

    def get_moment_columns(self):
        return [self.win, self.shots_per_total_players, self.kills_per_total_players,
                self.distance_traveled_per_player, self.max_distance_from_start, self.shots_per_kill,
                self.average_speed_while_shooting, self.num_players_alive_tick_before_explosion, self.numPlayers]


ctMomentsStrs = TeamMomentStrs(
    "ct win",
    "ct shots per total players",
    "ct kills per total players",
    "ct distance traveled per player",
    "ct max distance from start",
    "ct shots per kill",
    "ct average speed while shooting",
    "ct num players alive before explosion",
    "ct bot type",
    "ct num players"
)

tMomentsStrs = TeamMomentStrs(
    "t win",
    "t shots per total players",
    "t kills per total players",
    "t distance traveled per player",
    "t max distance from start",
    "t shots per kill",
    "t average speed while shooting",
    "t num players alive before explosion",
    "t bot type",
    "t num players"
)

validRoundIdStr = "id"
tickLengthStr = "tick length"
defusalIdStr = "defusalId"


@dataclass
class CTTPair:
    ct_column_name: str
    t_column_name: str


column_names: List[CTTPair] = \
    [CTTPair(ctStr, tStr) for (ctStr, tStr) in
     zip(ctMomentsStrs.get_moment_columns(), tMomentsStrs.get_moment_columns())]
