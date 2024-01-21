import itertools
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List, Optional, Union

import numpy as np
import pandas as pd
import trueskill
from matplotlib import pyplot as plt
from trueskill import Rating, rate

from learn_bot.latent.analyze.user_responses.user_constants import bot_types

plt.rc('font', family='Arial')

class PlayerRatings:
    human: Rating
    learned: Rating
    handcrafted: Rating
    default: Rating

    def __init__(self):
        self.human = Rating()
        self.learned = Rating()
        self.handcrafted = Rating()
        self.default = Rating()

    def __getitem__(self, item: Union[str, int]) -> Rating:
        if type(item) == int and item < len(bot_types):
            return self[bot_types[item]]
        elif item == 'human':
            return self.human
        elif item == 'learned':
            return self.learned
        elif item == 'hand-crafted':
            return self.handcrafted
        elif item == 'default':
            return self.default
        else:
            raise RuntimeError(f'invalid player rating string {item}')

    def get_ratings_for_game(self) -> List[Tuple[Rating]]:
        return [(self.human, ), (self.learned, ), (self.handcrafted, ), (self.default, )]

    def update_ratings(self, result: List[Tuple[Rating]]):
        self.human = result[0][0]
        self.learned = result[1][0]
        self.handcrafted = result[2][0]
        self.default = result[3][0]

    def __str__(self) -> str:
        return f"human {self.human}, learned {self.learned}, hand-crafted {self.handcrafted}, default {self.default}"

    def plot_ratings(self, plot_path: Path, title_prefix: Optional[str] = None):
        fig, ax = plt.subplots(figsize=(10, 6))
        mus = [self.human.mu, self.learned.mu, self.handcrafted.mu, self.default.mu]
        sigmas = [self.human.sigma, self.learned.sigma, self.handcrafted.sigma, self.default.sigma]

        ax.bar(bot_types, mus, yerr=sigmas, align='center', ecolor='black', color="#3f8f35", capsize=5)
        ax.set_ylim(0, 40)
        ax.set_yticks([0, 10, 20, 30, 40])
        ax.set_ylabel('TrueSkill Rating', fontsize=20)
        ax.set_xlabel('Player Type', fontsize=20)
        full_title = 'User Study TrueSkill Ratings'
        if title_prefix is not None:
            full_title = title_prefix + " " + full_title
        ax.set_title(full_title, fontsize=25)
        ax.tick_params(axis="x", labelsize=20)
        ax.tick_params(axis="y", labelsize=20)

        # remove right/top spine
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # remove veritcal grid lines, make horizontal dotted
        ax.yaxis.grid(True, color='#EEEEEE', dashes=[4, 1])
        ax.xaxis.grid(False)
        fig.tight_layout()
        plt.savefig(plot_path)

        with open(plot_path.with_suffix('.txt'), 'w') as f:
            f.write(str(self))

    def plot_win_probabilities(self, plot_path: Path, title_prefix: Optional[str] = None):
        np.ndarray
        for bot_type in bot_types


    def plot(self, plot_path: Path, title_prefix: Optional[str] = None):
        self.plot_ratings(plot_path, title_prefix)


def win_probability(team1, team2):
    delta_mu = sum(r.mu for r in team1) - sum(r.mu for r in team2)
    sum_sigma = sum(r.sigma ** 2 for r in itertools.chain(team1, team2))
    size = len(team1) + len(team2)
    denom = math.sqrt(size * (trueskill.BETA * trueskill.BETA) + sum_sigma)
    ts = trueskill.global_env()
    return ts.cdf(delta_mu / denom)


def build_trueskill(df: pd.DataFrame) -> PlayerRatings:
    player_ratings = PlayerRatings()

    for _, one_example_one_response in df.iterrows():
        ranks = one_example_one_response[bot_types]
        updated_ratings = rate(player_ratings.get_ratings_for_game(), ranks)
        player_ratings.update_ratings(updated_ratings)

    return player_ratings
