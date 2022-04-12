import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)

# return is a mapping from sequence
def organize_into_sequences(df):
    sort_cols = ['round id', 'source player id', 'engagement id', 'tick id']
    df.sort_values(sort_cols, inplace=True)
    for i in range(len(sort_cols) - 1):
        df.loc[:, 'prior ' + sort_cols[i]] = df.loc[:, sort_cols[i]].shift(1, fill_value=-1)
        if i == 0:
            df.loc[:, 'new engagement'] = df['prior ' + sort_cols[i]] != df[sort_cols[i]]
        else:
            df.loc[:, 'new engagement'] = df.loc[:, 'new engagement'] | (df['prior ' + sort_cols[i]] != df[sort_cols[i]])
    df.loc[:, 'new engagement'] = df.loc[:, 'new engagement'].astype(int)
    # subtract 1 so first sequence is index 0
    df.loc[:, 'global engagement id'] = df['new engagement'].cumsum() - 1
    df.reset_index(inplace=True)
    df.loc[:, 'index'] = df.index

    return df.groupby('global engagement id').agg(Min=('index', 'min'), Max=('index','max'));

def get_longest_sequences(df, name):
    seq_df = organize_into_sequences(df)
    seq_df['length'] = seq_df['Max'] - seq_df['Min']
    seq_df.sort_values('length', ascending=False, inplace=True)

    def print_seq_min_max(i):
        min_idx = seq_df.iloc[i].loc['Min']
        max_idx = seq_df.iloc[i].loc['Max']
        cols = ['demo name', 'source player name', 'game tick number', 'round id', 'engagement id', 'global engagement id']
        min_df = df.loc[min_idx, cols]
        max_df = df.loc[max_idx, cols]
        print(str(i) + ' ' + min_df['demo name'] + ' ' + min_df['source player name'] + ' ' +
              str(min_df['game tick number']) + ' ' + str(max_df['game tick number']) + ' ' +
              str(min_df['round id']) + ' ' + str(min_df['engagement id']) + ' ' + str(min_df['global engagement id']))

    print(name + " longest 5")
    for i in range(5):
        print_seq_min_max(i)

    print(name + " shortest 5")
    for i in range(5):
        print_seq_min_max(-1 * (i+1))

    print(' ')

    return seq_df

if __name__ == "__main__":
    orig_df = pd.read_csv(Path(__file__).parent / '..' / 'data' / 'explore_engagements' / 'train_engagement_dataset.csv')
    explore_df = pd.read_csv(Path(__file__).parent / '..' / 'data' / 'explore_engagements' / 'train_engagement_dataset.csv')
    explore_seq_df = get_longest_sequences(explore_df, "explore")
    explore_seq_df.hist('length', bins=100)
    plt.show()

    # can't split seuqences based on knife or no knife
    # you get a bunch of knifing followed by a switching to a non-knife
    # and then you get 1 tick non-knifing events wihtout any shooting that make no sense
    # must filter by weapon when doing actual sequence analysis
    # same issue occurs for enemies - if keep shooting, the engagmenet as of 4/10 isn't continued, so get a bad point
    #not_enemies_df = explore_df[(explore_df['enemy slot filled'] == 0) & (explore_df['shooter active weapon'] != 405)].copy()
    #not_enemies_seq_df = get_longest_sequences(not_enemies_df, "not enemies")

    #enemies_df = explore_df[(explore_df['enemy slot filled'] == 1) & (explore_df['shooter active weapon'] != 405)].copy()
    #enemies_seq_df = get_longest_sequences(enemies_df, "enemies")

    #no_knife_df = explore_df[explore_df['shooter active weapon'] != 405].copy()
    #no_knife_seq_df = get_longest_sequences(no_knife_df, "no knife")

    #issue_row = explore_df[(explore_df['source player name'] == 'blameF') & (explore_df['game tick number'] == 263919)]

    overlapping_sequences = {}
    #for i in range(len(enemies_seq_df)):
    #    for j in range(len(not_enemies_seq_df)):
    #        if enemies_seq_df.loc['Min'] >= not_enemies_seq_df['Min'] and ene
    x = 2
