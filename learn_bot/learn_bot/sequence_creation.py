import pandas as pd
from pathlib import Path

pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)

# return is a mapping from sequence
def organize_into_sequences(df):
    df.sort_values(['round id', 'source player id', 'tick id'], inplace=True)
    df.loc[:, 'prior round id'] = df.loc[:, 'round id'].shift(1, fill_value=-1)
    df.loc[:, 'prior source player id'] = df['source player id'].shift(1, fill_value=-1)
    df.loc[:, 'prior tick id'] = df['tick id'].shift(1, fill_value=-1)
    df.loc[:, 'new sequence'] = ((df['prior round id'] != df['round id']) |
                                 (df['prior source player id'] != df['source player id']) |
                                 (df['prior tick id'] + 1 != df['tick id'])).astype(int)
    # subtract 1 so first sequence is index 0
    df.loc[:, 'sequence number'] = df['new sequence'].cumsum() - 1
    df.reset_index(inplace=True)
    df.loc[:, 'index'] = df.index

    return df.groupby('sequence number').agg(Min=('index', 'min'), Max=('index','max'));

def get_longest_sequences(df, name):
    seq_df = organize_into_sequences(df)
    seq_df['length'] = seq_df['Max'] - seq_df['Min']
    seq_df.sort_values('length', ascending=False, inplace=True)

    def print_seq_min_max(i):
        min_idx = seq_df.iloc[i].loc['Min']
        max_idx = seq_df.iloc[i].loc['Max']
        min_df = df.loc[min_idx, ['demo name', 'source player name', 'game tick number']]
        max_df = df.loc[max_idx, ['demo name', 'source player name', 'game tick number']]
        print(str(i) + ' ' + min_df['demo name'] + ' ' + min_df['source player name'] + ' ' + str(min_df['game tick number']) + ' ' + str(max_df['game tick number']))

    print(name + " longest 5")
    for i in range(5):
        print_seq_min_max(i)

    print(name + " shortest 5")
    for i in range(5):
        print_seq_min_max(-1 * (i+1))

    print(' ')

    return seq_df

if __name__ == "__main__":
    explore_df = pd.read_csv(Path(__file__).parent / '..' / 'data' / 'explore_engagements' / 'train_engagement_dataset.csv')
    explore_seq_df = get_longest_sequences(explore_df, "explore")

    # can't split seuqences based on knife or no knife
    # you get a bunch of knifing followed by a switching to a non-knife
    # and then you get 1 tick non-knifing events wihtout any shooting that make no sense
    # must filter by weapon when doing actual sequence analysis
    not_enemies_df = explore_df[(explore_df['enemy slot filled'] == 0) & (explore_df['shooter active weapon'] != 405)].copy()
    not_enemies_seq_df = get_longest_sequences(not_enemies_df, "not enemies")

    enemies_df = explore_df[(explore_df['enemy slot filled'] == 1) & (explore_df['shooter active weapon'] != 405)].copy()
    enemies_seq_df = get_longest_sequences(enemies_df, "enemies")

    no_knife_df = explore_df[explore_df['shooter active weapon'] != 405].copy()
    no_knife_seq_df = get_longest_sequences(no_knife_df, "no knife")

    issue_row = explore_df[(explore_df['source player name'] == 'blameF') & (explore_df['game tick number'] == 263919)]

    overlapping_sequences = {}
    #for i in range(len(enemies_seq_df)):
    #    for j in range(len(not_enemies_seq_df)):
    #        if enemies_seq_df.loc['Min'] >= not_enemies_seq_df['Min'] and ene
    x = 2
