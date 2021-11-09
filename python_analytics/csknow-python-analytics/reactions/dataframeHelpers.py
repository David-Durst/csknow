import pandas as pd

plot_titles = []
visibility_techniques = ['Pixel Adjusted', 'Pixel Unadjusted', 'Bounding Box']
for i, visibility_technique in enumerate(visibility_techniques):
    plot_titles.append([])
    for hacking_approach in ['Pro', 'Hacking Amateur', 'Legit Amateur']:
        plot_titles[i].append(f'''{visibility_technique}, {hacking_approach}''')

class HackingTypeDataFrames:
    pro_df: pd.DataFrame
    hacks_df: pd.DataFrame
    legit_df: pd.DataFrame

    def __init__(self, visibility_technique_id, filtered_df):
        self.pro_df = filtered_df[(filtered_df['hacking'] == 2) &
                                  (filtered_df['visibility_technique_id'] == visibility_technique_id)]
        self.hacks_df = filtered_df[(filtered_df['hacking'] == 1) &
                                    (filtered_df['visibility_technique_id'] == visibility_technique_id)]
        self.legit_df = filtered_df[(filtered_df['hacking'] == 0) &
                                    (filtered_df['visibility_technique_id'] == visibility_technique_id)]

    def get_as_array(self):
        return [self.pro_df, self.hacks_df, self.legit_df]

    def print_size(self, df_outer_name):
        print(f'''{df_outer_name} pro size {len(self.pro_df)}, hacking size {len(self.hacks_df)}, ''' +
              f'''legit size {len(self.legit_df)}''')

    def get_hacks_union_legit(self):
        return pd.concat([self.hacks_df, self.legit_df]).reset_index(drop=True)


class VisibilityTechniqueDataFrames:
    unfiltered_df: pd.DataFrame
    filtered_df: pd.DataFrame
    pix_adjusted_dfs: HackingTypeDataFrames
    pix_unadjusted_dfs: HackingTypeDataFrames
    bbox_dfs: HackingTypeDataFrames

    def __init__(self, unfiltered_df, filtered_df):
        self.unfiltered_df = unfiltered_df
        self.filtered_df = filtered_df
        self.pix_adjusted_dfs = HackingTypeDataFrames(0, filtered_df)
        self.pix_unadjusted_dfs = HackingTypeDataFrames(1, filtered_df)
        self.bbox_dfs = HackingTypeDataFrames(2, filtered_df)

    def get_as_grid(self):
        return [self.pix_adjusted_dfs.get_as_array(), self.pix_unadjusted_dfs.get_as_array(),
                self.bbox_dfs.get_as_array()]

    def print_size(self):
        print(f'''unfiltered size {len(self.unfiltered_df)}''')
        print(f'''filtered size {len(self.filtered_df)}''')
        self.pix_adjusted_dfs.print_size('pixel adjusted')
        self.pix_unadjusted_dfs.print_size('pixel unadjusted')
        self.bbox_dfs.print_size('bbox')
