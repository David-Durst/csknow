from learn_bot.latent.analyze.run_comparison import *

bot_similarity_hdf5_data_path = Path(__file__).parent / '..' / '..' / '..' / '..' / 'analytics' / 'manual_outputs' / \
                                 'botTrajectorySimilarity.hdf5'

def aggregate_analyze_comparison():
    similarity_df = load_hdf5_to_pd(bot_similarity_hdf5_data_path)
    


if __name__ == "__main__":
    aggregate_analyze_comparison()

