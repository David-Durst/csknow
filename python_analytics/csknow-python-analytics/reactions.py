import psycopg2
import argparse
import pandas.io.sql as sqlio
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("password", help="database password",
                    type=str)
parser.add_argument("query_file", help="file containing query",
                    type=str)
parser.add_argument("plot_folder", help="folder_for_plots",
                    type=str)
args = parser.parse_args()

conn = psycopg2.connect(
    host="localhost",
    database="csknow",
    user="postgres",
    password=args.password,
    port=3125)
cur = conn.cursor()

with open(args.query_file, 'r') as query_file:
    cur.execute(query_file.read())

unfiltered_df = sqlio.read_sql_query("select * from react_ticks", conn)
hand_filtered_df = sqlio.read_sql_query("select * from react_final where hand_react_ms <= 3 and hand_react_ms >= -3", conn)
cpu_filtered_df = sqlio.read_sql_query("select * from react_final where cpu_react_ms <= 3 and cpu_react_ms >= -3", conn)

hacks_hand_filtered_df = hand_filtered_df[hand_filtered_df['hacking']]
hacks_cpu_filtered_df = cpu_filtered_df[cpu_filtered_df['hacking']]
legit_hand_filtered_df = hand_filtered_df[~hand_filtered_df['hacking']]
legit_cpu_filtered_df = cpu_filtered_df[~cpu_filtered_df['hacking']]

print(f"""total size {len(unfiltered_df)}, hacks hand size {len(hacks_hand_filtered_df)}, """ +
      f"""hacks cpu size {len(hacks_cpu_filtered_df)}, legit hand size {len(legit_hand_filtered_df)}, """ +
      f"""legit cpu size {len(legit_cpu_filtered_df)}""")

fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(16, 16))

hacks_hand_filtered_df.hist('hand_react_ms', bins=60, ax=ax[0][0])
hacks_cpu_filtered_df.hist('cpu_react_ms', bins=60, ax=ax[1][0])
legit_hand_filtered_df.hist('hand_react_ms', bins=60, ax=ax[0][1])
legit_cpu_filtered_df.hist('cpu_react_ms', bins=60, ax=ax[1][1])

for i in range(len(ax)):
    for j in range(len(ax[i])):
        ax[i][j].set_xlim(-3, 3)
        if i == 0:
            ax[i][j].set_ylim(0, 50)
        else:
            ax[i][j].set_ylim(0, 12)
        ax[i][j].set_xlabel('Reaction Time (s)', fontsize=14)
        ax[i][j].set_ylabel('Frequency', fontsize=14)

ax[0][0].set_title("Hand Labeled, Hacking", fontsize=18)
ax[0][1].set_title("Hand Labeled, Not Hacking", fontsize=18)
ax[1][0].set_title("CPU Labeled, Hacking", fontsize=18)
ax[1][1].set_title("CPU Labeled, Not Hacking", fontsize=18)

ax[0][0].annotate('total points: ' + str(len(hacks_hand_filtered_df)), (1.1,45), fontsize="14")
ax[0][1].annotate('total points: ' + str(len(legit_hand_filtered_df)), (1.1,45), fontsize="14")
ax[1][0].annotate('total points: ' + str(len(hacks_cpu_filtered_df)), (1.1,11.2), fontsize="14")
ax[1][1].annotate('total points: ' + str(len(legit_cpu_filtered_df)), (1.1,11.2), fontsize="14")

plt.tight_layout()
fig.savefig(args.plot_folder + "hand_vs_cpu__hacking_vs_legit.png")

