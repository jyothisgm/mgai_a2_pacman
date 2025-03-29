#%%
# Re-import required packages after code state reset
import pandas as pd
from collections import defaultdict
import ast
import glob

# Load and concatenate all matching CSV files
all_files = glob.glob("match_results_test_*.csv")
df_list = [pd.read_csv(f) for f in all_files]
df = pd.concat(df_list, ignore_index=True)

# Parse 'score' column from string to list
df['score'] = df['score'].apply(ast.literal_eval)

# Sort by match_number
df = df.sort_values(by='match_number').reset_index(drop=True)

# Initialize Elo scores and results tracker
elo_scores = defaultdict(lambda: 1000)
results = defaultdict(lambda: defaultdict(lambda: {'W': 0, 'D': 0, 'L': 0}))

# Elo update function
def update_elo(red_agent, blue_agent, elo_scores, red_score, blue_score, K=16):
    expected_red = 1 / (1 + 10 ** ((elo_scores[blue_agent] - elo_scores[red_agent]) / 400))
    expected_blue = 1 - expected_red

    elo_scores[red_agent] += K * (red_score - expected_red)
    elo_scores[blue_agent] += K * (blue_score - expected_blue)

    elo_scores[red_agent] = max(elo_scores[red_agent], 100)
    elo_scores[blue_agent] = max(elo_scores[blue_agent], 100)
    return elo_scores

# Process match results
for _, row in df.iterrows():
    red, blue = row['red_team'], row['blue_team']
    match_scores = row['score']

    for score in match_scores:
        if score > 0:
            red_score, blue_score = 1, 0
            results[red][blue]['W'] += 1
            results[blue][red]['L'] += 1
        elif score < 0:
            red_score, blue_score = 0, 1
            results[red][blue]['L'] += 1
            results[blue][red]['W'] += 1
        else:
            red_score = blue_score = 0.5
            results[red][blue]['D'] += 1
            results[blue][red]['D'] += 1

        elo_scores = update_elo(red, blue, elo_scores, red_score, blue_score)

# Final Elo rankings
final_elo_df = pd.DataFrame([{'Agent': agent, 'Elo': round(score, 2)} for agent, score in elo_scores.items()])
final_elo_df = final_elo_df.sort_values(by='Elo', ascending=False).reset_index(drop=True)

# W/D/L summary table
agents = sorted(set(df['red_team']).union(set(df['blue_team'])))
records = []
for agent1 in agents:
    row = {'Agent': agent1}
    for agent2 in agents:
        if agent1 == agent2:
            row[agent2] = "-"
        else:
            rec = results[agent1][agent2]
            row[agent2] = f"{rec['W']}/{rec['D']}/{rec['L']}"
    records.append(row)

head_to_head_df = pd.DataFrame(records)

# Display results
print(head_to_head_df)
print(final_elo_df)

# %%
head_to_head_df
#%%
final_elo_df
#%%
