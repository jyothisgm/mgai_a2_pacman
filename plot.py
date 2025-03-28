#%%
# Re-import required packages after code state reset
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from io import StringIO
import ast

# Load the uploaded CSV file again
df_uploaded = pd.read_csv("elo_match_history7.csv")
df_uploaded['score'] = df_uploaded['score'].apply(ast.literal_eval)

# Step 1: Head-to-Head W/D/L Summary
results = defaultdict(lambda: defaultdict(lambda: {'W': 0, 'D': 0, 'L': 0}))
elo_progression = []

for i, row in df_uploaded.iterrows():
    red, blue = row['red_agent'], row['blue_agent']
    scores = row['score']
    for score in scores:
        red_elo, blue_elo = row['elo_red_after'], row['elo_blue_after']

        if score > 0:
            results[red][blue]['W'] += 1
            results[blue][red]['L'] += 1
        elif score < 0:
            results[red][blue]['L'] += 1
            results[blue][red]['W'] += 1
        else:
            results[red][blue]['D'] += 1
            results[blue][red]['D'] += 1

        elo_progression.append({'match': i, 'agent': red, 'elo': red_elo})
        elo_progression.append({'match': i, 'agent': blue, 'elo': blue_elo})

# Convert results to DataFrame
agents = sorted(set(df_uploaded['red_agent']).union(set(df_uploaded['blue_agent'])))
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

# Step 2: Plot Elo progression
elo_df = pd.DataFrame(elo_progression)
plt.figure(figsize=(12, 6))
for agent in elo_df['agent'].unique():
    plt.plot(elo_df[elo_df['agent'] == agent]['match'],
             elo_df[elo_df['agent'] == agent]['elo'], label=agent)

plt.title('Elo Rating Progression (Uploaded CSV)')
plt.xlabel('Match Index')
plt.ylabel('Elo Rating')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Step 3: Dynamic K-Factor Elo Simulation
def dynamic_k_elo_simulation(df, base_k=32, scale_factor=400):
    elo_scores = defaultdict(lambda: 1000)
    progression = []

    for i, row in df.iterrows():
        red, blue = row['red_agent'], row['blue_agent']
            
        scores = row['score']
        for score in scores:

            red_elo, blue_elo = elo_scores[red], elo_scores[blue]
            expected_red = 1 / (1 + 10 ** ((blue_elo - red_elo) / 400))
            expected_blue = 1 - expected_red

            if score > 0:
                red_score, blue_score = 1, 0
            elif score < 0:
                red_score, blue_score = 0, 1
            else:
                red_score = blue_score = 0.5

            k_red = base_k * (2 / (1 + abs(red_elo - 1000) / scale_factor))
            k_blue = base_k * (2 / (1 + abs(blue_elo - 1000) / scale_factor))

            elo_scores[red] += k_red * (red_score - expected_red)
            elo_scores[blue] += k_blue * (blue_score - expected_blue)

            progression.append({'match': i, 'agent': red, 'elo': elo_scores[red]})
            progression.append({'match': i, 'agent': blue, 'elo': elo_scores[blue]})

    final_scores = pd.DataFrame(list(elo_scores.items()), columns=['agent', 'final_elo'])
    elo_trend = pd.DataFrame(progression)
    return final_scores.sort_values(by='final_elo', ascending=False), elo_trend

final_elos_dynamic_k, dynamic_k_trend = dynamic_k_elo_simulation(df_uploaded)

# Plot dynamic K elo progression
plt.figure(figsize=(12, 6))
for agent in dynamic_k_trend['agent'].unique():
    plt.plot(dynamic_k_trend[dynamic_k_trend['agent'] == agent]['match'],
             dynamic_k_trend[dynamic_k_trend['agent'] == agent]['elo'], label=agent)

plt.title('Dynamic K-Factor Elo Rating Progression (Uploaded CSV)')
plt.xlabel('Match Index')
plt.ylabel('Elo Rating')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Step 4: Show tables
print("Head-to-Head W/D/L Summary:")
print(head_to_head_df.to_string(index=False))

# print("\nFinal Elo Scores with Dynamic K-Factor:")
# print(elo_df.to_string(index=False))

print("\nFinal Elo Scores with Dynamic K-Factor:")
print(final_elos_dynamic_k.to_string(index=False))

# %%
