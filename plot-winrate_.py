import pandas as pd
from collections import defaultdict
import ast

# Load and parse the CSV
df = pd.read_csv("match_results_mcts_final2.csv")
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

# Compute final Elo rankings with W/D/L + Win Rate
agents = sorted(set(df['red_team']).union(set(df['blue_team'])))
records = []

for agent in agents:
    total_W = total_D = total_L = 0
    for opponent in agents:
        if agent == opponent:
            continue
        rec = results[agent][opponent]
        total_W += rec['W']
        total_D += rec['D']
        total_L += rec['L']
    total_matches = total_W + total_D + total_L
    win_rate = (total_W + 0.5 * total_D) / total_matches if total_matches > 0 else 0

    records.append({
        'Agent': agent,
        'Elo': round(elo_scores[agent], 2),
        'Wins/Draws/Losses': f"{total_W}/{total_D}/{total_L}",
        'Win Rate (%)': round(win_rate * 100, 2)
    })

final_elo_df = pd.DataFrame(records)
final_elo_df = final_elo_df.sort_values(by='Elo', ascending=False).reset_index(drop=True)

# Create head-to-head table
head_to_head_records = []
for agent1 in agents:
    row = {'Agent': agent1}
    for agent2 in agents:
        if agent1 == agent2:
            row[agent2] = "-"
        else:
            rec = results[agent1][agent2]
            w, d, l = rec['W'], rec['D'], rec['L']
            total = w + d + l
            win_rate = (w + 0.5 * d) / total if total > 0 else 0
            row[agent2] = f"{w}/{d}/{l} ({round(win_rate * 100, 1)}%)"
    head_to_head_records.append(row)

head_to_head_df = pd.DataFrame(head_to_head_records)

# Display outputs
print("\n=== Final Elo Rankings with W/D/L and Win Rate ===")
print(final_elo_df.to_string(index=False))

print("\n=== Head-to-Head Results (W/D/L) with Win Rate ===")
print(head_to_head_df.to_string(index=False))
