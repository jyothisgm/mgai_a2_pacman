import os
import csv
import multiprocessing
import random
import traceback
from itertools import combinations
from capture import readCommand, runGames

AGENT_FOLDER = "heutour"
MATCHES_PER_PAIR = 500
MAX_PARALLEL = 20
OUTPUT_CSV = "match_results.csv"
LAYOUT = "default"

csv_lock = multiprocessing.Lock()

def list_agents():
    return [f[:-3] for f in os.listdir(AGENT_FOLDER) if f.endswith(".py")]

def run_single_match(args):
    match_number, red_file, blue_file, layout = args

    print(f"\nRunning match: RED = {red_file} vs BLUE = {blue_file}\n")

    red_path = os.path.join(AGENT_FOLDER, red_file + ".py")
    blue_path = os.path.join(AGENT_FOLDER, blue_file + ".py")

    try:
        args = readCommand([
            '-r', red_path,
            '-b', blue_path,
            '--quiet',
            '-c'
        ])
        games = runGames(**args)
        score = [g.state.data.score for g in games]
    except Exception as e:
        score = "Error"
        print(f"Error in match {match_number}: {e}")
        traceback.print_exc()  # 🔥 Print full traceback

    row = {
        "match_number": match_number,
        "red_team": red_file,
        "blue_team": blue_file,
        "score": score
    }

    with csv_lock:
        write_row_to_csv(row)

    return f"Match {match_number} complete"

def write_row_to_csv(row):
    file_exists = os.path.exists(OUTPUT_CSV)
    with open(OUTPUT_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["match_number", "red_team", "blue_team", "score"])
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

if __name__ == "__main__":
    agents = list_agents()
    if len(agents) < 2:
        raise ValueError("Need at least 2 agents in the heutour folder.")

    match_args = []
    match_id = 1
    for a1, a2 in combinations(agents, 2):
        for i in range(MATCHES_PER_PAIR // 2):
            match_args.append((match_id, a1, a2, f"RANDOM{i}"))  # a1 red, a2 blue
            match_id += 1
            match_args.append((match_id, a2, a1, f"RANDOM{i}"))  # a2 red, a1 blue
            match_id += 1
    random.shuffle(match_args)
    if os.path.exists(OUTPUT_CSV):
        os.remove(OUTPUT_CSV)

    with multiprocessing.Pool(processes=MAX_PARALLEL) as pool:
        pool.map(run_single_match, match_args)

    print(f"\n✅ All matches complete. Results saved to {OUTPUT_CSV}")
