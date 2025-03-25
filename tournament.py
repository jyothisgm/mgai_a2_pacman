
import os
import argparse
from itertools import combinations
import sys
import pandas as pd
from capture import readCommand, runGames, save_score

import random

def get_agent_files(folder_path,  leaderboard_f, match_history_f):
    agents = []
    for file in os.listdir(folder_path):
        if file.endswith(".py"):
            agents.append(folder_path+"/"+file) 
    
    all_elo_scores = choose_combination_(agents, leaderboard_f, match_history_f)
    return all_elo_scores

def update_elo(red_agent, blue_agent, elo_scores, red_score, blue_score, K=32):
    expected_red = 1 / (1 + 10 ** ((elo_scores[blue_agent] - elo_scores[red_agent]) / 400))
    expected_blue = 1 - expected_red

    elo_scores[red_agent] += K * (red_score - expected_red)
    elo_scores[blue_agent] += K * (blue_score - expected_blue)

    # Optional minimum rating
    elo_scores[red_agent] = max(elo_scores[red_agent], 100)
    elo_scores[blue_agent] = max(elo_scores[blue_agent], 100)

def get_elo_score(red, blue, elo_scores, scores_capture):
    for score in scores_capture:
        
    
        if score > 0:
            red_score, blue_score = 1, 0
         
        elif score < 0:
            red_score, blue_score = 0, 1
   
        else:
            red_score = blue_score = 0.5  # Tie
  

        update_elo(red, blue, elo_scores, red_score, blue_score)
    
    
def choose_combination_(agents,   leaderboard_file, match_history_file):
    
    elo_scores = {agent: 1500 for agent in agents}
    for _ in range(20):
        matchups = list(combinations(agents, 2))
        random.shuffle(matchups)
        
        match_history = []
        match_count =1
        max_elo = 1500
        
        for red, blue in matchups:
            
        
            scores = run_match(red, blue, )
            get_elo_score(red, blue, elo_scores, scores)
            
            match_history.append({
                    "match_number": match_count,
                    "red_agent":  os.path.basename(red),
                    "blue_agent": os.path.basename(blue),
                
                    "elo_red_after": round(elo_scores[red], 2),
                    "elo_blue_after": round(elo_scores[blue], 2)
                })
            match_count +=1
            
            scores = run_match(blue, red)
            get_elo_score(blue, red, elo_scores, scores)
            
            match_history.append({
                    "match_number": match_count,
                    "red_agent": os.path.basename(blue),
                    "blue_agent": os.path.basename(red),
                
                    "elo_red_after": round(elo_scores[blue], 2),
                    "elo_blue_after": round(elo_scores[red], 2)
                })
            match_count +=1
    df = pd.DataFrame(match_history)
    
    df.to_csv(match_history_file, index=False)
    print(f" Match history saved to {match_history_file}")
    
    sorted_elo = sorted(elo_scores.items(), key=lambda x: -x[1])

        # Build a list of dicts for each agent
    ranking_data = [
        {"rank": i + 1, "agent": os.path.basename(agent), "elo_score": round(score, 2)}
        for i, (agent, score) in enumerate(sorted_elo)
    ]

    # Convert to DataFrame
    elo_ranking_df = pd.DataFrame(ranking_data)
    
    elo_ranking_df.to_csv(leaderboard_file, index=False)
    print(f"Leaderboard saved to {leaderboard_file}")

    
    


def run_match(red_file, blue_file):
    # Build command-line-like args
    args = [
        "-r", red_file,
        "-b", blue_file,
  
        # "--numGames", str(num_games),
        "-q"
      
    ]

    print(f"\nRunning match: RED = {red_file} vs BLUE = {blue_file}\n")
    options = readCommand(args)
    games = runGames(**options)
    return [g.state.data.score for g in games]
    


def main():
    # Setup argument parser
    parser = argparse.ArgumentParser(description="Run Pacman agent tournament")
    parser.add_argument(
        "--folder", type=str, default="tourna_files",
        help="Folder containing agent files"
    )
    # parser.add_argument(
    #     "--games", type=int, default=2,
    #     help="Number of games per match"
    # )
    args = parser.parse_args()

    # Construct agent folder path
    base_dir = os.path.dirname(__file__)
    agents_dir = os.path.join(base_dir, args.folder)
  
    leaderboard_file = "elo_leaderboard.csv"
    match_history_file = "elo_match_history.csv"

    # Run tournament
    agent_tournament = get_agent_files(agents_dir, leaderboard_file, match_history_file)
     
if __name__ == "__main__":
    main()


