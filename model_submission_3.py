# Pokemon Battles Prediction 2025 - Enhanced Version
# FDS Kaggle Competition Solution - Advanced Feature Engineering
# This file will be imported as a module: model_submission_3.py

"""
===============================================================================
                          Feature Engineering Summary
===============================================================================

This solution uses approximately 233 carefully designed features, divided into three categories:

[1. Static Features (60 features)] - From team and Pokémon base stats
    
    1. P1 Team Stat Aggregates (30 features)
       - 6 base stats: HP, ATK, DEF, SPA, SPD, SPE
       - 5 aggregation methods: sum, mean, max, min, std
       - Examples: p1_team_hp_sum, p1_team_atk_mean, p1_team_spe_max
       
    2. P2 Lead Pokémon Stats (6 features)
       - p2_lead_hp, p2_lead_atk, p2_lead_def, p2_lead_spa, p2_lead_spd, p2_lead_spe
       
    3. Stat Comparison Features (12 features)
       - Advantage (6): hp_advantage, atk_advantage, def_advantage, etc.
       - Ratio (6): hp_ratio, atk_ratio, def_ratio, etc.
       
    4. Type Advantage Features (4 features)
       - type_adv_mean: Team's average type advantage against P2 lead
       - type_adv_max: Maximum type advantage
       - type_adv_min: Minimum type advantage
       - type_adv_std: Standard deviation of type advantage
       
    5. Team Diversity Features (2 features)
       - type_diversity: Type diversity (unique types / total types)
       - stat_diversity: Stat diversity (coefficient of variation)
       
    6. Team Balance Features (3 features)
       - physical_special_atk_ratio: Physical/Special Atk ratio
       - physical_special_def_ratio: Physical/Special Def ratio
       - offense_defense_ratio: Offense/Defense ratio
       
    7. Overall Strength Features (3 features)
       - p1_total_stats: P1 team total stats
       - p2_total_stats: P2 lead total stats
       - total_stats_advantage: Total stat advantage

[2. Dynamic Features (174 features)] - From battle timeline analysis

    1. Basic Info (1 feature)
       - total_turns: Total battle turns
       
    2. Overall HP Stats (14 features)
       - 7 per player: start, end, min, max, avg, std, trend (slope)
       - Examples: p1_hp_start, p2_hp_end, p1_hp_trend
       
    3. Phased HP Stats (12 features) *New*
       - Three phases: early(1/3), mid(1/3), late(1/3)
       - 2 per player per phase: avg(HP), min(HP)
       - Examples: p1_early_hp_avg, p2_late_hp_min
       
    4. HP Loss Rate (4 features)
       - p1_avg_hp_loss, p1_max_hp_loss: P1 avg and max HP loss
       - p2_avg_hp_loss, p2_max_hp_loss: P2 avg and max HP loss
       
    5. HP Advantage Metrics (3 features)
       - hp_advantage_start: Starting HP advantage
       - hp_advantage_end: Ending HP advantage
       - hp_advantage_avg: Average HP advantage
       
    6. Move Diversity (2 features)
       - p1_move_diversity: P1 move diversity (unique moves / total moves)
       - p2_move_diversity: P2 move diversity
       
    7. Total Status Changes (2 features)
       - p1_status_changes: P1 status abnormal turns
       - p2_status_changes: P2 status abnormal turns
       
    8. Abnormal Status Stats (1 feature) *New*
       - abnormal_status_count_ratio: (P1 Pokémon with abnormal status in first 30 turns) / (P2 Pokémon with abnormal status + 1)
       
    9. Counter Move Stats (2 features) *New*
       - p1_counter_invalid: P1 used Counter ineffectively
       - p2_counter_invalid: P2 used Counter ineffectively
       
   10. Move Power Stats (6 features)
       - 3 per player: avg_move_power, max_move_power, min_move_power
       
   11. Move Accuracy (2 features)
       - p1_avg_accuracy, p2_avg_accuracy
       
   12. Move Category Ratio (6 features)
       - 3 per player: physical_move_ratio, special_move_ratio, status_move_ratio
       
   13. Switching Stats (10 features) *Enhanced*
       - p1_switch_count, p2_switch_count
       - p1_early_switch_ratio, p2_early_switch_ratio
       - p1_move_null_switch, p2_move_null_switch (null move due to switch)
       - p1_move_null_status, p2_move_null_status (null move due to status)
       
   14. Move Usage Count (80 features) *New*
       - 40 moves x 2 players
       - Format: p1_move_{skill_name}_count, p2_move_{skill_name}_count
       
   15. Consecutive Attack Stats (2 features) *New*
       - p1_max_consecutive_attacks, p2_max_consecutive_attacks
       
   16. Boost Stats (13 features)
       - 6 per player (final_boost_sum, atk, def, spa, spd, spe)
       - boost_advantage: Difference in final boost sums
       
   17. Field Effect Stats (2 features)
       - p1_effect_turns, p2_effect_turns
       
   18. Pokémon Count (first 30 turns) (2 features) *New*
       - p1_unique_pokemon_count_30turns
       - p2_unique_pokemon_count_30turns
       
   19. Total Team HP (first 30 turns) (3 features) *New*
       - p1_total_pokemon_hp_pct_30turns, p2_total_pokemon_hp_pct_30turns
       - total_pokemon_hp_pct_ratio_30turns (P1/P2 ratio)
       - Unseen Pokémon are counted as 1.0 HP
       
   20. Fainted Pokémon Count (2 features) *New*
       - fnt_count_ratio: p1_fnt_count / (p2_fnt_count + 1)
       - fnt_count_diff: p1_fnt_count - p2_fnt_count
       
   21. Type Effectiveness Multiplier (2 features) *New*
       - p1_avg: P1 team's avg type effectiveness vs P2 team
       - p2_avg: P2 team's avg type effectiveness vs P1 team
       
   22. Priority Move Sum (2 features) *New*
       - p1_num_priority_moves: Sum of P1's priority move values
       - p2_num_priority_moves: Sum of P2's priority move values

[3. Interaction Features (4 features)] - Combined features
    1. hp_boost_interaction: HP Advantage × Boost Advantage
    2. p1_effective_power: P1 Avg Power × P1 Avg Accuracy
    3. p2_effective_power: P2 Avg Power × P2 Avg Accuracy
    4. type_stats_interaction: Type Advantage Mean × Total Stats Advantage
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE, RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import warnings
import os
import sys
import optuna
from optuna.pruners import MedianPruner


def extract_final_pokemon_status(timeline):
    """
    Extracts the final status of Pokémon from the battle timeline.
    
    Returns:
    - p1_alive_pokemon_names: Set of P1's final surviving Pokémon names
    - p1_fnt_pokemon_names: Set of P1's final fainted Pokémon names
    - p2_alive_pokemon_names: Set of P2's final surviving Pokémon names
    """
    p1_alive_pokemon_names = set()
    p1_fnt_pokemon_names = set()
    p2_alive_pokemon_names = set()
    p1_last_status = {}
    p2_last_status = {}
    for turn in timeline:
        p1_pokemon_state = turn.get('p1_pokemon_state', {})
        p1_pokemon_name = p1_pokemon_state.get('name', '')
        if p1_pokemon_name:
            p1_last_status[p1_pokemon_name] = p1_pokemon_state.get('status', 'nostatus')

        p2_pokemon_state = turn.get('p2_pokemon_state', {})
        p2_pokemon_name = p2_pokemon_state.get('name', '')
        if p2_pokemon_name:
            p2_last_status[p2_pokemon_name] = p2_pokemon_state.get('status', 'nostatus')

    for pokemon_name, status in p1_last_status.items():
        if status == 'fnt':
            p1_fnt_pokemon_names.add(pokemon_name)
        else:
            p1_alive_pokemon_names.add(pokemon_name)

    for pokemon_name, status in p2_last_status.items():
        if status != 'fnt':
            p2_alive_pokemon_names.add(pokemon_name)

    return p1_alive_pokemon_names, p1_fnt_pokemon_names, p2_alive_pokemon_names

def extract_moves(data: dict):
    """
    Extracts the set of moves used by each player during the battle.
    
    This function iterates through the battle timeline, recording all moves used by each Pokémon. 
    If a Pokémon faints, its move records are removed, as fainted Pokémon 
    should not be included in the final feature calculation.
    
    Args:
        data (dict): Battle data dictionary, must contain "battle_timeline" key.
    
    Returns:
        p1_pokemon_moves (dict): Player 1's Pokémon moves dictionary
            Format: {pokemon_name: [move_details_1, move_details_2, ...]}
        p2_pokemon_moves (dict): Player 2's Pokémon moves dictionary
    """
    p1_pokemon_moves = {}
    p2_pokemon_moves = {}

    for turn in data.get("battle_timeline", []):
        p1_name = turn.get("p1_pokemon_state", {}).get("name", "")
        p2_name = turn.get("p2_pokemon_state", {}).get("name", "")

        p1_move = turn.get("p1_move_details")
        if p1_move is not None and (p1_name not in p1_pokemon_moves or p1_move not in p1_pokemon_moves[p1_name]):
            p1_pokemon_moves.setdefault(p1_name, []).append(p1_move)
        
        p2_move = turn.get("p2_move_details")
        if p2_move is not None and (p2_name not in p2_pokemon_moves or p2_move not in p2_pokemon_moves[p2_name]):
            p2_pokemon_moves.setdefault(p2_name, []).append(p2_move)

        p1_status = turn.get("p1_pokemon_state", {}).get("status", "")
        if p1_status == "fnt":
            p1_pokemon_moves.pop(p1_name, None)
        
        p2_status = turn.get("p2_pokemon_state", {}).get("status", "")
        if p2_status == "fnt":
            p2_pokemon_moves.pop(p2_name, None)

    return p1_pokemon_moves, p2_pokemon_moves

def type_multiplier(p1_moves: dict, p2_moves: dict):
    """
    Calculates the average type effectiveness multiplier for each team.
    
    Args:
        p1_moves (dict): Player 1's Pokémon moves dictionary.
        p2_moves (dict): Player 2's Pokémon moves dictionary.
    
    Returns:
        p1_team_avg (float): P1 team's average type effectiveness against P2 team.
        p2_team_avg (float): P2 team's average type effectiveness against P1 team.
    """
    type_pokemon1 = {}
    type_pokemon2 = {}

    for pokemon, moves in p1_moves.items():
        type_pokemon1[pokemon] = []
        for move in moves:
            if move and move.get("type") and move.get("base_power") and move.get("category") != "STATUS":
                type_pokemon1[pokemon].append({
                    "type": move["type"].capitalize(),
                    "power": move["base_power"] * move.get("accuracy", 1.0)
                })

    for pokemon, moves in p2_moves.items():
        type_pokemon2[pokemon] = []
        for move in moves:
            if move and move.get("type") and move.get("base_power") and move.get("category") != "STATUS":
                type_pokemon2[pokemon].append({
                    "type": move["type"].capitalize(),
                    "power": move["base_power"] * move.get("accuracy", 1.0)
                })

    diz_multiplier_my_pokemon = {}
    diz_multiplier_other_pokemon = {}

    for pokemon1, moves1 in type_pokemon1.items():
        total_effectiveness = []
        for pokemon2, moves2 in type_pokemon2.items():
            multiplier = 1.0
            for move in moves1:
                t_att = move["type"]
                base_power = move["power"]
                if t_att not in TABLE_TYPE:
                    continue
                super_eff, meno_eff, no_eff = TABLE_TYPE[t_att]
                for t_def in P_DEF_TYPE.get(pokemon2.lower(), []):
                    if t_def in no_eff:
                        multiplier *= 0.0
                    elif t_def in super_eff:
                        multiplier *= 2.0
                    elif t_def in meno_eff:
                        multiplier *= 0.5
                multiplier *= (base_power / 100.0)
            total_effectiveness.append(multiplier)
        if total_effectiveness:
            diz_multiplier_my_pokemon[pokemon1] = np.mean(total_effectiveness)
        else:
            diz_multiplier_my_pokemon[pokemon1] = 0.0

    for pokemon2, moves2 in type_pokemon2.items():
        total_effectiveness = []
        for pokemon1, moves1 in type_pokemon1.items():
            multiplier = 1.0
            for move in moves2:
                t_att = move["type"]
                base_power = move["power"]
                if t_att not in TABLE_TYPE:
                    continue
                super_eff, meno_eff, no_eff = TABLE_TYPE[t_att]
                for t_def in P_DEF_TYPE.get(pokemon1.lower(), []):
                    if t_def in no_eff:
                        multiplier *= 0.0
                    elif t_def in super_eff:
                        multiplier *= 2.0
                    elif t_def in meno_eff:
                        multiplier *= 0.5
                multiplier *= (base_power / 100.0)
            total_effectiveness.append(multiplier)
        if total_effectiveness:
            diz_multiplier_other_pokemon[pokemon2] = np.mean(total_effectiveness)
        else:
            diz_multiplier_other_pokemon[pokemon2] = 0.0

    if diz_multiplier_my_pokemon:
        p1_team_avg = np.mean(list(diz_multiplier_my_pokemon.values()))
    else:
        p1_team_avg = 0.0
    if diz_multiplier_other_pokemon:
        p2_team_avg = np.mean(list(diz_multiplier_other_pokemon.values()))
    else:
        p2_team_avg = 0.0

    return p1_team_avg, p2_team_avg

def count_priority_moves(pokemon_moves: dict) -> int:
    """
    Counts the total priority value of all priority moves used by the Pokémon.
    
    Args:
        pokemon_moves (dict): Pokémon moves dictionary.
    
    Returns:
        int: Total sum of priority values for all priority moves.
    """
    return sum(move.get("priority", 0) for moves in pokemon_moves.values() for move in moves)

# ============================================================================
# Type Effectiveness Chart (for type_multiplier feature)
# ============================================================================
TABLE_TYPE = {
    "Normal": ([], ["Rock", "Steel"], ["Ghost"]),
    "Fire": (["Grass", "Ice", "Bug", "Steel"],
             ["Fire", "Water", "Rock", "Dragon"], []),
    "Water": (["Fire", "Ground", "Rock"],
              ["Water", "Grass", "Dragon"], []),
    "Electric": (["Water", "Flying"],
                 ["Electric", "Grass", "Dragon"], ["Ground"]),
    "Grass": (["Water", "Ground", "Rock"],
              ["Fire", "Grass", "Poison", "Flying", "Bug", "Dragon", "Steel"], []),
    "Ice": (["Grass", "Ground", "Flying", "Dragon"],
            ["Fire", "Water", "Ice", "Steel"], []),
    "Fighting": (["Normal", "Ice", "Rock", "Dark", "Steel"],
                 ["Poison", "Flying", "Psychic", "Bug", "Fairy"], []),
    "Poison": (["Grass", "Fairy"],
               ["Poison", "Ground", "Rock", "Ghost"], []),
    "Ground": (["Fire", "Electric", "Poison", "Rock", "Steel"],
               ["Grass", "Bug"], ["Flying"]),
    "Flying": (["Grass", "Fighting", "Bug"],
               ["Electric", "Rock", "Steel"], []),
    "Psychic": (["Fighting", "Poison"],
                ["Psychic", "Steel"], ["Dark"]),
    "Bug": (["Grass", "Psychic", "Dark"],
            ["Fire", "Fighting", "Poison", "Flying", "Ghost", "Steel", "Fairy"], []),
    "Rock": (["Fire", "Ice", "Flying", "Bug"],
             ["Fighting", "Ground", "Steel"], []),
    "Ghost": (["Psychic", "Ghost"],
              ["Dark"], ["Normal"]),
    "Dragon": (["Dragon"],
               ["Steel"], ["Fairy"]),
    "Dark": (["Psychic", "Ghost"],
             ["Fighting", "Dark", "Fairy"], []),
    "Steel": (["Ice", "Rock", "Fairy"],
              ["Fire", "Water", "Electric", "Steel"], []),
    "Fairy": (["Fighting", "Dragon", "Dark"],
              ["Fire", "Poison", "Steel"], [])
}

# Pokémon Defense Types (Gen 1 common Pokémon)
P_DEF_TYPE = {
    "starmie": ["psychic", "water"],
    "exeggutor": ["grass", "psychic"],
    "chansey": ["normal"],
    "snorlax": ["normal"],
    "tauros": ["normal"],
    "alakazam": ["psychic"],
    "jynx": ["ice", "psychic"],
    "slowbro": ["psychic", "water"],
    "gengar": ["ghost", "poison"],
    "rhydon": ["ground", "rock"],
    "zapdos": ["electric", "flying"],
    "cloyster": ["ice", "water"],
    "golem": ["ground", "rock"],
    "jolteon": ["electric"],
    "articuno": ["flying", "ice"],
    "persian": ["normal"],
    "lapras": ["ice", "water"],
    "dragonite": ["dragon", "flying"],
    "victreebel": ["grass", "poison"],
    "charizard": ["fire", "flying"]
}

# ============================================================================
# Logging System: Saves print output to a file
# ============================================================================
_log_file_handle = None
_original_print = print

def setup_logging(log_dir='print_log'):
    """
    Sets up the logging system to save print output to a file.
    
    Args:
    - log_dir: Directory to save log files (default 'print_log')
    
    Returns:
    - Log file path
    """
    global _log_file_handle
    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file_path = os.path.join(log_dir, f'print_log_{timestamp}.log')
    
    _log_file_handle = open(log_file_path, 'w', encoding='utf-8')
    
    return log_file_path

def log_print(*args, **kwargs):
    """
    Custom print function that outputs to both console and log file.
    """
    global _log_file_handle, _original_print
    
    file_param = kwargs.get('file', None)
    
    _original_print(*args, **kwargs)
    
    if _log_file_handle is not None and file_param is None:
        try:
            sep = kwargs.get('sep', ' ')
            end = kwargs.get('end', '\n')
            
            message = sep.join(str(arg) for arg in args) + end
            
            _log_file_handle.write(message)
            _log_file_handle.flush()
        except Exception as e:
            _original_print(f"Warning: Failed to write to log file: {e}", file=sys.stderr)

def close_logging():
    """Closes the log file."""
    global _log_file_handle
    if _log_file_handle is not None:
        _log_file_handle.close()
        _log_file_handle = None

print = log_print

class PokemonBattlePredictorEnhanced:
    """Pokemon Battle Predictor Main Class - Enhanced Version"""

    def __init__(self):
        self.train_data = None
        self.test_data = None
        self.features = None
        self.target = None
        self.models = {}
        self.feature_importance = {}
        self.scaler = StandardScaler()
        self.feature_selector = None
        self.X_val = None
        self.y_val = None
        self.best_model = None
        self.best_model_name = None
        self.all_moves_list = None  # Stores all move list
        # self.turn30_winrates = None  # Stores turn 30 Pokémon combo win rates
        
        # Feature removal configuration list
        self.features_to_remove = []

        # Validation set split configuration
        self.validation_split = 0.1

        # Optuna hyperparameter tuning configuration
        self.use_optuna_tuning = False
        self.optuna_n_trials = 50
        self.optuna_timeout = 3600
        self.optuna_cv_folds = 3
        self.optuna_pruner_warmup = 5
        self.optuna_pruner_interval = 1

        self.optuna_best_params = {}
        self.optuna_studies = {}

    def load_data(self, train_path, test_path):
        """Loads training and test data"""
        print("Loading data...")

        train_records = []
        with open(train_path, 'r', encoding='utf-8') as f:
            for line in f:
                train_records.append(json.loads(line.strip()))

        test_records = []
        with open(test_path, 'r', encoding='utf-8') as f:
            for line in f:
                test_records.append(json.loads(line.strip()))

        self.train_data = pd.DataFrame(train_records)
        self.test_data = pd.DataFrame(test_records)

        print(f"Training data loaded: {len(self.train_data)} records")
        print(f"Test data loaded: {len(self.test_data)} records")

        return self.train_data, self.test_data

    def get_complete_type_effectiveness(self):
        """Complete Pokemon type effectiveness chart (Gen 1-9)"""
        effectiveness = {
            'normal': {'rock': 0.5, 'ghost': 0.0, 'steel': 0.5},
            'fire': {'fire': 0.5, 'water': 0.5, 'grass': 2.0, 'ice': 2.0, 'bug': 2.0, 'rock': 0.5, 'dragon': 0.5, 'steel': 2.0},
            'water': {'fire': 2.0, 'water': 0.5, 'grass': 0.5, 'ground': 2.0, 'rock': 2.0, 'dragon': 0.5},
            'electric': {'water': 2.0, 'electric': 0.5, 'grass': 0.5, 'ground': 0.0, 'flying': 2.0, 'dragon': 0.5},
            'grass': {'fire': 0.5, 'water': 2.0, 'grass': 0.5, 'poison': 0.5, 'ground': 2.0, 'flying': 0.5, 'bug': 0.5, 'rock': 2.0, 'dragon': 0.5, 'steel': 0.5},
            'ice': {'fire': 0.5, 'water': 0.5, 'grass': 2.0, 'ice': 0.5, 'ground': 2.0, 'flying': 2.0, 'dragon': 2.0, 'steel': 0.5},
            'fighting': {'normal': 2.0, 'ice': 2.0, 'poison': 0.5, 'flying': 0.5, 'psychic': 0.5, 'bug': 0.5, 'rock': 2.0, 'ghost': 0.0, 'dark': 2.0, 'steel': 2.0, 'fairy': 0.5},
            'poison': {'grass': 2.0, 'poison': 0.5, 'ground': 0.5, 'rock': 0.5, 'ghost': 0.5, 'steel': 0.0, 'fairy': 2.0},
            'ground': {'fire': 2.0, 'electric': 2.0, 'grass': 0.5, 'poison': 2.0, 'flying': 0.0, 'bug': 0.5, 'rock': 2.0, 'steel': 2.0},
            'flying': {'electric': 0.5, 'grass': 2.0, 'fighting': 2.0, 'bug': 2.0, 'rock': 0.5, 'steel': 0.5},
            'psychic': {'fighting': 2.0, 'poison': 2.0, 'psychic': 0.5, 'dark': 0.0, 'steel': 0.5},
            'bug': {'fire': 0.5, 'grass': 2.0, 'fighting': 0.5, 'poison': 0.5, 'flying': 0.5, 'psychic': 2.0, 'ghost': 0.5, 'dark': 2.0, 'steel': 0.5, 'fairy': 0.5},
            'rock': {'fire': 2.0, 'ice': 2.0, 'fighting': 0.5, 'ground': 0.5, 'flying': 2.0, 'bug': 2.0, 'steel': 0.5},
            'ghost': {'normal': 0.0, 'psychic': 2.0, 'ghost': 2.0, 'dark': 0.5},
            'dragon': {'dragon': 2.0, 'steel': 0.5, 'fairy': 0.0},
            'dark': {'fighting': 0.5, 'psychic': 2.0, 'ghost': 2.0, 'dark': 0.5, 'fairy': 0.5},
            'steel': {'fire': 0.5, 'water': 0.5, 'electric': 0.5, 'ice': 2.0, 'rock': 2.0, 'steel': 0.5, 'fairy': 2.0},
            'fairy': {'fire': 0.5, 'fighting': 2.0, 'poison': 0.5, 'dragon': 2.0, 'dark': 2.0, 'steel': 0.5}
        }
        return effectiveness

    def calculate_team_type_advantage(self, p1_team, p2_lead):
        """
        Calculates team-wide type advantage.
        
        Args:
        - p1_team: P1 team list, each element is a Pokémon dict
        - p2_lead: P2 lead Pokémon dict
        
        Returns:
        - dict with type_adv_mean, type_adv_max, type_adv_min, type_adv_std
        """
        effectiveness = self.get_complete_type_effectiveness()

        advantages = []
        p2_types = p2_lead.get('types', []) if p2_lead else []

        for pokemon in p1_team:
            if not pokemon:
                continue
            p1_types = pokemon.get('types', [])

            adv = 1.0
            for p1_type in p1_types:
                if p1_type.lower() in effectiveness:
                    for p2_type in p2_types:
                        if p2_type.lower() in effectiveness[p1_type.lower()]:
                            adv *= effectiveness[p1_type.lower()][p2_type.lower()]
            advantages.append(adv)

        if not advantages:
            return {'type_adv_mean': 1.0, 'type_adv_max': 1.0, 'type_adv_min': 1.0}

        return {
            'type_adv_mean': np.mean(advantages),
            'type_adv_max': max(advantages),
            'type_adv_min': min(advantages),
            'type_adv_std': np.std(advantages) if len(advantages) > 1 else 0
        }

    def calculate_alive_teams_type_advantage(self, p1_alive_team, p2_alive_team):
        """
        Calculate type advantage of P1's surviving Pokémon against P2's surviving Pokémon.
        
        Args:
        - p1_alive_team: List of P1's surviving Pokémon dicts
        - p2_alive_team: List of P2's surviving Pokémon dicts
        
        Returns:
        - dict with type_adv_mean, type_adv_max, type_adv_min, type_adv_std
        """
        if not p1_alive_team or not p2_alive_team:
            return {
                'type_adv_mean': 1.0,
                'type_adv_max': 1.0,
                'type_adv_min': 1.0,
                'type_adv_std': 0.0
            }
        
        all_advantages = []
        for p2_pokemon in p2_alive_team:
            type_adv = self.calculate_team_type_advantage(p1_alive_team, p2_pokemon)
            all_advantages.append(type_adv.get('type_adv_mean', 1.0))
        
        if all_advantages:
            return {
                'type_adv_mean': np.mean(all_advantages),
                'type_adv_max': np.max(all_advantages),
                'type_adv_min': np.min(all_advantages),
                'type_adv_std': np.std(all_advantages) if len(all_advantages) > 1 else 0.0
            }
        else:
            return {
                'type_adv_mean': 1.0,
                'type_adv_max': 1.0,
                'type_adv_min': 1.0,
                'type_adv_std': 0.0
            }

    def calculate_team_diversity(self, team):
        """
        Calculates team diversity.
        
        Args:
        - team: List of Pokémon objects, each containing types and base stats
        
        Returns:
        - dict: {'type_diversity': ..., 'stat_diversity': ...}
        """
        if not team:
            return {'type_diversity': 0, 'stat_diversity': 0}

        all_types = []
        for pokemon in team:
            if pokemon and 'types' in pokemon:
                all_types.extend(pokemon['types'])
        type_diversity = len(set(all_types)) / max(len(all_types), 1)

        stats_matrix = []
        for pokemon in team:
            if pokemon:
                stats = [
                    pokemon.get('base_hp', 0),
                    pokemon.get('base_atk', 0),
                    pokemon.get('base_def', 0),
                    pokemon.get('base_spa', 0),
                    pokemon.get('base_spd', 0),
                    pokemon.get('base_spe', 0)
                ]
                stats_matrix.append(stats)

        stat_diversity = 0
        if stats_matrix:
            stats_matrix = np.array(stats_matrix)
            for i in range(6):
                col = stats_matrix[:, i]
                if np.mean(col) > 0:
                    stat_diversity += np.std(col) / np.mean(col)
            stat_diversity /= 6

        return {'type_diversity': type_diversity, 'stat_diversity': stat_diversity}

    def extract_static_features(self):
        """
        Extracts static features (team and Pokémon stats) - Enhanced Version
        """
        print("\n=== Extracting static features (Enhanced Version) ===")
        
        pokemon_db = {}
        try:
            if os.path.exists('pokemon_stats_20.json'):
                with open('pokemon_stats_20.json', 'r', encoding='utf-8') as f:
                    pokemon_list = json.load(f)
                    for pokemon in pokemon_list:
                        pokemon_name = pokemon.get('name', '').lower()
                        if pokemon_name:
                            pokemon_db[pokemon_name] = pokemon
                print(f"✓ Loaded {len(pokemon_db)} Pokémon stats records")
        except Exception as e:
            print(f"⚠️ Failed to load Pokémon stats database: {e}")
            pokemon_db = {}

        def extract_pokemon_stats(pokemon):
            """Extracts stats for a single Pokémon"""
            if not pokemon:
                return [0] * 6
            return [
                pokemon.get('base_hp', 0),
                pokemon.get('base_atk', 0),
                pokemon.get('base_def', 0),
                pokemon.get('base_spa', 0),
                pokemon.get('base_spd', 0),
                pokemon.get('base_spe', 0)
            ]

        def calculate_team_stats(alive_pokemon_names=None):
            """
            Calculates team-wide stats.
            
            Args:
            - alive_pokemon_names: Set of surviving Pokémon names
            """
            if not alive_pokemon_names:
                return {'sum': [0]*6, 'mean': [0]*6, 'max': [0]*6, 'min': [0]*6, 'std': [0]*6}

            stats_matrix = []
            for pokemon_name in alive_pokemon_names:
                pokemon_name_lower = pokemon_name.lower()
                if pokemon_name_lower in pokemon_db:
                    pokemon = pokemon_db[pokemon_name_lower]
                    stats = extract_pokemon_stats(pokemon)
                    stats_matrix.append(stats)

            if not stats_matrix:
                return {'sum': [0]*6, 'mean': [0]*6, 'max': [0]*6, 'min': [0]*6, 'std': [0]*6}

            stats_matrix = np.array(stats_matrix)

            return {
                'sum': np.sum(stats_matrix, axis=0).tolist(),
                'mean': np.mean(stats_matrix, axis=0).tolist(),
                'max': np.max(stats_matrix, axis=0).tolist(),
                'min': np.min(stats_matrix, axis=0).tolist(),
                'std': np.std(stats_matrix, axis=0).tolist()
            }

        def extract_single_static_features(row):
            """
            Extracts static features for a single sample (for train and test).
            
            Args:
            - row: A single DataFrame row
            
            Returns:
            - features: Dictionary of all static features
            """
            features = {}

            p1_team = row.get('p1_team_details', [])
            
            timeline = row.get('battle_timeline', [])
            (
                p1_alive_pokemon_names,
                p1_fnt_pokemon_names,
                p2_alive_pokemon_names,
            ) = extract_final_pokemon_status(timeline)
            
            p1_team_names = {pokemon.get('name', '') for pokemon in p1_team 
                           if pokemon and pokemon.get('name', '') not in p1_fnt_pokemon_names}
            p1_alive_pokemon_names = p1_team_names | p1_alive_pokemon_names

            # Player 1 Team Features (surviving Pokémon only)
            p1_team_stats = calculate_team_stats(alive_pokemon_names=p1_alive_pokemon_names)

            # Player 2 Team Features (surviving Pokémon only)
            p2_team_stats = calculate_team_stats(alive_pokemon_names=p2_alive_pokemon_names)

            # Stat Comparison Features
            for i, stat_name in enumerate(['hp', 'atk', 'def', 'spa', 'spd', 'spe']):
                features[f'{stat_name}_advantage'] = p1_team_stats['mean'][i] - p2_team_stats['mean'][i]
                features[f'{stat_name}_ratio'] = p1_team_stats['mean'][i] / (p2_team_stats['mean'][i] + 1)

            # Type Advantage Features (Enhanced)
            p1_alive_team = []
            for pokemon_name in p1_alive_pokemon_names:
                pokemon_name_lower = pokemon_name.lower()
                if pokemon_name_lower in pokemon_db:
                    p1_alive_team.append(pokemon_db[pokemon_name_lower])
            
            p2_alive_team = []
            for pokemon_name in p2_alive_pokemon_names:
                pokemon_name_lower = pokemon_name.lower()
                if pokemon_name_lower in pokemon_db:
                    p2_alive_team.append(pokemon_db[pokemon_name_lower])
            
            type_adv = self.calculate_alive_teams_type_advantage(p1_alive_team, p2_alive_team)
            
            features['type_adv_mean'] = type_adv['type_adv_mean']
            features['type_adv_max'] = type_adv['type_adv_max']
            features['type_adv_min'] = type_adv['type_adv_min']
            features['type_adv_std'] = type_adv['type_adv_std']

            # Team Diversity
            p1_team_for_diversity = []
            for pokemon_name in p1_alive_pokemon_names:
                pokemon_name_lower = pokemon_name.lower()
                if pokemon_name_lower in pokemon_db:
                    p1_team_for_diversity.append(pokemon_db[pokemon_name_lower])
            p1_diversity = self.calculate_team_diversity(p1_team_for_diversity)
            features.update(p1_diversity)
            
            p2_team_for_diversity = []
            for pokemon_name in p2_alive_pokemon_names:
                pokemon_name_lower = pokemon_name.lower()
                if pokemon_name_lower in pokemon_db:
                    p2_team_for_diversity.append(pokemon_db[pokemon_name_lower])
            p2_diversity = self.calculate_team_diversity(p2_team_for_diversity)
            features.update({f'p2_{k}': v for k, v in p2_diversity.items()})
            
            p1_type_div = p1_diversity.get('type_diversity', 0)
            p2_type_div = p2_diversity.get('type_diversity', 0)
            p1_stat_div = p1_diversity.get('stat_diversity', 0)
            p2_stat_div = p2_diversity.get('stat_diversity', 0)
            
            features['type_diversity_ratio'] = p1_type_div / (p2_type_div + 1e-6)
            features['stat_diversity_ratio'] = p1_stat_div / (p2_stat_div + 1e-6)

            # Team Balance Features
            p1_physical_atk = p1_team_stats['mean'][1]
            p1_special_atk = p1_team_stats['mean'][3]
            p1_physical_def = p1_team_stats['mean'][2]
            p1_special_def = p1_team_stats['mean'][4]

            features['physical_special_atk_ratio'] = p1_physical_atk / (p1_special_atk + 1)
            features['physical_special_def_ratio'] = p1_physical_def / (p1_special_def + 1)
            features['offense_defense_ratio'] = (p1_physical_atk + p1_special_atk) / (p1_physical_def + p1_special_def + 1)
            
            p2_physical_atk = p2_team_stats['mean'][1]
            p2_special_atk = p2_team_stats['mean'][3]
            p2_physical_def = p2_team_stats['mean'][2]
            p2_special_def = p2_team_stats['mean'][4]

            features['p2_physical_special_atk_ratio'] = p2_physical_atk / (p2_special_atk + 1)
            features['p2_physical_special_def_ratio'] = p2_physical_def / (p2_special_def + 1)
            features['p2_offense_defense_ratio'] = (p2_physical_atk + p2_special_atk) / (p2_physical_def + p2_special_def + 1)
            
            features['physical_special_atk_ratio_p1_p2'] = features['physical_special_atk_ratio'] / (features['p2_physical_special_atk_ratio'] + 1e-6)
            features['physical_special_def_ratio_p1_p2'] = features['physical_special_def_ratio'] / (features['p2_physical_special_def_ratio'] + 1e-6)
            features['offense_defense_ratio_p1_p2'] = features['offense_defense_ratio'] / (features['p2_offense_defense_ratio'] + 1e-6)

            # Overall Strength Metrics (using mean)
            p1_total_stats_mean = sum(p1_team_stats['mean'])
            p2_total_stats_mean = sum(p2_team_stats['mean'])
            features['p1_total_stats_mean'] = p1_total_stats_mean
            features['p2_total_stats_mean'] = p2_total_stats_mean
            features['total_stats_advantage'] = p1_total_stats_mean - p2_total_stats_mean

            return features

        train_features = [extract_single_static_features(row) for _, row in self.train_data.iterrows()]
        test_features = [extract_single_static_features(row) for _, row in self.test_data.iterrows()]

        self.train_static_features = pd.DataFrame(train_features)
        self.test_static_features = pd.DataFrame(test_features)

        print(f"Static feature extraction completed: {self.train_static_features.shape[1]} features")
        return self.train_static_features, self.test_static_features

    def extract_dynamic_features(self):
        """
        Extracts dynamic features (from battle timeline) - Enhanced Version
        """
        print("\n=== Extracting dynamic features (Enhanced Version) ===")
        
        pokemon_db = {}
        try:
            if os.path.exists('pokemon_stats_20.json'):
                with open('pokemon_stats_20.json', 'r', encoding='utf-8') as f:
                    pokemon_list = json.load(f)
                    for pokemon in pokemon_list:
                        pokemon_name = pokemon.get('name', '').lower()
                        if pokemon_name:
                            pokemon_db[pokemon_name] = pokemon
                print(f"✓ Loaded {len(pokemon_db)} Pokémon stats records")
        except Exception as e:
            print(f"⚠️ Failed to load Pokémon stats database: {e}")
            pokemon_db = {}
        
        WEIGHT_HP = 1.0
        WEIGHT_ATTACK = 1.5
        WEIGHT_DEFENSE = 1.0
        WEIGHT_SPEED = 1.75
        
        SPECIAL_POKEMON_MODIFIERS = {
            "chansey": 1.2, "alakazam": 1.25, "snorlax": 1.15, "dragonite": 1.2,
            "zapdos": 1.1, "starmie": 1.05, "exeggutor": 1.05, "gengar": 1.1,
            "rhydon": 1.05, "cloyster": 1.05, "golem": 1.05, "jolteon": 1.05,
            "articuno": 1.05, "persian": 1.0, "lapras": 1.05, "charizard": 1.0,
            "victreebel": 1.0, "jynx": 1.0, "slowbro": 1.05, "tauros": 1.05
        }
        
        def calculate_pokemon_strength(pokemon_data):
            """Calculates a Pokémon's strength value"""
            if not pokemon_data:
                return 0.0
            
            base_hp = pokemon_data.get('base_hp', 0)
            base_atk = pokemon_data.get('base_atk', 0)
            base_def = pokemon_data.get('base_def', 0)
            base_spa = pokemon_data.get('base_spa', 0)
            base_spd = pokemon_data.get('base_spd', 0)
            base_spe = pokemon_data.get('base_spe', 0)
            
            hp_score = base_hp * WEIGHT_HP
            atk_score = base_atk * WEIGHT_ATTACK
            def_score = base_def * WEIGHT_DEFENSE
            spa_score = base_spa * WEIGHT_ATTACK
            spd_score = base_spd * WEIGHT_DEFENSE
            spe_score = base_spe * WEIGHT_SPEED
            
            strength = hp_score + atk_score + def_score + spa_score + spd_score + spe_score
            
            pokemon_name = pokemon_data.get('name', '').lower()
            if pokemon_name in SPECIAL_POKEMON_MODIFIERS:
                strength *= SPECIAL_POKEMON_MODIFIERS[pokemon_name]
            
            return strength

        def analyze_battle_timeline(row, timeline, pokemon_db=None, turn30_winrates=None):
            """Analyzes the battle timeline - includes early/mid/late phase analysis"""
            if not timeline:
                return {}
            
            if pokemon_db is None: pokemon_db = {}
            if turn30_winrates is None: turn30_winrates = {}

            features = {}
            total_turns = len(timeline)
            
            p1_pokemon_status_dict = {}
            p2_pokemon_status_dict = {}
            p1_counter_invalid = 0
            p2_counter_invalid = 0
            p1_move_accuracies = []
            p2_move_accuracies = []
            p1_switch_count = 0
            p2_switch_count = 0
            p1_move_null_switch = 0
            p2_move_null_switch = 0
            p1_move_null_status = 0
            p2_move_null_status = 0
            p1_hp_losses = []
            p2_hp_losses = []

            early_end = total_turns // 3
            
            p1_switch_turns = []
            p2_switch_turns = []
            
            p1_pokemon_appeared_30turns = set()
            p2_pokemon_appeared_30turns = set()
            
            p1_pokemon_hp_dict = {}
            p2_pokemon_hp_dict = {}

            p1_prev_pokemon_name = None
            p2_prev_pokemon_name = None
            p1_prev_hp = None
            p2_prev_hp = None

            for i, turn in enumerate(timeline):
                p1_hp = turn.get('p1_pokemon_state', {}).get('hp_pct', 0)
                p2_hp = turn.get('p2_pokemon_state', {}).get('hp_pct', 0)
                
                if i < 30:
                    p1_pokemon_name = turn.get('p1_pokemon_state', {}).get('name', '')
                    p2_pokemon_name = turn.get('p2_pokemon_state', {}).get('name', '')
                    
                    if p1_pokemon_name:
                        p1_pokemon_appeared_30turns.add(p1_pokemon_name)
                        if p1_pokemon_name not in p1_pokemon_hp_dict:
                            p1_pokemon_hp_dict[p1_pokemon_name] = []
                        p1_pokemon_hp_dict[p1_pokemon_name].append(p1_hp)
                    
                    if p2_pokemon_name:
                        p2_pokemon_appeared_30turns.add(p2_pokemon_name)
                        if p2_pokemon_name not in p2_pokemon_hp_dict:
                            p2_pokemon_hp_dict[p2_pokemon_name] = []
                        p2_pokemon_hp_dict[p2_pokemon_name].append(p2_hp)

                if p1_prev_hp is not None:
                    p1_hp_losses.append(p1_prev_hp - p1_hp)
                if p2_prev_hp is not None:
                    p2_hp_losses.append(p2_prev_hp - p2_hp)
                
                p1_prev_hp = p1_hp
                p2_prev_hp = p2_hp

                p1_move = turn.get('p1_move_details')
                p2_move = turn.get('p2_move_details')
                
                p1_current_pokemon_name = turn.get('p1_pokemon_state', {}).get('name', '')
                p2_current_pokemon_name = turn.get('p2_pokemon_state', {}).get('name', '')
                
                if p1_move is None:
                    if i > 0 and p1_prev_pokemon_name is not None and p1_current_pokemon_name != p1_prev_pokemon_name:
                        p1_move_null_switch += 1
                    else:
                        p1_move_null_status += 1
                
                if p2_move is None:
                    if i > 0 and p2_prev_pokemon_name is not None and p2_current_pokemon_name != p2_prev_pokemon_name:
                        p2_move_null_switch += 1
                    else:
                        p2_move_null_status += 1
                
                p1_prev_pokemon_name = p1_current_pokemon_name
                p2_prev_pokemon_name = p2_current_pokemon_name
                
                if p1_move and p1_move.get('name', '').lower() == 'counter':
                    if not p2_move or p2_move.get('category', 'STATUS') != 'PHYSICAL':
                        p1_counter_invalid += 1
                
                if p2_move and p2_move.get('name', '').lower() == 'counter':
                    if not p1_move or p1_move.get('category', 'STATUS') != 'PHYSICAL':
                        p2_counter_invalid += 1

                if p1_move:
                    accuracy = p1_move.get('accuracy', 1.0)
                    p1_move_accuracies.append(accuracy)
                else:
                    p1_switch_count += 1
                    p1_switch_turns.append(i)

                if p2_move:
                    accuracy = p2_move.get('accuracy', 1.0)
                    p2_move_accuracies.append(accuracy)
                else:
                    p2_switch_count += 1
                    p2_switch_turns.append(i)

                p1_pokemon_name = turn.get('p1_pokemon_state', {}).get('name', '')
                p1_status = turn.get('p1_pokemon_state', {}).get('status', 'nostatus')
                if p1_pokemon_name:
                    p1_pokemon_status_dict[p1_pokemon_name] = p1_status

                p2_pokemon_name = turn.get('p2_pokemon_state', {}).get('name', '')
                p2_status = turn.get('p2_pokemon_state', {}).get('status', 'nostatus')
                if p2_pokemon_name:
                    p2_pokemon_status_dict[p2_pokemon_name] = p2_status

            # --- Feature Calculation ---
            
            if p1_hp_losses:
                features['p1_avg_hp_loss'] = np.mean(p1_hp_losses)
                features['p1_max_hp_loss'] = max(p1_hp_losses)
            else:
                features['p1_avg_hp_loss'] = 0
                features['p1_max_hp_loss'] = 0

            if p2_hp_losses:
                features['p2_avg_hp_loss'] = np.mean(p2_hp_losses)
                features['p2_max_hp_loss'] = max(p2_hp_losses)
            else:
                features['p2_avg_hp_loss'] = 0
                features['p2_max_hp_loss'] = 0

            p1_abnormal_status_count = 0
            for pokemon_name in p1_pokemon_appeared_30turns:
                status = p1_pokemon_status_dict.get(pokemon_name, 'nostatus')
                if status != 'nostatus' and status != 'fnt':
                    p1_abnormal_status_count += 1
            
            p2_abnormal_status_count = 0
            for pokemon_name in p2_pokemon_appeared_30turns:
                status = p2_pokemon_status_dict.get(pokemon_name, 'nostatus')
                if status != 'nostatus' and status != 'fnt':
                    p2_abnormal_status_count += 1
            
            features['p1_abnormal_status_count'] = p1_abnormal_status_count
            features['p2_abnormal_status_count'] = p2_abnormal_status_count
            features['abnormal_status_count_ratio'] = p1_abnormal_status_count / (p2_abnormal_status_count + 1.0)
            
            features['p1_counter_invalid'] = p1_counter_invalid
            features['p2_counter_invalid'] = p2_counter_invalid

            features['p1_avg_accuracy'] = np.mean(p1_move_accuracies) if p1_move_accuracies else 1.0
            features['p2_avg_accuracy'] = np.mean(p2_move_accuracies) if p2_move_accuracies else 1.0

            features['p1_switch_count'] = p1_switch_count
            features['p2_switch_count'] = p2_switch_count
            
            features['p1_move_null_switch'] = p1_move_null_switch
            features['p2_move_null_switch'] = p2_move_null_switch
            features['p1_move_null_status'] = p1_move_null_status
            features['p2_move_null_status'] = p2_move_null_status
            
            if p1_switch_turns:
                early_switches = sum(1 for t in p1_switch_turns if t < early_end)
                features['p1_early_switch_ratio'] = early_switches / max(p1_switch_count, 1)
            else:
                features['p1_early_switch_ratio'] = 0

            if p2_switch_turns:
                early_switches = sum(1 for t in p2_switch_turns if t < early_end)
                features['p2_early_switch_ratio'] = early_switches / max(p2_switch_count, 1)
            else:
                features['p2_early_switch_ratio'] = 0

            features['p1_unique_pokemon_count_30turns'] = len(p1_pokemon_appeared_30turns)
            features['p2_unique_pokemon_count_30turns'] = len(p2_pokemon_appeared_30turns)
            
            p1_weighted_strength_sum = 0.0
            p1_strength_list = []
            for pokemon_name, hp_list in p1_pokemon_hp_dict.items():
                if hp_list:
                    last_hp = hp_list[-1]
                    if last_hp > 0:
                        pokemon_name_lower = pokemon_name.lower()
                        if pokemon_name_lower in pokemon_db:
                            pokemon_data = pokemon_db[pokemon_name_lower]
                            strength = calculate_pokemon_strength(pokemon_data)
                            p1_weighted_strength_sum += last_hp * strength
                            p1_strength_list.append(strength)
            
            p1_appeared_count = len(p1_pokemon_hp_dict)
            p1_missing_count = 6 - p1_appeared_count
            if p1_missing_count > 0 and len(p1_strength_list) > 0:
                p1_avg_strength = np.mean(p1_strength_list)
                p1_weighted_strength_sum += p1_missing_count * 1.0 * p1_avg_strength
            
            p2_weighted_strength_sum = 0.0
            p2_strength_list = []
            for pokemon_name, hp_list in p2_pokemon_hp_dict.items():
                if hp_list:
                    last_hp = hp_list[-1]
                    if last_hp > 0:
                        pokemon_name_lower = pokemon_name.lower()
                        if pokemon_name_lower in pokemon_db:
                            pokemon_data = pokemon_db[pokemon_name_lower]
                            strength = calculate_pokemon_strength(pokemon_data)
                            p2_weighted_strength_sum += last_hp * strength
                            p2_strength_list.append(strength)
            
            p2_appeared_count = len(p2_pokemon_hp_dict)
            p2_missing_count = 6 - p2_appeared_count
            if p2_missing_count > 0 and len(p2_strength_list) > 0:
                p2_avg_strength = np.mean(p2_strength_list)
                p2_weighted_strength_sum += p2_missing_count * 1.0 * p2_avg_strength
            
            features['p1_weighted_strength_sum'] = p1_weighted_strength_sum
            features['p2_weighted_strength_sum'] = p2_weighted_strength_sum
            features['weighted_strength_ratio_30turns'] = p1_weighted_strength_sum / (p2_weighted_strength_sum + 1e-6)
            
            p1_fnt_count = 0
            for pokemon_name, status in p1_pokemon_status_dict.items():
                if status == 'fnt':
                    p1_fnt_count += 1
            
            p2_fnt_count = 0
            for pokemon_name, status in p2_pokemon_status_dict.items():
                if status == 'fnt':
                    p2_fnt_count += 1
            
            features['fnt_count_ratio'] = p1_fnt_count / (p2_fnt_count + 1.0)
            features['fnt_count_diff'] = p1_fnt_count - p2_fnt_count
            
            battle_data = {'battle_timeline': timeline}
            p1_moves, p2_moves = extract_moves(battle_data)
            
            p1_team_avg, p2_team_avg = type_multiplier(p1_moves, p2_moves)
            features['p1_avg'] = p1_team_avg
            features['p2_avg'] = p2_team_avg
            
            features['p1_num_priority_moves'] = count_priority_moves(p1_moves)
            features['p2_num_priority_moves'] = count_priority_moves(p2_moves)

            return features

        train_battle_features = []
        for idx, row in self.train_data.iterrows():
            timeline = row.get('battle_timeline', [])
            battle_features = analyze_battle_timeline(row, timeline, pokemon_db, None)
            train_battle_features.append(battle_features)

        test_battle_features = []
        for idx, row in self.test_data.iterrows():
            timeline = row.get('battle_timeline', [])
            battle_features = analyze_battle_timeline(row, timeline, pokemon_db, None)
            test_battle_features.append(battle_features)

        self.train_dynamic_features = pd.DataFrame(train_battle_features)
        self.test_dynamic_features = pd.DataFrame(test_battle_features)

        print(f"Dynamic feature extraction completed: {self.train_dynamic_features.shape[1]} features")
        return self.train_dynamic_features, self.test_dynamic_features

    def create_interaction_features(self, df):
        """
        Creates feature interactions.
        """
        df_copy = df.copy()

        if 'p1_avg_move_power' in df_copy.columns and 'p1_avg_accuracy' in df_copy.columns:
            df_copy['p1_effective_power'] = df_copy['p1_avg_move_power'] * df_copy['p1_avg_accuracy']

        if 'p2_avg_move_power' in df_copy.columns and 'p2_avg_accuracy' in df_copy.columns:
            df_copy['p2_effective_power'] = df_copy['p2_avg_move_power'] * df_copy['p2_avg_accuracy']

        if 'type_adv_mean' in df_copy.columns and 'total_stats_advantage' in df_copy.columns:
            df_copy['type_stats_interaction'] = df_copy['type_adv_mean'] * df_copy['total_stats_advantage']

        return df_copy

    def combine_features(self):
        """Combines all features and creates interactions"""
        print("\n=== Combining features and creating interactions ===")

        self.train_combined = pd.concat([
            self.train_static_features.reset_index(drop=True),
            self.train_dynamic_features.reset_index(drop=True)
        ], axis=1)

        self.test_combined = pd.concat([
            self.test_static_features.reset_index(drop=True),
            self.test_dynamic_features.reset_index(drop=True)
        ], axis=1)

        self.train_combined = self.create_interaction_features(self.train_combined)
        self.test_combined = self.create_interaction_features(self.test_combined)

        if self.features_to_remove:
            print(f"\n=== Removing configured features ({len(self.features_to_remove)}) ===")
            features_to_remove_actual = []
            for feature in self.features_to_remove:
                if feature in self.train_combined.columns:
                    features_to_remove_actual.append(feature)
                else:
                    print(f"  Warning: Feature '{feature}' does not exist, skipping removal")
            
            if features_to_remove_actual:
                self.train_combined = self.train_combined.drop(columns=features_to_remove_actual)
                self.test_combined = self.test_combined.drop(columns=features_to_remove_actual)
                print(f"  ✓ Removed {len(features_to_remove_actual)} features: {features_to_remove_actual}")
            else:
                print("  ⚠️ No features found to remove")

        self.train_combined = self.train_combined.fillna(0)
        self.test_combined = self.test_combined.fillna(0)

        train_cols = set(self.train_combined.columns)
        test_cols = set(self.test_combined.columns)

        missing_in_test = train_cols - test_cols
        missing_in_train = test_cols - train_cols

        for col in missing_in_test:
            self.test_combined[col] = 0
        for col in missing_in_train:
            self.train_combined[col] = 0

        self.test_combined = self.test_combined[self.train_combined.columns]

        print(f"Feature combination completed:")
        print(f"Training feature shape: {self.train_combined.shape}")
        print(f"Test feature shape: {self.test_combined.shape}")

        return self.train_combined, self.test_combined

    def select_features_rfecv(self, X_train, y_train, estimator=None, cv=5, scoring='accuracy', min_features_to_select=10, n_jobs=-1):
        """
        Performs feature selection using RFECV (Recursive Feature Elimination with Cross-Validation).
        """
        print(f"\n=== RFECV Feature Selection ===")
        print(f"Initial feature count: {X_train.shape[1]}")
        print(f"Cross-validation folds: {cv}")
        print(f"Minimum features to select: {min_features_to_select}")
        print("Calculating optimal feature count (this may take a few minutes)...")
        
        if estimator is None:
            estimator = xgb.XGBClassifier(
                n_estimators=100, max_depth=6, learning_rate=0.1, subsample=0.8,
                colsample_bytree=0.8, min_child_weight=1, gamma=0, reg_alpha=0.01,
                reg_lambda=0.01, random_state=42, eval_metric='logloss', n_jobs=n_jobs
            )
        
        rfecv = RFECV(
            estimator=estimator, step=1, cv=cv, scoring=scoring,
            min_features_to_select=min_features_to_select, n_jobs=n_jobs
        )
        
        rfecv.fit(X_train, y_train)
        
        self.feature_selector = rfecv
        
        selected_mask = rfecv.support_
        selected_features = X_train.columns[selected_mask].tolist()
        optimal_n_features = rfecv.n_features_
        
        X_train_selected = rfecv.transform(X_train)
        
        if isinstance(X_train, pd.DataFrame):
            X_train_selected = pd.DataFrame(
                X_train_selected, columns=selected_features, index=X_train.index
            )
        
        all_features = X_train.columns.tolist()
        removed_features = [feat for feat in all_features if feat not in selected_features]
        
        print(f"✓ RFECV Feature Selection completed")
        print(f"  Optimal feature count: {optimal_n_features}")
        print(f"  Selected feature count: {len(selected_features)}")
        print(f"  Features reduced: {X_train.shape[1] - optimal_n_features} ({(1 - optimal_n_features/X_train.shape[1])*100:.1f}%)")
        
        if removed_features:
            print(f"\nRemoved features ({len(removed_features)}):")
            removed_features_sorted = sorted(removed_features)
            for i in range(0, len(removed_features_sorted), 5):
                features_line = removed_features_sorted[i:i+5]
                print(f"  {', '.join(features_line)}")
        else:
            print(f"\nNo features removed")
        
        print(f"\nCross-validation score vs feature count:")
        print(f"  Highest score: {rfecv.cv_results_['mean_test_score'].max():.4f}")
        print(f"  Score at optimal feature count: {rfecv.cv_results_['mean_test_score'][rfecv.n_features_ - min_features_to_select]:.4f}")
        
        return X_train_selected, selected_features, optimal_n_features

    def save_misclassified_samples(self, y_true_indices, y_pred, y_true, model_name='best_model'):
        """
        Saves misclassified samples to a JSON file.
        """
        print(f"\n=== Saving {model_name} misclassified samples ===")
        
        misclassified_mask = y_pred != y_true.values
        misclassified_indices = y_true_indices[misclassified_mask]
        
        if len(misclassified_indices) == 0:
            print("No misclassified samples!")
            return None, None
        
        print(f"Found {len(misclassified_indices)} misclassified samples")
        
        y_pred_wrong = y_pred[misclassified_mask]
        y_true_wrong = y_true.values[misclassified_mask]
        
        misclassified_samples = []
        for i, idx in enumerate(misclassified_indices):
            sample = self.train_data.iloc[idx].to_dict()
            sample['prediction_info'] = {
                'predicted': int(y_pred_wrong[i]),
                'actual': int(y_true_wrong[i]),
                'model': model_name,
                'original_index': int(idx)
            }
            misclassified_samples.append(sample)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f'misclassified/misclassified_samples_{model_name}_{timestamp}.json'
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(misclassified_samples, f, indent=2, ensure_ascii=False)
        
        print(f"Misclassified samples saved to: {output_file}")
        
        stats = {
            'total_samples': len(y_true),
            'misclassified_count': len(misclassified_indices),
            'accuracy': 1 - (len(misclassified_indices) / len(y_true)),
            'error_rate': len(misclassified_indices) / len(y_true),
            'false_positive': int(((y_pred == 1) & (y_true == 0)).sum()),
            'false_negative': int(((y_pred == 0) & (y_true == 1)).sum()),
            'timestamp': timestamp,
            'model_name': model_name
        }
        
        stats_file = f'misclassified/misclassified_stats_{model_name}_{timestamp}.json'
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
        
        print(f"Statistics saved to: {stats_file}")
        print(f"Accuracy: {stats['accuracy']:.4f}")
        print(f"Error rate: {stats['error_rate']:.4f}")
        print(f"False positive (predicted win but actual loss): {stats['false_positive']}")
        print(f"False negative (predicted loss but actual win): {stats['false_negative']}")
        
        return misclassified_samples, stats

    def optimize_xgboost_hyperparams(self, X_train, y_train):
        """Uses Optuna + MedianPruner to optimize XGBoost hyperparameters"""
        print(f"\n=== XGBoost Hyperparameter Optimization ===")
        print(f"Trials: {self.optuna_n_trials}")
        print(f"CV Folds: {self.optuna_cv_folds}")
        print(f"Timeout: {self.optuna_timeout}s")

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 200, 1000, step=100),
                'max_depth': trial.suggest_int('max_depth', 4, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 0, 0.5),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.001, 1.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.001, 1.0, log=True),
                'random_state': 42, 'eval_metric': 'logloss', 'n_jobs': -1
            }
            model = xgb.XGBClassifier(**params)
            cv = StratifiedKFold(n_splits=self.optuna_cv_folds, shuffle=True, random_state=42)
            scores = []
            for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
                X_fold_train = X_train.iloc[train_idx] if hasattr(X_train, 'iloc') else X_train[train_idx]
                y_fold_train = y_train.iloc[train_idx] if hasattr(y_train, 'iloc') else y_train[train_idx]
                X_fold_val = X_train.iloc[val_idx] if hasattr(X_train, 'iloc') else X_train[val_idx]
                y_fold_val = y_train.iloc[val_idx] if hasattr(y_train, 'iloc') else y_train[val_idx]
                model.fit(X_fold_train, y_fold_train)
                y_pred = model.predict(X_fold_val)
                fold_accuracy = accuracy_score(y_fold_val, y_pred)
                scores.append(fold_accuracy)
                trial.report(fold_accuracy, fold_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            return np.mean(scores)

        pruner = MedianPruner(n_startup_trials=self.optuna_pruner_warmup, n_warmup_steps=self.optuna_pruner_interval)
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction='maximize', pruner=pruner, study_name='xgboost_optimization')
        study.optimize(objective, n_trials=self.optuna_n_trials, timeout=self.optuna_timeout, show_progress_bar=True)
        
        self.optuna_studies['XGBoost'] = study
        self.optuna_best_params['XGBoost'] = study.best_params
        
        print(f"\n✓ XGBoost Optimization Complete")
        print(f"  Best accuracy: {study.best_value:.4f}")
        print(f"  Best params:")
        for key, value in study.best_params.items():
            print(f"    {key}: {value}")
        print(f"  Trials completed: {len(study.trials)}")
        print(f"  Trials pruned: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
        return study.best_params

    def optimize_lightgbm_hyperparams(self, X_train, y_train):
        """Uses Optuna + MedianPruner to optimize LightGBM hyperparameters"""
        print(f"\n=== LightGBM Hyperparameter Optimization ===")
        print(f"Trials: {self.optuna_n_trials}")
        print(f"CV Folds: {self.optuna_cv_folds}")
        print(f"Timeout: {self.optuna_timeout}s")

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 200, 1000, step=100),
                'max_depth': trial.suggest_int('max_depth', 4, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.001, 1.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.001, 1.0, log=True),
                'random_state': 42, 'verbose': -1, 'n_jobs': -1
            }
            model = lgb.LGBMClassifier(**params)
            cv = StratifiedKFold(n_splits=self.optuna_cv_folds, shuffle=True, random_state=42)
            scores = []
            for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
                X_fold_train = X_train.iloc[train_idx] if hasattr(X_train, 'iloc') else X_train[train_idx]
                y_fold_train = y_train.iloc[train_idx] if hasattr(y_train, 'iloc') else y_train[train_idx]
                X_fold_val = X_train.iloc[val_idx] if hasattr(X_train, 'iloc') else X_train[val_idx]
                y_fold_val = y_train.iloc[val_idx] if hasattr(y_train, 'iloc') else y_train[val_idx]
                model.fit(X_fold_train, y_fold_train)
                y_pred = model.predict(X_fold_val)
                fold_accuracy = accuracy_score(y_fold_val, y_pred)
                scores.append(fold_accuracy)
                trial.report(fold_accuracy, fold_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            return np.mean(scores)

        pruner = MedianPruner(n_startup_trials=self.optuna_pruner_warmup, n_warmup_steps=self.optuna_pruner_interval)
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction='maximize', pruner=pruner, study_name='lightgbm_optimization')
        study.optimize(objective, n_trials=self.optuna_n_trials, timeout=self.optuna_timeout, show_progress_bar=True)

        self.optuna_studies['LightGBM'] = study
        self.optuna_best_params['LightGBM'] = study.best_params

        print(f"\n✓ LightGBM Optimization Complete")
        print(f"  Best accuracy: {study.best_value:.4f}")
        print(f"  Best params:")
        for key, value in study.best_params.items():
            print(f"    {key}: {value}")
        print(f"  Trials completed: {len(study.trials)}")
        print(f"  Trials pruned: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
        return study.best_params

    def optimize_catboost_hyperparams(self, X_train, y_train):
        """Uses Optuna + MedianPruner to optimize CatBoost hyperparameters"""
        print(f"\n=== CatBoost Hyperparameter Optimization ===")
        print(f"Trials: {self.optuna_n_trials}")
        print(f"CV Folds: {self.optuna_cv_folds}")
        print(f"Timeout: {self.optuna_timeout}s")

        def objective(trial):
            params = {
                'iterations': trial.suggest_int('iterations', 200, 1000, step=100),
                'depth': trial.suggest_int('depth', 4, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'l2_leaf_reg': trial.suggest_int('l2_leaf_reg', 1, 10),
                'random_seed': 42, 'verbose': False, 'thread_count': -1
            }
            model = CatBoostClassifier(**params)
            cv = StratifiedKFold(n_splits=self.optuna_cv_folds, shuffle=True, random_state=42)
            scores = []
            for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
                X_fold_train = X_train.iloc[train_idx] if hasattr(X_train, 'iloc') else X_train[train_idx]
                y_fold_train = y_train.iloc[train_idx] if hasattr(y_train, 'iloc') else y_train[train_idx]
                X_fold_val = X_train.iloc[val_idx] if hasattr(X_train, 'iloc') else X_train[val_idx]
                y_fold_val = y_train.iloc[val_idx] if hasattr(y_train, 'iloc') else y_train[val_idx]
                model.fit(X_fold_train, y_fold_train)
                y_pred = model.predict(X_fold_val)
                fold_accuracy = accuracy_score(y_fold_val, y_pred)
                scores.append(fold_accuracy)
                trial.report(fold_accuracy, fold_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            return np.mean(scores)

        pruner = MedianPruner(n_startup_trials=self.optuna_pruner_warmup, n_warmup_steps=self.optuna_pruner_interval)
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction='maximize', pruner=pruner, study_name='catboost_optimization')
        study.optimize(objective, n_trials=self.optuna_n_trials, timeout=self.optuna_timeout, show_progress_bar=True)
        
        self.optuna_studies['CatBoost'] = study
        self.optuna_best_params['CatBoost'] = study.best_params
        
        print(f"\n✓ CatBoost Optimization Complete")
        print(f"  Best accuracy: {study.best_value:.4f}")
        print(f"  Best params:")
        for key, value in study.best_params.items():
            print(f"    {key}: {value}")
        print(f"  Trials completed: {len(study.trials)}")
        print(f"  Trials pruned: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
        return study.best_params

    def save_optuna_results(self, output_dir='optuna_results'):
        """Saves Optuna optimization results to files."""
        if not self.optuna_studies:
            print("⚠️ No Optuna results to save.")
            return

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        print(f"\n=== Saving Optuna Optimization Results ===")
        print(f"Output directory: {output_dir}")

        for model_name, study in self.optuna_studies.items():
            best_params_file = os.path.join(output_dir, f'{model_name}_best_params_{timestamp}.json')
            with open(best_params_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'model': model_name,
                    'best_value': study.best_value,
                    'best_params': study.best_params,
                    'n_trials': len(study.trials),
                    'n_pruned': len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
                    'timestamp': timestamp
                }, f, indent=2, ensure_ascii=False)
            print(f"✓ {model_name} best params saved: {best_params_file}")

            trials_df = study.trials_dataframe()
            history_file = os.path.join(output_dir, f'{model_name}_optimization_history_{timestamp}.csv')
            trials_df.to_csv(history_file, index=False)
            print(f"✓ {model_name} optimization history saved: {history_file}")

            print(f"\n{model_name} Optimization Stats:")
            print(f"  Total trials: {len(study.trials)}")
            print(f"  Completed trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
            print(f"  Pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
            print(f"  Failed trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])}")
            print(f"  Best accuracy: {study.best_value:.4f}")
            print(f"  Best params: {study.best_params}")

        print(f"\n✓ All Optuna results saved to {output_dir}")

    def train_models(self):
        """
        Trains multiple models and uses Stacking ensemble.
        """
        print("\n=== Model Training (Stacking Ensemble) ===")

        X = self.train_combined
        y = self.train_data['player_won']

        xgb_cv_params = dict(
            n_estimators=800, max_depth=8, learning_rate=0.03, subsample=0.85,
            colsample_bytree=0.85, min_child_weight=2, gamma=0.05, reg_alpha=0.1,
            reg_lambda=0.1, random_state=42, eval_metric='logloss', n_jobs=-1
        )
        print("\nRunning XGBoost 4-fold CV evaluation (all labeled samples)...")
        xgb_cv_model = xgb.XGBClassifier(**xgb_cv_params)
        xgb_cv_splitter = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
        xgb_cv_scores = cross_val_score(
            xgb_cv_model, X, y, cv=xgb_cv_splitter, scoring='accuracy', n_jobs=-1
        )
        print(f"XGBoost 4-fold CV accuracy: {xgb_cv_scores.mean():.4f} ± {xgb_cv_scores.std():.4f}")

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.validation_split, random_state=42, stratify=y
        )
        
        self.X_val = X_val
        self.y_val = y_val
        val_indices = y_val.index.to_numpy()

        use_rfecv = False
        if use_rfecv:
            X_train, selected_features, optimal_n = self.select_features_rfecv(
                X_train, y_train, cv=4, min_features_to_select=20, n_jobs=-1
            )
            X_val = self.feature_selector.transform(X_val)
            print(f"\nTraining models with RFECV selected features")
            print(f"Training feature shape: {X_train.shape}, Validation feature shape: {X_val.shape}")

        xgb_best_params = None
        lgb_best_params = None
        cat_best_params = None

        if self.use_optuna_tuning:
            print(f"\n{'='*80}\n=== Starting Optuna Hyperparameter Optimization ===\n{'='*80}")
            xgb_best_params = self.optimize_xgboost_hyperparams(X_train, y_train)
            lgb_best_params = self.optimize_lightgbm_hyperparams(X_train, y_train)
            cat_best_params = self.optimize_catboost_hyperparams(X_train, y_train)
            self.save_optuna_results(output_dir='optuna_results')
            print(f"\n{'='*80}\n=== Optuna Hyperparameter Optimization Complete ===\n{'='*80}")
        else:
            print("\n⚠️ Optuna hyperparameter optimization is disabled. Using default parameters.")

        X_train_final = X_train.copy()
        y_train_final = y_train.copy()

        print(f"\n=== Formal Model Training ===")
        print(f"Training data size: {len(X_train_final)} samples")

        if self.use_optuna_tuning and xgb_best_params:
            print("\n✓ Using Optuna-optimized XGBoost parameters")
            xgb_params = {**xgb_best_params, 'random_state': 42, 'eval_metric': 'logloss', 'n_jobs': -1}
        else:
            xgb_params = xgb_cv_params

        if self.use_optuna_tuning and lgb_best_params:
            print("✓ Using Optuna-optimized LightGBM parameters")
            lgb_params = {**lgb_best_params, 'random_state': 42, 'verbose': -1, 'n_jobs': -1}
        else:
            lgb_params = {
                'n_estimators': 800, 'max_depth': 10, 'learning_rate': 0.03, 'subsample': 0.85,
                'colsample_bytree': 0.85, 'min_child_samples': 15, 'reg_alpha': 0.1,
                'reg_lambda': 0.1, 'num_leaves': 50, 'random_state': 42, 'verbose': -1, 'n_jobs': -1
            }

        if self.use_optuna_tuning and cat_best_params:
            print("✓ Using Optuna-optimized CatBoost parameters")
            cat_params = {**cat_best_params, 'random_seed': 42, 'verbose': False, 'thread_count': -1}
        else:
            cat_params = {
                'iterations': 800, 'depth': 9, 'learning_rate': 0.03, 'l2_leaf_reg': 2,
                'random_seed': 42, 'verbose': False, 'thread_count': -1
            }

        base_models = {
            'XGBoost': xgb.XGBClassifier(**xgb_params),
            'LightGBM': lgb.LGBMClassifier(**lgb_params),
            'CatBoost': CatBoostClassifier(**cat_params)
        }

        for name, model in base_models.items():
            print(f"Training {name}...")
            model.fit(X_train_final, y_train_final)
            y_pred = model.predict(X_val)
            accuracy = accuracy_score(y_val, y_pred)
            print(f"{name} Validation accuracy: {accuracy:.4f}")
            self.models[name] = model
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[name] = model.feature_importances_

        print("\nTraining Stacking ensemble model...")
        stacking_estimators = [
            ('XGBoost', base_models['XGBoost']),
            ('LightGBM', base_models['LightGBM']),
            ('CatBoost', base_models['CatBoost'])
        ]

        stacking_model = StackingClassifier(
            estimators=stacking_estimators,
            final_estimator=LogisticRegression(C=1.0, max_iter=500, random_state=42),
            cv=2,
            n_jobs=1
        )

        try:
            stacking_model.fit(X_train_final, y_train_final)
            y_pred_stacking = stacking_model.predict(X_val)
            stacking_accuracy = accuracy_score(y_val, y_pred_stacking)
            print(f"Stacking ensemble model validation accuracy: {stacking_accuracy:.4f}")
            self.models['Stacking'] = stacking_model
        except Exception as e:
            print(f"Stacking training failed: {e}")
            print("Skipping Stacking model, using Voting as main ensemble method")

        if 'Stacking' in self.models:
            self.best_model_name = 'Stacking'
            self.best_model = self.models['Stacking']
            best_predictions = self.models['Stacking'].predict(X_val)
        else:
            self.best_model_name = 'Voting'
            # (Assuming Voting model was defined and trained if Stacking fails)
            # self.best_model = self.models['Voting'] 
            # best_predictions = y_pred_voting
            # Fallback if Voting is not defined:
            self.best_model_name = 'XGBoost' # Fallback
            self.best_model = self.models['XGBoost']
            best_predictions = self.models['XGBoost'].predict(X_val)


        return self.models

    def make_predictions(self):
        """Generates predictions"""
        print("\n=== Generating predictions ===")

        test_features = self.test_combined
        if hasattr(self, 'feature_selector') and self.feature_selector is not None:
            print("Applying feature selection to test data...")
            test_features_array = self.feature_selector.transform(self.test_combined)
            if isinstance(self.test_combined, pd.DataFrame):
                selected_features = self.test_combined.columns[self.feature_selector.support_].tolist()
                test_features = pd.DataFrame(
                    test_features_array, columns=selected_features, index=self.test_combined.index
                )
            else:
                test_features = test_features_array
            print(f"Test feature shape: {test_features.shape}")

        predictions = {}
        for name, model in self.models.items():
            pred = model.predict(test_features)
            predictions[name] = pred
            print(f"{name} Prediction completed")

        if 'Stacking' in predictions:
            best_predictions = predictions['Stacking']
            print("Using Stacking model for final predictions")
        else:
            best_predictions = predictions['Voting'] # Assuming Voting exists
            print("Using Voting model for final predictions")

        submission = pd.DataFrame({
            'battle_id': self.test_data['battle_id'],
            'player_won': best_predictions.astype(int)
        })

        submission.to_csv('submission_enhanced_v4.csv', index=False)
        print("Submission file generated: submission_enhanced_v4.csv")

        return submission, predictions

    def get_sample_features(self, original_index):
        """Gets all features for a specific sample index"""
        if self.train_combined is None:
            print("Error: Please run feature extraction first (extract_static_features, extract_dynamic_features, combine_features)")
            return None
        if original_index >= len(self.train_combined):
            print(f"Error: Index out of range (Maximum: {len(self.train_combined)-1})")
            return None
        return self.train_combined.iloc[original_index].to_dict()

    def explain_single_prediction(self, sample_index, model_name='XGBoost', top_k=20, use_shap=True):
        """
        Explains the prediction for a single sample.
        """
        print("\n" + "="*80)
        print(f"Explaining sample #{sample_index} prediction")
        print("="*80)
        
        if self.train_combined is None:
            print("Error: Please run feature extraction first")
            return None
        
        if model_name not in self.models:
            print(f"Error: Model '{model_name}' does not exist")
            print(f"Available models: {list(self.models.keys())}")
            return None
        
        model = self.models[model_name]
        
        if sample_index >= len(self.train_combined):
            print(f"Error: Index out of range (Maximum: {len(self.train_combined)-1})")
            return None
        
        sample_features = self.train_combined.iloc[sample_index:sample_index+1]
        feature_names = self.train_combined.columns.tolist()
        
        prediction = model.predict(sample_features)[0]
        prediction_proba = model.predict_proba(sample_features)[0]
        
        print(f"\n📊 Prediction results:")
        print(f"   Predicted class: {prediction} ({'P1 wins' if prediction == 1 else 'P2 wins'})")
        print(f"   Prediction probability: P1 wins={prediction_proba[1]:.4f}, P2 wins={prediction_proba[0]:.4f}")
        print(f"   Confidence: {max(prediction_proba):.4f}")
        
        if use_shap:
            try:
                import shap
                print(f"\n🔍 Using SHAP values to explain prediction (showing top {top_k} features)...")
                
                explainer = shap.TreeExplainer(model)
                
                if self.X_val is not None and len(self.X_val) > 100:
                    background_data = self.X_val.iloc[:100]
                    shap_values = explainer.shap_values(sample_features, background_data)
                else:
                    shap_values = explainer.shap_values(sample_features)
                
                if isinstance(shap_values, list):
                    shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
                
                if len(shap_values.shape) > 1:
                    sample_shap = shap_values[0]
                else:
                    sample_shap = shap_values
                
                feature_contributions = pd.DataFrame({
                    'feature': feature_names,
                    'shap_value': sample_shap,
                    'feature_value': sample_features.iloc[0].values,
                    'abs_shap': np.abs(sample_shap)
                })
                
                feature_contributions = feature_contributions.sort_values('abs_shap', ascending=False)
                
                print(f"\n📈 Top {top_K} features contributing to this prediction:")
                print("-" * 80)
                print(f"{'Rank':<6} {'Feature':<40} {'Value':<15} {'SHAP Value':<12} {'Direction':<10}")
                print("-" * 80)
                
                for i, (idx, row) in enumerate(feature_contributions.head(top_k).iterrows(), 1):
                    direction = "↑ Supports P1" if row['shap_value'] > 0 else "↓ Supports P2"
                    print(f"{i:<6} {row['feature']:<40} {row['feature_value']:<15.4f} {row['shap_value']:<12.6f} {direction:<10}")
                
                positive_contrib = feature_contributions[feature_contributions['shap_value'] > 0]['shap_value'].sum()
                negative_contrib = feature_contributions[feature_contributions['shap_value'] < 0]['shap_value'].sum()
                
                print(f"\n💡 Contribution summary:")
                print(f"   Total contribution supporting P1 win: {positive_contrib:.4f}")
                print(f"   Total contribution supporting P2 win: {abs(negative_contrib):.4f}")
                print(f"   Net contribution: {positive_contrib + negative_contrib:.4f}")
                
                return {
                    'sample_index': sample_index,
                    'prediction': int(prediction),
                    'prediction_proba': prediction_proba,
                    'feature_contributions': feature_contributions,
                    'method': 'SHAP'
                }
                
            except ImportError:
                print("⚠️ SHAP library not installed, using feature importance method...")
                use_shap = False
            except Exception as e:
                print(f"⚠️ SHAP calculation failed: {e}")
                print("Using feature importance method...")
                use_shap = False
        
        return None

    def analyze_results(self):
        """Analyzes results"""
        print("\n=== Results analysis ===")

        if self.feature_importance:
            print("\nFeature importance analysis:")
            for model_name, importance in self.feature_importance.items():
                print(f"\n{model_name} Top 15 important features:")
                feature_names = self.train_combined.columns
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importance
                }).sort_values('importance', ascending=False)
                print(importance_df.head(15))

        return self.feature_importance