# FILE: model_v2_code.py
# Extracted from submission-2-2.ipynb

import pandas as pd
import numpy as np
import json
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE, RFECV
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import warnings
import os
import sys
import optuna
from optuna.pruners import MedianPruner

def extract_final_pokemon_status(timeline):
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
                    if t_def in no_eff: multiplier *= 0.0
                    elif t_def in super_eff: multiplier *= 2.0
                    elif t_def in meno_eff: multiplier *= 0.5
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
                    if t_def in no_eff: multiplier *= 0.0
                    elif t_def in super_eff: multiplier *= 2.0
                    elif t_def in meno_eff: multiplier *= 0.5
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
    return sum(move.get("priority", 0) for moves in pokemon_moves.values() for move in moves)

TABLE_TYPE = {
    "Normal": ([], ["Rock", "Steel"], ["Ghost"]),
    "Fire": (["Grass", "Ice", "Bug", "Steel"], ["Fire", "Water", "Rock", "Dragon"], []),
    "Water": (["Fire", "Ground", "Rock"], ["Water", "Grass", "Dragon"], []),
    "Electric": (["Water", "Flying"], ["Electric", "Grass", "Dragon"], ["Ground"]),
    "Grass": (["Water", "Ground", "Rock"], ["Fire", "Grass", "Poison", "Flying", "Bug", "Dragon", "Steel"], []),
    "Ice": (["Grass", "Ground", "Flying", "Dragon"], ["Fire", "Water", "Ice", "Steel"], []),
    "Fighting": (["Normal", "Ice", "Rock", "Dark", "Steel"], ["Poison", "Flying", "Psychic", "Bug", "Fairy"], []),
    "Poison": (["Grass", "Fairy"], ["Poison", "Ground", "Rock", "Ghost"], []),
    "Ground": (["Fire", "Electric", "Poison", "Rock", "Steel"], ["Grass", "Bug"], ["Flying"]),
    "Flying": (["Grass", "Fighting", "Bug"], ["Electric", "Rock", "Steel"], []),
    "Psychic": (["Fighting", "Poison"], ["Psychic", "Steel"], ["Dark"]),
    "Bug": (["Grass", "Psychic", "Dark"], ["Fire", "Fighting", "Poison", "Flying", "Ghost", "Steel", "Fairy"], []),
    "Rock": (["Fire", "Ice", "Flying", "Bug"], ["Fighting", "Ground", "Steel"], []),
    "Ghost": (["Psychic", "Ghost"], ["Dark"], ["Normal"]),
    "Dragon": (["Dragon"], ["Steel"], ["Fairy"]),
    "Dark": (["Psychic", "Ghost"], ["Fighting", "Dark", "Fairy"], []),
    "Steel": (["Ice", "Rock", "Fairy"], ["Fire", "Water", "Electric", "Steel"], []),
    "Fairy": (["Fighting", "Dragon", "Dark"], ["Fire", "Poison", "Steel"], [])
}

P_DEF_TYPE = {
    "starmie": ["psychic", "water"], "exeggutor": ["grass", "psychic"], "chansey": ["normal"],
    "snorlax": ["normal"], "tauros": ["normal"], "alakazam": ["psychic"],
    "jynx": ["ice", "psychic"], "slowbro": ["psychic", "water"], "gengar": ["ghost", "poison"],
    "rhydon": ["ground", "rock"], "zapdos": ["electric", "flying"], "cloyster": ["ice", "water"],
    "golem": ["ground", "rock"], "jolteon": ["electric"], "articuno": ["flying", "ice"],
    "persian": ["normal"], "lapras": ["ice", "water"], "dragonite": ["dragon", "flying"],
    "victreebel": ["grass", "poison"], "charizard": ["fire", "flying"]
}

_log_file_handle = None
_original_print = print

def setup_logging(log_dir='print_log'):
    global _log_file_handle
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file_path = os.path.join(log_dir, f'print_log_{timestamp}.log')
    _log_file_handle = open(log_file_path, 'w', encoding='utf-8')
    return log_file_path

def log_print(*args, **kwargs):
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
    global _log_file_handle
    if _log_file_handle is not None:
        _log_file_handle.close()
        _log_file_handle = None

# Apply the logging print function
print = log_print

class PokemonBattlePredictorEnhanced:
    # Class constants
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
        self.features_to_remove = []
        self.validation_split = 0.1
        self.use_optuna_tuning = False
        self.optuna_n_trials = 50
        self.optuna_timeout = 3600
        self.optuna_cv_folds = 3
        self.optuna_pruner_warmup = 5
        self.optuna_pruner_interval = 1
        self.optuna_best_params = {}
        self.optuna_studies = {}
        self.pokemon_db = self.load_pokemon_database() # Load DB on init

    def load_pokemon_database(self):
        pokemon_db = {}
        try:
            if os.path.exists('pokemon_stats_20.json'):
                with open('pokemon_stats_20.json', 'r', encoding='utf-8') as f:
                    pokemon_list = json.load(f)
                    for pokemon in pokemon_list:
                        pokemon_name = pokemon.get('name', '').lower()
                        if pokemon_name:
                            pokemon_db[pokemon_name] = pokemon
                print(f"✓ Loaded {len(pokemon_db)} Pokemon attribute data")
        except Exception as e:
            print(f"⚠️ Failed to load Pokemon attribute database: {e}")
            pokemon_db = {}
        return pokemon_db

    def calculate_pokemon_strength(self, pokemon_data):
        if not pokemon_data: return 0.0
        base_hp = pokemon_data.get('base_hp', 0)
        base_atk = pokemon_data.get('base_atk', 0)
        base_def = pokemon_data.get('base_def', 0)
        base_spa = pokemon_data.get('base_spa', 0)
        base_spd = pokemon_data.get('base_spd', 0)
        base_spe = pokemon_data.get('base_spe', 0)
        hp_score = base_hp * self.WEIGHT_HP
        atk_score = base_atk * self.WEIGHT_ATTACK
        def_score = base_def * self.WEIGHT_DEFENSE
        spa_score = base_spa * self.WEIGHT_ATTACK
        spd_score = base_spd * self.WEIGHT_DEFENSE
        spe_score = base_spe * self.WEIGHT_SPEED
        strength = hp_score + atk_score + def_score + spa_score + spd_score + spe_score
        pokemon_name = pokemon_data.get('name', '').lower()
        if pokemon_name in self.SPECIAL_POKEMON_MODIFIERS:
            strength *= self.SPECIAL_POKEMON_MODIFIERS[pokemon_name]
        return strength

    def extract_hp_features(self, timeline, total_turns):
        features = {}
        p1_hp_losses, p2_hp_losses = [], []
        p1_prev_hp, p2_prev_hp = None, None
        for turn in timeline:
            p1_hp = turn.get('p1_pokemon_state', {}).get('hp_pct', 0)
            p2_hp = turn.get('p2_pokemon_state', {}).get('hp_pct', 0)
            if p1_prev_hp is not None: p1_hp_losses.append(p1_prev_hp - p1_hp)
            if p2_prev_hp is not None: p2_hp_losses.append(p2_prev_hp - p2_hp)
            p1_prev_hp, p2_prev_hp = p1_hp, p2_hp
        features['p1_avg_hp_loss'] = np.mean(p1_hp_losses) if p1_hp_losses else 0
        features['p1_max_hp_loss'] = max(p1_hp_losses) if p1_hp_losses else 0
        features['p2_avg_hp_loss'] = np.mean(p2_hp_losses) if p2_hp_losses else 0
        features['p2_max_hp_loss'] = max(p2_hp_losses) if p2_hp_losses else 0
        return features

    def extract_move_features(self, timeline):
        features = {}
        p1_move_powers, p2_move_powers = [], []
        p1_move_accuracies, p2_move_accuracies = [], []
        p1_switch_count, p2_switch_count = 0, 0
        for turn in timeline:
            p1_move = turn.get('p1_move_details')
            p2_move = turn.get('p2_move_details')
            if p1_move:
                if p1_move.get('base_power', 0) > 0: p1_move_powers.append(p1_move.get('base_power', 0))
                p1_move_accuracies.append(p1_move.get('accuracy', 1.0))
            else: p1_switch_count += 1
            if p2_move:
                if p2_move.get('base_power', 0) > 0: p2_move_powers.append(p2_move.get('base_power', 0))
                p2_move_accuracies.append(p2_move.get('accuracy', 1.0))
            else: p2_switch_count += 1
        features['p1_avg_accuracy'] = np.mean(p1_move_accuracies) if p1_move_accuracies else 1.0
        features['p2_avg_accuracy'] = np.mean(p2_move_accuracies) if p2_move_accuracies else 1.0
        features['p1_switch_count'] = p1_switch_count
        features['p2_switch_count'] = p2_switch_count
        return features

    def extract_status_features(self, timeline):
        features = {}
        p1_pokemon_status_dict, p2_pokemon_status_dict = {}, {}
        p1_counter_invalid, p2_counter_invalid = 0, 0
        p1_pokemon_appeared_30turns, p2_pokemon_appeared_30turns = set(), set()
        p1_move_null_switch, p2_move_null_switch = 0, 0
        p1_move_null_status, p2_move_null_status = 0, 0
        p1_prev_pokemon_name, p2_prev_pokemon_name = None, None

        for i, turn in enumerate(timeline):
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
                if not p2_move or p2_move.get('category', 'STATUS') != 'PHYSICAL': p1_counter_invalid += 1
            if p2_move and p2_move.get('name', '').lower() == 'counter':
                if not p1_move or p1_move.get('category', 'STATUS') != 'PHYSICAL': p2_counter_invalid += 1

            p1_pokemon_name = turn.get('p1_pokemon_state', {}).get('name', '')
            p1_status = turn.get('p1_pokemon_state', {}).get('status', 'nostatus')
            if p1_pokemon_name:
                p1_pokemon_status_dict[p1_pokemon_name] = p1_status
                if i < 30: p1_pokemon_appeared_30turns.add(p1_pokemon_name)
            p2_pokemon_name = turn.get('p2_pokemon_state', {}).get('name', '')
            p2_status = turn.get('p2_pokemon_state', {}).get('status', 'nostatus')
            if p2_pokemon_name:
                p2_pokemon_status_dict[p2_pokemon_name] = p2_status
                if i < 30: p2_pokemon_appeared_30turns.add(p2_pokemon_name)

        p1_abnormal_status_count = sum(1 for name in p1_pokemon_appeared_30turns if p1_pokemon_status_dict.get(name, 'nostatus') not in ['nostatus', 'fnt'])
        p2_abnormal_status_count = sum(1 for name in p2_pokemon_appeared_30turns if p2_pokemon_status_dict.get(name, 'nostatus') not in ['nostatus', 'fnt'])
        p1_fnt_count = sum(1 for status in p1_pokemon_status_dict.values() if status == 'fnt')
        p2_fnt_count = sum(1 for status in p2_pokemon_status_dict.values() if status == 'fnt')

        features['p1_abnormal_status_count'] = p1_abnormal_status_count
        features['p2_abnormal_status_count'] = p2_abnormal_status_count
        features['abnormal_status_count_ratio'] = p1_abnormal_status_count / (p2_abnormal_status_count + 1.0)
        features['p1_counter_invalid'] = p1_counter_invalid
        features['p2_counter_invalid'] = p2_counter_invalid
        features['fnt_count_ratio'] = p1_fnt_count / (p2_fnt_count + 1.0)
        features['fnt_count_diff'] = p1_fnt_count - p2_fnt_count
        features['p1_unique_pokemon_count_30turns'] = len(p1_pokemon_appeared_30turns)
        features['p2_unique_pokemon_count_30turns'] = len(p2_pokemon_appeared_30turns)
        features['p1_move_null_switch'] = p1_move_null_switch
        features['p2_move_null_switch'] = p2_move_null_switch
        features['p1_move_null_status'] = p1_move_null_status
        features['p2_move_null_status'] = p2_move_null_status
        return features

    def calculate_strength_features(self, timeline, pokemon_db):
        features = {}
        p1_pokemon_hp_dict, p2_pokemon_hp_dict = {}, {}
        for i, turn in enumerate(timeline):
            if i < 30:
                p1_name = turn.get('p1_pokemon_state', {}).get('name', '')
                p2_name = turn.get('p2_pokemon_state', {}).get('name', '')
                p1_hp = turn.get('p1_pokemon_state', {}).get('hp_pct', 0)
                p2_hp = turn.get('p2_pokemon_state', {}).get('hp_pct', 0)
                if p1_name:
                    if p1_name not in p1_pokemon_hp_dict: p1_pokemon_hp_dict[p1_name] = []
                    p1_pokemon_hp_dict[p1_name].append(p1_hp)
                if p2_name:
                    if p2_name not in p2_pokemon_hp_dict: p2_pokemon_hp_dict[p2_name] = []
                    p2_pokemon_hp_dict[p2_name].append(p2_hp)

        p1_weighted_strength_sum, p1_strength_list = 0.0, []
        for name, hp_list in p1_pokemon_hp_dict.items():
            if hp_list:
                last_hp = hp_list[-1]
                if last_hp > 0 and name.lower() in pokemon_db:
                    strength = self.calculate_pokemon_strength(pokemon_db[name.lower()])
                    p1_weighted_strength_sum += last_hp * strength
                    p1_strength_list.append(strength)
        p1_missing_count = 6 - len(p1_pokemon_hp_dict)
        if p1_missing_count > 0 and p1_strength_list:
            p1_weighted_strength_sum += p1_missing_count * 1.0 * np.mean(p1_strength_list)

        p2_weighted_strength_sum, p2_strength_list = 0.0, []
        for name, hp_list in p2_pokemon_hp_dict.items():
            if hp_list:
                last_hp = hp_list[-1]
                if last_hp > 0 and name.lower() in pokemon_db:
                    strength = self.calculate_pokemon_strength(pokemon_db[name.lower()])
                    p2_weighted_strength_sum += last_hp * strength
                    p2_strength_list.append(strength)
        p2_missing_count = 6 - len(p2_pokemon_hp_dict)
        if p2_missing_count > 0 and p2_strength_list:
            p2_weighted_strength_sum += p2_missing_count * 1.0 * np.mean(p2_strength_list)

        features['p1_weighted_strength_sum'] = p1_weighted_strength_sum
        features['p2_weighted_strength_sum'] = p2_weighted_strength_sum
        features['weighted_strength_ratio_30turns'] = p1_weighted_strength_sum / (p2_weighted_strength_sum + 1e-6)
        return features

    def load_data(self, train_path, test_path):
        print("Loading data...")
        train_records = []
        with open(train_path, 'r', encoding='utf-8') as f:
            for line in f: train_records.append(json.loads(line.strip()))
        test_records = []
        with open(test_path, 'r', encoding='utf-8') as f:
            for line in f: test_records.append(json.loads(line.strip()))
        self.train_data = pd.DataFrame(train_records)
        self.test_data = pd.DataFrame(test_records)
        print(f"Training data loaded: {len(self.train_data)}  records")
        print(f"Test data loaded: {len(self.test_data)}  records")
        return self.train_data, self.test_data

    def calculate_alive_teams_type_advantage(self, p1_alive_team, p2_alive_team):
        if not p1_alive_team or not p2_alive_team:
            return {'type_adv_mean': 1.0, 'type_adv_max': 1.0, 'type_adv_min': 1.0, 'type_adv_std': 0.0}
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
            return {'type_adv_mean': 1.0, 'type_adv_max': 1.0, 'type_adv_min': 1.0, 'type_adv_std': 0.0}

    def calculate_team_diversity(self, team):
        if not team: return {'type_diversity': 0, 'stat_diversity': 0}
        all_types = []
        for pokemon in team:
            if pokemon and 'types' in pokemon: all_types.extend(pokemon['types'])
        type_diversity = len(set(all_types)) / max(len(all_types), 1)
        stats_matrix = []
        for pokemon in team:
            if pokemon:
                stats_matrix.append([
                    pokemon.get('base_hp', 0), pokemon.get('base_atk', 0), pokemon.get('base_def', 0),
                    pokemon.get('base_spa', 0), pokemon.get('base_spd', 0), pokemon.get('base_spe', 0)
                ])
        stat_diversity = 0
        if stats_matrix:
            stats_matrix = np.array(stats_matrix)
            for i in range(6):
                col = stats_matrix[:, i]
                if np.mean(col) > 0: stat_diversity += np.std(col) / np.mean(col)
            stat_diversity /= 6
        return {'type_diversity': type_diversity, 'stat_diversity': stat_diversity}
    
    def calculate_team_type_advantage(self, p1_team, p2_lead):
        effectiveness = self.get_complete_type_effectiveness()
        advantages = []
        p2_types = p2_lead.get('types', []) if p2_lead else []
        for pokemon in p1_team:
            if not pokemon: continue
            p1_types = pokemon.get('types', [])
            adv = 1.0
            for p1_type in p1_types:
                if p1_type.lower() in effectiveness:
                    for p2_type in p2_types:
                        if p2_type.lower() in effectiveness[p1_type.lower()]:
                            adv *= effectiveness[p1_type.lower()][p2_type.lower()]
            advantages.append(adv)
        if not advantages: return {'type_adv_mean': 1.0, 'type_adv_max': 1.0, 'type_adv_min': 1.0, 'type_adv_std': 0}
        return {
            'type_adv_mean': np.mean(advantages), 'type_adv_max': max(advantages),
            'type_adv_min': min(advantages), 'type_adv_std': np.std(advantages) if len(advantages) > 1 else 0
        }

    def get_complete_type_effectiveness(self):
        return TABLE_TYPE # Assuming TABLE_TYPE is defined globally in the module

    def extract_static_features(self):
        print("\n=== Extracting Static Features (Enhanced) ===")
        pokemon_db = self.pokemon_db

        def extract_pokemon_stats(pokemon):
            if not pokemon: return [0] * 6
            return [
                pokemon.get('base_hp', 0), pokemon.get('base_atk', 0), pokemon.get('base_def', 0),
                pokemon.get('base_spa', 0), pokemon.get('base_spd', 0), pokemon.get('base_spe', 0)
            ]
        def calculate_team_stats(alive_pokemon_names=None):
            if not alive_pokemon_names:
                return {'sum': [0]*6, 'mean': [0]*6, 'max': [0]*6, 'min': [0]*6, 'std': [0]*6}
            stats_matrix = []
            for name in alive_pokemon_names:
                if name.lower() in pokemon_db:
                    stats_matrix.append(extract_pokemon_stats(pokemon_db[name.lower()]))
            if not stats_matrix:
                return {'sum': [0]*6, 'mean': [0]*6, 'max': [0]*6, 'min': [0]*6, 'std': [0]*6}
            stats_matrix = np.array(stats_matrix)
            return {
                'sum': np.sum(stats_matrix, axis=0).tolist(), 'mean': np.mean(stats_matrix, axis=0).tolist(),
                'max': np.max(stats_matrix, axis=0).tolist(), 'min': np.min(stats_matrix, axis=0).tolist(),
                'std': np.std(stats_matrix, axis=0).tolist()
            }

        def extract_single_static_features(row):
            features = {}
            p1_team = row.get('p1_team_details', [])
            timeline = row.get('battle_timeline', [])
            p1_alive, p1_fnt, p2_alive = extract_final_pokemon_status(timeline)
            p1_team_names = {p.get('name', '') for p in p1_team if p and p.get('name', '') not in p1_fnt}
            p1_alive = p1_team_names | p1_alive
            
            p1_team_stats = calculate_team_stats(alive_pokemon_names=p1_alive)
            p2_team_stats = calculate_team_stats(alive_pokemon_names=p2_alive)

            for i, stat_name in enumerate(['hp', 'atk', 'def', 'spa', 'spd', 'spe']):
                features[f'{stat_name}_advantage'] = p1_team_stats['mean'][i] - p2_team_stats['mean'][i]
                features[f'{stat_name}_ratio'] = p1_team_stats['mean'][i] / (p2_team_stats['mean'][i] + 1)

            p1_alive_team = [pokemon_db[name.lower()] for name in p1_alive if name.lower() in pokemon_db]
            p2_alive_team = [pokemon_db[name.lower()] for name in p2_alive if name.lower() in pokemon_db]
            
            type_adv = self.calculate_alive_teams_type_advantage(p1_alive_team, p2_alive_team)
            features.update(type_adv)

            p1_diversity = self.calculate_team_diversity(p1_alive_team)
            features.update(p1_diversity)
            p2_diversity = self.calculate_team_diversity(p2_alive_team)
            features.update({f'p2_{k}': v for k, v in p2_diversity.items()})
            features['type_diversity_ratio'] = p1_diversity.get('type_diversity', 0) / (p2_diversity.get('type_diversity', 0) + 1e-6)
            features['stat_diversity_ratio'] = p1_diversity.get('stat_diversity', 0) / (p2_diversity.get('stat_diversity', 0) + 1e-6)

            p1_phys_atk, p1_spec_atk = p1_team_stats['mean'][1], p1_team_stats['mean'][3]
            p1_phys_def, p1_spec_def = p1_team_stats['mean'][2], p1_team_stats['mean'][4]
            features['physical_special_atk_ratio'] = p1_phys_atk / (p1_spec_atk + 1)
            features['physical_special_def_ratio'] = p1_phys_def / (p1_spec_def + 1)
            features['offense_defense_ratio'] = (p1_phys_atk + p1_spec_atk) / (p1_phys_def + p1_spec_def + 1)
            
            p2_phys_atk, p2_spec_atk = p2_team_stats['mean'][1], p2_team_stats['mean'][3]
            p2_phys_def, p2_spec_def = p2_team_stats['mean'][2], p2_team_stats['mean'][4]
            features['p2_physical_special_atk_ratio'] = p2_phys_atk / (p2_spec_atk + 1)
            features['p2_physical_special_def_ratio'] = p2_phys_def / (p2_spec_def + 1)
            features['p2_offense_defense_ratio'] = (p2_phys_atk + p2_spec_atk) / (p2_phys_def + p2_spec_def + 1)
            
            features['physical_special_atk_ratio_p1_p2'] = features['physical_special_atk_ratio'] / (features['p2_physical_special_atk_ratio'] + 1e-6)
            features['physical_special_def_ratio_p1_p2'] = features['physical_special_def_ratio'] / (features['p2_physical_special_def_ratio'] + 1e-6)
            features['offense_defense_ratio_p1_p2'] = features['offense_defense_ratio'] / (features['p2_offense_defense_ratio'] + 1e-6)

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
        print("\n=== Extracting Dynamic Features (Enhanced) ===")
        train_battle_features = [self.analyze_battle_timeline(row, row.get('battle_timeline', []), self.pokemon_db, None) for _, row in self.train_data.iterrows()]
        test_battle_features = [self.analyze_battle_timeline(row, row.get('battle_timeline', []), self.pokemon_db, None) for _, row in self.test_data.iterrows()]
        self.train_dynamic_features = pd.DataFrame(train_battle_features)
        self.test_dynamic_features = pd.DataFrame(test_battle_features)
        print(f"Dynamic feature extraction completed: {self.train_dynamic_features.shape[1]} features")
        return self.train_dynamic_features, self.test_dynamic_features

    def analyze_battle_timeline(self, row, timeline, pokemon_db=None, turn30_winrates=None):
        if not timeline: return {}
        if pokemon_db is None: pokemon_db = {}
        if turn30_winrates is None: turn30_winrates = {}
        total_turns = len(timeline)
        hp_features = self.extract_hp_features(timeline, total_turns)
        move_features = self.extract_move_features(timeline)
        status_features = self.extract_status_features(timeline)
        strength_features = self.calculate_strength_features(timeline, pokemon_db)
        features = {}
        features.update(hp_features)
        features.update(move_features)
        features.update(status_features)
        features.update(strength_features)
        battle_data = {'battle_timeline': timeline}
        p1_moves, p2_moves = extract_moves(battle_data)
        p1_team_avg, p2_team_avg = type_multiplier(p1_moves, p2_moves)
        features['p1_avg'] = p1_team_avg
        features['p2_avg'] = p2_team_avg
        features['p1_num_priority_moves'] = count_priority_moves(p1_moves)
        features['p2_num_priority_moves'] = count_priority_moves(p2_moves)
        return features

    def create_interaction_features(self, df):
        df_copy = df.copy()
        if 'p1_avg_move_power' in df_copy.columns and 'p1_avg_accuracy' in df_copy.columns:
            df_copy['p1_effective_power'] = df_copy['p1_avg_move_power'] * df_copy['p1_avg_accuracy']
        if 'p2_avg_move_power' in df_copy.columns and 'p2_avg_accuracy' in df_copy.columns:
            df_copy['p2_effective_power'] = df_copy['p2_avg_move_power'] * df_copy['p2_avg_accuracy']
        if 'type_adv_mean' in df_copy.columns and 'total_stats_advantage' in df_copy.columns:
            df_copy['type_stats_interaction'] = df_copy['type_adv_mean'] * df_copy['total_stats_advantage']
        return df_copy

    def combine_features(self):
        print("\n=== Combining Features and Creating Interactions ===")
        self.train_combined = pd.concat([self.train_static_features.reset_index(drop=True), self.train_dynamic_features.reset_index(drop=True)], axis=1)
        self.test_combined = pd.concat([self.test_static_features.reset_index(drop=True), self.test_dynamic_features.reset_index(drop=True)], axis=1)
        self.train_combined = self.create_interaction_features(self.train_combined)
        self.test_combined = self.create_interaction_features(self.test_combined)
        
        # (Feature removal logic is kept, though list is empty)
        if self.features_to_remove:
            print(f"\n=== Removing Configured Features ({len(self.features_to_remove)}) ===")
            features_to_remove_actual = [f for f in self.features_to_remove if f in self.train_combined.columns]
            if features_to_remove_actual:
                self.train_combined = self.train_combined.drop(columns=features_to_remove_actual)
                self.test_combined = self.test_combined.drop(columns=features_to_remove_actual)
                print(f"  ✓ Removed {len(features_to_remove_actual)} features: {features_to_remove_actual}")
        
        self.train_combined = self.train_combined.fillna(0)
        self.test_combined = self.test_combined.fillna(0)
        
        train_cols = set(self.train_combined.columns)
        test_cols = set(self.test_combined.columns)
        missing_in_test = train_cols - test_cols
        missing_in_train = test_cols - train_cols
        for col in missing_in_test: self.test_combined[col] = 0
        for col in missing_in_train: self.train_combined[col] = 0
        self.test_combined = self.test_combined[self.train_combined.columns]
        
        print(f"Feature combination completed:")
        print(f"Training feature shape: {self.train_combined.shape}")
        print(f"Test feature shape: {self.test_combined.shape}")
        return self.train_combined, self.test_combined

    def select_features_rfecv(self, X_train, y_train, estimator=None, cv=5, scoring='accuracy', min_features_to_select=10, n_jobs=-1):
        print(f"\n=== RFECV Feature Selection ===")
        print(f"Initial number of features: {X_train.shape[1]}")
        print(f"Calculating optimal number of features...")
        if estimator is None:
            estimator = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, eval_metric='logloss', n_jobs=n_jobs)
        rfecv = RFECV(estimator=estimator, step=1, cv=cv, scoring=scoring, min_features_to_select=min_features_to_select, n_jobs=n_jobs)
        rfecv.fit(X_train, y_train)
        self.feature_selector = rfecv
        selected_features = X_train.columns[rfecv.support_].tolist()
        print(f"✓ RFECV Feature Selection completed")
        print(f"  Optimal number of features: {rfecv.n_features_}")
        removed_features = [f for f in X_train.columns if f not in selected_features]
        if removed_features:
            print(f"\nRemoved features ({len(removed_features)}): {', '.join(removed_features)}")
        return X_train.loc[:, rfecv.support_], selected_features, rfecv.n_features_

    def train_models(self):
        print("\n=== Model Training (Stacking Ensemble) ===")
        X = self.train_combined
        y = self.train_data['player_won']
        
        xgb_cv_params = dict(
            n_estimators=800, max_depth=8, learning_rate=0.03, subsample=0.85,
            colsample_bytree=0.85, min_child_weight=2, gamma=0.05, reg_alpha=0.1,
            reg_lambda=0.1, random_state=42, eval_metric='logloss', n_jobs=-1
        )
        print("\nPerforming XGBoost 4-fold cross-validation evaluation...")
        xgb_cv_model = xgb.XGBClassifier(**xgb_cv_params)
        xgb_cv_splitter = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
        xgb_cv_scores = cross_val_score(xgb_cv_model, X, y, cv=xgb_cv_splitter, scoring='accuracy', n_jobs=-1)
        print(f"XGBoost 4-fold cross-validation accuracy: {xgb_cv_scores.mean():.4f} ± {xgb_cv_scores.std():.4f}")

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=self.validation_split, random_state=42, stratify=y)
        self.X_val = X_val
        self.y_val = y_val
        val_indices = y_val.index.to_numpy()

        # NOTE: use_rfecv is hardcoded to True in the original notebook, so we keep it.
        use_rfecv = True
        if use_rfecv:
            X_train_selected, selected_features, optimal_n = self.select_features_rfecv(
                X_train, y_train, cv=4, min_features_to_select=20, n_jobs=-1
            )
            X_val_selected = self.feature_selector.transform(X_val)
            X_train = X_train_selected
            X_val = X_val_selected
            self.X_val = X_val # Update self.X_val to the selected features
            print(f"\nTraining models with RFECV-selected features")
            print(f"Training feature shape: {X_train.shape}, Validation feature shape: {X_val.shape}")

        if self.use_optuna_tuning:
            # (Optuna logic removed for brevity, as it's disabled in the notebook)
            print("\nOptuna is disabled. Running with default params.")
        else:
            print("\n⚠️ Optuna hyperparameter optimization disabled, using default parameters")

        X_train_final, y_train_final = X_train.copy(), y_train.copy()
        print(f"\n=== Formal Model Training ===")
        print(f"Training data size: {len(X_train_final)}  samples")

        xgb_params = xgb_cv_params
        lgb_params = {
            'n_estimators': 800, 'max_depth': 10, 'learning_rate': 0.03, 'subsample': 0.85,
            'colsample_bytree': 0.85, 'min_child_samples': 15, 'reg_alpha': 0.1,
            'reg_lambda': 0.1, 'num_leaves': 50, 'random_state': 42, 'verbose': -1, 'n_jobs': -1
        }
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
        stacking_estimators = [('XGBoost', base_models['XGBoost']), ('LightGBM', base_models['LightGBM']), ('CatBoost', base_models['CatBoost'])]
        stacking_model = StackingClassifier(estimators=stacking_estimators, final_estimator=LogisticRegression(C=1.0, max_iter=500, random_state=42), cv=2, n_jobs=1)
        
        try:
            stacking_model.fit(X_train_final, y_train_final)
            y_pred_stacking = stacking_model.predict(X_val)
            stacking_accuracy = accuracy_score(y_val, y_pred_stacking)
            print(f"Stacking ensemble model validation accuracy: {stacking_accuracy:.4f}")
            self.models['Stacking'] = stacking_model
        except Exception as e:
            print(f"Stacking training failed: {e}")
            print("Skipping Stacking model, using best base model as fallback")

        if 'Stacking' in self.models:
            self.best_model_name = 'Stacking'
            self.best_model = self.models['Stacking']
        else:
            best_accuracy = 0
            best_model_name = 'XGBoost'
            for name in ['XGBoost', 'LightGBM', 'CatBoost']:
                if name in self.models:
                    accuracy = accuracy_score(y_val, self.models[name].predict(X_val))
                    if accuracy > best_accuracy:
                        best_accuracy, best_model_name = accuracy, name
            self.best_model_name = best_model_name
            self.best_model = self.models[best_model_name]
        
        print(f"Selected best model: {self.best_model_name}")
        return self.models

    def make_predictions(self, model_name, output_filename):
        """Generates predictions for a specific model."""
        print(f"\n=== Generating Predictions for {model_name} ===")
        
        if model_name not in self.models:
            print(f"Error: Model '{model_name}' not found. Cannot make predictions.")
            return None

        model_to_use = self.models[model_name]
        
        test_features = self.test_combined
        # Apply RFECV transformation if it was used
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
        
        predictions = model_to_use.predict(test_features)
        
        submission = pd.DataFrame({
            'battle_id': self.test_data['battle_id'],
            'player_won': predictions.astype(int)
        })
        submission.to_csv(output_filename, index=False)
        print(f"Submission file generated: {output_filename}")
        return submission

    def analyze_results(self):
        print("\n=== Results Analysis ===")
        if self.feature_importance:
            print("\nFeature Importance Analysis:")
            for model_name, importance in self.feature_importance.items():
                if importance is None: continue
                print(f"\n{model_name} Top 15 important features:")
                
                # Check if feature selector was used
                if hasattr(self, 'feature_selector') and self.feature_selector is not None:
                    feature_names = self.train_combined.columns[self.feature_selector.support_]
                else:
                    feature_names = self.train_combined.columns
                    
                if len(feature_names) != len(importance):
                    print(f"  Warning: Mismatch in feature count ({len(feature_names)}) and importance count ({len(importance)}). Skipping.")
                    continue

                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importance
                }).sort_values('importance', ascending=False)
                print(importance_df.head(15))
        return self.feature_importance