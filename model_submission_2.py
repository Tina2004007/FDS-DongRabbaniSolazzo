# Pokemon Battles Prediction 2025 - Enhanced Version
# FDS Kaggle Competition Solution - Advanced Feature Engineering
# This file will be imported as a module: model_submission_2.py

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

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

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

# We keep the print override here, as it's part of the module's logic
print = log_print

class PokemonBattlePredictorEnhanced:

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
        self.all_moves_list = None
        
        self.features_to_remove = ['p1_unique_pokemon_count_30turns','p2_unique_pokemon_count_30turns','p1_switch_count','p2_switch_count','p1_hp_min','p2_hp_min', 'p2_move_null_switch','p1_move_null_switch']

    def get_all_moves_list(self, max_samples=10000):
        if self.all_moves_list is not None:
            return self.all_moves_list
        
        print("\nCollecting all move list...")
        all_moves = set()
        
        sample_size = min(max_samples, len(self.train_data))
        for idx in range(sample_size):
            row = self.train_data.iloc[idx]
            timeline = row.get('battle_timeline', [])
            
            for turn in timeline:
                p1_move = turn.get('p1_move_details')
                p2_move = turn.get('p2_move_details')
                
                if p1_move and p1_move.get('name'):
                    all_moves.add(p1_move.get('name'))
                
                if p2_move and p2_move.get('name'):
                    all_moves.add(p2_move.get('name'))
        
        self.all_moves_list = sorted(list(all_moves))
        print(f"✓ Found {len(self.all_moves_list)}  unique moves")
        
        return self.all_moves_list

    def load_data(self, train_path, test_path):
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

        print(f"Training data loaded: {len(self.train_data)}  records")
        print(f"Test data loaded: {len(self.test_data)}  records")

        return self.train_data, self.test_data

    def get_complete_type_effectiveness(self):
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

    def calculate_team_diversity(self, team):
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
        print("\n=== Extracting static features (Enhanced) ===")

        def extract_pokemon_stats(pokemon):
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

        def calculate_team_stats(team, fnt_pokemon_names=None):
            if not team:
                return {'sum': [0]*6, 'mean': [0]*6, 'max': [0]*6, 'min': [0]*6, 'std': [0]*6}

            stats_matrix = []
            for pokemon in team:
                if fnt_pokemon_names is not None:
                    pokemon_name = pokemon.get('name', '') if pokemon else ''
                    if pokemon_name in fnt_pokemon_names:
                        continue
                
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

        train_features = []
        for idx, row in self.train_data.iterrows():
            features = {}

            p1_fnt_pokemon_names = set()
            timeline = row.get('battle_timeline', [])
            for turn in timeline:
                p1_status = turn.get('p1_pokemon_state', {}).get('status', 'nostatus')
                if p1_status == 'fnt':
                    p1_pokemon_name = turn.get('p1_pokemon_state', {}).get('name', '')
                    if p1_pokemon_name:
                        p1_fnt_pokemon_names.add(p1_pokemon_name)

            p1_team = row.get('p1_team_details', [])
            p1_team_stats = calculate_team_stats(p1_team, fnt_pokemon_names=p1_fnt_pokemon_names)

            p2_lead = row.get('p2_lead_details', {})
            p2_stats = extract_pokemon_stats(p2_lead)

            for agg_type in ['sum', 'mean', 'max', 'min', 'std']:
                for i, stat_name in enumerate(['hp', 'atk', 'def', 'spa', 'spd', 'spe']):
                    features[f'p1_team_{stat_name}_{agg_type}'] = p1_team_stats[agg_type][i]

            for i, stat_name in enumerate(['hp', 'atk', 'def', 'spa', 'spd', 'spe']):
                features[f'p2_lead_{stat_name}'] = p2_stats[i]

            for i, stat_name in enumerate(['hp', 'atk', 'def', 'spa', 'spd', 'spe']):
                features[f'{stat_name}_advantage'] = p1_team_stats['mean'][i] - p2_stats[i]
                features[f'{stat_name}_ratio'] = p1_team_stats['mean'][i] / (p2_stats[i] + 1)

            type_adv = self.calculate_team_type_advantage(p1_team, p2_lead)
            features.update(type_adv)

            diversity = self.calculate_team_diversity(p1_team)
            features.update(diversity)

            physical_atk = p1_team_stats['mean'][1]
            special_atk = p1_team_stats['mean'][3]
            physical_def = p1_team_stats['mean'][2]
            special_def = p1_team_stats['mean'][4]

            features['physical_special_atk_ratio'] = physical_atk / (special_atk + 1)
            features['physical_special_def_ratio'] = physical_def / (special_def + 1)
            features['offense_defense_ratio'] = (physical_atk + special_atk) / (physical_def + special_def + 1)

            features['p1_total_stats'] = sum(p1_team_stats['sum'])
            features['p2_total_stats'] = sum(p2_stats)
            features['total_stats_advantage'] = features['p1_total_stats'] - features['p2_total_stats']

            train_features.append(features)

        test_features = []
        for idx, row in self.test_data.iterrows():
            features = {}

            p1_fnt_pokemon_names = set()
            timeline = row.get('battle_timeline', [])
            for turn in timeline:
                p1_status = turn.get('p1_pokemon_state', {}).get('status', 'nostatus')
                if p1_status == 'fnt':
                    p1_pokemon_name = turn.get('p1_pokemon_state', {}).get('name', '')
                    if p1_pokemon_name:
                        p1_fnt_pokemon_names.add(p1_pokemon_name)

            p1_team = row.get('p1_team_details', [])
            p1_team_stats = calculate_team_stats(p1_team, fnt_pokemon_names=p1_fnt_pokemon_names)

            p2_lead = row.get('p2_lead_details', {})
            p2_stats = extract_pokemon_stats(p2_lead)

            for agg_type in ['sum', 'mean', 'max', 'min', 'std']:
                for i, stat_name in enumerate(['hp', 'atk', 'def', 'spa', 'spd', 'spe']):
                    features[f'p1_team_{stat_name}_{agg_type}'] = p1_team_stats[agg_type][i]

            for i, stat_name in enumerate(['hp', 'atk', 'def', 'spa', 'spd', 'spe']):
                features[f'p2_lead_{stat_name}'] = p2_stats[i]

            for i, stat_name in enumerate(['hp', 'atk', 'def', 'spa', 'spd', 'spe']):
                features[f'{stat_name}_advantage'] = p1_team_stats['mean'][i] - p2_stats[i]
                features[f'{stat_name}_ratio'] = p1_team_stats['mean'][i] / (p2_stats[i] + 1)

            type_adv = self.calculate_team_type_advantage(p1_team, p2_lead)
            features.update(type_adv)

            diversity = self.calculate_team_diversity(p1_team)
            features.update(diversity)

            physical_atk = p1_team_stats['mean'][1]
            special_atk = p1_team_stats['mean'][3]
            physical_def = p1_team_stats['mean'][2]
            special_def = p1_team_stats['mean'][4]

            features['physical_special_atk_ratio'] = physical_atk / (special_atk + 1)
            features['physical_special_def_ratio'] = physical_def / (special_def + 1)
            features['offense_defense_ratio'] = (physical_atk + special_atk) / (physical_def + special_def + 1)

            features['p1_total_stats'] = sum(p1_team_stats['sum'])
            features['p2_total_stats'] = sum(p2_stats)
            features['total_stats_advantage'] = features['p1_total_stats'] - features['p2_total_stats']

            test_features.append(features)

        self.train_static_features = pd.DataFrame(train_features)
        self.test_static_features = pd.DataFrame(test_features)

        print(f"Static features extraction completed: {self.train_static_features.shape[1]}  features")
        return self.train_static_features, self.test_static_features

    def extract_dynamic_features(self):
        print("\n=== Extracting dynamic features (Enhanced) ===")
        
        all_moves_list = self.get_all_moves_list()
        
        def sanitize_move_name(move_name):
            sanitized = move_name.replace(' ', '_').replace('-', '_').replace("'", '').replace('.', '')
            sanitized = ''.join(c if c.isalnum() or c == '_' else '_' for c in sanitized)
            return sanitized.lower()
        
        move_to_feature = {move: sanitize_move_name(move) for move in all_moves_list}

        def analyze_battle_timeline(timeline):
            if not timeline:
                return {}

            features = {}
            total_turns = len(timeline)
            features['total_turns'] = total_turns

            p1_hp_changes = []
            p2_hp_changes = []
            p1_moves = []
            p2_moves = []
            p1_pokemon_status_dict = {}
            p2_pokemon_status_dict = {}
            p1_counter_invalid = 0
            p2_counter_invalid = 0
            p1_move_powers = []
            p2_move_powers = []
            p1_move_accuracies = []
            p2_move_accuracies = []
            p1_physical_moves = 0
            p2_physical_moves = 0
            p1_special_moves = 0
            p2_special_moves = 0
            p1_status_moves = 0
            p2_status_moves = 0
            p1_switch_count = 0
            p2_switch_count = 0
            p1_move_null_switch = 0
            p2_move_null_switch = 0
            p1_move_null_status = 0
            p2_move_null_status = 0
            
            p1_move_counts = {move: 0 for move in all_moves_list}
            p2_move_counts = {move: 0 for move in all_moves_list}
            
            p1_boost_changes = []
            p2_boost_changes = []
            p1_final_boosts = {'atk': 0, 'def': 0, 'spa': 0, 'spd': 0, 'spe': 0}
            p2_final_boosts = {'atk': 0, 'def': 0, 'spa': 0, 'spd': 0, 'spe': 0}
            p1_effect_turns = 0
            p2_effect_turns = 0
            p1_hp_losses = []
            p2_hp_losses = []

            early_end = total_turns // 3
            mid_end = 2 * total_turns // 3

            p1_early_hp = []
            p2_early_hp = []
            p1_mid_hp = []
            p2_mid_hp = []
            p1_late_hp = []
            p2_late_hp = []

            p1_consecutive_attacks = 0
            p2_consecutive_attacks = 0
            p1_max_consecutive = 0
            p2_max_consecutive = 0
            p1_last_was_attack = False
            p2_last_was_attack = False

            p1_switch_turns = []
            p2_switch_turns = []
            
            p1_pokemon_appeared_30turns = set()
            p2_pokemon_appeared_30turns = set()
            
            p1_pokemon_hp_dict = {}
            p2_pokemon_hp_dict = {}

            p1_prev_pokemon_name = None
            p2_prev_pokemon_name = None

            for i, turn in enumerate(timeline):
                p1_hp = turn.get('p1_pokemon_state', {}).get('hp_pct', 0)
                p2_hp = turn.get('p2_pokemon_state', {}).get('hp_pct', 0)
                p1_hp_changes.append(p1_hp)
                p2_hp_changes.append(p2_hp)
                
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

                if i < early_end:
                    p1_early_hp.append(p1_hp)
                    p2_early_hp.append(p2_hp)
                elif i < mid_end:
                    p1_mid_hp.append(p1_hp)
                    p2_mid_hp.append(p2_hp)
                else:
                    p1_late_hp.append(p1_hp)
                    p2_late_hp.append(p2_hp)

                if i > 0:
                    p1_hp_loss = p1_hp_changes[i-1] - p1_hp
                    p2_hp_loss = p2_hp_changes[i-1] - p2_hp
                    p1_hp_losses.append(p1_hp_loss)
                    p2_hp_losses.append(p2_hp_loss)

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
                    if not p2_move:
                        p1_counter_invalid += 1
                    elif p2_move.get('category', 'STATUS') != 'PHYSICAL':
                        p1_counter_invalid += 1
                
                if p2_move and p2_move.get('name', '').lower() == 'counter':
                    if not p1_move:
                        p2_counter_invalid += 1
                    elif p1_move.get('category', 'STATUS') != 'PHYSICAL':
                        p2_counter_invalid += 1

                if p1_move:
                    move_name = p1_move.get('name', '')
                    p1_moves.append(move_name)
                    
                    if move_name in p1_move_counts:
                        p1_move_counts[move_name] += 1
                    
                    power = p1_move.get('base_power', 0)
                    accuracy = p1_move.get('accuracy', 1.0)
                    category = p1_move.get('category', 'STATUS')

                    if power > 0:
                        p1_move_powers.append(power)
                        p1_consecutive_attacks += 1
                        p1_last_was_attack = True
                    else:
                        if p1_last_was_attack:
                            p1_max_consecutive = max(p1_max_consecutive, p1_consecutive_attacks)
                            p1_consecutive_attacks = 0
                        p1_last_was_attack = False

                    p1_move_accuracies.append(accuracy)

                    if category == 'PHYSICAL':
                        p1_physical_moves += 1
                    elif category == 'SPECIAL':
                        p1_special_moves += 1
                    elif category == 'STATUS':
                        p1_status_moves += 1
                else:
                    p1_switch_count += 1
                    p1_switch_turns.append(i)
                    if p1_last_was_attack:
                        p1_max_consecutive = max(p1_max_consecutive, p1_consecutive_attacks)
                        p1_consecutive_attacks = 0
                    p1_last_was_attack = False

                if p2_move:
                    move_name = p2_move.get('name', '')
                    p2_moves.append(move_name)
                    
                    if move_name in p2_move_counts:
                        p2_move_counts[move_name] += 1
                    
                    power = p2_move.get('base_power', 0)
                    accuracy = p2_move.get('accuracy', 1.0)
                    category = p2_move.get('category', 'STATUS')

                    if power > 0:
                        p2_move_powers.append(power)
                        p2_consecutive_attacks += 1
                        p2_last_was_attack = True
                    else:
                        if p2_last_was_attack:
                            p2_max_consecutive = max(p2_max_consecutive, p2_consecutive_attacks)
                            p2_consecutive_attacks = 0
                        p2_last_was_attack = False

                    p2_move_accuracies.append(accuracy)

                    if category == 'PHYSICAL':
                        p2_physical_moves += 1
                    elif category == 'SPECIAL':
                        p2_special_moves += 1
                    elif category == 'STATUS':
                        p2_status_moves += 1
                else:
                    p2_switch_count += 1
                    p2_switch_turns.append(i)
                    if p2_last_was_attack:
                        p2_max_consecutive = max(p2_max_consecutive, p2_consecutive_attacks)
                        p2_consecutive_attacks = 0
                    p2_last_was_attack = False

                p1_pokemon_name = turn.get('p1_pokemon_state', {}).get('name', '')
                p1_status = turn.get('p1_pokemon_state', {}).get('status', 'nostatus')
                if p1_pokemon_name:
                    p1_pokemon_status_dict[p1_pokemon_name] = p1_status

                p2_pokemon_name = turn.get('p2_pokemon_state', {}).get('name', '')
                p2_status = turn.get('p2_pokemon_state', {}).get('status', 'nostatus')
                if p2_pokemon_name:
                    p2_pokemon_status_dict[p2_pokemon_name] = p2_status

                p1_boosts = turn.get('p1_pokemon_state', {}).get('boosts', {})
                p2_boosts = turn.get('p2_pokemon_state', {}).get('boosts', {})

                if p1_boosts:
                    p1_final_boosts = p1_boosts
                    boost_sum = sum(p1_boosts.values())
                    p1_boost_changes.append(boost_sum)

                if p2_boosts:
                    p2_final_boosts = p2_boosts
                    boost_sum = sum(p2_boosts.values())
                    p2_boost_changes.append(boost_sum)

                p1_effects = turn.get('p1_pokemon_state', {}).get('effects', ['noeffect'])
                p2_effects = turn.get('p2_pokemon_state', {}).get('effects', ['noeffect'])

                if p1_effects and 'noeffect' not in p1_effects:
                    p1_effect_turns += 1
                if p2_effects and 'noeffect' not in p2_effects:
                    p2_effect_turns += 1

            if p1_hp_changes:
                features['p1_hp_start'] = p1_hp_changes[0]
                features['p1_hp_end'] = p1_hp_changes[-1]
                features['p1_hp_min'] = min(p1_hp_changes)
                features['p1_hp_max'] = max(p1_hp_changes)
                features['p1_hp_avg'] = np.mean(p1_hp_changes)
                features['p1_hp_std'] = np.std(p1_hp_changes)
                if len(p1_hp_changes) > 1:
                    features['p1_hp_trend'] = np.polyfit(range(len(p1_hp_changes)), p1_hp_changes, 1)[0]
                else:
                    features['p1_hp_trend'] = 0

            if p2_hp_changes:
                features['p2_hp_start'] = p2_hp_changes[0]
                features['p2_hp_end'] = p2_hp_changes[-1]
                features['p2_hp_min'] = min(p2_hp_changes)
                features['p2_hp_max'] = max(p2_hp_changes)
                features['p2_hp_avg'] = np.mean(p2_hp_changes)
                features['p2_hp_std'] = np.std(p2_hp_changes)
                if len(p2_hp_changes) > 1:
                    features['p2_hp_trend'] = np.polyfit(range(len(p2_hp_changes)), p2_hp_changes, 1)[0]
                else:
                    features['p2_hp_trend'] = 0

            for phase, p1_hp_list, p2_hp_list in [
                ('early', p1_early_hp, p2_early_hp),
                ('mid', p1_mid_hp, p2_mid_hp),
                ('late', p1_late_hp, p2_late_hp)
            ]:
                if p1_hp_list:
                    features[f'p1_{phase}_hp_avg'] = np.mean(p1_hp_list)
                    features[f'p1_{phase}_hp_min'] = min(p1_hp_list)
                else:
                    features[f'p1_{phase}_hp_avg'] = 0
                    features[f'p1_{phase}_hp_min'] = 0

                if p2_hp_list:
                    features[f'p2_{phase}_hp_avg'] = np.mean(p2_hp_list)
                    features[f'p2_{phase}_hp_min'] = min(p2_hp_list)
                else:
                    features[f'p2_{phase}_hp_avg'] = 0
                    features[f'p2_{phase}_hp_min'] = 0

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

            features['hp_advantage_start'] = features.get('p1_hp_start', 0) - features.get('p2_hp_start', 0)
            features['hp_advantage_end'] = features.get('p1_hp_end', 0) - features.get('p2_hp_end', 0)
            features['hp_advantage_avg'] = features.get('p1_hp_avg', 0) - features.get('p2_hp_avg', 0)


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
            
            features['abnormal_status_count_ratio'] = p1_abnormal_status_count / (p2_abnormal_status_count + 1.0)
            
            features['p1_counter_invalid'] = p1_counter_invalid
            features['p2_counter_invalid'] = p2_counter_invalid

            if p1_move_powers:
                features['p1_avg_move_power'] = np.mean(p1_move_powers)
                features['p1_max_move_power'] = max(p1_move_powers)
                features['p1_min_move_power'] = min(p1_move_powers)
            else:
                features['p1_avg_move_power'] = 0
                features['p1_max_move_power'] = 0
                features['p1_min_move_power'] = 0

            if p2_move_powers:
                features['p2_avg_move_power'] = np.mean(p2_move_powers)
                features['p2_max_move_power'] = max(p2_move_powers)
                features['p2_min_move_power'] = min(p2_move_powers)
            else:
                features['p2_avg_move_power'] = 0
                features['p2_max_move_power'] = 0
                features['p2_min_move_power'] = 0

            if p1_move_accuracies:
                features['p1_avg_accuracy'] = np.mean(p1_move_accuracies)
            else:
                features['p1_avg_accuracy'] = 1.0

            if p2_move_accuracies:
                features['p2_avg_accuracy'] = np.mean(p2_move_accuracies)
            else:
                features['p2_avg_accuracy'] = 1.0

            total_p1_moves = max(p1_physical_moves + p1_special_moves + p1_status_moves, 1)
            total_p2_moves = max(p2_physical_moves + p2_special_moves + p2_status_moves, 1)

            features['p1_physical_move_ratio'] = p1_physical_moves / total_p1_moves
            features['p1_special_move_ratio'] = p1_special_moves / total_p1_moves
            features['p1_status_move_ratio'] = p1_status_moves / total_p1_moves

            features['p2_physical_move_ratio'] = p2_physical_moves / total_p2_moves
            features['p2_special_move_ratio'] = p2_special_moves / total_p2_moves
            features['p2_status_move_ratio'] = p2_status_moves / total_p2_moves

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

            p1_max_consecutive = max(p1_max_consecutive, p1_consecutive_attacks)
            p2_max_consecutive = max(p2_max_consecutive, p2_consecutive_attacks)
            features['p1_max_consecutive_attacks'] = p1_max_consecutive
            features['p2_max_consecutive_attacks'] = p2_max_consecutive

            features['p1_final_boost_sum'] = sum(p1_final_boosts.values())
            features['p2_final_boost_sum'] = sum(p2_final_boosts.values())
            features['p1_final_atk_boost'] = p1_final_boosts.get('atk', 0)
            features['p1_final_def_boost'] = p1_final_boosts.get('def', 0)
            features['p1_final_spa_boost'] = p1_final_boosts.get('spa', 0)
            features['p1_final_spd_boost'] = p1_final_boosts.get('spd', 0)
            features['p1_final_spe_boost'] = p1_final_boosts.get('spe', 0)

            features['p2_final_atk_boost'] = p2_final_boosts.get('atk', 0)
            features['p2_final_def_boost'] = p2_final_boosts.get('def', 0)
            features['p2_final_spa_boost'] = p2_final_boosts.get('spa', 0)
            features['p2_final_spd_boost'] = p2_final_boosts.get('spd', 0)
            features['p2_final_spe_boost'] = p2_final_boosts.get('spe', 0)

            features['boost_advantage'] = features['p1_final_boost_sum'] - features['p2_final_boost_sum']

            features['p1_effect_turns'] = p1_effect_turns
            features['p2_effect_turns'] = p2_effect_turns
            
            features['p1_unique_pokemon_count_30turns'] = len(p1_pokemon_appeared_30turns)
            features['p2_unique_pokemon_count_30turns'] = len(p2_pokemon_appeared_30turns)
            
            p1_total_hp_pct = 0.0
            p1_appeared_count = len(p1_pokemon_hp_dict)
            for pokemon_name, hp_list in p1_pokemon_hp_dict.items():
                if hp_list:
                    last_hp = hp_list[-1]
                    p1_total_hp_pct += last_hp
            
            p1_missing_count = 6 - p1_appeared_count
            if p1_missing_count > 0:
                p1_total_hp_pct += p1_missing_count * 1.0
            
            p2_total_hp_pct = 0.0
            p2_appeared_count = len(p2_pokemon_hp_dict)
            for pokemon_name, hp_list in p2_pokemon_hp_dict.items():
                if hp_list:
                    last_hp = hp_list[-1]
                    p2_total_hp_pct += last_hp
            
            p2_missing_count = 6 - p2_appeared_count
            if p2_missing_count > 0:
                p2_total_hp_pct += p2_missing_count * 1.0
            
            
            if p2_total_hp_pct > 0:
                features['total_pokemon_hp_pct_ratio_30turns'] = p1_total_hp_pct / p2_total_hp_pct
            else:
                features['total_pokemon_hp_pct_ratio_30turns'] = 0.0
            
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

            return features

        train_battle_features = []
        for idx, row in self.train_data.iterrows():
            timeline = row.get('battle_timeline', [])
            battle_features = analyze_battle_timeline(timeline)
            train_battle_features.append(battle_features)

        test_battle_features = []
        for idx, row in self.test_data.iterrows():
            timeline = row.get('battle_timeline', [])
            battle_features = analyze_battle_timeline(timeline)
            test_battle_features.append(battle_features)

        self.train_dynamic_features = pd.DataFrame(train_battle_features)
        self.test_dynamic_features = pd.DataFrame(test_battle_features)

        print(f"Dynamic features extraction completed: {self.train_dynamic_features.shape[1]}  features")
        return self.train_dynamic_features, self.test_dynamic_features

    def create_interaction_features(self, df):
        df_copy = df.copy()

        if 'hp_advantage_end' in df_copy.columns and 'boost_advantage' in df_copy.columns:
            df_copy['hp_boost_interaction'] = df_copy['hp_advantage_end'] * df_copy['boost_advantage']

        if 'p1_avg_move_power' in df_copy.columns and 'p1_avg_accuracy' in df_copy.columns:
            df_copy['p1_effective_power'] = df_copy['p1_avg_move_power'] * df_copy['p1_avg_accuracy']

        if 'p2_avg_move_power' in df_copy.columns and 'p2_avg_accuracy' in df_copy.columns:
            df_copy['p2_effective_power'] = df_copy['p2_avg_move_power'] * df_copy['p2_avg_accuracy']

        if 'type_adv_mean' in df_copy.columns and 'total_stats_advantage' in df_copy.columns:
            df_copy['type_stats_interaction'] = df_copy['type_adv_mean'] * df_copy['total_stats_advantage']

        return df_copy

    def combine_features(self):
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
                    print(f"  Warning: Feature '{feature}'  does not exist, skipping removal")
            
            if features_to_remove_actual:
                self.train_combined = self.train_combined.drop(columns=features_to_remove_actual)
                self.test_combined = self.test_combined.drop(columns=features_to_remove_actual)
                print(f"  ✓ Removed {len(features_to_remove_actual)}  features: {features_to_remove_actual}")
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
        print(f"\n=== RFECV Feature Selection ===")
        print(f"Initial feature count: {X_train.shape[1]}")
        print(f"Cross-validation folds: {cv}")
        print(f"Minimum features to select: {min_features_to_select}")
        print("Calculating optimal feature count (this may take a few minutes)...")
        
        if estimator is None:
            estimator = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=1,
                gamma=0,
                reg_alpha=0.01,
                reg_lambda=0.01,
                random_state=42,
                eval_metric='logloss',
                n_jobs=n_jobs
            )
        
        rfecv = RFECV(
            estimator=estimator,
            step=1,
            cv=cv,
            scoring=scoring,
            min_features_to_select=min_features_to_select,
            n_jobs=n_jobs
        )
        
        rfecv.fit(X_train, y_train)
        
        self.feature_selector = rfecv
        
        selected_mask = rfecv.support_
        selected_features = X_train.columns[selected_mask].tolist()
        optimal_n_features = rfecv.n_features_
        
        X_train_selected = rfecv.transform(X_train)
        
        if isinstance(X_train, pd.DataFrame):
            X_train_selected = pd.DataFrame(
                X_train_selected,
                columns=selected_features,
                index=X_train.index
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
        print(f"\n=== Saving{model_name} misclassified samples ===")
        
        misclassified_mask = y_pred != y_true.values
        misclassified_indices = y_true_indices[misclassified_mask]
        
        if len(misclassified_indices) == 0:
            print("No misclassified samples!")
            return None, None
        
        print(f"Found {len(misclassified_indices)}  misclassified samples")
        
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

    def train_models(self):
        print("\n=== Model Training (Stacking Ensemble) ===")

        X = self.train_combined
        y = self.train_data['player_won']

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.1, random_state=42, stratify=y
        )
        
        self.X_val = X_val
        self.y_val = y_val
        
        val_indices = y_val.index.to_numpy()

        
        use_rfecv = False
        if use_rfecv:
            X_train_selected, selected_features, optimal_n = self.select_features_rfecv(
                X_train, y_train,
                estimator=None,
                cv=4,
                scoring='accuracy',
                min_features_to_select=20,
                n_jobs=-1
            )
            X_val_selected = self.feature_selector.transform(X_val)
            X_train = X_train_selected
            X_val = X_val_selected
            print(f"\nTraining models with RFECV selected features")
            print(f"Training feature shape: {X_train.shape}, Validation feature shape: {X_val.shape}")

        base_models = {
            'XGBoost': xgb.XGBClassifier(
                n_estimators=800,
                max_depth=8,
                learning_rate=0.03,
                subsample=0.85,
                colsample_bytree=0.85,
                min_child_weight=2,
                gamma=0.05,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=42,
                eval_metric='logloss',
                n_jobs=-1
            ),
            'LightGBM': lgb.LGBMClassifier(
                n_estimators=800,
                max_depth=10,
                learning_rate=0.03,
                subsample=0.85,
                colsample_bytree=0.85,
                min_child_samples=15,
                reg_alpha=0.1,
                reg_lambda=0.1,
                num_leaves=50,
                random_state=42,
                verbose=-1,
                n_jobs=-1
            ),
            'CatBoost': CatBoostClassifier(
                iterations=800,
                depth=9,
                learning_rate=0.03,
                l2_leaf_reg=2,
                random_seed=42,
                verbose=False,
                thread_count=-1
            )
        }

        for name, model in base_models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)

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
            cv='prefit',
            n_jobs=1
        )

        try:
            stacking_model.fit(X_train, y_train)
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
            self.best_model = self.models['Voting']
            best_predictions = y_pred_voting
        
        # self.save_misclassified_samples(val_indices, best_predictions, y_val, self.best_model_name)
        

        return self.models

    def make_predictions(self):
        print("\n=== Generating predictions ===")

        test_features = self.test_combined
        if hasattr(self, 'feature_selector') and self.feature_selector is not None:
            print("Applying feature selection to test data...")
            test_features_array = self.feature_selector.transform(self.test_combined)
            if isinstance(self.test_combined, pd.DataFrame):
                selected_features = self.test_combined.columns[self.feature_selector.support_].tolist()
                test_features = pd.DataFrame(
                    test_features_array,
                    columns=selected_features,
                    index=self.test_combined.index
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
            best_predictions = predictions['Voting']
            print("Using Voting model for final predictions")

        submission = pd.DataFrame({
            'battle_id': self.test_data['battle_id'],
            'player_won': best_predictions.astype(int)
        })

        submission.to_csv('submission_enhanced_v2.csv', index=False)
        print("Submission file generated: submission_enhanced_v2.csv")

        return submission, predictions

    def get_sample_features(self, original_index):
        if self.train_combined is None:
            print("Error: Please run feature extraction first (extract_static_features, extract_dynamic_features, combine_features)")
            return None

        if original_index >= len(self.train_combined):
            print(f"Error: Index out of range (Maximum: {len(self.train_combined)-1})")
            return None

        return self.train_combined.iloc[original_index].to_dict()

    def explain_single_prediction(self, sample_index, model_name='XGBoost', top_k=20, use_shap=True):
        print("\n" + "="*80)
        print(f"Explaining sample #{sample_index} prediction")
        print("="*80)
        
        if self.train_combined is None:
            print("Error: Please run feature extraction first")
            return None
        
        if model_name not in self.models:
            print(f"Error: Model '{model_name}'  does not exist")
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
        print(f"   Predicted class: {prediction} ({'P1 wins' if prediction == 1 else 'P2 wins'})\")")
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
                
                print(f"\n📈 Top {top_k} features contributing to this prediction:")
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

    def analyze_feature_usefulness(self):
        print("\n" + "="*80)
        print("Feature Usefulness Analysis")
        print("="*80)
        
        if self.X_val is None or self.y_val is None or self.best_model is None:
            print("Error: Please train model first")
            return None
        
        feature_names = self.train_combined.columns.tolist()
        n_features = len(feature_names)
        
        print("\n[1/4] Calculating tree-based feature importance...")
        
        tree_importance_dict = {}
        
        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                tree_importance_dict[name] = model.feature_importances_
        
        print("\n[2/4] Calculating permutation importance (this may take a few minutes)...")
        
        try:
            perm_importance = permutation_importance(
                self.best_model, 
                self.X_val, 
                self.y_val,
                n_repeats=10,
                random_state=42,
                n_jobs=-1
            )
            
            perm_importance_mean = perm_importance.importances_mean
            perm_importance_std = perm_importance.importances_std
            
            print(f"✓ Permutation importance calculation completed")
        except Exception as e:
            print(f"✗ Permutation importance calculation failed: {e}")
            perm_importance_mean = np.zeros(n_features)
            perm_importance_std = np.zeros(n_features)
        
        print("\n[3/4] Analyzing feature-target correlation...")
        
        correlations = []
        for col in feature_names:
            try:
                corr = np.corrcoef(self.X_val[col], self.y_val)[0, 1]
                correlations.append(abs(corr))
            except:
                correlations.append(0)
        
        correlations = np.array(correlations)
        
        print("\n[4/4] Summarizing analysis results...")
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'perm_importance_mean': perm_importance_mean,
            'perm_importance_std': perm_importance_std,
            'correlation': correlations
        })
        
        for name, importance in tree_importance_dict.items():
            importance_df[f'{name}_importance'] = importance
        
        scores = []
        for col in importance_df.columns:
            if col != 'feature' and 'std' not in col:
                values = importance_df[col].values
                if values.max() > 0:
                    normalized = values / values.max()
                else:
                    normalized = values
                scores.append(normalized)
        
        if scores:
            importance_df['comprehensive_score'] = np.mean(scores, axis=0)
        else:
            importance_df['comprehensive_score'] = 0
        
        importance_df = importance_df.sort_values('comprehensive_score', ascending=False)
        
        print("\n" + "="*80)
        print("Feature Classification Results")
        print("="*80)
        
        useful_threshold = 0.01
        very_useful_threshold = 0.1
        
        very_useful_features = importance_df[importance_df['comprehensive_score'] > very_useful_threshold]
        useful_features = importance_df[
            (importance_df['comprehensive_score'] > useful_threshold) & 
            (importance_df['comprehensive_score'] <= very_useful_threshold)
        ]
        useless_features = importance_df[importance_df['comprehensive_score'] <= useful_threshold]
        
        print(f"\n🏆 Very useful features (score > {very_useful_threshold}): {len(very_useful_features)}")
        print("-" * 80)
        print(very_useful_features[['feature', 'comprehensive_score', 'perm_importance_mean', 'correlation']].head(20).to_string(index=False))
        
        print(f"\n✅ Useful features ({useful_threshold} < score <= {very_useful_threshold}): {len(useful_features)}")
        print("-" * 80)
        if len(useful_features) > 0:
            print(useful_features[['feature', 'comprehensive_score']].head(15).to_string(index=False))
        
        print(f"\n❌ Potentially useless features (score <= {useful_threshold}): {len(useless_features)}")
        print("-" * 80)
        if len(useless_features) > 0:
            print(useless_features[['feature', 'comprehensive_score']].to_string(index=False))
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = f'feature_importance_report_{timestamp}.csv'
        importance_df.to_csv(report_file, index=False, encoding='utf-8-sig')
        print(f"\n📄 Detailed report saved to: {report_file}")
        
        
        print("\n" + "="*80)
        print("Feature Optimization Suggestions")
        print("="*80)
        
        if len(useless_features) > 10:
            print(f"\n⚠️ Found {len(useless_features)} potentially useless features, suggestions:")
            print(f"   1. Removing these features may improve training speed")
            print(f"   2. Does not affect model accuracy (may even slightly improve)")
            print(f"   3. Reduces overfitting risk")
        elif len(useless_features) > 0:
            print(f"\n✓ Found {len(useless_features)}  low importance features, minimal impact")
        else:
            print(f"\n✓ All features have some contribution, feature engineering is well done!")
        
        print(f"\n💡 Top 10 most important features:")
        top10 = importance_df.head(10)
        for i, row in enumerate(top10.iterrows(), 1):
            idx, data = row
            print(f"   {i:2d}. {data['feature']:40s} (Score: {data['comprehensive_score']:.4f})")
        
        return importance_df


    def test_feature_removal_impact(self, top_n=20, n_repeats=2):
        print("\n" + "="*80)
        print("Feature Removal Impact Test")
        print("="*80)
        print("⚠️ Note: This will retrain models and may take a long time")
        print(f"   Will test top {top_n} features, each feature repeated {n_repeats} times")
        
        if self.X_val is None or self.y_val is None:
            print("Error: Please train model first")
            return None
        
        if hasattr(self, 'feature_importance') and self.feature_importance:
            if 'XGBoost' in self.feature_importance:
                importances = self.feature_importance['XGBoost']
                feature_names = self.train_combined.columns.tolist()
                top_indices = np.argsort(importances)[-top_n:][::-1]
            else:
                feature_names = self.train_combined.columns.tolist()
                top_indices = list(range(min(top_n, len(feature_names))))
        else:
            feature_names = self.train_combined.columns.tolist()
            top_indices = list(range(min(top_n, len(feature_names))))
        
        X_full = pd.concat([self.train_combined, self.X_val], axis=0)
        y_full = pd.concat([self.train_data['player_won'], self.y_val], axis=0)
        
        print("\n[Step 1/3] Calculating baseline accuracy (using all features)...")
        baseline_accuracies = []
        for repeat in range(n_repeats):
            X_train_base, X_val_base, y_train_base, y_val_base = train_test_split(
                X_full, y_full, test_size=0.2, random_state=42+repeat, stratify=y_full
            )
            
            base_model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                random_state=42,
                eval_metric='logloss',
                n_jobs=-1
            )
            base_model.fit(X_train_base, y_train_base)
            y_pred_base = base_model.predict(X_val_base)
            acc_base = accuracy_score(y_val_base, y_pred_base)
            baseline_accuracies.append(acc_base)
        
        baseline_acc = np.mean(baseline_accuracies)
        baseline_std = np.std(baseline_accuracies)
        print(f"✓ Baseline accuracy: {baseline_acc:.4f} ± {baseline_std:.4f}")
        
        print(f"\n[Step 2/3] Testing removal of top {len(top_indices)} features...")
        results = []
        
        for idx, feature_idx in enumerate(top_indices, 1):
            feature_name = feature_names[feature_idx]
            print(f"\n[{idx}/{len(top_indices)}] Testing removal of feature: {feature_name}")
            
            X_full_removed = X_full.drop(columns=[feature_name])
            
            accuracies_removed = []
            for repeat in range(n_repeats):
                X_train_rem, X_val_rem, y_train_rem, y_val_rem = train_test_split(
                    X_full_removed, y_full, test_size=0.2, random_state=42+repeat, stratify=y_full
                )
                
                model_removed = xgb.XGBClassifier(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.05,
                    random_state=42,
                    eval_metric='logloss',
                    n_jobs=-1
                )
                model_removed.fit(X_train_rem, y_train_rem)
                y_pred_rem = model_removed.predict(X_val_rem)
                acc_rem = accuracy_score(y_val_rem, y_pred_rem)
                accuracies_removed.append(acc_rem)
            
            acc_removed = np.mean(accuracies_removed)
            std_removed = np.std(accuracies_removed)
            impact = baseline_acc - acc_removed
            
            results.append({
                'feature': feature_name,
                'baseline_acc': baseline_acc,
                'removed_acc': acc_removed,
                'impact': impact,
                'impact_std': std_removed,
                'relative_impact': impact / baseline_acc * 100
            })
            
            status = "✅ Useful" if impact > 0.001 else "❌ Useless" if impact < -0.001 else "➖ Neutral"
            print(f"   Accuracy after removal: {acc_removed:.4f} ± {std_removed:.4f}")
            print(f"   Impact: {impact:+.4f} ({impact/baseline_acc*100:+.2f}%) {status}")
        
        print("\n[Step 3/3] Summarizing results...")
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('impact', ascending=False)
        
        useful_features = results_df[results_df['impact'] > 0.001]
        harmful_features = results_df[results_df['impact'] < -0.001]
        neutral_features = results_df[
            (results_df['impact'] >= -0.001) & (results_df['impact'] <= 0.001)
        ]
        
        print("\n" + "="*80)
        print("Feature Removal Impact Analysis Results")
        print("="*80)
        
        print(f"\n✅ Useful features (accuracy drops > 0.1% after removal): {len(useful_features)}")
        print("-" * 80)
        if len(useful_features) > 0:
            print(useful_features[['feature', 'impact', 'relative_impact']].to_string(index=False))
        
        print(f"\n❌ Potentially useless features (accuracy unchanged or improved after removal): {len(harmful_features) + len(neutral_features)}")
        print("-" * 80)
        if len(harmful_features) > 0:
            print("\nFeatures that improve accuracy after removal (potentially harmful):")
            print(harmful_features[['feature', 'impact', 'relative_impact']].to_string(index=False))
        if len(neutral_features) > 0:
            print("\nFeatures with minimal impact after removal (potentially redundant):")
            print(neutral_features[['feature', 'impact', 'relative_impact']].head(10).to_string(index=False))
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = f'feature_removal_test_{timestamp}.csv'
        results_df.to_csv(results_file, index=False, encoding='utf-8-sig')
        print(f"\n📄 Detailed results saved to: {results_file}")
        
        
        print("\n" + "="*80)
        print("Optimization Suggestions")
        print("="*80)
        
        if len(harmful_features) > 0:
            print(f"\n⚠️ Found {len(harmful_features)}  potentially harmful features:")
            print("   Removing these features may improve model performance!")
            print("   Suggestion: Consider removing these features from the feature set")
        
        if len(neutral_features) > 5:
            print(f"\n💡 Found {len(neutral_features)}  redundant features:")
            print("   Removing these features won't affect performance, but can:")
            print("   - Speed up training")
            print("   - Reduces overfitting risk")
            print("   - Simplify model")
        
        if len(useful_features) == len(results_df):
            print(f"\n✓ All tested features are useful, feature engineering is well done!")
        
        return results_df