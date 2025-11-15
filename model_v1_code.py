# FILE: model_v1_code.py
# Extracted from submission-1.ipynb

import pandas as pd
import numpy as np
import json
import warnings
import os
import time
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy.stats import linregress
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

warnings.filterwarnings('ignore')

# --- Generation 1 Type Chart ---
GEN_1_TYPE_CHART = {
    'normal': {'rock': 0.5, 'ghost': 0.0},
    'fire': {'fire': 0.5, 'water': 0.5, 'grass': 2.0, 'ice': 2.0, 'bug': 2.0, 'rock': 0.5, 'dragon': 0.5},
    'water': {'fire': 2.0, 'water': 0.5, 'grass': 0.5, 'ground': 2.0, 'rock': 2.0, 'dragon': 0.5},
    'electric': {'water': 2.0, 'electric': 0.5, 'grass': 0.5, 'ground': 0.0, 'flying': 2.0, 'dragon': 0.5},
    'grass': {'fire': 0.5, 'water': 2.0, 'grass': 0.5, 'poison': 0.5, 'ground': 2.0, 'flying': 0.5, 'bug': 0.5, 'rock': 2.0, 'dragon': 0.5},
    'ice': {'water': 0.5, 'grass': 2.0, 'ground': 2.0, 'flying': 2.0, 'dragon': 2.0},
    'fighting': {'normal': 2.0, 'ice': 2.0, 'rock': 2.0, 'poison': 0.5, 'flying': 0.5, 'psychic': 0.5, 'bug': 0.5, 'ghost': 0.0},
    'poison': {'grass': 2.0, 'poison': 0.5, 'ground': 0.5, 'rock': 0.5, 'ghost': 0.5},
    'ground': {'fire': 2.0, 'electric': 2.0, 'grass': 0.5, 'poison': 2.0, 'flying': 0.0, 'bug': 0.5, 'rock': 2.0},
    'flying': {'electric': 0.5, 'grass': 2.0, 'fighting': 2.0, 'bug': 2.0, 'rock': 0.5},
    'psychic': {'fighting': 2.0, 'poison': 2.0, 'psychic': 0.5, 'ghost': 0.0}, # Gen 1 Bug
    'bug': {'fire': 0.5, 'grass': 2.0, 'fighting': 0.5, 'poison': 0.5, 'flying': 0.5, 'psychic': 2.0, 'ghost': 0.5},
    'rock': {'fire': 2.0, 'ice': 2.0, 'fighting': 0.5, 'ground': 0.5, 'flying': 2.0, 'bug': 2.0},
    'ghost': {'normal': 0.0, 'psychic': 0.0, 'ghost': 2.0}, # Gen 1 Bug
    'dragon': {'dragon': 2.0}
}


class AdvancedPokemonPredictor:
    """
    Advanced Pokemon Battle Predictor (from submission-1)
    Implements an A/B test to compare a Linear vs. Ensemble strategy.
    """
    
    def __init__(self):
        self.train_data = None
        self.test_data = None
        self.train_features_df = None
        self.test_features_df = None
        self.models = {}
        self.selected_feature_names = []
        self.all_feature_names = []
        self.final_model_name = None
        self.cv_scores = {}
        
    def load_data(self):
        print("Loading data for Model V1...")
        
        base_path = '../input/fds-pokemon-battles-prediction-2025'
        if not os.path.exists(base_path):
            base_path = './' # Fallback for local/different env

        train_file_path = os.path.join(base_path, 'train.jsonl')
        test_file_path = os.path.join(base_path, 'test.jsonl')
        
        train_records = []
        try:
            with open(train_file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    train_records.append(json.loads(line.strip()))
            
            test_records = []
            with open(test_file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    test_records.append(json.loads(line.strip()))
            
            self.train_data = pd.DataFrame(train_records)
            self.test_data = pd.DataFrame(test_records)
            
            print(f"Training data: {len(self.train_data)} records")
            print(f"Test data: {len(self.test_data)} records")
            
        except FileNotFoundError:
            print(f"ERROR: Could not find the training file at '{train_file_path}'.")
            return False
            
        return True
    
    @staticmethod
    def backward_selection_to_n(X, y, base_pipe, target_n=15, cv=5, verbose=True):
        print(f"\nRunning Backward Selection to find top {target_n} features...")
        kfold = KFold(n_splits=cv, shuffle=True, random_state=42)
        selected = list(X.columns)
        history = []

        best_score = np.mean(cross_val_score(base_pipe, X[selected], y, cv=kfold, scoring='accuracy', n_jobs=1))
        print(f"Initial mean CV accuracy: {best_score:.4f} with {len(selected)} features")

        iteration = 0
        while len(selected) > target_n:
            iteration += 1
            scores_with_candidates = []
            for f in selected:
                candidate_features = [feat for feat in selected if feat != f]
                score = np.mean(cross_val_score(base_pipe, X[candidate_features], y, cv=kfold, scoring='accuracy', n_jobs=1))
                scores_with_candidates.append((f, score))

            worst_candidate, worst_candidate_score = max(scores_with_candidates, key=lambda x: x[1])
            delta = worst_candidate_score - best_score
            selected.remove(worst_candidate)
            best_score = worst_candidate_score

            if verbose:
                print(f"➖ Removed '{worst_candidate}' → mean CV={best_score:.4f} (Δ={delta:+.4f}) | Remaining: {len(selected)}")
            history.append((iteration, len(selected), best_score))

            if len(selected) <= target_n:
                if verbose:
                    print(f"\n🏁 Reached target of {target_n} features. Stopping.")
                break
        
        print(f"Selection complete. Final CV score: {best_score:.4f}")
        return selected, pd.DataFrame(history, columns=["iteration", "n_features", "cv_accuracy"])

    def extract_advanced_features(self):
        print("Extracting advanced features (A/B Test Strategy)...")
        
        def get_pokemon_stats(pokemon_dict):
            if not pokemon_dict or not isinstance(pokemon_dict, dict): return [0] * 6
            stats = [
                pokemon_dict.get('base_hp', 0), pokemon_dict.get('base_atk', 0),
                pokemon_dict.get('base_def', 0), pokemon_dict.get('base_spa', 0),
                pokemon_dict.get('base_spd', 0), pokemon_dict.get('base_spe', 0)
            ]
            return [s if s is not None else 0 for s in stats]

        def get_team_stats(team_list):
            if not team_list or not isinstance(team_list, list): return [0] * 6
            total_stats = np.array([0.0] * 6)
            for pokemon in team_list:
                total_stats += np.array(get_pokemon_stats(pokemon))
            return total_stats

        def calculate_type_effectiveness(p1_types, p2_types):
            p1_types = [str(t).lower() for t in p1_types if t != 'notype']
            p2_types = [str(t).lower() for t in p2_types if t != 'notype']
            if not p1_types or not p2_types: return 1.0
            scores = []
            for p1_type in p1_types:
                if p1_type in GEN_1_TYPE_CHART:
                    type_score = 1.0
                    for p2_type in p2_types:
                        type_score *= GEN_1_TYPE_CHART[p1_type].get(p2_type, 1.0)
                    scores.append(type_score)
            return np.mean(scores) if scores else 1.0
        
        def get_battle_duration(timeline):
            try:
                return len([t for t in timeline if t['p1_pokemon_state']['hp_pct'] > 0 and t['p2_pokemon_state']['hp_pct'] > 0])
            except: return 0

        def get_final_hp_states(timeline):
            p1_hp_final, p2_hp_final = {}, {}
            for t in timeline:
                if t.get('p1_pokemon_state'): p1_hp_final[t['p1_pokemon_state']['name']] = t['p1_pokemon_state']['hp_pct']
                if t.get('p2_pokemon_state'): p2_hp_final[t['p2_pokemon_state']['name']] = t['p2_pokemon_state']['hp_pct']
            return p1_hp_final, p2_hp_final

        def analyze_timeline_aggregates(timeline):
            if not timeline: return {}
            features = {}
            p1_hp, p2_hp = [], []
            p1_status, p2_status = [], []
            p1_move_power, p2_move_power = [], []
            
            for t in timeline:
                if t.get('p1_pokemon_state'): p1_hp.append(t['p1_pokemon_state']['hp_pct'])
                if t.get('p2_pokemon_state'): p2_hp.append(t['p2_pokemon_state']['hp_pct'])
                p1_status.append(t.get('p1_pokemon_state', {}).get('status', 'nostatus'))
                p2_status.append(t.get('p2_pokemon_state', {}).get('status', 'nostatus'))
                p1_move = t.get('p1_move_details')
                p2_move = t.get('p2_move_details')
                if p1_move: p1_move_power.append(p1_move.get('base_power', 0) or 0)
                if p2_move: p2_move_power.append(p2_move.get('base_power', 0) or 0)

            min_len = min(len(p1_hp), len(p2_hp))
            if min_len < 2: return {}
                
            p1_hp, p2_hp = p1_hp[:min_len], p2_hp[:min_len]
            p1_status, p2_status = p1_status[:min_len], p2_status[:min_len]
            hp_delta = np.array(p1_hp) - np.array(p2_hp)

            features['p1_hp_volatility'] = np.std(p1_hp)
            features['p2_hp_volatility'] = np.std(p2_hp)
            features['hp_delta_trend'] = np.polyfit(range(len(hp_delta)), hp_delta, 1)[0]
            features['hp_delta_std'] = np.std(hp_delta)
            features['p1_hp_advantage_mean'] = np.mean(hp_delta > 0)
            negative_status = {'brn', 'par', 'psn', 'tox', 'frz', 'slp'}
            p1_neg_status_mean = np.mean([s in negative_status for s in p1_status])
            p2_neg_status_mean = np.mean([s in negative_status for s in p2_status])
            features['p1_bad_status_advantage'] = p2_neg_status_mean - p1_neg_status_mean
            p1_status_change = np.sum(np.array(p1_status[1:]) != np.array(p1_status[:-1]))
            p2_status_change = np.sum(np.array(p2_status[1:]) != np.array(p2_status[:-1]))
            features['status_change_diff'] = p1_status_change - p2_status_change
            features['p1_avg_power'] = np.mean(p1_move_power) if p1_move_power else 0
            features['p2_avg_power'] = np.mean(p2_move_power) if p2_move_power else 0
            return features

        def process_data(data_rows):
            feature_list = []
            stat_names = ['hp', 'atk', 'def', 'spa', 'spd', 'spe']
            for _, row in data_rows.iterrows():
                features = {}
                p1_team = row.get('p1_team_details', [])
                p2_lead = row.get('p2_lead_details', {})
                timeline = row.get('battle_timeline', [])

                p1_team_stats = get_team_stats(p1_team)
                p2_lead_stats = get_pokemon_stats(p2_lead)
                for i, stat in enumerate(stat_names):
                    features[f'team_vs_lead_{stat}_diff'] = p1_team_stats[i] - p2_lead_stats[i]

                p1_lead_stats = get_pokemon_stats(p1_team[0]) if p1_team else [0]*6
                for i, stat in enumerate(stat_names):
                    features[f'lead_vs_lead_{stat}_diff'] = p1_lead_stats[i] - p2_lead_stats[i]

                p1_lead_types = p1_team[0].get('types', []) if p1_team else []
                p2_lead_types = p2_lead.get('types', [])
                features['lead_type_advantage_gen1'] = calculate_type_effectiveness(p1_lead_types, p2_lead_types)
                features['lead_type_disadvantage_gen1'] = calculate_type_effectiveness(p2_lead_types, p1_lead_types)

                momentum_features = analyze_timeline_aggregates(timeline)
                features.update(momentum_features)

                p1_hp_final, p2_hp_final = get_final_hp_states(timeline)
                p1_n_pokemon_used = len(p1_hp_final.keys())
                p2_n_pokemon_used = len(p2_hp_final.keys())
                features['diff_n_pokemon_used'] = p1_n_pokemon_used - p2_n_pokemon_used
                features['p1_fainted_count'] = np.sum([1 for hp in p1_hp_final.values() if hp == 0])
                features['p2_fainted_count'] = np.sum([1 for hp in p2_hp_final.values() if hp == 0])
                features['diff_fainted_count'] = features['p1_fainted_count'] - features['p2_fainted_count']
                
                p1_total_final_hp = np.sum(list(p1_hp_final.values())) + (6 - p1_n_pokemon_used)
                p2_total_final_hp = np.sum(list(p2_hp_final.values())) + (6 - p2_n_pokemon_used)
                features['final_hp_advantage'] = p1_total_final_hp - p2_total_final_hp

                features['battle_duration'] = get_battle_duration(timeline)
                features['hp_loss_rate'] = (features['final_hp_advantage'] / features['battle_duration'] if features['battle_duration'] > 0 else 0.0)
                features['total_turns'] = len(timeline)

                p1_lead_atk = p1_lead_stats[1]
                p1_lead_spa = p1_lead_stats[3]
                p2_lead_def = p2_lead_stats[2]
                p2_lead_spd = p2_lead_stats[4]
                features['p1_phys_pressure'] = p1_lead_atk / (p2_lead_def + 1)
                features['p1_spec_pressure'] = p1_lead_spa / (p2_lead_spd + 1)
                features['status_chaos'] = features.get('hp_delta_std', 0) * (features.get('p1_bad_status_advantage', 0) + 1)
                features['hp_gain_per_turn'] = features.get('hp_delta_trend', 0) * features.get('battle_duration', 0)

                features['battle_id'] = row.get('battle_id')
                if 'player_won' in row:
                    features['player_won'] = int(row['player_won'])
                
                feature_list.append(features)
            
            return pd.DataFrame(feature_list).fillna(0)

        print("Processing training data...")
        self.train_features_df = process_data(self.train_data)
        print("Processing test data...")
        self.test_features_df = process_data(self.test_data)
        
        train_cols = set(self.train_features_df.columns)
        test_cols = set(self.test_features_df.columns)
        
        missing_in_test = list(train_cols - test_cols - {'player_won'})
        for col in missing_in_test: self.test_features_df[col] = 0
        missing_in_train = list(test_cols - train_cols)
        for col in missing_in_train: self.train_features_df[col] = 0
        
        self.all_feature_names = [c for c in self.train_features_df.columns if c not in ['battle_id', 'player_won']]
        self.test_features_df = self.test_features_df[['battle_id'] + self.all_feature_names]
        
        print(f"Advanced feature extraction complete: {len(self.all_feature_names)} features created.")
        return True

    def run_ab_test_and_train(self, linear_target_n=17, linear_cv=7, ensemble_cv=5):
        print(f"\n--- A/B Test: Linear vs. Ensemble ---")
        
        if self.train_features_df is None:
            print("ERROR: Features not extracted. Run extract_advanced_features() first.")
            return False
            
        X_all = self.train_features_df[self.all_feature_names]
        y = self.train_data['player_won']
        
        kfold_linear = KFold(n_splits=linear_cv, shuffle=True, random_state=42)
        kfold_ensemble = KFold(n_splits=ensemble_cv, shuffle=True, random_state=42)
        
        self.cv_scores = {}

        # --- STRATEGY A: Linear Model on Selected Features ---
        print("\n--- Test A: Training Linear (LogReg) Strategy ---")
        try:
            selection_pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("logreg", LogisticRegression(max_iter=2000, random_state=42))
            ])
            
            selected, _ = self.backward_selection_to_n(
                X_all, y, base_pipe=selection_pipe,
                target_n=linear_target_n, cv=linear_cv, verbose=True
            )
            self.selected_feature_names = selected
            X_selected = self.train_features_df[self.selected_feature_names]
            
            linear_model_pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("logreg", LogisticRegression(max_iter=2000, random_state=42))
            ])
            cv_scores_linear = cross_val_score(linear_model_pipe, X_selected, y, cv=kfold_linear, scoring='accuracy', n_jobs=1)
            
            self.cv_scores['LogisticRegression'] = cv_scores_linear.mean()
            print(f"  - FINAL Linear CV Score (on {len(selected)} features): {self.cv_scores['LogisticRegression']:.4f} (+/- {cv_scores_linear.std() * 2:.4f})")
            
            linear_model_pipe.fit(X_selected, y)
            self.models['Linear_Model'] = linear_model_pipe
            
        except Exception as e:
            print(f"ERROR during Linear Strategy: {e}")
            self.cv_scores['LogisticRegression'] = 0.0

        # --- STRATEGY B: Ensemble Model on All Features ---
        print("\n--- Test B: Training Full Ensemble (RF+XGB+LGBM) Strategy ---")
        try:
            rf_pipe = Pipeline([('model', RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42, n_jobs=1))])
            xgb_pipe = Pipeline([('model', xgb.XGBClassifier(n_estimators=400, learning_rate=0.1, max_depth=5, random_state=42, eval_metric='logloss', use_label_encoder=False, n_jobs=1))])
            lgbm_pipe = Pipeline([('model', lgb.LGBMClassifier(n_estimators=400, learning_rate=0.1, max_depth=5, random_state=42, verbose=-1, n_jobs=1))])
            
            models_to_test = {'RandomForest': rf_pipe, 'XGBoost': xgb_pipe, 'LightGBM': lgbm_pipe}

            print(f"  Testing individual ensemble models on all {len(self.all_feature_names)} features...")
            for name, pipe in models_to_test.items():
                start_time_model = time.time()
                cv_scores = cross_val_score(pipe, X_all, y, cv=kfold_ensemble, scoring='accuracy', n_jobs=1)
                self.cv_scores[name] = cv_scores.mean()
                print(f"    - {name} CV Score: {self.cv_scores[name]:.4f} (+/- {cv_scores.std() * 2:.4f}) [Time: {time.time() - start_time_model:.1f}s]")

            print("  Testing full (RF+XGB+LGBM) Ensemble...")
            ensemble_model_pipe = VotingClassifier(estimators=[('rf', rf_pipe), ('xgb', xgb_pipe), ('lgbm', lgbm_pipe)], voting='soft')
            
            start_time_ensemble = time.time()
            cv_scores_ensemble = cross_val_score(ensemble_model_pipe, X_all, y, cv=kfold_ensemble, scoring='accuracy', n_jobs=1)
            self.cv_scores['Ensemble'] = cv_scores_ensemble.mean()
            print(f"    - FINAL Ensemble CV Score: {self.cv_scores['Ensemble']:.4f} (+/- {cv_scores_ensemble.std() * 2:.4f}) [Time: {time.time() - start_time_ensemble:.1f}s]")

            ensemble_model_pipe.fit(X_all, y)
            self.models['Ensemble_Model'] = ensemble_model_pipe
        
        except Exception as e:
            print(f"ERROR during Ensemble Strategy: {e}")
            self.cv_scores['Ensemble'] = 0.0

        # --- A/B Test Conclusion ---
        print("\n--- A/B Test Results (Full Comparison) ---")
        best_model_name = ""
        best_score = 0.0
        for name, score in self.cv_scores.items():
            print(f"  - {name}: {score:.4f}")
            if score > best_score:
                best_score = score
                best_model_name = name
        
        if best_model_name == 'LogisticRegression':
            self.final_model_name = 'Linear_Model'
        else:
            self.final_model_name = 'Ensemble_Model'
            
        print(f"\nWINNER (based on CV): {best_model_name} (Using {self.final_model_name} for submission)")
        return True

    def analyze_feature_importance(self):
        print("\nAnalyzing winning model's feature importance...")
        
        if self.final_model_name is None:
            print("No model was selected as the winner.")
            return

        model_pipe = self.models[self.final_model_name]

        if self.final_model_name == 'Linear_Model':
            print(f"(Analyzing {self.final_model_name})")
            try:
                model = model_pipe.named_steps['logreg']
                coeffs = model.coef_[0]
                coeff_df = pd.DataFrame({
                    'feature': self.selected_feature_names,
                    'coefficient': coeffs,
                    'abs_coefficient': np.abs(coeffs)
                }).sort_values('abs_coefficient', ascending=False)
                print(f"Top {len(self.selected_feature_names)} most impactful features (Coefficients):")
                print(coeff_df)
            except Exception as e:
                print(f"Could not analyze coefficients: {e}")
        
        else:
            print(f"(Analyzing {self.final_model_name})")
            for name, pipe in model_pipe.named_steps['estimators']:
                try:
                    model = pipe.named_steps['model']
                    if hasattr(model, 'feature_importances_'):
                        print(f"\n--- Top 10 Features for: {name.upper()} (in Ensemble) ---")
                        importance_df = pd.DataFrame({
                            'feature': self.all_feature_names,
                            'importance': model.feature_importances_
                        }).sort_values('importance', ascending=False)
                        print(importance_df.head(10))
                except Exception as e:
                    print(f"Could not analyze importance for {name}: {e}")