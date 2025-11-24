import os
import random

import joblib
import numpy as np
import pandas as pd
import shap
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance
from tqdm import trange

random.seed(42)
np.random.seed(42)

import warnings

def save_feature_importance(df, path, filename):
    if not os.path.exists(path):
        os.makedirs(path)
    df.to_csv(os.path.join(path, filename), index=False)

def plot_perm_import(perm_importance_df, paths, file_name):
    plt.figure(figsize=(8, 6))
    plt.title(f'Permutation Feature Importance ({file_name})')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    # use seaborn to create a bar plot
    sns.barplot(x=perm_importance_df['Importance'], y=perm_importance_df['Feature'], orient='h')
    plt.tight_layout()
    plt.savefig(os.path.join(paths, f'{file_name}_permutation_feature_importance.png'), dpi=300, bbox_inches='tight')

def permutation_features(model, df, target_col, features, paths,file_name):
    X = df[features]
    y = df[target_col]
    perm_importance = permutation_importance(model, X, y, n_repeats=10, random_state=42, scoring='accuracy')
    perm_importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': perm_importance.importances_mean
    }).sort_values(by='Importance', ascending=False)
    save_feature_importance(perm_importance_df, paths, file_name+'_permutation_feature_importance.csv')
    plot_perm_import(perm_importance_df, paths, file_name)

def plot_shap_importance(shap_values_dfs, house_keeping_cols, paths, X, file_name):
    if isinstance(shap_values_dfs, list): 
        for i, shap_df in enumerate(shap_values_dfs):
            shap_df = shap_df.drop(columns=house_keeping_cols)
            # print(shap_df)
            plot_single_shap_importance(shap_df, paths, file_name, f"_class_{i}")

    else: 
        plot_single_shap_importance(shap_values_dfs, paths, file_name)

def plot_single_shap_importance(shap_df, paths, file_name, suffix=""):
    
    shap_importance = shap_df.abs().median().sort_values(ascending=False)
    plt.figure(figsize=(8, 6))
    plt.title(f'SHAP Feature Importance ({file_name}{suffix})')
    plt.xlabel('Absolute Median SHAP Value')
    plt.ylabel('Feature')
    sns.barplot(x=shap_importance.values, y=shap_importance.index, orient='h')
    plt.tight_layout()
    plt.savefig(os.path.join(paths,f'{file_name}_shap_feature_importance{suffix}.png'), dpi=300, bbox_inches='tight')

class GeneralWrapper:
    """A general wrapper for sklearn depedent models."""
    def __init__(self, model, feature_names, model_type):
        self.model = model
        self.feature_names = feature_names
        self.model_type = model_type
    
    def predictions(self, X):
        if isinstance(X, pd.Series):
            X = X.values.reshape(1, -1)
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names)
        predictions = self.model.predict_proba(X)
        return predictions


# unify to list-of-arrays [ (n_samples, n_features) per class ]
def _standardize_shap(raw, X, n_features):
    if hasattr(raw, "values"):
        arr = np.asarray(raw.values)
    else:
        arr = np.asarray(raw)
    if isinstance(raw, list):
        return [np.asarray(r) for r in raw]
    if arr.ndim == 2:
        return [arr]
    if arr.ndim == 3:
        if arr.shape[0] == X.shape[0] and arr.shape[1] == n_features:
            return [arr[:, :, i] for i in range(arr.shape[2])]
        if arr.shape[1] == X.shape[0] and arr.shape[2] == n_features:
            return [arr[i] for i in range(arr.shape[0])]
        if arr.shape[0] == X.shape[0] and arr.shape[2] == n_features:
            return [arr[:, i, :] for i in range(arr.shape[1])]
        return [arr[i] for i in range(arr.shape[0])]
    return [arr]
       
def calculate_shap_values(model, features, class_names, background_data, data, paths, model_type, do_boot):
    X = data[features].reset_index(drop=True)
    house_keeping_cols = data.drop(columns=features).columns.to_list()
    # print(house_keeping_cols)
    
    if do_boot:
        print("Bootstrapping SHAP values enabled.")

    else:
        med_full = background_data[features].reset_index(drop=True)
        
        print(f"Background rows used: {len(med_full)}")
    
        print('Using Kernel Explainer')
        wrapper = GeneralWrapper(model, features, model_type)
        explainer = shap.KernelExplainer(wrapper.predictions, med_full)
        shap_raw = explainer.shap_values(X, nsamples=5000)

        shap_values_list = _standardize_shap(shap_raw, X, len(features))

        try:
            joblib.dump(explainer, os.path.join(paths, f'{model_type}_shap_explainer_debug.joblib'))
        except Exception as e:
            print(f"Could not dump explainer: {e}")

        n_classes = len(shap_values_list)
        print(f'Interpreting SHAP as a {n_classes} class problem')

        # ---------- 1) Plotting + CSVs ----------
        if n_classes > 1:
            plt.figure(figsize=(6, max(6, len(features)*0.25 * n_classes)))
            bar_width = 0.8 / n_classes
            index = np.arange(len(features))
            mean_abs_per_class = [np.mean(np.abs(sv), axis=0) for sv in shap_values_list]
            combined_score = np.sum(mean_abs_per_class, axis=0)
            sorted_idx = np.argsort(combined_score)
            for i in range(n_classes):
                vals = mean_abs_per_class[i][sorted_idx]
                plt.barh(index + (i * bar_width), vals, bar_width, align='center', label=f'Class {i}')
            plt.yticks(index + (bar_width*(n_classes-1)/2), [features[j] for j in sorted_idx])
            plt.ylabel('Features'); plt.xlabel('Mean Absolute SHAP Value')
            plt.title('SHAP Summary Plot (mean abs) - all classes'); plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(paths, f'{model_type}_shap_summary_all_classes.png'), dpi=300)
            plt.close()

            shap_values_dfs = []
            for i in range(n_classes):
                arr = shap_values_list[i]
                try:
                    plt.figure(figsize=(8, 6))
                    shap.summary_plot(arr, X, feature_names=features, show=False)
                    plt.title(f'SHAP Summary Plot For Class ({i})')
                    plt.savefig(os.path.join(paths, f'{model_type}_shap_summary_class_{i}.png'), dpi=300, bbox_inches='tight')
                    plt.close()
                except Exception as e:
                    print(f"Could not create SHAP summary plot for class {i}: {e}")
                df_i = pd.DataFrame(arr, columns=features)
                df_i[house_keeping_cols] = data[house_keeping_cols]
                shap_values_dfs.append(df_i)
                df_i.to_csv(os.path.join(paths, f'shap_values_class_{i}.csv'), index=False)
            
        else:
            arr = shap_values_list[0]
            shap_values_dfs = [pd.DataFrame(arr, columns=features)]
            shap_values_dfs[0].to_csv(os.path.join(paths, 'shap_values.csv'), index=False)
            try:
                plt.figure(figsize=(8, 6))
                shap.summary_plot(arr, X, feature_names=features, show=False)
                plt.savefig(os.path.join(paths, f'{model_type}_shap_summary.png'), dpi=300, bbox_inches='tight')
                plt.close()
            except Exception as e:
                print(f"Could not create SHAP summary plot: {e}")
        
        # print(shap_values_dfs)
        plot_shap_importance(shap_values_dfs, house_keeping_cols, paths, X, model_type)
        return shap_values_dfs, explainer
    
    if do_boot:
        # ---------- 2) Bootstrap SHAP ----------
        RNG_SEED = 10
        N_BOOT = 200
        med_full = background_data[features].reset_index(drop=True)
        ksamples = min(len(med_full), 10)

        print(f"Bootstrapping SHAP for {N_BOOT} boots")
        print(f"Background rows used: {len(med_full)}")
        print(f"Using {ksamples} kmeans samples for Kernel SHAP background")
        
        wrap = GeneralWrapper(model, features, model_type)
        boot_seeds = np.random.default_rng(RNG_SEED).integers(0, 2**32 - 1, size=N_BOOT, dtype=np.uint32) #not used for kmeans
        
        def _one_boot(b):        
            np.random.seed(int(boot_seeds[b]))
            bg = shap.kmeans(med_full, ksamples)
            # bg = med_full

            ke = shap.KernelExplainer(wrap.predictions, bg)
            raw = ke.shap_values(X, nsamples="auto", silent=True)
            return _standardize_shap(raw, X, len(features))

        boot_accum = None  # list-of-arrays per class: (n_boot, n_samples, n_features)
        n_classes = None
        for b in trange(N_BOOT, desc='Bootstrapping SHAP', unit='boot'):
            s_list = _one_boot(b)
            n_classes = len(s_list)
            if boot_accum is None:
                boot_accum = [np.zeros((N_BOOT, s.shape[0], s.shape[1]), dtype=np.float64) for s in s_list]
            for k, s in enumerate(s_list):
                boot_accum[k][b] = s

        boot_mean_list = [blk.mean(axis=0) for blk in boot_accum]
        boot_lo_list = [np.percentile(blk,  2.5, axis=0) for blk in boot_accum]
        boot_hi_list = [np.percentile(blk, 97.5, axis=0) for blk in boot_accum]
        boot_se_list = [np.std(blk, axis=0, ddof=1) for blk in boot_accum]

        boot_mean_dfs, boot_lo_dfs, boot_hi_dfs, boot_se_dfs = [], [], [], []
 
        for i in range(n_classes):
            df_mean = pd.DataFrame(boot_mean_list[i], columns=features)
            df_mean[house_keeping_cols] = data[house_keeping_cols]

            df_lo = pd.DataFrame(boot_lo_list[i], columns=features)
            df_lo[house_keeping_cols] = data[house_keeping_cols]

            df_hi = pd.DataFrame(boot_hi_list[i], columns=features)
            df_hi[house_keeping_cols] = data[house_keeping_cols]

            df_se = pd.DataFrame(boot_se_list[i], columns=features)
            df_se[house_keeping_cols] = data[house_keeping_cols]

            boot_mean_dfs.append(df_mean)
            boot_lo_dfs.append(df_lo)
            boot_hi_dfs.append(df_hi)
            boot_se_dfs.append(df_se)

            df_mean.to_csv(os.path.join(paths, f'shap_boot_mean_class_{i}.csv'), index=False)
            df_lo.to_csv(  os.path.join(paths, f'shap_boot_ci_low_class_{i}.csv'), index=False)
            df_hi.to_csv(  os.path.join(paths, f'shap_boot_ci_high_class_{i}.csv'), index=False)
            df_se.to_csv(  os.path.join(paths, f'shap_boot_ci_halfwidth_class_{i}.csv'), index=False)

        boot_meta = {
            'n_boot': N_BOOT,
            'seed': boot_seeds,
            'explainer_type': model_type,
        }

        extras = {
            'boot_mean_dfs': boot_mean_dfs,
            'boot_ci_low_dfs': boot_lo_dfs,
            'boot_ci_high_dfs': boot_hi_dfs,
            'boot_se_dfs': boot_se_dfs,
            'boot_meta': boot_meta
        }

        return extras['boot_mean_dfs'], None

