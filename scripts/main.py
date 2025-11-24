import os
import glob
import argparse
import joblib
import numpy as np
import pandas as pd

from datetime import datetime

from autogluon.tabular import TabularPredictor
from sklearn.metrics import (
    balanced_accuracy_score,
    accuracy_score,
    f1_score,
    matthews_corrcoef,
)

from models import run_models
from feature_importance import calculate_shap_values

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")

def calculate_feature_stats(df_shap, features, dx_column='dx'):
    top_feats_by_dx_stats = {}
    for dx in df_shap[dx_column].unique():
        dx_group = df_shap[df_shap[dx_column] == dx]
        mean_abs_shap = dx_group[features].abs().mean().sort_values(ascending=False)[:]
        stats = []
        for feat in mean_abs_shap.index:
            mean_val = dx_group[feat].mean() * 100
            std_val = dx_group[feat].std()
            var_val = dx_group[feat].var()

            stats.append(f"{feat} (mean={mean_val:.3f}, std={std_val:.3f}, var={var_val:.3f})")
        top_feats_by_dx_stats[dx] = stats
    return top_feats_by_dx_stats

def calculate_het(df_shap_baseline, features, target_clf, df_raw):
    """
    Calculates heterogeneity (HET) score for each sample based on SHAP feature stats.

    Args:
        df_shap_baseline (pd.DataFrame): SHAP values dataframe with 'dx' column.
        features (list): List of feature column names.
        df_raw (pd.DataFrame): Input data (centered/scaled by CU) with 'dx' column.

    Returns:
        pd.DataFrame: Copy of df_raw with features multiplied by groupwise SHAP means
                      (Eq. 2) and an added 'HET' column (Eq. 3).
    """
    df = df_raw.copy()

    # Calculate groupwise SHAP stats 
    top_feats_stats = calculate_feature_stats(
        df_shap_baseline,
        features,
        dx_column=target_clf,
    )

    # Apply SHAP mean weights per group (Eq. 2)
    for dx, stats_list in top_feats_stats.items():
        feat_stats = {}
        for stat in stats_list:
            name = stat.split(" (")[0]
            mean_val = float(stat.split("mean=")[1].split(",")[0])
            std_val = float(stat.split("std=")[1].split(",")[0])
            var_val = float(stat.split("var=")[1].split(")")[0])
            feat_stats[name] = (mean_val, std_val, var_val)

        mask = df[target_clf] == dx
        for feat, (mean_val, _, _) in feat_stats.items():
            if feat in df.columns:
                df.loc[mask, feat] = df.loc[mask, feat] * mean_val

    # HET: average of weighted features across all regions (Eq. 3)
    df["HET"] = df[features].mean(axis=1)

    df = df.dropna()

    return df


def main(_paths, features, filenames, args):
    todays_run = args.todays_run
    scaledata = args.scaledata
    target_clf = args.target
    do_boot = args.do_boot

    if not todays_run:
        todays_run = datetime.today().strftime("%Y%m%d")

    run_dir = os.path.join(_paths["run_dir"], todays_run, args.model)
    os.makedirs(run_dir, exist_ok=True)

    if args.fresh_run:
        df_raw = pd.read_csv(os.path.join(_paths["data_path"], filenames["main_file_name"]))

        df_raw[target_clf] = df_raw[target_clf].map({'CON':'CON','MSA-C':'MSA','MSA-P':'MSA','PD':'PD'})

        if scaledata:
            print("Data z-scored by the baseline controls mean and std.")

            df_raw_con = df_raw[df_raw[target_clf] == "CON"]
            df_raw_con_mean = df_raw_con[features].mean()
            df_raw_con_std = df_raw_con[features].std()

            df_raw[features] = (df_raw[features] - df_raw_con_mean) / df_raw_con_std
        else:
            print("Data not scaled, used as raw.")

        df_raw_msa_pd = df_raw[df_raw[target_clf] != "CON"]
        df_raw_msa_pd[target_clf] = df_raw[target_clf].map({'MSA':0,'PD':1})

        df_raw_msa_pd.to_csv(os.path.join(run_dir, "training_data.csv"), index=False)

        # run_models(df_raw_msa_pd, target_clf, features, "ID", run_dir)

        print("----", f"Model Training DONE for {args.model}", "----")

    else:
        training_data = pd.read_csv(os.path.join(run_dir, "training_data.csv"))
        baseline_data = training_data[training_data["visit"] == 1]
        followup_data = training_data[training_data["visit"] != 1]

        joblib_model_files = glob.glob(os.path.join(run_dir, "*_best_classifier.joblib"))

        if joblib_model_files:
            best_classifier_path = joblib_model_files[0]
            print("->" * 5, "Loading best model from joblib file:", best_classifier_path)
            best_model_classifier = joblib.load(best_classifier_path)
            best_model_type = "custom"
            shap_trainer = "tree"
            model_classifier = None
        else:
            print("->" * 5, "Loading best model from AutoGluon predictor.")
            model_classifier = TabularPredictor.load(
                os.path.join(run_dir, "overall_best_AutoGluon_best_classifier"),
                require_version_match=False,
                require_py_version_match=False,
            )

            try:
                best_model_name = model_classifier.get_model_best()
            except AttributeError:
                lb = model_classifier.leaderboard(silent=True)
                if not lb.empty and "model" in lb.columns:
                    print("Leaderboard: \n", lb)
                    best_model_name = lb.iloc[0]["model"]
                else:
                    raise RuntimeError("Could not determine best AutoGluon model from leaderboard.")

            print(f"Top AutoGluon model: {best_model_name}")

            # Use the predictor directly so .predict and .predict_proba work as expected
            best_model_classifier = model_classifier
            best_model_type = "ag"
            shap_trainer = "kernel"

        files_to_analyse = {"baseline": baseline_data, "followup": followup_data}
        for key in files_to_analyse:
            os.makedirs(os.path.join(run_dir, key), exist_ok=True)

        performance_df = pd.DataFrame(
            columns=[
                "Model_Type",
                "Dataset",
                "Balanced_Accuracy",
                "Accuracy",
                "F1_Score",
                "Matthews_Correlation_Coefficient",
            ]
        )

        background_data_classification = training_data[features].copy()

        for key, data in files_to_analyse.items():
            data = data.copy()
            data.to_csv(os.path.join(run_dir, key, f"Data_{key}.csv"), index=False)

            print("->" * 5, f"Explaining for {key}")

            X_data = data[features].copy()

            if best_model_type == "ag":
                y_pred_classifier = best_model_classifier.predict(X_data)
                y_pred_proba_classifier = best_model_classifier.predict_proba(X_data)
                class_labels = best_model_classifier.class_labels
            else:
                y_pred_classifier = best_model_classifier.predict(X_data)
                y_pred_proba_classifier = best_model_classifier.predict_proba(X_data)
                class_labels = best_model_classifier.classes_

            data["Target_Predicted"] = np.asarray(y_pred_classifier)
            y_pred_proba_classifier = np.asarray(y_pred_proba_classifier)

            for i, class_label in enumerate(class_labels):
                data[f"Target_Probability_{class_label}"] = y_pred_proba_classifier[:, i]

            data.to_csv(os.path.join(run_dir, key, "classification_predictions.csv"), index=False)

            y_true_classifier = data[target_clf]
            balanced_accuracy = balanced_accuracy_score(y_true_classifier, y_pred_classifier)
            accuracy = accuracy_score(y_true_classifier, y_pred_classifier)
            f1 = f1_score(y_true_classifier, y_pred_classifier, average="weighted")
            matthews_corr = matthews_corrcoef(y_true_classifier, y_pred_classifier)

            classification_metrics = pd.DataFrame(
                {
                    "Model_Type": ["Classification"],
                    "Dataset": [key],
                    "Balanced_Accuracy": [balanced_accuracy],
                    "Accuracy": [accuracy],
                    "F1_Score": [f1],
                    "Matthews_Correlation_Coefficient": [matthews_corr],
                }
            )

            performance_df = pd.concat([performance_df, classification_metrics], ignore_index=True)

            class_names = list(class_labels)
            _shap_values_df_classification, _explainer_classifier = calculate_shap_values(
                best_model_classifier,
                features,
                class_names,
                background_data_classification,
                data,
                os.path.join(run_dir, key),
                shap_trainer,
                do_boot=do_boot,
            )

        performance_df.to_csv(
            os.path.join(run_dir, "combined_model_performance_metrics.csv"),
            index=False,
        )

        # lets compute HET for the whole data
        # read the baseline shap back from its directory
        baseline_SHAP_boot = pd.read_csv(os.path.join(run_dir, 'baseline', f"shap_boot_mean_class_0.csv"))
        training_data_het = calculate_het(baseline_SHAP_boot, features, target_clf, training_data)
        training_data_het.to_csv(os.path.join(run_dir, "training_data_het.csv"), index=False)

        print("----", "DONE", "----")

if __name__ == "__main__":
    todays_run = datetime.today().strftime('%Y%m%d')

    _paths = {
        "data_path": "GitHubUpload",   # TODO: set to your data folder
        "run_dir": "output",   # TODO: set to your preferred output folder
    }

    filenames = {
        "main_file_name": "example.csv",  # TODO: set to your main CSV file
    }

    # TODO: replace with actual feature column names from your CSV
    # features = ["feature_1", "feature_2", "feature_3"]
    # if your dataframe is structures as 'ID', 'visit', 'dx, 'feature_1', 'feature_2',...,'feature_n', then the below line will work for you, if not format acordingly
    features = (
        pd.read_csv(os.path.join(_paths["data_path"], filenames["main_file_name"]))
        .drop(columns=["ID", "visit", "dx"])
        .columns.tolist()
    )
    print("features: ", features)

    parser = argparse.ArgumentParser(description="Run model training with options for scaling and balancing classes.")
    parser.add_argument("--todays_run", type=str, required=False, help="Run ID (default: today's date as YYYYMMDD).")
    parser.add_argument("--scaledata", type=str2bool, default=True, required=False, help="Scale with CUs data before model training (True/False).")
    parser.add_argument("--fresh_run", type=str2bool, default=True, help="Start a fresh model training (True/False).")
    parser.add_argument("--target", type=str, default="dx", help="Target column for classification (e.g., 'dx').")
    parser.add_argument("--model", type=str, default="volume", help="Model/feature set label: 'volume', 'fa', 'md'")
    parser.add_argument("--do_boot", type=str2bool, default=True, help="Whether to perform SHAP bootstrapping (True/False).")

    args = parser.parse_args()
    args.todays_run = todays_run
    main(_paths, features, filenames, args)