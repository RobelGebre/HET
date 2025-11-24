import os
import shutil
import random
import secrets

import joblib
import matplotlib
import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    StratifiedGroupKFold,
    RandomizedSearchCV,
    cross_validate,
)
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from autogluon.tabular import TabularPredictor
from tqdm import tqdm

random.seed(42)
np.random.seed(42)

def save_models(models, in_path, prefix):
    """Utility to save a dict of models under `<in_path>/CV trained models`."""
    path = os.path.join(in_path, "CV trained models")
    os.makedirs(path, exist_ok=True)
    for name, model in models.items():
        joblib.dump(model, os.path.join(path, f"{prefix}_{name}.joblib"))


def run_models(df, target_clf, features, ids_column_name, run_dir):
    random.seed(42)
    np.random.seed(42)

    target_Y_clf = df[target_clf].values
    predictors_X_clf = df[features].values
    groupings = df[ids_column_name].values  # subject-wise grouping

    xgb_params = {
        "n_estimators": [200, 400],
        "max_depth": [3, 10, 50],
        "learning_rate": [0.001, 0.01, 0.05],
        "subsample": [0.5, 0.7, 0.9, 1.0],
        "colsample_bytree": [0.5, 0.7, 0.9, 1.0],
        "alpha": [0, 0.1, 0.5, 1, 5, 10],
        "lambda": [0, 0.1, 0.5, 1, 5, 10, 20],
    }
    rf_clf_params = {
        "n_estimators": [200, 500],
        "max_depth": [None, 50, 100, 200],
        "min_samples_leaf": [1, 2, 5, 10, 50, 100],
        "min_samples_split": [2, 5, 10, 20, 50, 100],
    }
    lgbm_params = {
        "n_estimators": [200, 400],
        "max_depth": [3, 6],
        "learning_rate": [0.01, 0.05],
        "num_leaves": [50, 100],
        "min_child_samples": [5, 10],
        "subsample": [0.7, 0.9],
        "colsample_bytree": [0.7, 0.9],
    }
    catboost_params = {
        "iterations": [200, 400],
        "learning_rate": [0.01, 0.05, 0.1],
        "depth": [3, 7, 10],
        "l2_leaf_reg": [3, 5, 10],
    }

    classifiers = {
        "XGBoost": (XGBClassifier(random_state=42), xgb_params),
        "RandomForest": (RandomForestClassifier(random_state=42), rf_clf_params),
        "LGBM": (LGBMClassifier(random_state=42, verbose=-1), lgbm_params),
        "CatBoost": (CatBoostClassifier(random_state=42, verbose=0), catboost_params),
    }

    # 10 different seeds for robustness
    seed_list = [secrets.randbits(32) for _ in range(10)]

    n_iter_indi = 10
    cv_splits_n = 3
    all_seeds_summary = []

    for rseed in seed_list:
        print(f"\n==== Seed {rseed} ====")
        seed_dir = os.path.join(run_dir, f"seed_{rseed}")
        os.makedirs(seed_dir, exist_ok=True)

        sgkf = StratifiedGroupKFold(
            n_splits=cv_splits_n,
            shuffle=True,
            random_state=rseed,
        )
        cv_splits = list(sgkf.split(predictors_X_clf, target_Y_clf, groups=groupings))

        final_results = {}
        per_fold_rows = []

        # ---------------- AutoGluon (per-fold) ----------------
        print("Training AutoGluon with StratifiedGroupKFold (subject-wise)")
        f1_scores_ag, acc_scores_ag, bacc_scores_ag = [], [], []
        ag_fold_paths, ag_fold_scores = [], []

        # safe CPU count for AutoGluon
        n_cpus = os.cpu_count() or 1
        ag_num_cpus = max(1, n_cpus // 2)

        for fold_id, (tr_idx, te_idx) in enumerate(
            tqdm(cv_splits, desc=f"AG SGKF seed={rseed}"), 1
        ):
            fold_dir = os.path.join(seed_dir, f"fold_{fold_id}")
            os.makedirs(fold_dir, exist_ok=True)

            # save fold splits
            df.iloc[tr_idx].to_csv(os.path.join(fold_dir, "train.csv"), index=False)
            df.iloc[te_idx].to_csv(os.path.join(fold_dir, "test.csv"), index=False)

            X_train, X_test = predictors_X_clf[tr_idx], predictors_X_clf[te_idx]
            y_train, y_test = target_Y_clf[tr_idx], target_Y_clf[te_idx]

            ag_train = pd.DataFrame(X_train, columns=features)
            ag_train[target_clf] = y_train

            problem_type = (
                "binary" if np.unique(y_train).size == 2 else "multiclass"
            )
            ag_path = os.path.join(fold_dir, "autogluon")

            ag_predictor = (
                TabularPredictor(
                    label=target_clf,
                    path=ag_path,
                    problem_type=problem_type,
                    eval_metric="f1",
                    verbosity=0,
                )
                .fit(
                    train_data=ag_train,
                    presets="medium_quality",
                    ag_args_fit={"num_cpus": ag_num_cpus},
                )
            )

            # test metrics
            X_test_df = pd.DataFrame(X_test, columns=features)
            pred_te = ag_predictor.predict(X_test_df)
            f1_te = f1_score(y_test, pred_te, average="weighted")
            acc_te = accuracy_score(y_test, pred_te)
            bacc_te = balanced_accuracy_score(y_test, pred_te)
            f1_scores_ag.append(f1_te)
            acc_scores_ag.append(acc_te)
            bacc_scores_ag.append(bacc_te)

            # train metrics
            X_train_df = pd.DataFrame(X_train, columns=features)
            pred_tr = ag_predictor.predict(X_train_df)
            f1_tr = f1_score(y_train, pred_tr, average="weighted")
            acc_tr = accuracy_score(y_train, pred_tr)
            bacc_tr = balanced_accuracy_score(y_train, pred_tr)

            pd.DataFrame(
                [
                    {
                        "fold": fold_id,
                        "f1_weighted": f1_te,
                        "accuracy": acc_te,
                        "balanced_accuracy": bacc_te,
                    }
                ]
            ).to_csv(
                os.path.join(fold_dir, "AutoGluon_metrics.csv"),
                index=False,
            )

            per_fold_rows += [
                {
                    "model": "AutoGluon",
                    "fold": fold_id,
                    "split": "train",
                    "f1_weighted": f1_tr,
                    "accuracy": acc_tr,
                    "balanced_accuracy": bacc_tr,
                },
                {
                    "model": "AutoGluon",
                    "fold": fold_id,
                    "split": "test",
                    "f1_weighted": f1_te,
                    "accuracy": acc_te,
                    "balanced_accuracy": bacc_te,
                },
            ]
            ag_fold_paths.append(ag_path)
            ag_fold_scores.append((f1_te, bacc_te))

        # pick best AG fold by F1 then balanced accuracy
        if ag_fold_scores:
            f1_arr = np.array([s[0] for s in ag_fold_scores])
            bacc_arr = np.array([s[1] for s in ag_fold_scores])
            cand = np.where(f1_arr == f1_arr.max())[0]
            best_idx = int(cand[np.argmax(bacc_arr[cand])])
            best_ag_path = ag_fold_paths[best_idx]
            best_ag_model = TabularPredictor.load(best_ag_path)
        else:
            best_ag_model = None

        final_results["AutoGluon"] = {
            "best_model": best_ag_model,
            "f1_score": float(np.mean(f1_scores_ag)) if f1_scores_ag else np.nan,
            "accuracy": float(np.mean(acc_scores_ag)) if acc_scores_ag else np.nan,
            "balanced_accuracy": float(np.mean(bacc_scores_ag)) if bacc_scores_ag else np.nan,
            "f1_score_std": float(np.std(f1_scores_ag)) if f1_scores_ag else np.nan,
        }

        # ---------------- sklearn classifiers (per-fold) ----------------
        print("Training classifiers with StratifiedGroupKFold (subject-wise)")
        for name, (classifier, param_grid) in tqdm(
            classifiers.items(), desc=f"GridSearch seed={rseed}"
        ):
            search = RandomizedSearchCV(
                estimator=classifier,
                param_distributions=param_grid,
                cv=cv_splits,
                n_iter=n_iter_indi,
                n_jobs=-1,
                verbose=1,
                random_state=rseed,
                scoring="f1_weighted",
                refit=True,
            )
            search.fit(predictors_X_clf, target_Y_clf)

            best_model = search.best_estimator_

            scores = cross_validate(
                best_model,
                predictors_X_clf,
                target_Y_clf,
                cv=cv_splits,
                scoring={
                    "f1": "f1_weighted",
                    "accuracy": "accuracy",
                    "balanced_accuracy": "balanced_accuracy",
                },
                n_jobs=-1,
                return_train_score=True,
            )

            # write per-fold CSVs
            for fold_id, (
                f1_tr,
                acc_tr,
                bacc_tr,
                f1_te,
                acc_te,
                bacc_te,
            ) in enumerate(
                zip(
                    scores["train_f1"],
                    scores["train_accuracy"],
                    scores["train_balanced_accuracy"],
                    scores["test_f1"],
                    scores["test_accuracy"],
                    scores["test_balanced_accuracy"],
                ),
                1,
            ):
                fold_dir = os.path.join(seed_dir, f"fold_{fold_id}")
                os.makedirs(fold_dir, exist_ok=True)
                out_path = os.path.join(fold_dir, f"{name}_metrics.csv")
                row = pd.DataFrame(
                    [
                        {
                            "fold": fold_id,
                            "model": name,
                            "f1_weighted": f1_te,
                            "accuracy": acc_te,
                            "balanced_accuracy": bacc_te,
                        }
                    ]
                )
                if os.path.exists(out_path):
                    row.to_csv(out_path, mode="a", header=False, index=False)
                else:
                    row.to_csv(out_path, index=False)

                per_fold_rows += [
                    {
                        "model": name,
                        "fold": fold_id,
                        "split": "train",
                        "f1_weighted": f1_tr,
                        "accuracy": acc_tr,
                        "balanced_accuracy": bacc_tr,
                    },
                    {
                        "model": name,
                        "fold": fold_id,
                        "split": "test",
                        "f1_weighted": f1_te,
                        "accuracy": acc_te,
                        "balanced_accuracy": bacc_te,
                    },
                ]

            final_results[name] = {
                "best_model": best_model,
                "f1_score": float(np.mean(scores["test_f1"])),
                "accuracy": float(np.mean(scores["test_accuracy"])),
                "balanced_accuracy": float(np.mean(scores["test_balanced_accuracy"])),
                "f1_score_std": float(np.std(scores["test_f1"])),
            }

        # save per-fold split metrics for this seed
        per_fold_df = (
            pd.DataFrame(per_fold_rows)
            .sort_values(["model", "fold", "split"])
            .reset_index(drop=True)
        )
        per_fold_df.to_csv(
            os.path.join(seed_dir, "per_fold_split_metrics.csv"),
            index=False,
        )

        # winner for this seed: maximize mean F1, then minimize F1 std
        best_name = max(
            final_results.keys(),
            key=lambda k: (final_results[k]["f1_score"], -final_results[k]["f1_score_std"]),
        )
        best_model_obj = final_results[best_name]["best_model"]
        print(f"[Seed {rseed}] Winner: {best_name}")

        # save winner into seed
        if best_name == "AutoGluon":
            if best_model_obj is None or not getattr(best_model_obj, "path", None):
                raise RuntimeError(
                    "AutoGluon chosen as best but best_model/path missing."
                )
            src_dir = best_model_obj.path
            dest_dir = os.path.join(seed_dir, f"{best_name}_best_classifier")
            if os.path.exists(dest_dir):
                shutil.rmtree(dest_dir)
            shutil.copytree(src_dir, dest_dir)
            saved_path = dest_dir
        else:
            model_path = os.path.join(seed_dir, f"{best_name}_best_classifier.joblib")
            joblib.dump(best_model_obj, model_path)
            saved_path = model_path

        # per-seed summary row
        all_seeds_summary.append(
            {
                "seed": rseed,
                "winner_model": best_name,
                "saved_to": saved_path,
                "f1_score": final_results[best_name]["f1_score"],
                "f1_score_std": final_results[best_name]["f1_score_std"],
                "accuracy": final_results[best_name]["accuracy"],
                "balanced_accuracy": final_results[best_name]["balanced_accuracy"],
            }
        )

    winners_df = pd.DataFrame(all_seeds_summary)
    winners_df.to_csv(
        os.path.join(run_dir, "seeds_winners_summary.csv"),
        index=False,
    )

    # pick overall best across seeds: max F1, then min F1 std
    best_row = max(
        all_seeds_summary,
        key=lambda r: (r["f1_score"], -r["f1_score_std"]),
    )
    overall_name = best_row["winner_model"]
    overall_src = best_row["saved_to"]

    if overall_name == "AutoGluon":
        dest_dir = os.path.join(run_dir, f"overall_best_{overall_name}_best_classifier")
        if os.path.exists(dest_dir):
            shutil.rmtree(dest_dir)
        shutil.copytree(overall_src, dest_dir)
        print("Overall best AutoGluon copied to:", dest_dir)
    else:
        dest_path = os.path.join(
            run_dir,
            f"overall_best_{overall_name}_best_classifier.joblib",
        )
        if os.path.abspath(overall_src) != os.path.abspath(dest_path):
            shutil.copy2(overall_src, dest_path)
        print("Overall best sklearn model copied to:", dest_path)
