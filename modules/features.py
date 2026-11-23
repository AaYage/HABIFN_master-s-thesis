import pandas as pd
from sklearn.model_selection import StratifiedKFold
from .path_utils import safe_path


def create_stratified_folds(input_csv: str, output_dir: str, n_splits=5, seed=42):
    """生成五折交叉验证"""
    df = pd.read_csv(safe_path(input_csv))
    feature_cols = [col for col in df.columns if col not in ["filename", "label"]]

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for fold, (train_idx, test_idx) in enumerate(skf.split(df[feature_cols], df["label"])):
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]

        train_df.to_csv(safe_path(f"{output_dir}/train_fold{fold}.csv"), index=False)
        test_df.to_csv(safe_path(f"{output_dir}/test_fold{fold}.csv"), index=False)