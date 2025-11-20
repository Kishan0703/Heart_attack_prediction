from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Sequence

import pandas as pd
from rich.console import Console
from rich.table import Table
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

console = Console()

TARGET_COLUMN = "Calculated_Troponin_ng_mL"
DEFAULT_DATASET = Path(__file__).resolve().parents[3] / "Heart_attack_dataset.csv"


def _should_skip_metadata_line(csv_path: Path) -> bool:
    with csv_path.open("r", encoding="utf-8") as handle:
        first_line = handle.readline()
    return "," not in first_line


def load_dataset(csv_path: Path) -> pd.DataFrame:
    resolved = csv_path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(
            f"Could not find dataset at {resolved}. Pass --data to point to a valid CSV file."
        )

    skiprows = 1 if _should_skip_metadata_line(resolved) else 0
    dataframe = pd.read_csv(resolved, skiprows=skiprows)
    dataframe.columns = [column.strip() for column in dataframe.columns]
    return dataframe


def summarize_dataframe(df: pd.DataFrame) -> None:
    console.rule("[bold cyan]Dataset Overview")
    console.print(f"Rows: {df.shape[0]:,} | Columns: {df.shape[1]} | Target: {TARGET_COLUMN}")
    numeric_summary = df.describe().transpose()

    table = Table(title="Numerical Summary", show_lines=True)
    for header in ("Column", "Mean", "Std", "Min", "Max"):
        table.add_column(header, justify="right")

    for column, row in numeric_summary.iterrows():
        table.add_row(
            column,
            f"{row['mean']:.3f}",
            f"{row['std']:.3f}",
            f"{row['min']:.3f}",
            f"{row['max']:.3f}",
        )

    console.print(table)


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    if TARGET_COLUMN not in df.columns:
        raise KeyError(
            f"Expected '{TARGET_COLUMN}' in dataset columns. Found: {list(df.columns)}"
        )

    numeric_df = df.select_dtypes(include="number")
    feature_frame = numeric_df.drop(columns=[TARGET_COLUMN], errors="ignore")
    target = numeric_df[TARGET_COLUMN]
    return feature_frame, target


def train_model(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    test_size: float,
    n_estimators: int,
) -> tuple[RandomForestRegressor, dict[str, float], pd.Series]:
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        shuffle=True,
        random_state=42,
    )

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "rmse": math.sqrt(mean_squared_error(y_test, predictions)),
        "r2": r2_score(y_test, predictions),
    }

    importances = pd.Series(model.feature_importances_, index=features.columns)
    return model, metrics, importances.sort_values(ascending=False)


def render_metrics(metrics: dict[str, float], importances: pd.Series) -> None:
    console.rule("[bold green]Model Performance")
    console.print(f"RMSE: {metrics['rmse']:.4f}")
    console.print(f"R²: {metrics['r2']:.4f}")

    table = Table(title="Top Feature Importances")
    table.add_column("Feature", justify="right")
    table.add_column("Importance", justify="right")

    for feature, value in importances.head(5).items():
        table.add_row(feature, f"{value:.4f}")

    console.print(table)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a quick regression model for the heart attack dataset."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATASET,
        help=f"Path to the CSV dataset (default: {DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test split fraction (default: 0.2)",
    )
    parser.add_argument(
        "--estimators",
        type=int,
        default=300,
        help="Number of trees for the random forest (default: 300)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    df = load_dataset(args.data)
    summarize_dataframe(df)

    features, target = prepare_features(df)
    model, metrics, importances = train_model(
        features,
        target,
        test_size=args.test_size,
        n_estimators=args.estimators,
    )

    render_metrics(metrics, importances)
    console.print(
        "\n[bold]Done![/bold] You can tweak hyperparameters via the CLI flags for different results."
    )


if __name__ == "__main__":
    main()

