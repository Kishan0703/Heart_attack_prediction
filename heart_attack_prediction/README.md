## heart-attack-prediction

This package exposes a CLI that summarizes the multiplex FRET heart attack dataset and trains a baseline regression model for `Calculated_Troponin_ng_mL`.

### Usage

```bash
uv run heart-attack-prediction --data ../Heart_attack_dataset.csv
```

Flags:

- `--data PATH` – CSV file to analyze (defaults to the dataset in the repository root).
- `--test-size FLOAT` – fraction reserved for evaluation (default `0.2`).
- `--estimators INT` – number of trees in the random forest (default `300`).

The command prints dataset stats, model performance (RMSE/R²), and the top feature importances. Use `uv sync` beforehand if dependencies have not been installed yet.

