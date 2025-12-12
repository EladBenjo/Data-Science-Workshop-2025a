# Data-Science-Workshop-2025a

This repository reorganizes the original exploratory notebook into a modular Python codebase. The goal is to make the speech scraping, text processing, and forecasting workflow reproducible outside Jupyter.

## Repository layout

- `src/data/loading.py` – helpers to load the scraped speeches CSV, the FRED inflation expectations series, and aggregate speech-level features to monthly values.
- `src/features/text_features.py` – reusable text processing utilities (sentence splitting, keyword filtering, TTR calculations) extracted from the notebook.
- `src/features/sentiment_features.py` – functions for running the FOMC-RoBERTa classifier results through the hawkish/dovish scoring logic used in the analysis.
- `src/models/architectures.py` – TensorFlow/Keras implementations of the LSTM and GRU regressors explored in the notebook.
- `src/models/training.py` – shared training utilities (reproducible seeding and callback configuration).
- `src/models/evaluation.py` – metric and plotting helpers.
- `scripts/pipeline_outline.py` – an end-to-end example showing how to stitch the modules together once the CSV inputs exist.

The original notebooks remain untouched for reference, but all reusable code now lives in `src/`.

## Running the example pipeline

1. Install the required dependencies (TensorFlow, spaCy, Transformers, pandas, scikit-learn, matplotlib, etc.).
2. Download or generate the scraped speeches CSV and the FRED EXPINF1YR CSV.
3. Run the pipeline outline:

```bash
python scripts/pipeline_outline.py path/to/speeches.csv path/to/EXPINF1YR.csv
```

The script will:

- generate sentence-level features from `speech_text` using spaCy,
- compute sentiment measures if the `sentiment_results` column is present,
- aggregate speeches to monthly features,
- merge those features with EXPINF1YR values,
- train a GRU-based regressor, and
- print train/validation metrics.

Adjust the feature list or model architecture in `scripts/pipeline_outline.py` to match your dataset and hardware constraints.
