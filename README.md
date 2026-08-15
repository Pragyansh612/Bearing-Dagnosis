# Multimodal physics-informed bearing fault diagnosis

Streamlit demo and PyTorch codebase for conditional health assessment on the **SCA Bearing Dataset** (Mendeley: [tdn96mkkpt](https://data.mendeley.com/datasets/tdn96mkkpt/2)).

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Open the app, upload a `.mat` file, or use **Demo samples** if the SCA data is installed locally under the paths expected in `app.py`.

## Repository layout

| Path | Purpose |
|------|---------|
| `app.py` | Streamlit UI and inference pipeline |
| `models/` | Trained checkpoints, scaler, prototypes, baselines (~6 MB total) |
| `src/` | Models, preprocessing, training scripts |
| `notebooks/` | Evaluation and analysis scripts |
| `results/` | Logged metrics, figures, and JSON summaries |

## Data

Do not commit `.mat` files. Download the dataset from Mendeley and point the demo paths or use **Upload .mat file** in the sidebar.

## Training

Training entry points live under `src/training/`. They expect the SCA folder layout and a configured Python environment with the same dependencies as above (optionally CUDA-enabled `torch`).

streamlit run app.py

## License
