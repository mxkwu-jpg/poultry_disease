# Poultry Disease Data-Entry Shiny App

Python Shiny app for recording flock/case details and classifying poultry images with the **ResNet18** model from `ANS128_group_project (2).ipynb`.

## Classes

| Code | Disease |
|------|---------|
| `cocci` | Coccidiosis |
| `healthy` | Healthy |
| `ncd` | Newcastle disease |
| `salmo` | Salmonella |

## Setup

1. **Export model weights** (after training ResNet18 in the notebook):

   Run the notebook cell that saves:

   ```python
   torch.save(model_resnet.state_dict(), 'ResNet18_poultry_disease.pth')
   ```

   Copy `ResNet18_poultry_disease.pth` into this project folder (same directory as `app.py`).

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app**:

   ```bash
   shiny run app.py
   ```

## Usage

1. Enter case ID, date, farm, flock size, and notes.
2. Upload a `.jpg` or `.png` poultry image.
3. Click **Run prediction & save entry** to classify the image and append a row to the session table.
4. Use **Download all entries (CSV)** to export records from the current session.

## Files

- `app.py` — Shiny UI and data-entry workflow
- `model_inference.py` — ResNet18 load/predict (matches notebook transforms)
- `requirements.txt` — Python dependencies
