"""
Poultry disease data-entry app — ResNet18 from ANS128_group_project notebook.
Run: shiny run app.py
"""

from __future__ import annotations

from datetime import date
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from shiny import App, reactive, render, ui

from model_inference import (
    CLASS_INFO,
    CLASS_NAMES,
    MODEL_FILENAME,
    ModelWeightsError,
    load_model,
    predict_image,
)

APP_DIR = Path(__file__).resolve().parent
try:
    _model, _model_status = load_model(APP_DIR)
    _model_ready = True
except ModelWeightsError as exc:
    _model = None
    _model_status = str(exc)
    _model_ready = False

ENTRY_COLUMNS = [
    "case_id",
    "entry_date",
    "farm_location",
    "flock_size",
    "notes",
    "image_name",
    "predicted_class",
    "confidence",
    "cocci_prob",
    "healthy_prob",
    "ncd_prob",
    "salmo_prob",
]

app_ui = ui.page_fluid(
    ui.tags.head(
        ui.tags.style(
            """
            .app-header { margin-bottom: 1rem; }
            .status-ok { color: #1a7f37; font-weight: 600; }
            .status-warn { color: #9a6700; font-weight: 600; }
            .pred-box { background: #f6f8fa; border-radius: 8px; padding: 1rem; margin-top: 0.5rem; }
            """
        )
    ),
    ui.div(
        {"class": "app-header"},
        ui.h2("Poultry Disease Screening — Data Entry"),
        ui.p(
            "Upload a poultry image and record flock details. Predictions use the "
            "ResNet18 model trained in ",
            ui.code("ANS128_group_project (2).ipynb"),
            " (4 classes: cocci, healthy, ncd, salmo).",
        ),
        ui.output_ui("model_status"),
    ),
    ui.layout_sidebar(
        ui.sidebar(
            ui.h4("New case entry"),
            ui.input_text("case_id", "Case / flock ID", placeholder="e.g. FLOCK-2026-042"),
            ui.input_date("entry_date", "Entry date", value=date.today()),
            ui.input_text("farm_location", "Farm / location", placeholder="Optional"),
            ui.input_numeric("flock_size", "Flock size (birds)", value=1000, min=1, step=1),
            ui.input_text_area("notes", "Clinical notes", rows=3, placeholder="Optional observations"),
            ui.input_file(
                "image",
                "Poultry image (.jpg, .png)",
                accept=[".jpg", ".jpeg", ".png"],
                multiple=False,
            ),
            ui.input_action_button("submit", "Run prediction & save entry", class_="btn-primary"),
            ui.hr(),
            ui.download_button("download_csv", "Download all entries (CSV)"),
            width=380,
        ),
        ui.layout_columns(
            ui.card(
                ui.card_header("Uploaded image"),
                ui.output_image("preview_image", height="280px"),
            ),
            ui.card(
                ui.card_header("Model prediction"),
                ui.output_ui("prediction_panel"),
                ui.output_plot("prob_chart"),
            ),
            col_widths=[5, 7],
        ),
        ui.card(
            ui.card_header("Saved entries (this session)"),
            ui.output_data_frame("entries_table"),
        ),
    ),
)


def server(input, output, session):
    entries = reactive.Value(pd.DataFrame(columns=ENTRY_COLUMNS))
    last_prediction = reactive.Value(None)
    last_image_bytes = reactive.Value(None)

    @output
    @render.ui
    def model_status():
        css = "status-ok" if _model_ready else "status-warn"
        return ui.HTML(f'<p class="{css}">{_model_status}</p>')

    @reactive.calc
    def uploaded_file():
        files = input.image()
        if not files:
            return None
        return files[0]

    @reactive.effect
    @reactive.event(input.image)
    def _cache_upload():
        f = uploaded_file()
        if f is None:
            last_image_bytes.set(None)
            return
        last_image_bytes.set(Path(f["datapath"]).read_bytes())

    @output
    @render.image
    def preview_image():
        data = last_image_bytes.get()
        if data is None:
            return {"src": None, "alt": "No image uploaded", "width": "100%", "height": "auto"}
        return {"src": data, "alt": "Uploaded poultry image", "width": "100%", "height": "auto"}

    @reactive.effect
    @reactive.event(input.submit)
    def run_prediction():
        f = uploaded_file()
        if f is None:
            ui.notification_show("Please upload an image first.", type="warning")
            return

        if not _model_ready or _model is None:
            ui.notification_show(
                f"Poultry model not loaded. Add valid `{MODEL_FILENAME}` next to app.py.",
                type="error",
                duration=8,
            )
            return

        case_id = (input.case_id() or "").strip()
        if not case_id:
            ui.notification_show("Enter a case / flock ID.", type="warning")
            return

        image_bytes = Path(f["datapath"]).read_bytes()
        last_image_bytes.set(image_bytes)

        try:
            result = predict_image(_model, image_bytes)
        except Exception as exc:
            ui.notification_show(f"Prediction failed: {exc}", type="error")
            return

        last_prediction.set(result)

        probs = result["probabilities"]
        new_row = {
            "case_id": case_id,
            "entry_date": str(input.entry_date()),
            "farm_location": (input.farm_location() or "").strip(),
            "flock_size": input.flock_size(),
            "notes": (input.notes() or "").strip(),
            "image_name": f["name"],
            "predicted_class": result["predicted_class"],
            "confidence": round(result["confidence"], 4),
            "cocci_prob": round(probs["cocci"], 4),
            "healthy_prob": round(probs["healthy"], 4),
            "ncd_prob": round(probs["ncd"], 4),
            "salmo_prob": round(probs["salmo"], 4),
        }

        df = entries.get()
        entries.set(pd.concat([df, pd.DataFrame([new_row])], ignore_index=True))
        ui.notification_show(
            f"Saved: {case_id} → {result['predicted_class']} ({result['confidence']:.1%})",
            type="message",
        )

    @output
    @render.ui
    def prediction_panel():
        result = last_prediction.get()
        if result is None:
            return ui.p("Upload an image and click **Run prediction & save entry**.")

        cls = result["predicted_class"]
        info = CLASS_INFO[cls]
        return ui.div(
            {"class": "pred-box"},
            ui.h3(f"{info['title']} ({cls})"),
            ui.p(ui.strong("Confidence: "), f"{result['confidence']:.1%}"),
            ui.p(info["summary"]),
            ui.tags.small("Probabilities by class are shown in the chart →"),
        )

    @output
    @render.plot
    def prob_chart():
        result = last_prediction.get()
        fig, ax = plt.subplots(figsize=(6, 3.5))
        if result is None:
            ax.text(0.5, 0.5, "No prediction yet", ha="center", va="center")
            ax.axis("off")
            return fig

        probs = result["probabilities"]
        labels = [CLASS_INFO[c]["title"] for c in CLASS_NAMES]
        values = [probs[c] for c in CLASS_NAMES]
        colors = ["#d62728" if c == result["predicted_class"] else "#4c78a8" for c in CLASS_NAMES]

        ax.barh(labels, values, color=colors)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Probability")
        ax.set_title("Class probabilities (ResNet18)")
        plt.tight_layout()
        return fig

    @output
    @render.data_frame
    def entries_table():
        df = entries.get()
        if df.empty:
            return pd.DataFrame({"message": ["No entries yet — submit a case above."]})
        display = df.copy()
        display["confidence"] = display["confidence"].map(lambda x: f"{x:.1%}")
        for col in ("cocci_prob", "healthy_prob", "ncd_prob", "salmo_prob"):
            display[col] = display[col].map(lambda x: f"{x:.1%}")
        return display

    @session.download(filename="poultry_disease_entries.csv")
    def download_csv():
        df = entries.get()
        buffer = BytesIO()
        df.to_csv(buffer, index=False)
        buffer.seek(0)
        yield buffer.getvalue()


app = App(app_ui, server)

