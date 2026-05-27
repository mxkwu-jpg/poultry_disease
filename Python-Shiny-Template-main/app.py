from pathlib import Path

from shiny import App, ui, render
import pandas as pd
import matplotlib.pyplot as plt

# Load celtuce data
data_path = Path(__file__).parent / "celtuce_data.csv"
df = pd.read_csv(data_path)
disease_col = [c for c in df.columns if "disease" in c.lower()][0]
df["disease"] = pd.to_numeric(df[disease_col], errors="coerce")
stem_cols = [c for c in df.columns if "Stem" in c and "length" in c]
df["stem_length"] = df[stem_cols].apply(pd.to_numeric, errors="coerce").max(axis=1)

app_ui = ui.page_fluid(
    ui.h2("Celtuce Traits by Line"),
    ui.input_select("trait", "Trait:", choices={"disease": "Disease", "stem_length": "Stem length (cm)"}),
    ui.output_plot("boxplot")
)

def server(input, output, session):
    @output
    @render.plot
    def boxplot():
        trait = input.trait()
        plot_df = df.dropna(subset=["Line", trait])

        # Order lines by mean trait (ascending for disease, descending for stem length)
        ascending = trait == "disease"
        line_order = (
            plot_df.groupby("Line")[trait]
            .mean()
            .sort_values(ascending=ascending)
            .index.tolist()
        )
        plot_data = [plot_df.loc[plot_df["Line"] == line, trait].values for line in line_order]

        fig, ax = plt.subplots(figsize=(14, 6))
        bp = ax.boxplot(plot_data, labels=line_order, patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor("lightsteelblue")
        if trait == "disease":
            ax.set_title("Disease Observations by Line (0=none, 5=worst)")
            ax.set_ylabel("Disease score")
            ax.set_ylim(-0.5, 5.5)
        else:
            ax.set_title("Stem Length by Line")
            ax.set_ylabel("Stem length (cm)")
        ax.set_xlabel("Line")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        return fig

app = App(app_ui, server) 