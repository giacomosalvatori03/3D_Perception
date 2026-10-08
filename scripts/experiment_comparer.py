import glob
import json
import os
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import pandas as pd

try:
    from IPython.display import display

    HAS_IPYTHON = True
except ImportError:
    HAS_IPYTHON = False


class ExperimentComparer:
    """Utility class to aggregate, compare, and plot mAP40 metrics across multiple

    subsampling experiments saved in the Google Drive experiments folder.
    """

    def __init__(
        self,
        experiments_root: str = "/content/drive/MyDrive/3D_Perception/experiments",
    ):
        self.experiments_root = experiments_root

    def load_all_results(self) -> pd.DataFrame:
        """Searches all subfolders in the experiments directory for

        map40_results.json files and aggregates them into a unified Pandas
        DataFrame.
        """
        json_files = glob.glob(
            os.path.join(self.experiments_root, "**/map40_results.json"),
            recursive=True,
        )

        records = []
        for jpath in sorted(json_files):
            exp_dir = os.path.dirname(jpath)
            exp_name = os.path.basename(os.path.normpath(exp_dir))

            with open(jpath, "r") as f:
                data = json.load(f)

            for cls_name, diff_dict in data.items():
                for diff_level, map_val in diff_dict.items():
                    records.append({
                        "Experiment": exp_name,
                        "Class": cls_name,
                        "Difficulty": diff_level,
                        "mAP40": map_val,
                    })

        df = pd.DataFrame(records)
        return df

    def get_summary_table(
        self, target_class: str = "Car"
    ) -> Optional[pd.DataFrame]:
        """Generates a summary DataFrame comparing Easy, Moderate, and Hard mAP40

        scores across all experiments for a given class.
        """
        df = self.load_all_results()
        if df.empty:
            print(" No experiment results found.")
            return None

        df_cls = df[df["Class"] == target_class]
        pivot_df = df_cls.pivot(
            index="Experiment", columns="Difficulty", values="mAP40"
        )

        # Reorder columns
        cols = [
            c for c in ["Easy", "Moderate", "Hard"] if c in pivot_df.columns
        ]
        pivot_df = pivot_df[cols]
        return pivot_df

    def plot_degradation_curves(
        self,
        exp_order: List[str],
        target_class: str = "Car",
        title: str = "mAP40 Performance Degradation",
        save_path: Optional[str] = None,
    ):
        """Plots line charts showing mAP40 degradation across ordered experiment configurations."""
        df = self.load_all_results()
        if df.empty:
            print(" No experiment results found.")
            return

        df_cls = df[df["Class"] == target_class]

        plt.figure(figsize=(9, 5))
        for diff in ["Easy", "Moderate", "Hard"]:
            subset = df_cls[df_cls["Difficulty"] == diff]
            # Map values according to ordered list
            subset = subset.set_index("Experiment").reindex(exp_order).dropna()

            plt.plot(
                subset.index,
                subset["mAP40"],
                marker="o",
                linewidth=2.5,
                label=f"{diff} Difficulty",
            )

        plt.title(f"{title} - Class: {target_class}", fontsize=13)
        plt.xlabel("Experiment / Subsampling Level", fontsize=11)
        plt.ylabel("mAP40 (%)", fontsize=11)
        plt.ylim(-5, 105)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(loc="best")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f" Plot saved to: {save_path}")

        plt.show()

    def compare_all_classes_bar_plot(
        self,
        exp_names: List[str],
        difficulty: str = "Moderate",
        save_path: Optional[str] = None,
    ):
        """Plots a grouped bar chart comparing Moderate mAP40 across classes for selected experiments."""
        df = self.load_all_results()
        if df.empty:
            return

        filtered_df = df[
            (df["Experiment"].isin(exp_names))
            & (df["Difficulty"] == difficulty)
        ]

        pivot_df = filtered_df.pivot(
            index="Experiment", columns="Class", values="mAP40"
        )
        ax = pivot_df.plot(kind="bar", figsize=(10, 5), width=0.8)

        plt.title(
            f"Class-wise Comparison ({difficulty} Difficulty)", fontsize=13
        )
        plt.ylabel("mAP40 (%)", fontsize=11)
        plt.xlabel("Experiment", fontsize=11)
        plt.ylim(0, 100)
        plt.grid(axis="y", linestyle="--", alpha=0.6)
        plt.xticks(rotation=15)
        plt.legend(title="Class")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f" Bar chart saved to: {save_path}")

        plt.show()