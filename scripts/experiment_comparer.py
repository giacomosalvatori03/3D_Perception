import glob
import json
import os
import re
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

    @staticmethod
    def _sort_experiment_names(exp_names: List[str]) -> List[str]:
        """Helper method to logically sort experiment tags.

        Puts 'baseline' first, then orders by decreasing numeric parameters
        (e.g., beam_32rings -> beam_16rings, or random_100perc ->
        random_50perc).
        """

        def extract_sort_key(name: str):
            name_lower = name.lower()
            if "baseline" in name_lower:
                return (0, 0)

            # Extract integer values (e.g., 32 from beam_32rings, 50 from random_50perc)
            numbers = re.findall(r"\d+", name)
            num_val = int(numbers[0]) if numbers else 0

            # Sort by decreasing magnitude (higher beam count or keep_ratio first)
            return (1, -num_val)

        return sorted(exp_names, key=extract_sort_key)

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
        for jpath in json_files:
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
        self,
        target_class: str = "Car",
        exp_order: Optional[List[str]] = None,
    ) -> Optional[pd.DataFrame]:
        """Generates a summary DataFrame comparing Easy, Moderate, and Hard mAP40

        scores across all experiments for a given class.
        """
        df = self.load_all_results()
        if df.empty:
            print("⚠️ No experiment results found.")
            return None

        df_cls = df[df["Class"] == target_class]
        pivot_df = df_cls.pivot(
            index="Experiment", columns="Difficulty", values="mAP40"
        )

        # Reorder columns by difficulty
        cols = [
            c for c in ["Easy", "Moderate", "Hard"] if c in pivot_df.columns
        ]
        pivot_df = pivot_df[cols]

        # Apply custom or logical automatic ordering to index rows
        if exp_order:
            ordered_index = [e for e in exp_order if e in pivot_df.index]
        else:
            ordered_index = self._sort_experiment_names(list(pivot_df.index))

        pivot_df = pivot_df.reindex(ordered_index)
        return pivot_df

    def plot_degradation_curves(
        self,
        exp_order: Optional[List[str]] = None,
        target_class: str = "Car",
        title: str = "mAP40 Performance Degradation",
        save_path: Optional[str] = None,
    ):
        """Plots line charts showing mAP40 degradation across ordered experiment configurations."""
        df = self.load_all_results()
        if df.empty:
            print("⚠️ No experiment results found.")
            return

        if exp_order is None:
            exp_order = self._sort_experiment_names(
                list(df["Experiment"].unique())
            )

        df_cls = df[df["Class"] == target_class]

        plt.figure(figsize=(9, 5))
        for diff in ["Easy", "Moderate", "Hard"]:
            subset = df_cls[df_cls["Difficulty"] == diff]
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
            print(f"✅ Line plot saved to: {save_path}")

        plt.show()

    def compare_all_classes_bar_plot(
        self,
        exp_names: Optional[List[str]] = None,
        difficulty: str = "Moderate",
        save_path: Optional[str] = None,
    ):
        """Plots a grouped bar chart comparing mAP40 across classes preserving the specified experiment order."""
        df = self.load_all_results()
        if df.empty:
            print("⚠️ No experiment results found.")
            return

        if exp_names is None:
            exp_names = self._sort_experiment_names(
                list(df["Experiment"].unique())
            )
        else:
            exp_names = [e for e in exp_names if e in df["Experiment"].values]

        filtered_df = df[
            (df["Experiment"].isin(exp_names))
            & (df["Difficulty"] == difficulty)
        ]

        pivot_df = filtered_df.pivot(
            index="Experiment", columns="Class", values="mAP40"
        )

        # Enforce exact experiment sequence order on x-axis
        pivot_df = pivot_df.reindex(
            [e for e in exp_names if e in pivot_df.index]
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
            print(f"✅ Bar chart saved to: {save_path}")

        plt.show()