import glob
import json
import os
import re
from typing import List, Optional
import matplotlib.pyplot as plt
import pandas as pd


class ExperimentComparer:
    """Utility class to compare mAP40 performance degradation across multiple

    subsampling experiments, supporting both global mAP40 and distance-range
    breakdowns.
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

            numbers = re.findall(r"\d+", name)
            num_val = int(numbers[0]) if numbers else 0
            return (1, -num_val)

        return sorted(exp_names, key=extract_sort_key)

    def load_global_results(self) -> pd.DataFrame:
        """Loads and aggregates all map40_results.json files from experiment subdirectories."""
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

        return pd.DataFrame(records)

    def load_range_results(self) -> pd.DataFrame:
        """Loads and aggregates all map40_range_results.json files from experiment subdirectories."""
        json_files = glob.glob(
            os.path.join(self.experiments_root, "**/map40_range_results.json"),
            recursive=True,
        )

        records = []
        for jpath in json_files:
            exp_dir = os.path.dirname(jpath)
            exp_name = os.path.basename(os.path.normpath(exp_dir))

            with open(jpath, "r") as f:
                data = json.load(f)

            for entry in data:
                entry_copy = dict(entry)
                entry_copy["Experiment"] = exp_name
                records.append(entry_copy)

        return pd.DataFrame(records)

    def plot_degradation_curves(
        self,
        exp_order: Optional[List[str]] = None,
        target_class: str = "Car",
        title: str = "Overall mAP40 Performance Degradation",
        save_path: Optional[str] = None,
    ):
        """Plots global mAP40 degradation curves across experiment configurations for Easy, Moderate, and Hard difficulties."""
        df = self.load_global_results()
        if df.empty:
            print(" No global experiment results found.")
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
            print(f" Overall degradation plot saved to: {save_path}")

        plt.show()

    def plot_range_degradation_curves(
        self,
        exp_order: Optional[List[str]] = None,
        target_class: str = "Car",
        difficulty: str = "Moderate",
        title: str = "Distance Range mAP40 Degradation",
        save_path: Optional[str] = None,
    ):
        """Plots distance-range mAP40 degradation curves (Near, Medium, Far) across subsampling levels."""
        df = self.load_range_results()
        if df.empty:
            print(" No range-based experiment results found.")
            return

        if exp_order is None:
            exp_order = self._sort_experiment_names(
                list(df["Experiment"].unique())
            )

        filtered = df[
            (df["Class"] == target_class) & (df["Difficulty"] == difficulty)
        ]

        if filtered.empty:
            print(
                f" No data matching Class={target_class} and Difficulty={difficulty}."
            )
            return

        pivot_df = filtered.pivot(
            index="Experiment", columns="Range", values="mAP40"
        )
        pivot_df = pivot_df.reindex(
            [e for e in exp_order if e in pivot_df.index]
        )

        plt.figure(figsize=(9, 5))
        for range_col in pivot_df.columns:
            plt.plot(
                pivot_df.index,
                pivot_df[range_col],
                marker="s",
                linewidth=2.5,
                label=range_col,
            )

        plt.title(
            f"{title} - Class: {target_class} ({difficulty})", fontsize=13
        )
        plt.xlabel("Experiment / Subsampling Level", fontsize=11)
        plt.ylabel("mAP40 (%)", fontsize=11)
        plt.ylim(-5, 105)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(title="Depth Range", loc="best")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f" Range degradation plot saved to: {save_path}")

        plt.show()