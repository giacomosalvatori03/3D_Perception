import glob
import json
import os
import re
from typing import List, Optional
import matplotlib.pyplot as plt
import pandas as pd


class ExperimentComparer:
    """Utility class to compare mAP40 performance across LiDAR-only and Late

    Fusion experiments.
    """

    def __init__(
        self,
        experiments_root: str = "/content/drive/MyDrive/3D_Perception/experiments",
        fusion_root: Optional[str] = "/content/drive/MyDrive/3D_Perception/experiments_fusion",
    ):
        self.experiments_root = experiments_root
        self.fusion_root = fusion_root

    @staticmethod
    def _sort_experiment_names(exp_names: List[str]) -> List[str]:
        """Helper method to logically sort experiment tags."""

        def extract_sort_key(name: str):
            clean_name = name.lower().replace("fusion_", "")
            if "baseline" in clean_name:
                return (0, 0)

            numbers = re.findall(r"\d+", clean_name)
            num_val = int(numbers[0]) if numbers else 0
            return (1, -num_val)

        return sorted(exp_names, key=extract_sort_key)

    def load_global_results(self, root_path: Optional[str] = None) -> pd.DataFrame:
        """Loads and aggregates all map40_results.json files from experiment subdirectories."""
        target_root = root_path or self.experiments_root
        json_files = glob.glob(
            os.path.join(target_root, "**/map40_results.json"),
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

    def load_range_results(self, root_path: Optional[str] = None) -> pd.DataFrame:
        """Loads and aggregates all map40_range_results.json files from experiment subdirectories."""
        target_root = root_path or self.experiments_root
        json_files = glob.glob(
            os.path.join(target_root, "**/map40_range_results.json"),
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

    def compute_gain_matrix(
        self,
        fusion_root: Optional[str] = None,
        target_class: str = "Car",
        difficulty: str = "Moderate",
        exp_order: Optional[List[str]] = None,
        exclude_64beams: bool = True,
        save_dir: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Computes absolute ΔmAP (points) and relative gain (%) of Late Fusion over LiDAR-Only,
        excluding 64-beam experiments by default and optionally saving reports to Drive.
        """
        f_root = fusion_root or self.fusion_root
        if not f_root or not os.path.exists(f_root):
            raise FileNotFoundError(f"Fusion root directory not found: {f_root}")

        df_lidar = self.load_global_results(self.experiments_root)
        df_fusion = self.load_global_results(f_root)

        if df_lidar.empty or df_fusion.empty:
            print(" Insufficient data to compute gain matrix.")
            return pd.DataFrame()

        # Filter by class and difficulty
        s_lidar = (
            df_lidar[
                (df_lidar["Class"] == target_class)
                & (df_lidar["Difficulty"] == difficulty)
            ]
            .set_index("Experiment")["mAP40"]
        )
        s_fusion = (
            df_fusion[
                (df_fusion["Class"] == target_class)
                & (df_fusion["Difficulty"] == difficulty)
            ]
            .set_index("Experiment")["mAP40"]
        )

        if exp_order is None:
            exp_order = self._sort_experiment_names(list(s_lidar.index))

        # Automatic exclusion of 64 beams from the experiments
        if exclude_64beams:
            exp_order = [e for e in exp_order if "64" not in e.lower()]

        rows = []
        for exp in exp_order:
            val_l = s_lidar.get(exp, None)
            val_f = s_fusion.get(f"fusion_{exp}", s_fusion.get(exp, None))

            if val_l is not None and val_f is not None:
                delta_abs = val_f - val_l
                delta_rel = (delta_abs / val_l * 100.0) if val_l > 0 else 0.0

                rows.append({
                    "Experiment": exp,
                    "LiDAR-Only (%)": round(val_l, 2),
                    "Late Fusion (%)": round(val_f, 2),
                    "Absolute Δ (% pts)": round(delta_abs, 2),
                    "Relative Δ (%)": round(delta_rel, 2),
                })

        df_res = pd.DataFrame(rows)

        # Save automatically in Drive if save_dir is specified
        if save_dir and not df_res.empty:
            os.makedirs(save_dir, exist_ok=True)
            base_name = f"gain_matrix_{target_class.lower()}_{difficulty.lower()}"
            csv_path = os.path.join(save_dir, f"{base_name}.csv")
            md_path = os.path.join(save_dir, f"{base_name}.md")

            df_res.to_csv(csv_path, index=False)
            df_res.to_markdown(md_path, index=False)
            print(f" Gain Matrix saved to Drive: {csv_path} and {md_path}")

        return df_res

    def plot_fusion_vs_lidar_overlay(
        self,
        fusion_root: Optional[str] = None,
        exp_order: Optional[List[str]] = None,
        target_class: str = "Car",
        difficulty: str = "Moderate",
        title: str = "Robustness Comparison: LiDAR-Only vs Late Fusion",
        save_path: Optional[str] = None,
        exclude_64beams: bool = True,
    ):
        """Plots overlaid mAP40 degradation curves comparing LiDAR-Only against Late Fusion,
        filtering out 64-beam redundant experiments by default.
        """
        f_root = fusion_root or self.fusion_root
        if not f_root or not os.path.exists(f_root):
            raise FileNotFoundError(f"Fusion root directory not found: {f_root}")

        df_lidar = self.load_global_results(self.experiments_root)
        df_fusion = self.load_global_results(f_root)

        s_lidar = (
            df_lidar[
                (df_lidar["Class"] == target_class)
                & (df_lidar["Difficulty"] == difficulty)
            ]
            .set_index("Experiment")["mAP40"]
        )
        s_fusion = (
            df_fusion[
                (df_fusion["Class"] == target_class)
                & (df_fusion["Difficulty"] == difficulty)
            ]
            .set_index("Experiment")["mAP40"]
        )

        if exp_order is None:
            exp_order = self._sort_experiment_names(list(s_lidar.index))

        # Automatic exclusion of 64 beams from the experiments
        if exclude_64beams:
            exp_order = [e for e in exp_order if "64" not in e.lower()]

        y_lidar = [s_lidar.get(e, None) for e in exp_order]
        y_fusion = [
            s_fusion.get(f"fusion_{e}", s_fusion.get(e, None)) for e in exp_order
        ]

        plt.figure(figsize=(10, 5.5))
        plt.plot(
            exp_order,
            y_lidar,
            marker="o",
            linewidth=2.5,
            color="crimson",
            label="LiDAR Only (PointPillars)",
        )
        plt.plot(
            exp_order,
            y_fusion,
            marker="s",
            linewidth=2.5,
            color="mediumseagreen",
            label="LiDAR + Camera (Late Fusion)",
        )

        plt.title(
            f"{title} - Class: {target_class} ({difficulty})",
            fontsize=12,
            fontweight="bold",
        )
        plt.xlabel("Experiment / Subsampling Level", fontsize=11)
        plt.ylabel("mAP40 (%)", fontsize=11)
        plt.ylim(-5, 105)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(fontsize=10, loc="best")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f" Overlay plot saved to: {save_path}")

        plt.show()

    def plot_degradation_curves(
        self,
        exp_order: Optional[List[str]] = None,
        target_class: str = "Car",
        title: str = "Overall mAP40 Performance Degradation",
        save_path: Optional[str] = None,
    ):
        """Plots global mAP40 degradation curves across experiment configurations."""
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
        """Plots distance-range mAP40 degradation curves (Near, Medium, Far)."""
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