# ======================================================
# Parallel Cellpose Morphology Metrics (3 Datasets)


import os
import cv2
import numpy as np
import pandas as pd
from skimage.measure import regionprops
from cellpose.models import CellposeModel
from concurrent.futures import ProcessPoolExecutor, as_completed


# -----------------------------
# User configuration
# -----------------------------
DATASETS = {
    "Dataset_A": "workspace/Project_Datasets_Thato/Unstained/Test"
    "Dataset_B": 
    "Dataset_C": "workspace/Project_Datasets_Thato/IHC/Test"
}
OUTPUT_CSV = "morphology_metrics_all.csv"
SUMMARY_CSV = "morphology_summary_per_dataset.csv"
MODEL_TYPE = "nuclei"  # 'cyto' or 'nuclei'

# -----------------------------
# Helper function: circularity
# -----------------------------
def calc_circularity(area, perimeter):
    if perimeter == 0:
        return 0
    return (4 * np.pi * area) / (perimeter ** 2)


# -----------------------------
# Core function per dataset
# -----------------------------
def analyze_dataset(dataset_name, folder_path, model_type):
    """Compute morphology metrics for one dataset folder."""
    print(f"\n🧠 Starting analysis for {dataset_name}...")

    # Each process loads its own model (avoids GPU conflicts)
    model = CellposeModel(gpu=True, model_type=model_type)
    dataset_results = []

    for filename in os.listdir(folder_path):
        if not filename.lower().endswith((".png", ".jpg", ".jpeg", ".tif")):
            continue

        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)

        if img is None:
            print(f"⚠️ {dataset_name}: Skipping unreadable image: {filename}")
            continue

        # Run Cellpose inference
        masks, flows, styles, diams = model.eval(img, diameter=None, channels=[0, 0])

        # Extract morphology metrics
        props = regionprops(masks)
        if not props:
            print(f"⚠️ {dataset_name}: No cells detected in {filename}")
            continue

        areas = [p.area for p in props]
        eccs = [p.eccentricity for p in props]
        circs = [calc_circularity(p.area, p.perimeter) for p in props if p.perimeter > 0]

        dataset_results.append({
            "dataset": dataset_name,
            "image": filename,
            "cell_count": len(areas),
            "avg_area": np.mean(areas),
            "avg_eccentricity": np.mean(eccs),
            "avg_circularity": np.mean(circs),
            "std_area": np.std(areas),
            "std_circularity": np.std(circs),
        })

    print(f"✅ Finished {dataset_name}: {len(dataset_results)} images processed.")
    return dataset_name, dataset_results


# -----------------------------
# Run all datasets in parallel
# -----------------------------
if __name__ == "__main__":
    all_results = []
    summaries = []

    with ProcessPoolExecutor(max_workers=len(DATASETS)) as executor:
        futures = {
            executor.submit(analyze_dataset, name, path, MODEL_TYPE): name
            for name, path in DATASETS.items()
        }

        for future in as_completed(futures):
            dataset_name = futures[future]
            try:
                ds_name, ds_results = future.result()
                all_results.extend(ds_results)

                # Create per-dataset summary
                df = pd.DataFrame(ds_results)
                summary = {
                    "dataset": ds_name,
                    "mean_cell_count": df["cell_count"].mean(),
                    "mean_area": df["avg_area"].mean(),
                    "mean_eccentricity": df["avg_eccentricity"].mean(),
                    "mean_circularity": df["avg_circularity"].mean(),
                }
                summaries.append(summary)

            except Exception as e:
                print(f"❌ Error processing {dataset_name}: {e}")

    # -----------------------------
    # Save all results
    # -----------------------------
    if all_results:
        df_all = pd.DataFrame(all_results)
        df_all.to_csv(OUTPUT_CSV, index=False)

        df_summary = pd.DataFrame(summaries)
        df_summary.to_csv(SUMMARY_CSV, index=False)

        print("\n🏁 All datasets processed successfully!")
        print(f"📊 Full per-image metrics: {OUTPUT_CSV}")
        print(f"📈 Summary per dataset: {SUMMARY_CSV}")
    else:
        print("⚠️ No valid images were processed.")
