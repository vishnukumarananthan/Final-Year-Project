"""
Final Project Run Script — Convenience launcher
Generates dataset, trains models, and launches the Streamlit app in one command.
"""
import os
import subprocess
import sys

def run():
    base = os.path.dirname(os.path.abspath(__file__))
    python = sys.executable

    print("=" * 60)
    print("  Smart Electricity Anomaly Detection System — Launcher")
    print("=" * 60)

    # Step 1: Generate dataset if missing
    csv_path = os.path.join(base, "dataset", "electricity_data.csv")
    if not os.path.exists(csv_path):
        print("[1/3] Generating synthetic benchmark dataset...")
        subprocess.run([python, os.path.join(base, "dataset", "generate_dataset.py")], check=True)
    else:
        print(f"[1/3] Dataset exists: {csv_path}")

    # Step 2: Train models if missing
    metadata_path = os.path.join(base, "models", "model_metadata.joblib")
    if not os.path.exists(metadata_path):
        print("[2/3] Training Isolation Forest models...")
        subprocess.run([python, os.path.join(base, "train_model.py")], check=True)
    else:
        print("[2/3] Trained models found. Skipping retraining.")

    # Step 3: Launch Streamlit
    print("[3/3] Launching Streamlit application at http://localhost:8501 ...")
    print("      Press Ctrl+C to stop.\n")
    env = os.environ.copy()
    env["HOME"] = base  # Use project dir as HOME for streamlit credentials
    streamlit_bin = os.path.join(base, ".venv", "bin", "streamlit")
    if not os.path.exists(streamlit_bin):
        # Windows path
        streamlit_bin = os.path.join(base, ".venv", "Scripts", "streamlit.exe")
    if not os.path.exists(streamlit_bin):
        # Fallback: use system streamlit
        streamlit_bin = "streamlit"

    subprocess.run(
        [streamlit_bin, "run", os.path.join(base, "app.py"), "--server.headless", "true"],
        env=env,
        cwd=base,
        check=True
    )

if __name__ == "__main__":
    run()
