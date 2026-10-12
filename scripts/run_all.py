"""End-to-end pipeline runner for Duplicate Question Retrieval."""
import os
import sys
import subprocess

SCRIPTS = [
    ("Download & Validate Dataset", "scripts/01_download_data.py"),
    ("Build Corpus & Splits", "scripts/02_build_corpus.py"),
    ("Build Retrieval Indexes", "scripts/03_build_indexes.py"),
    ("Run Evaluation Benchmark", "scripts/04_evaluate.py"),
]


def run_command(desc, script):
    print(f"\n=======================================================")
    print(f"Executing: {desc} ({script})")
    print(f"=======================================================")
    res = subprocess.run([sys.executable, script], check=True)
    if res.returncode != 0:
        print(f"Error executing {script}")
        sys.exit(res.returncode)


def main():
    for desc, script in SCRIPTS:
        run_command(desc, script)
    print("\n=======================================================")
    print("Full pipeline executed successfully!")
    print("You can now launch the Streamlit search interface with:")
    print("  streamlit run app.py")
    print("=======================================================\n")


if __name__ == "__main__":
    main()
