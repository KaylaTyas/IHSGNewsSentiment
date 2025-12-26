import subprocess, sys, time

PY = sys.executable
STEPS = [
    ("scrape", f"{PY} scripts/scrape_news.py"),
    ("process", f"{PY} scripts/process_news_batch.py"),
    ("combine", f"{PY} scripts/combine_features.py"),
    ("predict", f"{PY} scripts/predict_ihsg.py"),
]

def run_step(name, cmd):
    print(f"\n=== STEP: {name.upper()} ===")
    start = time.time()
    try:
        subprocess.run(cmd, shell=True, check=True)
        print(f"{name} finished in {time.time()-start:.1f}s")
    except subprocess.CalledProcessError as e:
        print(f"{name} FAILED")
        sys.exit(e.returncode)

if __name__ == "__main__":
    for name, cmd in STEPS:
        run_step(name, cmd)
    print("\n PIPELINE COMPLETED SUCCESSFULLY")
