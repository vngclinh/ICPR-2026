import os, sys, time, subprocess

PROJECT = r"C:\Users\OS\Downloads\ICPR-2026\MultiFrame-LPR"
checkpoint = os.path.join(PROJECT, "results", "restran_sr_v7_best.pth")
log_path = os.path.join(PROJECT, "results", "sr_v7_cpu_eval_watcher.log")
output_csv = os.path.join(PROJECT, "results", "test_predictions_restran_sr_v7_cpu.csv")
python = os.path.join(PROJECT, ".venv", "Scripts", "python.exe")
eval_script = os.path.join(PROJECT, "eval_test_cpu.py")

def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}\n"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line)
    print(line, end="", flush=True)

log("sr_v7 watcher started.")
last_mtime = None

while True:
    if os.path.exists(checkpoint):
        mtime = os.path.getmtime(checkpoint)
        if mtime != last_mtime:
            last_mtime = mtime
            ts = time.strftime("%H:%M:%S", time.localtime(mtime))
            log(f"Checkpoint updated (saved at {ts}), launching CPU eval...")
            cmd = [python, eval_script, checkpoint, output_csv, "feed_hr"]
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT)
            for line in (result.stdout + result.stderr).strip().splitlines():
                log(line)
    time.sleep(30)
