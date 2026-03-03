import csv

file_full = "results/submission_final_full.txt"
file_split = "results/submission_final.txt"
output_file = "results/submission_merged.txt"

def read_submission(path):
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            track_id, rest = line.split(",", 1)
            plate, conf = rest.split(";")
            data[track_id] = (plate, float(conf))
    return data


# Load 2 file
data_full = read_submission(file_full)
data_split = read_submission(file_split)

all_keys = sorted(data_full.keys())

diff_text = []
diff_conf = []
merged = {}

for k in all_keys:
    plate_f, conf_f = data_full[k]
    plate_s, conf_s = data_split[k]

    # Case 1: text giống nhau
    if plate_f == plate_s:
        # giữ cái có confidence cao hơn
        if conf_f >= conf_s:
            merged[k] = (plate_f, conf_f)
        else:
            merged[k] = (plate_s, conf_s)

        if abs(conf_f - conf_s) > 1e-6:
            diff_conf.append((k, plate_f, conf_f, conf_s))

    # Case 2: text khác nhau
    else:
        diff_text.append((k, plate_f, conf_f, plate_s, conf_s))

        # chọn cái có confidence cao hơn
        if conf_f >= conf_s:
            merged[k] = (plate_f, conf_f)
        else:
            merged[k] = (plate_s, conf_s)


# ---- Print summary ----
print("="*60)
print("📊 SUMMARY")
print("="*60)
print(f"Total tracks: {len(all_keys)}")
print(f"Text different: {len(diff_text)}")
print(f"Only confidence different: {len(diff_conf)}")

print("\n--- TEXT DIFFERENCES (first 20) ---")
for item in diff_text[:100]:
    k, pf, cf, ps, cs = item
    print(f"{k}: FULL={pf} ({cf:.4f}) | SPLIT={ps} ({cs:.4f}) | diff={abs(cf-cs):.4f}")

print("\n--- CONF DIFFERENCES (first 20) ---")
for item in diff_conf[:20]:
    k, p, cf, cs = item
    print(f"{k}: {p} | FULL={cf:.4f} | SPLIT={cs:.4f} | diff={abs(cf-cs):.4f}")


# ---- Save merged file ----
with open(output_file, "w", encoding="utf-8") as f:
    for k in all_keys:
        plate, conf = merged[k]
        f.write(f"{k},{plate};{conf:.4f}\n")

print("\n✅ Merged file saved to:", output_file)