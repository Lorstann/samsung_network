import pandas as pd
import joblib
import json
import requests
import time
import warnings
from datetime import datetime, timezone

warnings.filterwarnings('ignore')

MODEL_PATH = 'xgboost_ensemble_model.pkl'
PARQUET_PATH = 'NF-ToN-IoT-V2.parquet'
ES_URL = "http://localhost:9200/iot-ids-logs/_doc"
LIMIT_ROWS = 20000

# Model sinif eslesmesi
LABEL_MAP = {
    0: 'BACKDOOR',
    1: 'BENIGN',
    2: 'DDOS',
    3: 'DOS',
    4: 'INJECTION',
    5: 'PASSWORD',
    6: 'RANSOMWARE',
    7: 'SCANNING',
    8: 'XSS'
}


print("=" * 50)
print("       IOT IDS - SALDIRI TESPIT SISTEMI")
print("=" * 50)

# 1. Model Yukle
print("\n[1/5] Model yukleniyor...")
model = joblib.load(MODEL_PATH)
print("      Model yuklendi: XGBoost (Multi-class)")
print(f"      Siniflar: {list(LABEL_MAP.values())}")

# 2. Veri Yukle
print(f"\n[2/5] Veri yukleniyor ({LIMIT_ROWS} satir)...")
df = pd.read_parquet(PARQUET_PATH).head(LIMIT_ROWS)
print(f"      Veri yuklendi: {len(df)} satir")

# 3. Feature Ayir
FEATURE_COLUMNS = [
    'L4_SRC_PORT', 'L4_DST_PORT', 'PROTOCOL', 'L7_PROTO', 'IN_BYTES',
    'IN_PKTS', 'OUT_BYTES', 'OUT_PKTS', 'TCP_FLAGS', 'CLIENT_TCP_FLAGS',
    'SERVER_TCP_FLAGS', 'FLOW_DURATION_MILLISECONDS', 'DURATION_IN',
    'DURATION_OUT', 'MIN_TTL', 'MAX_TTL', 'LONGEST_FLOW_PKT',
    'SHORTEST_FLOW_PKT', 'MIN_IP_PKT_LEN', 'MAX_IP_PKT_LEN',
    'SRC_TO_DST_SECOND_BYTES', 'DST_TO_SRC_SECOND_BYTES',
    'RETRANSMITTED_IN_BYTES', 'RETRANSMITTED_IN_PKTS',
    'RETRANSMITTED_OUT_BYTES', 'RETRANSMITTED_OUT_PKTS',
    'SRC_TO_DST_AVG_THROUGHPUT', 'DST_TO_SRC_AVG_THROUGHPUT',
    'NUM_PKTS_UP_TO_128_BYTES', 'NUM_PKTS_128_TO_256_BYTES',
    'NUM_PKTS_256_TO_512_BYTES', 'NUM_PKTS_512_TO_1024_BYTES',
    'NUM_PKTS_1024_TO_1514_BYTES', 'TCP_WIN_MAX_IN', 'TCP_WIN_MAX_OUT',
    'ICMP_TYPE', 'ICMP_IPV4_TYPE', 'DNS_QUERY_ID', 'DNS_QUERY_TYPE',
    'DNS_TTL_ANSWER', 'FTP_COMMAND_RET_CODE'
]

X = df[FEATURE_COLUMNS].values
gercek_labels = df['Attack'].values

# 5. Model Tahmin Yap
print("\n[3/5] Model tahmin yapiyor...")
y_pred = model.predict(X)

# Tahminleri isme cevir
tahmin_labels = [LABEL_MAP[p] for p in y_pred]

# 6. SIEM'e Gonder (BENIGN disindaki tahminler)
print("[4/5] SIEM'e gonderiliyor (BENIGN haric)...\n")
headers = {"Content-Type": "application/json"}
sent_count = 0

for i in range(len(y_pred)):
    model_tahmin = tahmin_labels[i]
    gercek_label = gercek_labels[i]

    # Model BENIGN demediyse (saldiri tespit ettiyse)
    if model_tahmin != 'BENIGN':
        sent_count += 1

        # Dogru mu yanlis mi?
        if model_tahmin.upper() == gercek_label.upper():
            sonuc = "DOGRU"
        else:
            sonuc = f"YANLIS (Gercek: {gercek_label})"

        alert = {
            "@timestamp": datetime.now(timezone.utc).isoformat(),
            "src_port": int(df.iloc[i]['L4_SRC_PORT']),
            "dst_port": int(df.iloc[i]['L4_DST_PORT']),
            "protocol": int(df.iloc[i]['PROTOCOL']),
            "in_bytes": int(df.iloc[i]['IN_BYTES']),
            "out_bytes": int(df.iloc[i]['OUT_BYTES']),
            "model_prediction": model_tahmin,
            "real_label": gercek_label,
            "is_correct": model_tahmin.upper() == gercek_label.upper(),
            "severity": "High",
            "model": "XGBoost"
        }

        try:
            requests.post(ES_URL, headers=headers, data=json.dumps(alert), timeout=1)
            es_status = "ES OK"
        except:
            es_status = "ES YOK"

        if sent_count <= 30:
            print(f"  [{sent_count:3d}] Model: {model_tahmin:12s} | Gercek: {gercek_label:12s} | {sonuc:25s} | {es_status}")
        elif sent_count == 31:
            print(f"  ... ve daha fazlasi gonderiliyor")

        time.sleep(0.002)

# 7. Model vs Gercek Karsilastirma
print("\n" + "=" * 50)
print("     MODEL vs GERCEK KARSILASTIRMA")
print("=" * 50)

# Her sinif icin dogru tahmin sayisi
print("\n  Sinif Bazinda Performans:")
print("  " + "-" * 45)
total_correct = 0
total_count = 0

for label_id, label_name in LABEL_MAP.items():
    # Bu sinifin gercek sayisi
    gercek_mask = [g.upper() == label_name for g in gercek_labels]
    gercek_count = sum(gercek_mask)

    # Model bu sinifi kac kere dogru tahmin etti
    dogru_tahmin = sum(1 for i in range(len(y_pred))
                       if tahmin_labels[i] == label_name
                       and gercek_labels[i].upper() == label_name)

    total_correct += dogru_tahmin
    total_count += gercek_count

    if gercek_count > 0:
        accuracy = dogru_tahmin / gercek_count * 100
        print(f"  {label_name:12s} : {dogru_tahmin:5d} / {gercek_count:5d} dogru (%{accuracy:.1f})")

print("  " + "-" * 45)
overall_accuracy = total_correct / total_count * 100 if total_count > 0 else 0
print(f"  GENEL ACCURACY: %{overall_accuracy:.2f}")

# Saldiri tespiti ozeti
saldiri_tahmin = sum(1 for t in tahmin_labels if t != 'BENIGN')
benign_tahmin = sum(1 for t in tahmin_labels if t == 'BENIGN')

print(f"""
  Ozet:
  -----
  Toplam veri             : {len(y_pred)}
  Model saldiri dedi      : {saldiri_tahmin} (SIEM'e gonderildi)
  Model BENIGN dedi       : {benign_tahmin}
  Dogru tahmin toplam     : {total_correct}
""")

print("=" * 50)
print("                 TAMAMLANDI")
print("=" * 50)
