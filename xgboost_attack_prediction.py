import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import xgboost as xgb
import warnings
import time
warnings.filterwarnings('ignore')

# Görsel ayarlar
plt.rcParams['figure.figsize'] = (14, 8)
sns.set_style("whitegrid")

print("="*70)
print("XGBoost ile Attack Type Prediction - NF-ToN-IoT-V2")
print("="*70)

# 1. VERİYİ YÜKLE
print("\n[1/7] Parquet dosyası okunuyor...")
start_time = time.time()
df = pd.read_parquet('NF-ToN-IoT-V2.parquet')
load_time = time.time() - start_time
print(f"✓ Veri yüklendi! ({load_time:.2f} saniye)")
print(f"  - Toplam kayıt: {len(df):,}")
print(f"  - Toplam feature: {len(df.columns)}")

# 2. HEDEF DEĞİŞKENİ HAZIRLA
print("\n[2/7] Hedef değişken hazırlanıyor...")
target = df['Attack'].copy()

# Attack type'ları kontrol et ve düzelt
print("\nAttack type dağılımı:")
attack_counts = target.value_counts()
for attack, count in attack_counts.items():
    print(f"  {attack}: {count:,}")

# Attack type'ları standartlaştır (büyük/küçük harf farkını düzelt)
target = target.str.upper()
target = target.str.strip()

# Olası yazım hatalarını düzelt
attack_mapping = {
    'SCANNING': 'SCANNING',
    'XSS': 'XSS',
    'DDOS': 'DDOS',
    'DDoS': 'DDOS',
    'PASSWORD': 'PASSWORD',
    'INJECTION': 'INJECTION',
    'INJEJCTION': 'INJECTION',  # Yazım hatası düzeltme
    'DOS': 'DOS',
    'DoS': 'DOS',
    'BACKDOOR': 'BACKDOOR',
    'BACKDOOT': 'BACKDOOR',  # Yazım hatası düzeltme
    'MITM': 'MITM',
    'RANSOMWARE': 'RANSOMWARE',
    'BENIGN': 'BENIGN'
}

target = target.map(attack_mapping).fillna(target)

print(f"\n✓ Hedef değişken hazırlandı!")
print(f"  - Benzersiz attack type sayısı: {target.nunique()}")
print(f"\nFinal attack type dağılımı:")
final_counts = target.value_counts()
for attack, count in final_counts.items():
    print(f"  {attack}: {count:,} ({count/len(target)*100:.2f}%)")

# Target'ı numerik değerlere çevir (XGBoost için gerekli)
target_encoder = LabelEncoder()
target_encoded = pd.Series(target_encoder.fit_transform(target), index=target.index)

# Label mapping'i kaydet (sonuçları göstermek için)
label_mapping = dict(zip(target_encoder.classes_, range(len(target_encoder.classes_))))
reverse_label_mapping = {v: k for k, v in label_mapping.items()}

print(f"\n✓ Target encode edildi!")
print(f"  - Label mapping:")
for label, code in sorted(label_mapping.items(), key=lambda x: x[1]):
    print(f"    {code}: {label}")

# Target'ı encoded versiyonla değiştir
target = target_encoded

# 3. FEATURE'LARI HAZIRLA
print("\n[3/7] Feature'lar hazırlanıyor...")
# Label ve Attack sütunlarını çıkar
X = df.drop(['Label', 'Attack'], axis=1)

# Kategorik değişkenleri kontrol et
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
print(f"  - Kategorik sütunlar: {categorical_cols}")

# Kategorik değişkenleri encode et
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le
    print(f"  ✓ {col} encode edildi ({X[col].nunique()} benzersiz değer)")

# Numerik değişkenleri kontrol et
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
print(f"  - Numerik sütun sayısı: {len(numeric_cols)}")

# Sonsuz değerleri ve çok büyük değerleri temizle
print("\n  - Sonsuz ve çok büyük değerler temizleniyor...")
inf_cols = []
large_value_cols = []

# XGBoost için maksimum değer limiti (float32 max değerinden küçük)
max_safe_value = np.finfo(np.float32).max / 10  # Güvenli bir limit

for col in X.columns:
    if X[col].dtype in ['int64', 'int32', 'int16', 'int8', 'float64', 'float32']:
        # Sonsuz değerleri kontrol et
        inf_mask = np.isinf(X[col]) | np.isneginf(X[col])
        if inf_mask.any():
            inf_cols.append(col)
            inf_count = inf_mask.sum()
            print(f"    ⚠ {col}: {inf_count:,} sonsuz değer bulundu")
            # Sonsuz değerleri NaN'a çevir
            X.loc[inf_mask, col] = np.nan
        
        # Çok büyük değerleri kontrol et (float64 için)
        if X[col].dtype in ['float64', 'float32']:
            large_mask = np.abs(X[col]) > max_safe_value
            if large_mask.any():
                large_value_cols.append(col)
                large_count = large_mask.sum()
                print(f"    ⚠ {col}: {large_count:,} çok büyük değer bulundu")
                # Çok büyük değerleri NaN'a çevir
                X.loc[large_mask, col] = np.nan

if inf_cols or large_value_cols:
    print(f"\n  ⚠ Toplam {len(set(inf_cols + large_value_cols))} sütunda sorunlu değer bulundu")
    print(f"    - Sonsuz değer içeren: {len(inf_cols)} sütun")
    print(f"    - Çok büyük değer içeren: {len(large_value_cols)} sütun")

# NaN değerleri doldur
print("\n  - NaN değerler dolduruluyor...")
for col in X.columns:
    if X[col].isnull().sum() > 0:
        null_count = X[col].isnull().sum()
        if X[col].dtype in ['int64', 'int32', 'int16', 'int8', 'float64', 'float32']:
            # Numerik sütunlar için median kullan
            median_val = X[col].median()
            if pd.isna(median_val):
                median_val = 0  # Eğer median da NaN ise 0 kullan
            X[col].fillna(median_val, inplace=True)
            if null_count > 0:
                print(f"    ✓ {col}: {null_count:,} NaN değer median ile dolduruldu")
        else:
            # Kategorik sütunlar için mode kullan
            mode_val = X[col].mode()[0] if len(X[col].mode()) > 0 else 0
            X[col].fillna(mode_val, inplace=True)
            if null_count > 0:
                print(f"    ✓ {col}: {null_count:,} NaN değer mode ile dolduruldu")

# Son kontrol: Hala inf veya çok büyük değer var mı?
print("\n  - Final kontrol yapılıyor...")
for col in X.columns:
    if X[col].dtype in ['float64', 'float32']:
        if np.isinf(X[col]).any() or (np.abs(X[col]) > max_safe_value).any():
            print(f"    ⚠ UYARI: {col} hala sorunlu değerler içeriyor!")
            # Son çare: çok büyük değerleri kırp
            X[col] = np.clip(X[col], -max_safe_value, max_safe_value)
            X[col] = X[col].replace([np.inf, -np.inf], X[col].median())

print(f"  ✓ Veri temizleme tamamlandı!")

print(f"✓ Feature'lar hazırlandı!")
print(f"  - Toplam feature sayısı: {X.shape[1]}")

# 4. TRAIN/TEST SPLIT
print("\n[4/7] Train/Test split yapılıyor...")
# Veri çok büyük olduğu için stratified split kullanıyoruz
X_train, X_test, y_train, y_test = train_test_split(
    X, target, 
    test_size=0.2, 
    random_state=42, 
    stratify=target
)

print(f"✓ Split tamamlandı!")
print(f"  - Train set: {len(X_train):,} kayıt")
print(f"  - Test set: {len(X_test):,} kayıt")

# Split sonrası son kontrol
print("\n  - Train/Test set'lerde son kontrol yapılıyor...")
for dataset_name, dataset in [("Train", X_train), ("Test", X_test)]:
    for col in dataset.columns:
        if dataset[col].dtype in ['float64', 'float32']:
            if np.isinf(dataset[col]).any():
                inf_count = np.isinf(dataset[col]).sum()
                print(f"    ⚠ {dataset_name} - {col}: {inf_count} sonsuz değer, temizleniyor...")
                dataset[col] = dataset[col].replace([np.inf, -np.inf], dataset[col].median())
                if pd.isna(dataset[col].median()):
                    dataset[col] = dataset[col].fillna(0)
print("  ✓ Son kontrol tamamlandı!")

# 5. XGBOOST MODELİ OLUŞTUR
print("\n[5/7] XGBoost modeli oluşturuluyor...")

# Multi-class classification için XGBoost parametreleri
xgb_model = xgb.XGBClassifier(
    objective='multi:softprob',  # Multi-class classification
    n_estimators=200,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,  # Tüm CPU çekirdeklerini kullan
    tree_method='hist',  # Hızlı eğitim için
    eval_metric='mlogloss',
    verbosity=1,
    missing=np.nan  # NaN değerleri açıkça belirt
)

print(f"✓ Model oluşturuldu!")
print(f"  - Objective: multi:softprob")
print(f"  - Sınıf sayısı: {target.nunique()}")
print(f"  - Estimator sayısı: {xgb_model.n_estimators}")
print(f"  - Max depth: {xgb_model.max_depth}")

# 6. MODELİ EĞİT
print("\n[6/7] Model eğitiliyor...")
print("  (Bu işlem birkaç dakika sürebilir...)")

train_start = time.time()
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    verbose=50
)
train_time = time.time() - train_start

print(f"\n✓ Model eğitimi tamamlandı! ({train_time:.2f} saniye)")

# 7. TAHMİN VE DEĞERLENDİRME
print("\n[7/7] Model değerlendiriliyor...")

# Tahminler
y_pred = xgb_model.predict(X_test)
y_pred_proba = xgb_model.predict_proba(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\n{'='*70}")
print("MODEL PERFORMANSI")
print(f"{'='*70}")
print(f"\n✓ Test Accuracy: {accuracy*100:.4f}%")

# Classification Report
print(f"\n{'='*70}")
print("DETAYLI SINIFLANDIRMA RAPORU")
print(f"{'='*70}")
# Target names'ı reverse mapping'den al
target_names = [reverse_label_mapping[i] for i in sorted(target.unique())]
print("\n" + classification_report(y_test, y_pred, target_names=target_names))

# Precision, Recall, F1-Score
precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average=None, 
                                                                 labels=sorted(target.unique()))
print(f"\n{'='*70}")
print("SINIF BAZINDA METRİKLER")
print(f"{'='*70}")
print(f"\n{'Attack Type':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<12}")
print("-" * 70)
for i, class_code in enumerate(sorted(target.unique())):
    attack_type = reverse_label_mapping[class_code]
    print(f"{attack_type:<15} {precision[i]:<12.4f} {recall[i]:<12.4f} {f1[i]:<12.4f} {support[i]:<12,}")

# 8. CONFUSION MATRIX
print(f"\n{'='*70}")
print("CONFUSION MATRIX GÖRSELLEŞTİRMESİ")
print(f"{'='*70}")

cm = confusion_matrix(y_test, y_pred, labels=sorted(target.unique()))
# Label isimlerini reverse mapping'den al
cm_labels = [reverse_label_mapping[i] for i in sorted(target.unique())]
plt.figure(figsize=(14, 12))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=cm_labels,
            yticklabels=cm_labels,
            cbar_kws={'label': 'Kayıt Sayısı'})
plt.title('Confusion Matrix - Attack Type Prediction', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Tahmin Edilen', fontsize=12, fontweight='bold')
plt.ylabel('Gerçek', fontsize=12, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('Confusion_Matrix_XGBoost.png', dpi=300, bbox_inches='tight')
print("\n✓ Confusion Matrix kaydedildi: Confusion_Matrix_XGBoost.png")
plt.close()

# 9. FEATURE IMPORTANCE
print(f"\n{'='*70}")
print("FEATURE IMPORTANCE ANALİZİ")
print(f"{'='*70}")

feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\nTop 20 En Önemli Feature:")
print("-" * 70)
for idx, row in feature_importance.head(20).iterrows():
    print(f"{row['feature']:<35} {row['importance']:.6f}")

# Feature Importance Görselleştirme
plt.figure(figsize=(12, 10))
top_features = feature_importance.head(20)
sns.barplot(data=top_features, y='feature', x='importance', palette='viridis')
plt.title('Top 20 Feature Importance - XGBoost Model', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Importance Score', fontsize=12, fontweight='bold')
plt.ylabel('Feature', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('Feature_Importance_XGBoost.png', dpi=300, bbox_inches='tight')
print("\n✓ Feature Importance grafiği kaydedildi: Feature_Importance_XGBoost.png")
plt.close()

# 10. MODELİ KAYDET
print(f"\n{'='*70}")
print("MODEL KAYDEDİLİYOR")
print(f"{'='*70}")

import joblib
model_filename = 'xgboost_attack_model.pkl'
joblib.dump(xgb_model, model_filename)
print(f"\n✓ Model kaydedildi: {model_filename}")

# Label encoder'ı da kaydet
encoder_filename = 'target_encoder.pkl'
joblib.dump(target_encoder, encoder_filename)
print(f"✓ Label encoder kaydedildi: {encoder_filename}")

# Feature importance'ı da kaydet
feature_importance.to_csv('feature_importance.csv', index=False)
print(f"✓ Feature importance kaydedildi: feature_importance.csv")

# Özet
print(f"\n{'='*70}")
print("ANALİZ TAMAMLANDI!")
print(f"{'='*70}")
print(f"\nÖzet:")
print(f"  - Toplam kayıt: {len(df):,}")
print(f"  - Train set: {len(X_train):,}")
print(f"  - Test set: {len(X_test):,}")
print(f"  - Test Accuracy: {accuracy*100:.4f}%")
print(f"  - Eğitim süresi: {train_time:.2f} saniye")
print(f"\nOluşturulan dosyalar:")
print(f"  ✓ {model_filename}")
print(f"  ✓ feature_importance.csv")
print(f"  ✓ Confusion_Matrix_XGBoost.png")
print(f"  ✓ Feature_Importance_XGBoost.png")
print(f"\n{'='*70}")

