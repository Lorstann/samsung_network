import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
import warnings
warnings.filterwarnings('ignore')

# Türkçe karakter desteği ve görsel ayarlar
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10
sns.set_style("whitegrid")
sns.set_palette("husl")

# Parquet dosyasını oku
print("="*60)
print("NF-ToN-IoT-V2 Parquet Dosyası Analizi")
print("="*60)
print("\nParquet dosyası okunuyor...")
df = pd.read_parquet('NF-ToN-IoT-V2.parquet')

print(f"\n✓ Dosya başarıyla okundu!")
print(f"  - Toplam kayıt sayısı: {len(df):,}")
print(f"  - Toplam özellik sayısı: {len(df.columns)}")
print(f"  - Dosya boyutu (bellek): {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# Sütun isimlerini göster
print(f"\n{'='*60}")
print("Sütun İsimleri:")
print(f"{'='*60}")
for i, col in enumerate(df.columns, 1):
    print(f"{i:2d}. {col}")

# Veri tipleri ve eksik değerler
print(f"\n{'='*60}")
print("Veri Kalitesi Analizi:")
print(f"{'='*60}")
print(f"\nVeri Tipleri:")
print(df.dtypes.value_counts())

print(f"\nEksik Değerler:")
missing = df.isnull().sum()
if missing.sum() > 0:
    print(missing[missing > 0])
else:
    print("✓ Eksik değer yok!")

# Label ve Attack sütunlarını bul
label_cols = [col for col in df.columns if 'label' in col.lower()]
attack_cols = [col for col in df.columns if 'attack' in col.lower()]

print(f"\n{'='*60}")
print("Label ve Attack Sütunları:")
print(f"{'='*60}")
print(f"Label sütunları: {label_cols}")
print(f"Attack sütunları: {attack_cols}")

# 1. LABEL DAĞILIMI (Normal vs Attack)
if label_cols:
    label_col = label_cols[0]
    print(f"\n{'='*60}")
    print(f"1. LABEL DAĞILIMI ANALİZİ ({label_col})")
    print(f"{'='*60}")
    
    label_counts = df[label_col].value_counts()
    label_percentages = df[label_col].value_counts(normalize=True) * 100
    
    print("\nLabel Dağılımı:")
    for label, count in label_counts.items():
        print(f"  {label}: {count:,} ({label_percentages[label]:.2f}%)")
    
    # Pie Chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Pie chart
    colors = sns.color_palette("Set2", len(label_counts))
    wedges, texts, autotexts = ax1.pie(label_counts.values, 
                                       labels=label_counts.index,
                                       autopct='%1.2f%%',
                                       colors=colors,
                                       startangle=90,
                                       textprops={'fontsize': 12, 'weight': 'bold'})
    ax1.set_title(f'Label Dağılımı (Pasta Grafiği)\n{label_col}', 
                  fontsize=14, fontweight='bold', pad=20)
    
    # Bar chart
    bars = ax2.bar(range(len(label_counts)), label_counts.values, color=colors)
    ax2.set_xticks(range(len(label_counts)))
    ax2.set_xticklabels(label_counts.index, rotation=45, ha='right')
    ax2.set_ylabel('Kayıt Sayısı', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Label', fontsize=12, fontweight='bold')
    ax2.set_title(f'Label Dağılımı (Sütun Grafiği)\n{label_col}', 
                  fontsize=14, fontweight='bold', pad=20)
    ax2.grid(axis='y', alpha=0.3)
    
    # Değerleri bar üzerine yaz
    for i, (bar, count) in enumerate(zip(bars, label_counts.values)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{count:,}\n({label_percentages.iloc[i]:.2f}%)',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('Label_Distribution.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Grafik kaydedildi: Label_Distribution.png")
    plt.close()

# 2. ATTACK TYPE DAĞILIMI
if attack_cols:
    attack_col = attack_cols[0]
    print(f"\n{'='*60}")
    print(f"2. ATTACK TYPE DAĞILIMI ANALİZİ ({attack_col})")
    print(f"{'='*60}")
    
    attack_counts = df[attack_col].value_counts()
    attack_percentages = df[attack_col].value_counts(normalize=True) * 100
    
    print("\nAttack Type Dağılımı:")
    for attack, count in attack_counts.items():
        print(f"  {attack}: {count:,} ({attack_percentages[attack]:.2f}%)")
    
    # Horizontal bar chart (daha okunabilir)
    fig, ax = plt.subplots(figsize=(14, max(8, len(attack_counts) * 0.5)))
    colors = sns.color_palette("viridis", len(attack_counts))
    bars = ax.barh(range(len(attack_counts)), attack_counts.values, color=colors)
    
    ax.set_yticks(range(len(attack_counts)))
    ax.set_yticklabels(attack_counts.index, fontsize=11)
    ax.set_xlabel('Kayıt Sayısı', fontsize=12, fontweight='bold')
    ax.set_ylabel('Attack Type', fontsize=12, fontweight='bold')
    ax.set_title(f'Attack Type Dağılımı\n{attack_col}', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3)
    
    # Değerleri bar üzerine yaz
    for i, (bar, count) in enumerate(zip(bars, attack_counts.values)):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2.,
                f' {count:,} ({attack_percentages.iloc[i]:.2f}%)',
                ha='left', va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('Attack_Type_Distribution.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Grafik kaydedildi: Attack_Type_Distribution.png")
    plt.close()

# 3. NUMERİK FEATURE DAĞILIMLARI (İlk 12 önemli feature)
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
# Label ve attack sütunlarını çıkar
numeric_cols = [col for col in numeric_cols if col not in label_cols + attack_cols]

if len(numeric_cols) > 0:
    print(f"\n{'='*60}")
    print(f"3. NUMERİK FEATURE DAĞILIMLARI")
    print(f"{'='*60}")
    print(f"Toplam {len(numeric_cols)} numerik feature bulundu.")
    
    # İlk 12 feature'ı göster
    n_features = min(12, len(numeric_cols))
    selected_features = numeric_cols[:n_features]
    
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    axes = axes.flatten()
    
    for idx, feature in enumerate(selected_features):
        ax = axes[idx]
        data = df[feature].dropna()
        
        # Histogram
        ax.hist(data, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        ax.set_title(f'{feature}\n(Mean: {data.mean():.2f}, Std: {data.std():.2f})', 
                     fontsize=10, fontweight='bold')
        ax.set_xlabel('Değer', fontsize=9)
        ax.set_ylabel('Frekans', fontsize=9)
        ax.grid(alpha=0.3)
    
    plt.suptitle('NetFlow Feature Dağılımları (İlk 12 Feature)', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('Feature_Distributions.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Grafik kaydedildi: Feature_Distributions.png")
    plt.close()

# 4. LABEL'A GÖRE FEATURE KARŞILAŞTIRMASI (Box Plot)
if label_cols and len(numeric_cols) > 0:
    print(f"\n{'='*60}")
    print(f"4. LABEL'A GÖRE FEATURE KARŞILAŞTIRMASI")
    print(f"{'='*60}")
    
    label_col = label_cols[0]
    # İlk 6 feature'ı karşılaştır
    n_compare = min(6, len(numeric_cols))
    selected_features = numeric_cols[:n_compare]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, feature in enumerate(selected_features):
        ax = axes[idx]
        data_to_plot = [df[df[label_col] == label][feature].dropna() 
                        for label in df[label_col].unique()]
        
        bp = ax.boxplot(data_to_plot, labels=df[label_col].unique(), 
                       patch_artist=True, showmeans=True)
        
        # Renklendir
        colors_box = sns.color_palette("Set2", len(bp['boxes']))
        for patch, color in zip(bp['boxes'], colors_box):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_title(f'{feature}', fontsize=11, fontweight='bold')
        ax.set_ylabel('Değer', fontsize=10)
        ax.set_xlabel('Label', fontsize=10)
        ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Label\'a Göre Feature Dağılımları (Box Plot)', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('Feature_Comparison_by_Label.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Grafik kaydedildi: Feature_Comparison_by_Label.png")
    plt.close()

# 5. KORELASYON MATRİSİ (İlk 15 numerik feature)
if len(numeric_cols) > 1:
    print(f"\n{'='*60}")
    print(f"5. KORELASYON MATRİSİ")
    print(f"{'='*60}")
    
    n_corr = min(15, len(numeric_cols))
    selected_features = numeric_cols[:n_corr]
    corr_matrix = df[selected_features].corr()
    
    plt.figure(figsize=(16, 14))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Üst üçgen maskesi
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
                cmap='coolwarm', center=0, square=True, linewidths=0.5,
                cbar_kws={"shrink": 0.8}, vmin=-1, vmax=1,
                xticklabels=selected_features, yticklabels=selected_features)
    plt.title(f'NetFlow Feature Korelasyon Matrisi (İlk {n_corr} Feature)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig('Correlation_Matrix.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Grafik kaydedildi: Correlation_Matrix.png")
    plt.close()

# 6. İSTATİSTİKSEL ÖZET
print(f"\n{'='*60}")
print("6. İSTATİSTİKSEL ÖZET")
print(f"{'='*60}")
print("\nNumerik Feature İstatistikleri:")
print(df[numeric_cols].describe().T)

# Özet rapor
print(f"\n{'='*60}")
print("ANALİZ TAMAMLANDI!")
print(f"{'='*60}")
print("\nOluşturulan Grafikler:")
if label_cols:
    print("  ✓ Label_Distribution.png")
if attack_cols:
    print("  ✓ Attack_Type_Distribution.png")
if len(numeric_cols) > 0:
    print("  ✓ Feature_Distributions.png")
    if label_cols:
        print("  ✓ Feature_Comparison_by_Label.png")
    print("  ✓ Correlation_Matrix.png")

print(f"\n{'='*60}")
