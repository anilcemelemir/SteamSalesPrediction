# Gerekli kütüphaneleri yükleme
import pandas as pd
import numpy as np

# Veri setini yükleme
data_path = r"C:\Users\anil_\Documents\GitHub\SteamSalesPrediction\data\games.csv"
df = pd.read_csv(data_path)

# Veri setinin boyutlarını kontrol etme
print(f"Veri seti boyutları: {df.shape}\n")

# Sütun isimlerini ve türlerini görüntüleme
print("Sütunlar ve türleri:")
print(df.dtypes, "\n")

# Eksik veri analizi
print("Eksik veri analizi:")
print(df.isnull().sum(), "\n")

# Sütun bazında özet istatistikler (sayısal sütunlar)
print("Sayısal sütunların özet istatistikleri:")
print(df.describe(), "\n")

# Kategorik sütunlar için benzersiz değerler
print("Kategorik sütunların benzersiz değerleri:")
categorical_columns = df.select_dtypes(include=['object']).columns
for col in categorical_columns:
    print(f"{col}: {df[col].nunique()} benzersiz değer")

# İlk 5 satırı görüntüleme (örnek veri)
print("\nVeri setinin ilk 5 satırı:")
print(df.head())
# Gerekli kütüphaneleri yükleme
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

velihan özge sdlcöasdlşcöasdşlc