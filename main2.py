# Gerekli kütüphaneler
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Veri Yükleme
file_path = "SteamSalesPrediction\data\games_cleaned.csv"  # Datasetin tam yolunu buraya girin
data = pd.read_csv(file_path)

# 2. Gereksiz Sütunları Kaldırma
columns_to_drop = [
    "AppID", "Name", "Release date", "About the game", "Supported languages", 
    "Full audio languages", "Reviews", "Header image", "Website", "Support url", 
    "Support email", "Metacritic url", "Notes", "Screenshots", "Movies"
]
data = data.drop(columns=columns_to_drop, axis=1)

# 3. Eksik Verileri Temizleme
data = data.dropna()  # Eksik verileri tamamen çıkar

# 4. Kategorik Verileri Dönüştürme
categorical_columns = ["Windows", "Mac", "Linux", "Categories", "Genres", "Tags", "Developers", "Publishers"]
encoder = LabelEncoder()

for col in categorical_columns:
    if col in data.columns:
        data[col] = encoder.fit_transform(data[col].astype(str))
        
# Yeni bir Net Sentiment (Net Duygu) özelliği oluşturma
data['Net_Sentiment'] = data['Positive'] - data['Negative']

# Yeni özellikleri kullanarak veriyi bölme ve model eğitme
X = data.drop(columns=["Estimated owners", "Positive", "Negative"])  # Artık Positive ve Negative'yi kullanmıyoruz
y = data["Estimated owners"]

# 5. Veri Bölme
X = data.drop(columns=["Estimated owners"])  # Özellikler
y = data["Estimated owners"]  # Tahmin edilecek değer

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10000)

# 6. Özellik Ölçekleme
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 7. Model Eğitimi
model = RandomForestRegressor(random_state=10000)
model.fit(X_train, y_train)

# 8. Tahmin ve Performans Değerlendirme
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared Score: {r2}")

# 9. Yeni oyun tahmini için sütun adlarını güncelle
def predict_new_game(new_game_data):
    # Eksik sütunları tamamla
    for col in X.columns:
        if col not in new_game_data.columns:
            new_game_data[col] = 0  # Eksik sütunlar için varsayılan değer

    # Özellikleri ölçekle ve tahmin yap
    new_game_scaled = scaler.transform(new_game_data[X.columns])
    sales_prediction = model.predict(new_game_scaled)
    return sales_prediction[0]

# 10. Veri Görselleştirme Fonksiyonu
def plot_predictions(y_test, y_pred, sales_prediction):
    plt.figure(figsize=(10, 6))

    # Gerçek satışlar ve tahmin edilen satışlar için scatter plot
    sns.scatterplot(x=y_test, y=y_pred, label="Test Data Prediction", color='blue')

    # Yeni oyunun tahminini görselleştirmek için ekleme
    sns.scatterplot(x=[y_test.mean()], y=[sales_prediction], label="New Game Prediction", color='red', s=100, marker='X')

    plt.xlabel("Gerçek Satışlar")
    plt.ylabel("Tahmin Edilen Satışlar")
    plt.title("Gerçek ve Tahmin Edilen Satışların Karşılaştırması")
    plt.legend()
    plt.show()

# Yeni oyun için başlangıç değerleri
new_game = pd.DataFrame({
    "Price": [99.99],
    "Discount": [0.5],
    "DLC_count": [2],
    "Windows": [1],
    "Mac": [0],
    "Linux": [0],
    "Metacritic_score": [5],
    "User_score": [4.5],
    "Positive": [10],
    "Negative": [89],
    "Score_rank": [15],
    "Achievements": [20],
    "Recommendations": [300],
    "Average_playtime_forever": [2],
    "Average_playtime_two_weeks": [5],
    "Median_playtime_forever": [1],
    "Median_playtime_two_weeks": [4],
    "Developers": [1],
    "Publishers": [5],
    "Categories": [5],
    "Genres": [1],
    "Tags": [5]
})

# Yeni oyun tahminini hesapla
sales_prediction = predict_new_game(new_game)

# 11. Grafik Güncellemesi

# Burada yeni oyun verilerini değiştirdiğinizde, grafikte değişiklik olacaktır
# Örneğin:
new_game["Price"] = [79.99]  # Fiyatı değiştir
sales_prediction = predict_new_game(new_game)
plot_predictions(y_test, y_pred, sales_prediction)



# 1. Özelliklerin Önemini Hesapla
feature_importances = model.feature_importances_

# 2. Özellikler ve Önem Derecelerini Birlikte Göster
feature_names = X.columns
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances
})

# 3. Önem Derecelerine Göre Azalan Sıralama
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# 4. Sonuçları Yazdır
print(importance_df)

# 5. Grafik ile Görselleştirme
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('Feature Importance')
plt.show()



velihan özhge sdcasdcascasd