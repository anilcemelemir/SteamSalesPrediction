import sys
import os

# Modül yolunu ekle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'models')))

from src.preprocess import preprocess_data
from models import train_model, predict_sales_pre_release
from src.visualization import (
    plot_actual_vs_predicted,
    plot_feature_importances,
    plot_residuals,
    plot_actual_vs_predicted_comparison,
    plot_genre_vs_ownership
)

# Veri işleme
file_path = "data/steam.csv"
X, y, data = preprocess_data(file_path)

# Model eğitimi
model, X_test, y_test, y_pred = train_model(X, y)

# Grafiksel analizler
plot_actual_vs_predicted(y_test, y_pred)
plot_feature_importances(model, X.columns)
plot_residuals(y_test, y_pred)
plot_actual_vs_predicted_comparison(y_test, y_pred)
plot_genre_vs_ownership(data)

# Örnek tahmin
prediction = predict_sales_pre_release(
    price=29.99,
    average_playtime=500,
    achievements=10,
    tags=["Action"],
    categories=["Single-player"],
    genres=["Adventure"],
    developers=["Valve"],
    publishers=["Electronic Arts"],
    platforms="Windows",
    positive_ratings_range=[100, 500],
    negative_ratings_range=[20, 50],
    english=1,
    required_age=18,
    release_month=11,
    X=X,
    model=model
)
print("Tahmin edilen satış:", prediction)
