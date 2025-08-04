"""
bozkod_model.py

TEKNOFEST 2025 E-Ticaret Hackathonu için Bozkod ekibi tarafından geliştirilmiştir.
Amaç: Kullanıcının ürün tıklama veya sipariş davranışını tahmin eden bir makine öğrenmesi modeli geliştirmek.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def load_data(csv_path):
    """
    Veriyi CSV formatında yükler.
    Args:
        csv_path (str): Veri dosyasının yolu.
    Returns:
        pandas.DataFrame: Yüklenen veri
    """
    df = pd.read_csv(csv_path)
    return df

def preprocess_data(df):
    """
    Veriyi modele uygun hale getirmek için temizler.
    Args:
        df (pd.DataFrame): Girdi veri çerçevesi
    Returns:
        X (pd.DataFrame): Özellikler
        y (pd.Series): Etiketler (tıklama: 0/1 gibi)
    """
    df = df.dropna()  # basit boş veri temizliği

    # Örnek olarak bazı kategorik alanları one-hot encoding'e çevirelim
    df = pd.get_dummies(df, columns=["device_type", "product_category"])

    X = df.drop("clicked", axis=1)
    y = df["clicked"]

    return X, y

def train_model(X, y):
    """
    Makine öğrenmesi modelini eğitir.
    Args:
        X (pd.DataFrame): Özellikler
        y (pd.Series): Etiketler
    Returns:
        model: Eğitilmiş model
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    return model

# Bu blok dosya direkt çalıştırıldığında devreye girer (modül gibi import edilirse çalışmaz)
if __name__ == "__main__":
    # Örnek veri yolu (dosya repo içinde olmayabilir, siz eklersiniz)
    df = load_data("data/sample_user_clicks.csv")
    X, y = preprocess_data(df)
    model = train_model(X, y)
