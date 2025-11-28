# train_model.py
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib

def main():
    # 1. Carrega dataset de exemplo
    data = load_diabetes()
    X = data.data
    y = data.target

    # 2. Split treino/teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3. Treina um modelo simples
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )
    model.fit(X_train, y_train)  # Correção aqui

    # 4. Avalia rapidamente
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"MAE no conjunto de teste: {mae:.2f}")

    # 5. Salva o modelo treinado
    joblib.dump(model, "model.joblib")
    print("Modelo salvo em 'model.joblib'.")

if __name__ == "__main__":
    main()