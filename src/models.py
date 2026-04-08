from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from data_preprocessing import load_data, preprocess_data, split_data
from utils import get_basic_info, check_missing_values, check_class_distribution


def train_model():
    # Load data
    df = load_data("../data/customer_data.csv")

    # Basic EDA
    get_basic_info(df)
    print("\nMissing Values:\n", check_missing_values(df))

    # Preprocess
    X, y = preprocess_data(df)

    # Check target distribution
    check_class_distribution(y)

    # Split
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Model
    model = GradientBoostingClassifier()
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy:.2f}")


if __name__ == "__main__":
    train_model()