from sklearn.linear_model import LinearRegression
import numpy as np

def train_model():
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([2, 4, 6, 8, 10])
    
    model = LinearRegression()
    model.fit(X, y)
    
    return model

if __name__ == "__main__":
    model = train_model()
    print(f"Model coefficient: {model.coef_[0]}, Intercept: {model.intercept_}")

def test_regression():
    model = train_model()
    assert round(model.coef_[0], 2) == 2.0  # Expecting a slope of 2