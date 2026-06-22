from src.app import predict_quality

def test_predict_quality_detects_missing_features():
    features = {
        "fixed acidity": 7.4,
        "volatile acidity": None,
        "citric acid": 0.0,
        "residual sugar": 1.9,
        "chlorides": 0.076,
        "free sulfur dioxide": 11,
        "total sulfur dioxide": 34,
        "density": 0.9978,
        "pH": 3.51,
        "sulphates": 0.56,
        "alcohol": 9.4,
    }

    prediction, missing = predict_quality(features)

    assert prediction is None
    assert "volatile acidity" in missing

def test_predict_quality_returns_prediction():
    features = {
        "fixed acidity": 7.4,
        "volatile acidity": 0.7,
        "citric acid": 0.0,
        "residual sugar": 1.9,
        "chlorides": 0.076,
        "free sulfur dioxide": 11,
        "total sulfur dioxide": 34,
        "density": 0.9978,
        "pH": 3.51,
        "sulphates": 0.56,
        "alcohol": 9.4,
    }

    prediction, missing = predict_quality(features)

    assert prediction is not None
    assert len(missing) == 0