from sklearn.base import BaseEstimator, TransformerMixin

class FieldExtractor(BaseEstimator, TransformerMixin):
    """Reads a precomputed field from each data dict."""

    def __init__(self, field: str):
        self.field = field

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [d.get(self.field, '') or '' for d in X]