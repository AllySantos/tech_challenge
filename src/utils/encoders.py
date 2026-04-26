import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


# Label encoder que funciona com mais de uma coluna
class CustomLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, true_values=None, false_values=None):
        super().__init__()
        self.true_values = true_values or ["male", "yes", 1, True]
        self.false_values = false_values or ["female", "no", 0, False]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_array = np.asarray(X)

        if X_array.ndim == 1:
            X_array = X_array.reshape(-1, 1)

        transformed = np.empty(X_array.shape, dtype=np.int64)

        for row_idx in range(X_array.shape[0]):
            for col_idx in range(X_array.shape[1]):
                val = X_array[row_idx, col_idx]
                normalized = val.lower() if isinstance(val, str) else val

                if normalized in self.true_values:
                    transformed[row_idx, col_idx] = 1
                elif normalized in self.false_values:
                    transformed[row_idx, col_idx] = 0
                else:
                    raise ValueError(f"Valor '{val}' nao reconhecido como binario.")

        return transformed


class CustomOneHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, drop_first=False, sparse=False, target=None):
        self.drop_first = drop_first
        self.sparse = sparse
        self.target = target

    def fit(self, X, y=None):
        X = np.array(X).flatten()
        X_list = [str(x) for x in X]

        self.categories_ = list(dict.fromkeys(X_list))
        self.categories_.sort()

        if self.target is not None and str(self.target) in self.categories_:
            self.categories_.remove(str(self.target))

        if self.drop_first and len(self.categories_) > 0:
            self.categories_ = self.categories_[1:]

        return self

    def transform(self, X):
        X = np.array(X).flatten()
        X_list = [str(x) for x in X]

        result = []
        for val in X_list:
            if val not in self.categories_ and not self.drop_first:
                raise ValueError(f"Valor '{val}' nao visto no fit.")

            row = np.array([1 if cat == val else 0 for cat in self.categories_], dtype=int)
            result.append(row)

        return np.array(result)

    def get_feature_names_out(self, input_features=None):
        prefix = input_features[0] if input_features else "x"
        return np.array([f"{prefix}_{cat}" for cat in self.categories_])
