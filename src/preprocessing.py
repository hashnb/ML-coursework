"""Build unfitted preparation steps for use inside model pipelines."""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def make_preprocessor(scale_numeric=True):
    """Impute numeric medians and encode categories using each training fold."""
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    return ColumnTransformer([
        ("numeric", Pipeline(numeric_steps), NUMERIC_FEATURES),
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
         CATEGORICAL_FEATURES),
    ], remainder="drop")


def make_cv():
    """Share the same five stratified development folds across comparisons."""
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
