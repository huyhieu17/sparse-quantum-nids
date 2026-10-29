"""Classical baselines on the same 64-dimensional front-end.

Reviewers of a quantum-kernel IDS paper always ask "how does this compare to a
plain classical model?".  These baselines consume exactly the same
``Features`` output and the same 2,000-sample training subset, so the only
difference is the classifier.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

BASELINES = ["RBF-SVM", "RandomForest"]


def train_classical(X_train, y_train, seed=11):
    models = {
        "RBF-SVM": SVC(C=1, kernel="rbf", gamma="scale", class_weight="balanced"),
        "RandomForest": RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                               random_state=seed, n_jobs=1),
    }
    for m in models.values():
        m.fit(X_train, y_train)
    return models


def classical_score(model, X):
    """A continuous score whose sign reproduces the model decision."""
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X), float)
    proba = model.predict_proba(X)
    return proba[:, 1] - 0.5
