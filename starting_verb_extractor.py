import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        """
        No fitting needed for this transformer.
        """
        return self

    def transform(self, X):
        """
        Extracts whether the first word in the text is a verb or not.
        """
        def starting_verb(text):
            # Check for empty or whitespace-only strings
            if not text or not text.strip():
                return 0
            words = text.split()
            # Check if the first word is title case
            if len(words) > 0 and words[0].istitle():
                return 1
            return 0

        # Ensure input X is iterable
        if not isinstance(X, (list, np.ndarray)):
            X = X.tolist()

        # Apply the starting_verb function to each element in X
        features = [starting_verb(text) for text in X]

        # Return a 2D NumPy array (scikit-learn requires this format)
        return np.array(features).reshape(-1, 1)