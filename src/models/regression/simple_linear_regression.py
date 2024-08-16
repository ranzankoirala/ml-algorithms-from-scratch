# ---------------- Author Details ----------------
# * @File: simple_linear_regression.py
# * @Author: Ranjan Koirala
# * @Version: 1.1
# * @Contact: ranjan.koirala1998@gmail.com

import numpy as np


class SimpleLinearRegression:
    """
    A simple linear regression model that fits a line to data points using the least squares method.

    The model represents the equation:
        y = β0 + β1 * X

    where:
        - β0 is the intercept
        - β1 is the slope

    Attributes:
        beta0 (float): The intercept of the linear regression line. Initialized as None.
        beta1 (float): The slope of the linear regression line. Initialized as None.

    Methods:
        fit(X: np.ndarray, y: np.ndarray) -> None:
            Fits the linear regression model to the provided data.

        predict(X: np.ndarray) -> np.ndarray:
            Predicts the response variable for the given explanatory variables using the fitted model.

    Example:
        Sample data
        >>> X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        >>> y = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

        Initialize and fit the model
        >>> model = SimpleLinearRegression()
        >>> model.fit(X=X, y=y)

        Predict on new data
        >>> X_new = np.array([21, 22])
        >>> predicted_y = model.predict(X_new)

        Evaluate the model
        >>> model.evaluate(X=X, y=y)
    """

    def __init__(self) -> None:
        """
        Initializes the coefficients for the linear regression model.
        The model represents the equation: y = β0 + β1 * X
        """
        self.beta0 = None  # Intercept
        self.beta1 = None  # Slope

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fits the linear regression model to the provided data.

        Args:
            X (np.ndarray): Array of explanatory variables (features).
            y (np.ndarray): Array of response variables (targets).

        Raises:
            ValueError: If the size of X and y do not match.
        """
        if len(X) != len(y):
            raise ValueError(
                "The size of Explanatory Variable (X) and Response Variable (y) must match."
            )

        # Calculate means
        mean_X = np.mean(X)
        mean_y = np.mean(y)

        # Calculate variance of X and covariance of X and y
        variance_X = np.sum((X - mean_X) ** 2)
        covariance_X_y = np.sum((X - mean_X) * (y - mean_y))

        # Calculate coefficients
        self.beta1 = covariance_X_y / variance_X
        self.beta0 = mean_y - self.beta1 * mean_X

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the response variable for the given explanatory variables using the fitted model.

        Args:
            X (np.ndarray): Array of explanatory variables (features).

        Returns:
            np.ndarray: Predicted response variables.

        Raises:
            ValueError: If the model has not been fitted yet.
        """
        if self.beta0 is None or self.beta1 is None:
            raise ValueError(
                "Model has not been fitted yet. Call 'fit' method before 'predict'."
            )

        return self.beta0 + self.beta1 * X

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Evaluates the performance of the model using various metrics.

        Args:
            X (np.ndarray): Array of explanatory variables (features).
            y (np.ndarray): Array of true response variables.

        Returns:
            dict: A dictionary containing the MAE, MSE, RMSE, and R-squared metrics.

        Notes:
            This method calculates the following evaluation metrics:
                - MAE (Mean Absolute Error)
                - MSE (Mean Squared Error)
                - RMSE (Root Mean Squared Error)
                - R-squared (Coefficient of Determination)
        """
        predictions = self.predict(X)

        # Calculate metrics
        residuals = y - predictions
        mae = np.mean(np.abs(residuals))
        mse = np.mean(residuals**2)
        rmse = np.sqrt(mse)
        total_variance = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (np.sum(residuals**2) / total_variance)

        return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R-squared": r_squared}


if __name__ == "__main__":
    pass
