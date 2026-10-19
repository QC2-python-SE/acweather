import math
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from acweather import read_dataset, clean_data, train_model, calculate_average, calculate_median, calculate_range, calculate_variance, calculate_standard_deviation

@pytest.fixture
def mock_data():
    """
    Creates a mock dataset for testing.
    """
    data = {
        'temperature': np.random.uniform(15, 35, 100),
        'humidity': np.random.uniform(40, 90, 100),
        'wind_speed': np.random.uniform(0, 20, 100),
        'precipitation': np.random.randint(0, 2, 100)
    }
    df = pd.DataFrame(data)
    return df

def test_read_dataset(mock_data, tmpdir):
    # Save the DataFrame to a temporary CSV file
    temp_file = tmpdir.join("test_weather_data.csv")
    mock_data.to_csv(temp_file, index=False)

    # Read the dataset
    result = read_dataset(temp_file)
    
    assert isinstance(result, pd.DataFrame)
    assert result.shape == mock_data.shape  # Check if the dimensions match

def test_clean_data(mock_data):
    # Test the data cleaning process
    cleaned_df = clean_data(mock_data, ['temperature', 'humidity', 'wind_speed'], 'precipitation')
    
    assert 'temperature' in cleaned_df.columns
    assert 'humidity' in cleaned_df.columns
    assert cleaned_df.isnull().sum().sum() == 0  # Ensure no NaN values
    assert cleaned_df.duplicated().sum() == 0  # Ensure no duplicates

def test_train_model(mock_data):
    # Clean data before training
    cleaned_df = clean_data(mock_data, ['temperature', 'humidity', 'wind_speed'], 'precipitation')
    
    # Train the model
    model, features, target = train_model(cleaned_df)
    
    assert isinstance(model, RandomForestRegressor)  # Check if the model is a RandomForestRegressor
    assert len(features) > 0  # Ensure features were returned
    assert (cleaned_df.iloc[:, -1].values == target.values).all()  # Ensure target matches the last column

def test_calculate_standard_deviation():
    # Test the standard deviation calculation

    for vals, ref in (([1, 4, 7], 3), ([1,] * 117, 0),([-1, -4, -7], 3)):
        std_dev = calculate_standard_deviation(vals)
        assert isinstance(std_dev, float)  # Check if the result is a float
        assert std_dev == ref  # Check if the standard deviation is correct
        assert std_dev >= 0  # Standard deviation should be non-negative
        assert math.isfinite(std_dev)  # Check if the standard deviation is finite

class TestCalculateRange:

    def test_simple_range(self):
        assert calculate_range([4, 30, 27, 4]) == 26

    def test_negative_range(self):
        assert calculate_range([4, 30, 27, -4]) == 34

    def test_float_range(self):
        assert calculate_range([4.5, 30.0, 27.0, 4.5]) == 25.5

    def test_negative_float_range(self):
        assert calculate_range([-4.5, 30.0, 27.0, 4.0]) == 34.5

    def test_float_int_range(self):
        assert calculate_range([4.5, 30, 27, 4.5]) == 25.5

# PLACEHOLDER: test_calculate_median()

# PLACEHOLDER: test_calculate_variance()
def test_calculate_variance():
    """
    Using pytest to test the calculate_variance function with various inputs.

    Test 1: A simple list of integers.
    Test 2: A list of zero values.
    Test 3: A list of negative and positive float values.
    Test 4: A list containing NaN values.
    Test 5: An empty list (should raise ZeroDivisionError).

    Args:
        None
    
    Returns:
        None
    
    Raises:
        AssertionError: If the calculated variance does not match the expected output.  
    """
    input_1 = [1, 2, 3, 4, 5]
    expected_output_1 = 2.0  # Variance of [1, 2, 3, 4, 5] is 2.5
    assert calculate_variance(input_1) == expected_output_1, "variance of f{input_1} != f{expected_output_1}"

    input_2 = [0,0,0,0,0,0]
    expected_output_2 = 0.0  # Variance of [0, 0, 0, 0, 0, 0] is 0.0
    assert calculate_variance(input_2) == expected_output_2, "variance of f{input_2} != f{expected_output_2}"

    input_3 = [-3.7, -2.1, 0.0, 1.5, 3.2]
    expected_output_3 = 6.0696 # Variance of [-3.7, -2.1, 0.0, 1.5, 3.2] is approximately 6.0696
    assert calculate_variance(input_3) == pytest.approx(expected_output_3), "variance of f{input_3} != f{expected_output_3}"

    input_4 = [float('nan'), 1, 2, 3, 4, float('nan')]
    expected_output_4 = float('nan')  # Variance with NaN values should return NaN
    assert calculate_variance(input_4) == pytest.approx(float("nan"), nan_ok=True), "variance of f{input_4} != f{expected_output_4}"

    input_5 = []
    expected_output_5 = 0  # Variance of an empty list should return ZeroDivisionError
    with pytest.raises(ZeroDivisionError):
        calculate_variance(input_5)