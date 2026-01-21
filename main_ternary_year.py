# barebones_voo_predictor.py

import numpy as np
import yfinance as yf
from CTW import CTW  # your CTW implementation

# Download VOO data
voo = yf.download('VOO', start='2024-01-01', end='2024-12-31', progress=False, auto_adjust=True)
prices = voo['Close'].values.flatten()

# Calculate returns
returns = np.diff(np.log(prices))

# Let's see the data distribution first
print("=== Data Analysis ===")
print(f"Mean return: {np.mean(returns):.4f}")
print(f"Std return: {np.std(returns):.4f}")
print(f"Min return: {np.min(returns):.4f}")
print(f"Max return: {np.max(returns):.4f}")

# Use a smaller threshold or percentile-based approach
# Discretize to ternary: 0=DOWN, 1=FLAT, 2=UP
# Using tertiles instead of fixed threshold
lower_tercile = np.percentile(returns, 33.33)
upper_tercile = np.percentile(returns, 66.67)

print(f"Lower tercile: {lower_tercile:.4f}")
print(f"Upper tercile: {upper_tercile:.4f}")

symbols = np.ones(len(returns), dtype=int)  # default to FLAT
symbols[returns < lower_tercile] = 0  # DOWN
symbols[returns > upper_tercile] = 2  # UP

# Convert to Python int list
symbols = [int(s) for s in symbols]

# Check distribution
print(f"\nSymbol distribution in full data:")
print(f"DOWN: {symbols.count(0)} ({symbols.count(0)/len(symbols)*100:.1f}%)")
print(f"FLAT: {symbols.count(1)} ({symbols.count(1)/len(symbols)*100:.1f}%)")
print(f"UP: {symbols.count(2)} ({symbols.count(2)/len(symbols)*100:.1f}%)")

# Split: train on Jan-Nov, test on Dec
train_end = 231
train_symbols = symbols[:train_end]
test_symbols = symbols[train_end:]

print(f"\nTrain set distribution:")
print(f"DOWN: {train_symbols.count(0)}, FLAT: {train_symbols.count(1)}, UP: {train_symbols.count(2)}")

# Create and train CTW
depth = 10
ctw = CTW(depth=depth, symbols=3)

# Train: predict sequence on training data
print("\nTraining CTW...")
ctw.predict_sequence(train_symbols)

# Predict December
predictions = []
context = list(reversed(train_symbols[-depth:]))

print(f"\nInitial context: {context}")
print("\nPredicting December...")

for i in range(len(test_symbols)):
    # Get prediction distribution
    dist = ctw.get_distribution()
    pred_symbol = np.argmax(dist)
    predictions.append(pred_symbol)
    
    # Debug first few predictions
    if i < 5:
        print(f"Day {i+1}: dist={dist}, pred={pred_symbol}, actual={test_symbols[i]}")
    
    # Update with actual
    actual = int(test_symbols[i])
    ctw.update(actual, context)
    
    # Update context
    context.insert(0, actual)
    context = context[:depth]

# Calculate hit rate
predictions = np.array(predictions)
test_symbols_array = np.array(test_symbols)
hitrate = np.mean(predictions == test_symbols_array)

print(f"\n=== December 2024 Results ===")
print(f"Hit Rate: {hitrate:.2%}")
print(f"Total predictions: {len(predictions)}")
print(f"Correct: {np.sum(predictions == test_symbols_array)}")

# Distribution of predictions vs actuals
print(f"\nPredicted - DOWN: {np.sum(predictions == 0)}, FLAT: {np.sum(predictions == 1)}, UP: {np.sum(predictions == 2)}")
print(f"Actual    - DOWN: {np.sum(test_symbols_array == 0)}, FLAT: {np.sum(test_symbols_array == 1)}, UP: {np.sum(test_symbols_array == 2)}")

# Baseline accuracy (if always predicted most common class)
most_common = np.argmax([np.sum(test_symbols_array == 0), 
                         np.sum(test_symbols_array == 1), 
                         np.sum(test_symbols_array == 2)])
baseline = np.max([np.sum(test_symbols_array == 0), 
                   np.sum(test_symbols_array == 1), 
                   np.sum(test_symbols_array == 2)]) / len(test_symbols_array)
print(f"\nBaseline (always predict most common): {baseline:.2%}")