# barebones_voo_predictor.py

import numpy as np
import yfinance as yf
from CTW import CTW  # your CTW implementation

# Download VOO data
voo = yf.download('VOO', start='2024-01-01', end='2024-12-31', progress=False, auto_adjust=True)
prices = voo['Close'].values.flatten()

# Calculate returns
returns = np.diff(np.log(prices))

# Use percentile-based discretization
lower_tercile = np.percentile(returns, 33.33)
upper_tercile = np.percentile(returns, 66.67)

print(f"Lower tercile: {lower_tercile:.4f}")
print(f"Upper tercile: {upper_tercile:.4f}")

symbols = np.ones(len(returns), dtype=int)
symbols[returns < lower_tercile] = 0  # DOWN
symbols[returns > upper_tercile] = 2  # UP

# Convert to Python int list
symbols = [int(s) for s in symbols]

print(f"\nTotal trading days: {len(symbols)}")

# Define train/test splits (approximately 21 trading days per month)
# Train 3 months, predict 1 month
splits = [
    {'name': 'April', 'train_start': 0, 'train_end': 63, 'test_end': 84},      # Jan-Mar → Apr
    {'name': 'August', 'train_start': 84, 'train_end': 147, 'test_end': 168},  # May-Jul → Aug
    {'name': 'December', 'train_start': 168, 'train_end': 231, 'test_end': 250} # Sep-Nov → Dec
]

all_results = []

for split in splits:
    print(f"\n{'='*60}")
    print(f"Predicting {split['name']} 2024")
    print(f"{'='*60}")
    
    # Get train and test data
    train_symbols = symbols[split['train_start']:split['train_end']]
    test_symbols = symbols[split['train_end']:split['test_end']]
    
    print(f"Training period: days {split['train_start']}-{split['train_end']} ({len(train_symbols)} days)")
    print(f"Testing period: days {split['train_end']}-{split['test_end']} ({len(test_symbols)} days)")
    
    print(f"\nTrain distribution:")
    print(f"  DOWN: {train_symbols.count(0)}, FLAT: {train_symbols.count(1)}, UP: {train_symbols.count(2)}")
    
    print(f"Test distribution:")
    print(f"  DOWN: {test_symbols.count(0)}, FLAT: {test_symbols.count(1)}, UP: {test_symbols.count(2)}")
    
    # Create and train CTW
    depth = 10
    ctw = CTW(depth=depth, symbols=3)
    
    # Train
    ctw.predict_sequence(train_symbols)
    
    # Predict test period
    predictions = []
    context = list(reversed(train_symbols[-depth:]))
    
    print(f"\nInitial context: {context}")
    
    for i in range(len(test_symbols)):
        # Get prediction distribution
        dist = ctw.get_distribution()
        pred_symbol = np.argmax(dist)
        predictions.append(pred_symbol)
        
        # Debug first few predictions
        if i < 3:
            print(f"  Day {i+1}: dist={dist}, pred={pred_symbol}, actual={test_symbols[i]}")
        
        # Update with actual
        actual = int(test_symbols[i])
        ctw.update(actual, context)
        
        # Update context
        context.insert(0, actual)
        context = context[:depth]
    
    # Calculate metrics
    predictions = np.array(predictions)
    test_symbols_array = np.array(test_symbols)
    hitrate = np.mean(predictions == test_symbols_array)
    
    # Baseline (most common class)
    baseline = np.max([np.sum(test_symbols_array == 0), 
                       np.sum(test_symbols_array == 1), 
                       np.sum(test_symbols_array == 2)]) / len(test_symbols_array)
    
    print(f"\n--- {split['name']} Results ---")
    print(f"Hit Rate: {hitrate:.2%}")
    print(f"Baseline: {baseline:.2%}")
    print(f"Total predictions: {len(predictions)}")
    print(f"Correct: {np.sum(predictions == test_symbols_array)}")
    
    print(f"\nPredicted - DOWN: {np.sum(predictions == 0)}, FLAT: {np.sum(predictions == 1)}, UP: {np.sum(predictions == 2)}")
    print(f"Actual    - DOWN: {np.sum(test_symbols_array == 0)}, FLAT: {np.sum(test_symbols_array == 1)}, UP: {np.sum(test_symbols_array == 2)}")
    
    # Store results
    all_results.append({
        'month': split['name'],
        'hitrate': hitrate,
        'baseline': baseline,
        'correct': np.sum(predictions == test_symbols_array),
        'total': len(predictions)
    })

# Summary
print(f"\n{'='*60}")
print("OVERALL SUMMARY")
print(f"{'='*60}")
total_correct = sum(r['correct'] for r in all_results)
total_predictions = sum(r['total'] for r in all_results)
overall_hitrate = total_correct / total_predictions

print(f"Overall Hit Rate: {overall_hitrate:.2%}")
print(f"Total Predictions: {total_predictions}")
print(f"Total Correct: {total_correct}")

print("\nMonth-by-month:")
for r in all_results:
    beat_baseline = "✓" if r['hitrate'] > r['baseline'] else "✗"
    print(f"  {r['month']:10s}: {r['hitrate']:.2%} (baseline: {r['baseline']:.2%}) {beat_baseline}")