import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from CTW import CTW


def get_sp500_with_vix(start_date="2025-10-01", end_date="2025-12-01"):
    """
    Télécharge S&P 500 + VIX et transforme en séquences binaires.
    Returns:
        - binary_seq: Direction du S&P (0=baisse, 1=hausse)
        - vix_seq: État de volatilité (0=faible, 1=moyenne, 2=élevée)
    """
    print(f"Chargement S&P 500 + VIX ({start_date} à {end_date})...")
    
    sp500 = yf.download("NVDA", start=start_date, end=end_date, progress=False)
    vix = yf.download("^VIX", start=start_date, end=end_date, progress=False)
    
    # Aligner les indices temporels
    common_index = sp500.index.intersection(vix.index)
    sp500 = sp500.loc[common_index, 'Close'].values
    vix_values = vix.loc[common_index, 'Close'].values
    
    # S&P 500 en binaire (direction)
    binary_seq = []
    for i in range(1, len(sp500)):
        binary_seq.append(1 if sp500[i] >= sp500[i-1] else 0)
    
    # VIX en 3 états (volatilité faible/moyenne/élevée)
    vix_seq = []
    for i in range(1, len(vix_values)):
        v = vix_values[i]
        if v < 15:
            vix_seq.append(0)  # Calme
        elif v < 25:
            vix_seq.append(1)  # Normal
        else:
            vix_seq.append(2)  # Stress
    
    print(f"  → {len(binary_seq)} jours récupérés")
    print(f"  → Distribution VIX: {np.bincount(vix_seq)} (calme/normal/stress)")
    
    return binary_seq, vix_seq


def get_sp500_with_volume(start_date="2020-01-01", end_date="2024-01-01"):
    """
    S&P 500 + Volume relatif comme side-info.
    """
    print(f"Chargement S&P 500 + Volume ({start_date} à {end_date})...")
    
    df = yf.download("^GSPC", start=start_date, end=end_date, progress=False)
    prices = df['Close'].values
    volumes = df['Volume'].values
    
    binary_seq = []
    for i in range(1, len(prices)):
        binary_seq.append(1 if prices[i] >= prices[i-1] else 0)
    
    volume_seq = []
    window = 20
    for i in range(1, len(volumes)):
        if i < window:
            volume_seq.append(1)
        else:
            avg_vol = np.mean(volumes[i-window:i])
            ratio = volumes[i] / avg_vol
            if ratio < 0.8:
                volume_seq.append(0)
            elif ratio < 1.2:
                volume_seq.append(1)
            else:
                volume_seq.append(2)
    
    print(f"  → {len(binary_seq)} jours récupérés")
    print(f"  → Distribution Volume: {np.bincount(volume_seq)}")
    
    return binary_seq, volume_seq


def get_sp500_with_momentum(start_date="2020-01-01", end_date="2024-01-01"):
    """
    S&P 500 + Momentum (tendance récente) comme side-info.
    """
    print(f"Chargement S&P 500 + Momentum ({start_date} à {end_date})...")
    
    df = yf.download("^GSPC", start=start_date, end=end_date, progress=False)
    prices = df['Close'].values
    
    binary_seq = []
    for i in range(1, len(prices)):
        binary_seq.append(1 if prices[i] >= prices[i-1] else 0)
    
    # Momentum = prix actuel vs moyenne 10 jours
    momentum_seq = []
    window = 10
    for i in range(1, len(prices)):
        if i < window:
            momentum_seq.append(1)  # Neutre
        else:
            ma = np.mean(prices[i-window:i])
            if prices[i] < ma * 0.98:
                momentum_seq.append(0)  # Baissier
            elif prices[i] > ma * 1.02:
                momentum_seq.append(2)  # Haussier
            else:
                momentum_seq.append(1)  # Neutre
    
    print(f"  → {len(binary_seq)} jours récupérés")
    print(f"  → Distribution Momentum: {np.bincount(momentum_seq)} (baissier/neutre/haussier)")
    
    return binary_seq, momentum_seq


def evaluate_ctw(sequence, sideseq, depth, sidesymbols, name="Test"):
    """
    Évalue CTW avec ou sans side-information.
    """
    if sideseq is None:
        model = CTW(depth=depth, symbols=2, sidesymbols=1)
        distributions = model.predict_sequence(sequence)
    else:
        model = CTW(depth=depth, symbols=2, sidesymbols=sidesymbols)
        distributions = model.predict_sequence(sequence, sideseq=sideseq)
    
    correct = 0
    total = distributions.shape[1]
    history_probs = []
    
    for i in range(total):
        real_idx = i + depth
        prob_up = distributions[1, i]
        history_probs.append(prob_up)
        
        pred = 1 if prob_up >= 0.5 else 0
        real = sequence[real_idx]
        
        if pred == real:
            correct += 1
    
    accuracy = (correct / total) * 100
    print(f"[{name}] Précision: {accuracy:.2f}%")
    
    return accuracy, history_probs


def run_comprehensive_comparison():
    """
    Compare CTW avec différentes sources de side-information.
    """
    depth = 5
    
    print("\n" + "="*70)
    print("COMPARAISON CTW: IMPACT DE LA SIDE-INFORMATION")
    print("="*70)
    
    # 1. Baseline: S&P 500 seul
    print("\n--- TEST 1: S&P 500 SEUL (baseline) ---")
    sp500_seq, _ = get_sp500_with_vix()
    acc1, probs1 = evaluate_ctw(sp500_seq, None, depth, 1, "S&P seul")
    
    # 2. S&P 500 + VIX
    print("\n--- TEST 2: S&P 500 + VIX ---")
    sp500_seq2, vix_seq = get_sp500_with_vix()
    acc2, probs2 = evaluate_ctw(sp500_seq2, vix_seq, depth, 3, "S&P + VIX")
    
    # 3. S&P 500 + Volume
    print("\n--- TEST 3: S&P 500 + VOLUME ---")
    sp500_seq3, volume_seq = get_sp500_with_volume()
    acc3, probs3 = evaluate_ctw(sp500_seq3, volume_seq, depth, 3, "S&P + Volume")
    
    # 4. S&P 500 + Momentum
    print("\n--- TEST 4: S&P 500 + MOMENTUM ---")
    sp500_seq4, momentum_seq = get_sp500_with_momentum()
    acc4, probs4 = evaluate_ctw(sp500_seq4, momentum_seq, depth, 3, "S&P + Momentum")
    
    # Visualisation
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f'Impact de la side-information sur CTW (Depth={depth})', 
                 fontsize=14, fontweight='bold')
    
    tests = [
        (probs1, f"S&P 500 seul\nPrécision: {acc1:.2f}%", axes[0, 0]),
        (probs2, f"S&P 500 + VIX\nPrécision: {acc2:.2f}%", axes[0, 1]),
        (probs3, f"S&P 500 + Volume\nPrécision: {acc3:.2f}%", axes[1, 0]),
        (probs4, f"S&P 500 + Momentum\nPrécision: {acc4:.2f}%", axes[1, 1])
    ]
    
    for probs, title, ax in tests:
        display_n = 900
        ax.plot(probs[-display_n:], label="P(Hausse)", color='blue', linewidth=1.5)
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label="Seuil 0.5")
        ax.set_title(title, fontweight='bold', fontsize=11)
        ax.set_xlabel("Jours")
        ax.set_ylabel("Probabilité")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.show()
    
    # Résumé
    print("\n" + "="*70)
    print("RÉSUMÉ DES PERFORMANCES")
    print("="*70)
    print(f"{'Configuration':<30} {'Précision':<12} {'Gain vs baseline'}")
    print("-"*70)
    print(f"{'S&P 500 seul':<30} {acc1:>6.2f}%     {'—'}")
    print(f"{'S&P 500 + VIX':<30} {acc2:>6.2f}%     {acc2-acc1:>+5.2f}%")
    print(f"{'S&P 500 + Volume':<30} {acc3:>6.2f}%     {acc3-acc1:>+5.2f}%")
    print(f"{'S&P 500 + Momentum':<30} {acc4:>6.2f}%     {acc4-acc1:>+5.2f}%")
    print("="*70)
    
    best_acc = max(acc2, acc3, acc4)
    best_name = ["VIX", "Volume", "Momentum"][np.argmax([acc2, acc3, acc4])]
    
    print("\n💡 ANALYSE:")
    if best_acc > acc1 + 2:
        print(f"   ✓ Meilleure side-info: {best_name} (+{best_acc-acc1:.2f}%)")
        print("   → La side-information apporte de l'information prédictive!")
        print("   → Le marché contient des patterns conditionnels exploitables")
    elif best_acc > acc1 + 0.5:
        print(f"   ~ Gain marginal avec {best_name} (+{best_acc-acc1:.2f}%)")
        print("   → Léger signal mais insuffisant pour une stratégie robuste")
    else:
        print("   ✗ Aucune side-info n'améliore significativement les prédictions")
        print("   → Le marché reste quasi-aléatoire même avec contexte additionnel")
        print("   → Les patterns sont trop faibles ou déjà arbitrés")


if __name__ == "__main__":
    run_comprehensive_comparison()
