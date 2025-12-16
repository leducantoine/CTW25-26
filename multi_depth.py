import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from CTW import CTW


def calculate_rsi(prices, window=14):
    """Calcule le RSI (Relative Strength Index) - Version simplifiée."""
    rsi = np.ones(len(prices)) * 50.0  # Valeur neutre par défaut
    
    if len(prices) <= window:
        return rsi
    
    # Calculer les variations
    deltas = np.diff(prices)
    
    for i in range(window, len(prices)):
        # Prendre les 'window' dernières variations
        period_deltas = deltas[i-window:i]
        
        # Séparer gains et pertes
        gains = period_deltas[period_deltas > 0]
        losses = -period_deltas[period_deltas < 0]
        
        # Calculer les moyennes
        avg_gain = np.mean(gains) if len(gains) > 0 else 0.0
        avg_loss = np.mean(losses) if len(losses) > 0 else 0.0
        
        # Calculer RSI
        if avg_loss == 0.0:
            rsi[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi


def get_advanced_features(ticker="^GSPC", start_date="2020-01-01", end_date="2024-01-01"):
    """
    Récupère un ticker avec plusieurs features techniques avancées.
    Returns:
        - binary_seq: Direction (0=baisse, 1=hausse)
        - rsi_seq: RSI discrétisé (0=survendu, 1=neutre, 2=suracheté)
        - bb_seq: Position Bollinger Bands (0=bas, 1=milieu, 2=haut)
        - macd_seq: Signal MACD (0=bearish, 1=neutre, 2=bullish)
        - volume_seq: Volume relatif (0=faible, 1=normal, 2=fort)
    """
    print(f"Chargement {ticker} ({start_date} à {end_date})...")
    
    df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
    prices = df['Close'].values
    volumes = df['Volume'].values
    
    # 1. Séquence binaire principale
    binary_seq = []
    for i in range(1, len(prices)):
        binary_seq.append(1 if prices[i] >= prices[i-1] else 0)
    
    # 2. RSI (14 jours) - Indicateur de momentum
    rsi = calculate_rsi(prices, window=14)
    rsi_seq = []
    for i in range(1, len(prices)):
        if rsi[i] < 30:
            rsi_seq.append(0)  # Survendu (potentiel rebond)
        elif rsi[i] < 70:
            rsi_seq.append(1)  # Neutre
        else:
            rsi_seq.append(2)  # Suracheté (potentiel correction)
    
    # 3. Bollinger Bands - Indicateur de volatilité
    window_bb = 20
    bb_seq = []
    for i in range(1, len(prices)):
        if i < window_bb:
            bb_seq.append(1)  # Neutre
        else:
            sma = np.mean(prices[i-window_bb:i])
            std = np.std(prices[i-window_bb:i])
            upper_band = sma + 2 * std
            lower_band = sma - 2 * std
            
            if prices[i] <= lower_band:
                bb_seq.append(0)  # Bande basse (oversold)
            elif prices[i] >= upper_band:
                bb_seq.append(2)  # Bande haute (overbought)
            else:
                bb_seq.append(1)  # Milieu
    
    # 4. MACD - Indicateur de tendance
    exp1 = 12
    exp2 = 26
    exp_signal = 9
    
    # Calcul EMA
    def calculate_ema(data, period):
        ema = np.zeros(len(data))
        ema[0] = data[0]
        multiplier = 2.0 / (period + 1.0)
        for i in range(1, len(data)):
            ema[i] = (data[i] - ema[i-1]) * multiplier + ema[i-1]
        return ema
    
    ema12 = calculate_ema(prices, exp1)
    ema26 = calculate_ema(prices, exp2)
    macd_line = ema12 - ema26
    signal_line = calculate_ema(macd_line, exp_signal)
    
    macd_seq = []
    for i in range(1, len(prices)):
        histogram = macd_line[i] - signal_line[i]
        # Normaliser par rapport au prix pour éviter les seuils fixes
        normalized_hist = (histogram / prices[i]) * 100.0
        
        if normalized_hist < -0.1:
            macd_seq.append(0)  # Signal bearish
        elif normalized_hist > 0.1:
            macd_seq.append(2)  # Signal bullish
        else:
            macd_seq.append(1)  # Neutre
    
    # 5. Volume relatif
    volume_seq = []
    window_vol = 20
    for i in range(1, len(volumes)):
        if i < window_vol:
            volume_seq.append(1)
        else:
            avg_vol = np.mean(volumes[i-window_vol:i])
            ratio = volumes[i] / avg_vol if avg_vol > 0 else 1.0
            
            if ratio < 0.7:
                volume_seq.append(0)  # Faible volume
            elif ratio < 1.3:
                volume_seq.append(1)  # Normal
            else:
                volume_seq.append(2)  # Fort volume (conviction)
    
    print(f"  → {len(binary_seq)} périodes récupérées")
    print(f"  → RSI: {np.bincount(rsi_seq)} (survendu/neutre/suracheté)")
    print(f"  → Bollinger: {np.bincount(bb_seq)} (bas/milieu/haut)")
    print(f"  → MACD: {np.bincount(macd_seq)} (bearish/neutre/bullish)")
    print(f"  → Volume: {np.bincount(volume_seq)} (faible/normal/fort)")
    
    return binary_seq, rsi_seq, bb_seq, macd_seq, volume_seq


def evaluate_ctw(sequence, sideseq, depth, sidesymbols, name="Test"):
    """Évalue CTW avec side-information."""
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
    
    return accuracy, history_probs


def run_depth_comparison():
    """
    Compare CTW avec différentes profondeurs et features.
    """
    print("\n" + "="*80)
    print("ANALYSE COMPLÈTE: PROFONDEURS + FEATURES AVANCÉES")
    print("="*80)
    
    # Récupérer les données avec toutes les features
    ticker = "^GSPC"
    binary_seq, rsi_seq, bb_seq, macd_seq, volume_seq = get_advanced_features(
        ticker=ticker, start_date="2020-01-01", end_date="2024-01-01"
    )
    
    # Test sur plusieurs profondeurs
    depths = [3, 5, 7, 10]
    features = [
        (None, 1, "Sans side-info"),
        (rsi_seq, 3, "RSI"),
        (bb_seq, 3, "Bollinger Bands"),
        (macd_seq, 3, "MACD"),
        (volume_seq, 3, "Volume")
    ]
    
    # Matrice de résultats
    results = {}
    all_probs = {}
    
    print("\n" + "-"*80)
    print("TESTS EN COURS...")
    print("-"*80)
    
    for depth in depths:
        results[depth] = {}
        all_probs[depth] = {}
        
        for sideseq, sidesymbols, feature_name in features:
            name = f"D={depth}, {feature_name}"
            acc, probs = evaluate_ctw(binary_seq, sideseq, depth, sidesymbols, name)
            results[depth][feature_name] = acc
            all_probs[depth][feature_name] = probs
    
    # Tableau récapitulatif
    print("\n" + "="*80)
    print("TABLEAU RÉCAPITULATIF DES PERFORMANCES")
    print("="*80)
    print(f"{'Feature':<20}", end="")
    for depth in depths:
        print(f"D={depth:2d}    ", end="")
    print("Moyenne")
    print("-"*80)
    
    feature_avgs = {}
    for _, _, feature_name in features:
        print(f"{feature_name:<20}", end="")
        accuracies = []
        for depth in depths:
            acc = results[depth][feature_name]
            print(f"{acc:>5.2f}%  ", end="")
            accuracies.append(acc)
        avg = np.mean(accuracies)
        feature_avgs[feature_name] = avg
        print(f"{avg:>5.2f}%")
    
    print("="*80)
    
    # Trouver la meilleure configuration
    best_config = None
    best_acc = 0
    for depth in depths:
        for feature_name, acc in results[depth].items():
            if acc > best_acc:
                best_acc = acc
                best_config = (depth, feature_name)
    
    print(f"\n🏆 MEILLEURE CONFIGURATION: Depth={best_config[0]}, Feature={best_config[1]}")
    print(f"   Précision: {best_acc:.2f}%")
    
    # Visualisation: Comparaison des features pour la meilleure profondeur
    best_depth = max(depths, key=lambda d: max(results[d].values()))
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Comparaison des features (Depth={best_depth}) - {ticker}', 
                 fontsize=14, fontweight='bold')
    axes = axes.flatten()
    
    for idx, (sideseq, sidesymbols, feature_name) in enumerate(features):
        ax = axes[idx]
        probs = all_probs[best_depth][feature_name]
        acc = results[best_depth][feature_name]
        
        display_n = min(500, len(probs))
        ax.plot(probs[-display_n:], label="P(Hausse)", color='blue', linewidth=1.5)
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label="Seuil 0.5")
        ax.set_title(f"{feature_name}\nPrécision: {acc:.2f}%", fontweight='bold', fontsize=10)
        ax.set_xlabel("Jours")
        ax.set_ylabel("Probabilité")
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
    
    # Masquer le dernier subplot
    axes[-1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Visualisation 2: Impact de la profondeur
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    
    for _, _, feature_name in features:
        accs = [results[d][feature_name] for d in depths]
        ax2.plot(depths, accs, marker='o', label=feature_name, linewidth=2, markersize=8)
    
    ax2.axhline(y=50, color='gray', linestyle=':', alpha=0.5, label="Hasard (50%)")
    ax2.set_xlabel("Profondeur (D)", fontsize=12)
    ax2.set_ylabel("Précision (%)", fontsize=12)
    ax2.set_title(f"Impact de la profondeur sur la performance - {ticker}", fontsize=14, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(depths)
    
    plt.tight_layout()
    plt.show()
    
    # Analyse finale
    print("\n" + "="*80)
    print("💡 ANALYSE DÉTAILLÉE")
    print("="*80)
    
    best_feature = max(feature_avgs.items(), key=lambda x: x[1])
    print(f"\n1. Meilleure feature en moyenne: {best_feature[0]} ({best_feature[1]:.2f}%)")
    
    best_depth_avg = max(depths, key=lambda d: np.mean(list(results[d].values())))
    print(f"2. Meilleure profondeur en moyenne: D={best_depth_avg}")
    
    baseline_avg = feature_avgs["Sans side-info"]
    best_side_info = max((k, v) for k, v in feature_avgs.items() if k != "Sans side-info")
    improvement = best_side_info[1] - baseline_avg
    
    print(f"\n3. Impact de la side-information:")
    print(f"   - Baseline (sans side-info): {baseline_avg:.2f}%")
    print(f"   - Meilleure side-info ({best_side_info[0]}): {best_side_info[1]:.2f}%")
    print(f"   - Amélioration: {improvement:+.2f}%")
    
    if improvement > 2:
        print("\n   ✓ La side-information apporte un gain significatif!")
        print("   → Patterns conditionnels détectés, stratégie potentiellement exploitable")
    elif improvement > 0.5:
        print("\n   ~ Gain marginal avec side-information")
        print("   → Signal faible, tester sur d'autres actifs ou périodes")
    else:
        print("\n   ✗ Aucune amélioration significative")
        print("   → Le marché reste très efficient, CTW binaire atteint ses limites")
    
    print("="*80)


if __name__ == "__main__":
    run_depth_comparison()
