import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from CTW import CTW

def get_sp500_data(start_date="2020-01-01", end_date="2024-01-01"):
    """
    Télécharge le S&P 500 (^GSPC) et transforme les prix en séquence binaire.
    0 = Baisse, 1 = Hausse
    """
    print(f"Chargement des données S&P 500 du {start_date} au {end_date}...")
    # Téléchargement via yfinance
    df = yf.download("^GSPC", start=start_date, end=end_date, progress=False)
    
    # On ne garde que le prix de clôture ('Close')
    # .values convertit la colonne pandas en un simple tableau numpy
    prices = df['Close'].values.flatten() # flatten() assure que c'est une liste 1D
    
    print(f"{len(prices)} jours de bourse récupérés.")

    # Transformation en binaire (0 ou 1)
    binary_seq = []
    real_prices = [] # On garde les prix correspondants pour l'affichage si besoin
    
    for i in range(1, len(prices)):
        # Si le prix d'aujourd'hui est >= prix d'hier -> 1 (Hausse)
        # Sinon -> 0 (Baisse)
        if prices[i] >= prices[i-1]:
            binary_seq.append(1)
        else:
            binary_seq.append(0)
        real_prices.append(prices[i])
            
    return binary_seq, real_prices

def run_prediction():
    # 1. Récupération des données réelles
    sequence, prices = get_sp500_data()
    
    # 2. Configuration du CTW
    # Profondeur D : Combien de jours passés le modèle regarde-t-il ?
    # D=5 est un bon début pour capturer une semaine de trading.
    depth = 5 
    model = CTW(depth=depth, symbols=2)
    
    print(f"Lancement du CTW (Profondeur D={depth})...")
    
    # 3. Prédiction
    # distributions sera de taille (2, N - D)
    distributions = model.predict_sequence(sequence)
    
    # 4. Évaluation
    # On compare la prédiction (proba > 0.5) à la réalité
    # Attention : Le tableau 'distributions' commence à l'index 'depth' de la séquence originale
    correct_predictions = 0
    total_predictions = distributions.shape[1]
    
    history_probs = [] # Pour tracer le graphique

    for i in range(total_predictions):
        # Index réel dans la séquence binaire
        real_idx = i + depth
        
        # Probabilité prédite de hausse (symbole 1)
        prob_up = distributions[1, i]
        history_probs.append(prob_up)
        
        # Notre pari : si proba > 0.5 on dit 1, sinon 0
        prediction = 1 if prob_up >= 0.5 else 0
        reality = sequence[real_idx]
        
        if prediction == reality:
            correct_predictions += 1

    accuracy = (correct_predictions / total_predictions) * 100
    print(f"\n=== RÉSULTATS ===")
    print(f"Précision globale : {accuracy:.2f}%")
    print("Note : 50% = hasard complet (pile ou face).")
    
    # 5. Visualisation (Zoom sur les 100 derniers jours)
    plt.figure(figsize=(12, 6))
    last_n = 100
    
    # La ligne de confiance (proba de hausse)
    plt.plot(history_probs[-last_n:], label="Probabilité de Hausse (CTW)", color='blue')
    # La ligne de neutralité
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label="Incertitude (0.5)")
    
    # On ajoute des points pour la réalité (1=Hausse en haut, 0=Baisse en bas)
    real_vals = sequence[-last_n:]
    plt.scatter(range(last_n), real_vals, c='green', marker='x', label="Réalité (0 ou 1)", alpha=0.6)
    
    plt.title(f"Prédiction CTW sur S&P 500 (D={depth}) - {last_n} derniers jours")
    plt.xlabel("Jours")
    plt.ylabel("Probabilité P(Hausse)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    run_prediction()