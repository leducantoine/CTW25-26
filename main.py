import numpy as np
from CTW import CTW  # Importe la classe depuis votre fichier ctw.py

def test_ctw_simple():
    print("=== DÉBUT DU TEST CTW ===")

    # 1. Configuration des paramètres
    # D=2 : L'arbre regarde jusqu'à 2 coups en arrière (ex: contexte "0,1")
    depth = 2
    symbols = 2      # Binaire : 0 ou 1
    sidesymbols = 1  # Pas d'info latérale pour ce test simple
    
    # Initialisation du modèle
    model = CTW(depth=depth, symbols=symbols, sidesymbols=sidesymbols)
    
    # 2. Création d'une séquence déterministe
    # Motif : 0, 1, 0, 1... Le modèle devrait vite comprendre l'alternance.
    sequence = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    print(f"Séquence totale à traiter : {sequence}")
    print(f"Les {depth} premiers symboles servent à initialiser le contexte.\n")

    # 3. Lancement de la prédiction
    # sideseq=None car on teste le fonctionnement de base sans signal externe
    distributions = model.predict_sequence(sequence, sideseq=None)

    # 4. Affichage des résultats étape par étape
    # On commence la boucle après la profondeur D (les premiers éléments sont le contexte initial)
    nb_predictions = distributions.shape[1]
    
    for i in range(nb_predictions):
        # L'index réel dans la séquence d'origine
        real_index = i + depth
        
        # Le contexte que le modèle voit (les D symboles précédents)
        context_view = sequence[real_index-depth : real_index]
        
        # La vraie valeur qui vient d'arriver
        observation = sequence[real_index]
        
        # La prédiction faite par le modèle (avant de voir l'observation)
        probs = distributions[:, i]
        p0 = probs[0]
        p1 = probs[1]
        
        print(f"--- Étape {i+1} (Index {real_index}) ---")
        print(f"Contexte passé : {context_view}")
        print(f"Prédictions    : P(0)={p0:.3f} | P(1)={p1:.3f}")
        print(f"Observation    : {observation}")
        
        # Analyse rapide
        predicted_best = np.argmax(probs)
        if predicted_best == observation:
            # Si la probabilité du bon symbole est > 50%
            confidence = probs[observation]
            print(f"✅ SUCCÈS : Le modèle avait prédit '{predicted_best}' avec {confidence:.1%} de confiance.")
        else:
            print(f"⚠️ APPRENTISSAGE : Le modèle s'est trompé (ou est incertain).")

if __name__ == "__main__":
    test_ctw_simple()