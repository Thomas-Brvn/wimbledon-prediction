def mean(arr):
    """Calcule la moyenne d'une liste. Retourne 0.5 si vide."""
    if len(arr) == 0:
        return 0.5
    return sum(arr) / len(arr)

def getWinnerLoserIDS(p1_id, p2_id, result):
    """Retourne (winner_id, loser_id) selon le resultat."""
    if result == 1 or result == "1":
        return p1_id, p2_id
    return p2_id, p1_id
