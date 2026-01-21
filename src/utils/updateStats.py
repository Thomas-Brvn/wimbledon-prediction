from datetime import datetime
from typing import Sequence, List

BASE_ELO = 1500

def createStats(base_elo=BASE_ELO):
    """Initialise le dictionnaire de statistiques pour tous les joueurs."""
    from collections import defaultdict, deque

    prev_stats = {}
    prev_stats["elo_players"] = defaultdict(lambda: base_elo)
    prev_stats["elo_surface_players"] = defaultdict(lambda: defaultdict(lambda: base_elo))
    prev_stats["elo_grad_players"] = defaultdict(lambda: deque(maxlen=1000))
    prev_stats["last_k_matches"] = defaultdict(lambda: deque(maxlen=1000))
    prev_stats["last_k_matches_stats"] = defaultdict(lambda: defaultdict(lambda: deque(maxlen=1000)))
    prev_stats["matches_played"] = defaultdict(int)
    prev_stats["matches_surface_played"] = defaultdict(lambda: defaultdict(int))
    prev_stats["h2h"] = defaultdict(int)
    prev_stats["h2h_surface"] = defaultdict(lambda: defaultdict(int))
    prev_stats["last_tourney"] = defaultdict(lambda: deque(maxlen=20))
    prev_stats["last_tourney_surface"] = defaultdict(lambda: defaultdict(lambda: deque(maxlen=20)))

    return prev_stats


def days_between(date1: str, date2: str) -> int:
    """Calcule le nombre de jours entre deux dates au format YYYYMMDD."""
    d1 = datetime.strptime(date1, "%Y%m%d")
    d2 = datetime.strptime(date2, "%Y%m%d")
    return abs((d2 - d1).days)


def gaps_between_tourneys(played: Sequence[str]) -> List[int]:
    """Retourne les ecarts en jours entre les tournois, du plus recent au plus ancien."""
    if len(played) < 2:
        return []
    sorted_dates = sorted(played)
    gaps = [days_between(sorted_dates[i-1], sorted_dates[i])
            for i in range(1, len(sorted_dates))]
    gaps.reverse()
    return gaps


def average_days_between_tourneys(played: Sequence[str]) -> float:
    """Calcule l'ecart moyen en jours entre les tournois."""
    gaps = gaps_between_tourneys(played)
    return (sum(gaps) / len(gaps)) if gaps else 0.0


def k_bonus_after_layoff(played: Sequence[str], threshold: int = 100) -> float:
    """
    Bonus multiplicateur pour le K-factor apres une longue pause.
    Commence a 1.5x si la derniere pause >= threshold jours.
    Reduit de 0.05 pour chaque pause courte consecutive.
    """
    gaps = gaps_between_tourneys(played)
    if len(gaps) <= 15:
        return 1.0

    bonus = 1.5
    for g in gaps:
        if g < threshold:
            bonus -= 0.05
        else:
            break
    return max(1.0, bonus)


def calculate_k_factor(n_games: int,
                       base_k: float,
                       max_k: float,
                       div_number: float,
                       last_played: Sequence[str],
                       bonus_after_layoff: bool) -> float:
    """
    Calcule le K-factor dynamique pour l'ELO.
    K plus eleve pour les nouveaux joueurs, plus bas pour les joueurs etablis.
    """
    if bonus_after_layoff:
        bonus_mul = k_bonus_after_layoff(last_played)
    else:
        bonus_mul = 1

    k = (base_k + div_number / (n_games + 1)) * bonus_mul
    return min(k, max_k)


def round_importance(code: str) -> int:
    """Convertit le code du tour en valeur numerique (1=Finale, 13=Q1)."""
    ROUND_RANK = {
        'F': 1, 'BR': 2, 'SF': 3, 'QF': 4,
        'R16': 5, 'R32': 6, 'R64': 7, 'R128': 8,
        'RR': 9, 'ER': 10, 'Q3': 11, 'Q2': 12, 'Q1': 13,
    }
    return ROUND_RANK[code]


def updateStats(match, prev_stats, k_factor, base_k_factor, max_k_factor, div_number, bonus_after_layoff):
    """
    Met a jour les statistiques apres un match.
    Calcule les nouveaux ELO et enregistre les stats de service/retour.
    """
    from utils.common import mean, getWinnerLoserIDS
    import numpy as np

    p1_id, p2_id, surface, result = match.p1_id, match.p2_id, match.surface, match.RESULT
    w_id, l_id = getWinnerLoserIDS(p1_id, p2_id, result)

    # Date du dernier tournoi
    prev_stats["last_tourney"][w_id].append(str(match.tourney_date))
    prev_stats["last_tourney"][l_id].append(str(match.tourney_date))
    prev_stats["last_tourney_surface"][surface][w_id].append(str(match.tourney_date))
    prev_stats["last_tourney_surface"][surface][l_id].append(str(match.tourney_date))

    # ELO avant le match
    initial_elo = BASE_ELO
    elo_w = prev_stats["elo_players"].get(w_id, initial_elo)
    elo_l = prev_stats["elo_players"].get(l_id, initial_elo)
    elo_surface_w = prev_stats["elo_surface_players"][surface].get(w_id, initial_elo)
    elo_surface_l = prev_stats["elo_surface_players"][surface].get(l_id, initial_elo)

    # Calcul du K-factor
    if k_factor is not None:
        k_factor_w = k_factor_l = k_factor_surface_w = k_factor_surface_l = k_factor
    else:
        w_seq_all = prev_stats["last_tourney"][w_id]
        l_seq_all = prev_stats["last_tourney"][l_id]
        w_seq_srf = prev_stats["last_tourney_surface"][surface][w_id]
        l_seq_srf = prev_stats["last_tourney_surface"][surface][l_id]

        k_factor_w = calculate_k_factor(prev_stats["matches_played"][w_id], base_k_factor, max_k_factor, div_number, w_seq_all, bonus_after_layoff)
        k_factor_l = calculate_k_factor(prev_stats["matches_played"][l_id], base_k_factor, max_k_factor, div_number, l_seq_all, bonus_after_layoff)
        k_factor_surface_w = calculate_k_factor(prev_stats["matches_surface_played"][surface][w_id], base_k_factor, max_k_factor, div_number, w_seq_srf, bonus_after_layoff)
        k_factor_surface_l = calculate_k_factor(prev_stats["matches_surface_played"][surface][l_id], base_k_factor, max_k_factor, div_number, l_seq_srf, bonus_after_layoff)

    # Probabilites attendues
    exp_w = 1/(1+(10**((elo_l-elo_w)/400)))
    exp_l = 1/(1+(10**((elo_w-elo_l)/400)))
    exp_surface_w = 1/(1+(10**((elo_surface_l-elo_surface_w)/400)))
    exp_surface_l = 1/(1+(10**((elo_surface_w-elo_surface_l)/400)))

    # Mise a jour ELO
    elo_w += k_factor_w*(1-exp_w)
    elo_l += k_factor_l*(0-exp_l)
    elo_surface_w += k_factor_surface_w*(1-exp_surface_w)
    elo_surface_l += k_factor_surface_l*(0-exp_surface_l)

    prev_stats["elo_players"][w_id] = elo_w
    prev_stats["elo_players"][l_id] = elo_l
    prev_stats["elo_surface_players"][surface][w_id] = elo_surface_w
    prev_stats["elo_surface_players"][surface][l_id] = elo_surface_l

    # Historique ELO pour le gradient
    prev_stats["elo_grad_players"][w_id].append(elo_w)
    prev_stats["elo_grad_players"][l_id].append(elo_l)

    # Nombre de matchs joues
    prev_stats["matches_played"][w_id] += 1
    prev_stats["matches_played"][l_id] += 1
    prev_stats["matches_surface_played"][surface][w_id] += 1
    prev_stats["matches_surface_played"][surface][l_id] += 1

    # Victoires/defaites recentes
    prev_stats["last_k_matches"][w_id].append(1)
    prev_stats["last_k_matches"][l_id].append(0)

    # Face a face
    prev_stats["h2h"][(w_id, l_id)] += 1
    prev_stats["h2h_surface"][surface][(w_id, l_id)] += 1

    # Stats de service et retour
    if p1_id == getWinnerLoserIDS(p1_id, p2_id, result)[0]:
        w_ace, l_ace = match.p1_ace, match.p2_ace
        w_df, l_df = match.p1_df, match.p2_df
        w_svpt, l_svpt = match.p1_svpt, match.p2_svpt
        w_1stIn, l_1stIn = match.p1_1stIn, match.p2_1stIn
        w_1stWon, l_1stWon = match.p1_1stWon, match.p2_1stWon
        w_2ndWon, l_2ndWon = match.p1_2ndWon, match.p2_2ndWon
        w_bpSaved, l_bpSaved = match.p1_bpSaved, match.p2_bpSaved
        w_bpFaced, l_bpFaced = match.p1_bpFaced, match.p2_bpFaced
    else:
        w_ace, l_ace = match.p2_ace, match.p1_ace
        w_df, l_df = match.p2_df, match.p1_df
        w_svpt, l_svpt = match.p2_svpt, match.p1_svpt
        w_1stIn, l_1stIn = match.p2_1stIn, match.p1_1stIn
        w_1stWon, l_1stWon = match.p2_1stWon, match.p1_1stWon
        w_2ndWon, l_2ndWon = match.p2_2ndWon, match.p1_2ndWon
        w_bpSaved, l_bpSaved = match.p2_bpSaved, match.p1_bpSaved
        w_bpFaced, l_bpFaced = match.p2_bpFaced, match.p1_bpFaced

    # Stats de service
    if (w_svpt != 0) and (w_svpt != w_1stIn):
        prev_stats["last_k_matches_stats"][w_id]["p_ace"].append(100*(w_ace/w_svpt))
        prev_stats["last_k_matches_stats"][w_id]["p_df"].append(100*(w_df/w_svpt))
        prev_stats["last_k_matches_stats"][w_id]["p_1stIn"].append(100*(w_1stIn/w_svpt))
        prev_stats["last_k_matches_stats"][w_id]["p_2ndWon"].append(100*(w_2ndWon/(w_svpt-w_1stIn)))
    if l_svpt != 0 and (l_svpt != l_1stIn):
        prev_stats["last_k_matches_stats"][l_id]["p_ace"].append(100*(l_ace/l_svpt))
        prev_stats["last_k_matches_stats"][l_id]["p_df"].append(100*(l_df/l_svpt))
        prev_stats["last_k_matches_stats"][l_id]["p_1stIn"].append(100*(l_1stIn/l_svpt))
        prev_stats["last_k_matches_stats"][l_id]["p_2ndWon"].append(100*(l_2ndWon/(l_svpt-l_1stIn)))

    if w_1stIn != 0:
        prev_stats["last_k_matches_stats"][w_id]["p_1stWon"].append(100*(w_1stWon/w_1stIn))
    if l_1stIn != 0:
        prev_stats["last_k_matches_stats"][l_id]["p_1stWon"].append(100*(l_1stWon/l_1stIn))

    if w_bpFaced != 0:
        prev_stats["last_k_matches_stats"][w_id]["p_bpSaved"].append(100*(w_bpSaved/w_bpFaced))
    if l_bpFaced != 0:
        prev_stats["last_k_matches_stats"][l_id]["p_bpSaved"].append(100*(l_bpSaved/l_bpFaced))

    # Stats de retour
    if l_svpt != 0:
        w_rpw = (l_svpt - l_1stWon - l_2ndWon) / l_svpt
        prev_stats["last_k_matches_stats"][w_id]["p_rpw"].append(100*(w_rpw))
        prev_stats["last_k_matches_stats"][w_id]["p_retAceAgainst"].append(100*(l_ace / l_svpt))
    if l_1stIn != 0:
        prev_stats["last_k_matches_stats"][w_id]["p_ret1stWon"].append(100*((l_1stIn - l_1stWon) / l_1stIn))
    if (l_svpt - l_1stIn) != 0:
        prev_stats["last_k_matches_stats"][w_id]["p_ret2ndWon"].append(100*(((l_svpt - l_1stIn) - l_2ndWon) / (l_svpt - l_1stIn)))
    if l_bpFaced != 0:
        prev_stats["last_k_matches_stats"][w_id]["p_bpConv"].append(100*((l_bpFaced - l_bpSaved) / l_bpFaced))

    if w_svpt != 0:
        l_rpw = (w_svpt - w_1stWon - w_2ndWon) / w_svpt
        prev_stats["last_k_matches_stats"][l_id]["p_rpw"].append(100*(l_rpw))
        prev_stats["last_k_matches_stats"][l_id]["p_retAceAgainst"].append(100*(w_ace / w_svpt))
    if w_1stIn != 0:
        prev_stats["last_k_matches_stats"][l_id]["p_ret1stWon"].append(100*((w_1stIn - w_1stWon) / w_1stIn))
    if (w_svpt - w_1stIn) != 0:
        prev_stats["last_k_matches_stats"][l_id]["p_ret2ndWon"].append(100*(((w_svpt - w_1stIn) - w_2ndWon) / (w_svpt - w_1stIn)))
    if w_bpFaced != 0:
        prev_stats["last_k_matches_stats"][l_id]["p_bpConv"].append(100*((w_bpFaced - w_bpSaved) / w_bpFaced))

    # Total points gagnes
    total_pts = w_svpt + l_svpt
    if total_pts != 0:
        w_tpw = (w_1stWon + w_2ndWon) + (l_svpt - l_1stWon - l_2ndWon)
        prev_stats["last_k_matches_stats"][w_id]["p_totalPtsWon"].append(100*(w_tpw / total_pts))
        l_tpw = (l_1stWon + l_2ndWon) + (w_svpt - w_1stWon - w_2ndWon)
        prev_stats["last_k_matches_stats"][l_id]["p_totalPtsWon"].append(100*(l_tpw / total_pts))

    # Ratio de dominance = (% points retour gagnes) / (% points service perdus)
    if (w_svpt != 0) and (l_svpt != 0):
        w_spw = (w_1stWon + w_2ndWon) / w_svpt
        l_spw = (l_1stWon + l_2ndWon) / l_svpt
        w_spl = 1.0 - w_spw
        l_spl = 1.0 - l_spw
        if w_spl > 0:
            prev_stats["last_k_matches_stats"][w_id]["dominance_ratio"].append(100*(w_rpw / w_spl))
        if l_spl > 0:
            prev_stats["last_k_matches_stats"][l_id]["dominance_ratio"].append(100*(l_rpw / l_spl))

    return prev_stats


def getStats(player1, player2, match, prev_stats):
    """
    Calcule les features pour un match entre deux joueurs.
    Retourne un dict avec les differences de stats entre joueur1 et joueur2.
    """
    from utils.common import mean, getWinnerLoserIDS
    import numpy as np

    output = {}
    PLAYER1_ID = player1["ID"]
    PLAYER2_ID = player2["ID"]
    SURFACE = match["SURFACE"]

    # Infos du match
    output["BEST_OF"] = match["BEST_OF"]
    output["DRAW_SIZE"] = match["DRAW_SIZE"]
    output["ROUND"] = round_importance(match["ROUND"])
    output["AGE_DIFF"] = player1["AGE"]-player2["AGE"]
    output["HEIGHT_DIFF"] = player1["HEIGHT"]-player2["HEIGHT"]
    output["ATP_RANK_DIFF"] = player1["ATP_RANK"]-player2["ATP_RANK"]

    elo_players = prev_stats["elo_players"]
    elo_surface_players = prev_stats["elo_surface_players"]
    elo_grad_players = prev_stats["elo_grad_players"]
    last_k_matches = prev_stats["last_k_matches"]
    last_k_matches_stats = prev_stats["last_k_matches_stats"]
    matches_played = prev_stats["matches_played"]
    h2h = prev_stats["h2h"]
    h2h_surface = prev_stats["h2h_surface"]

    # Differences ELO et H2H
    output["ELO_DIFF"] = elo_players[PLAYER1_ID] - elo_players[PLAYER2_ID]
    output["ELO_SURFACE_DIFF"] = elo_surface_players[SURFACE][PLAYER1_ID] - elo_surface_players[SURFACE][PLAYER2_ID]
    output["N_GAMES_DIFF"] = matches_played[PLAYER1_ID] - matches_played[PLAYER2_ID]
    output["H2H_DIFF"] = h2h[(PLAYER1_ID, PLAYER2_ID)] - h2h[(PLAYER2_ID, PLAYER1_ID)]
    output["H2H_SURFACE_DIFF"] = h2h_surface[SURFACE][(PLAYER1_ID, PLAYER2_ID)] - h2h_surface[SURFACE][(PLAYER2_ID, PLAYER1_ID)]

    # Stats sur les k derniers matchs
    for k in [3, 10, 25, 50, 100]:
        # Victoires recentes
        if len(last_k_matches[PLAYER1_ID]) >= k and len(last_k_matches[PLAYER2_ID]) >= k:
            output["WIN_LAST_"+str(k)+"_DIFF"] = sum(list(last_k_matches[PLAYER1_ID])[-k:])-sum(list(last_k_matches[PLAYER2_ID])[-k:])
        else:
            output["WIN_LAST_"+str(k)+"_DIFF"] = 0

        # Gradient ELO (tendance)
        if len(elo_grad_players[PLAYER1_ID]) >= k and len(elo_grad_players[PLAYER2_ID]) >= k:
            elo_grad_p1 = list(elo_grad_players[PLAYER1_ID])[-k:]
            elo_grad_p2 = list(elo_grad_players[PLAYER2_ID])[-k:]
            slope_1 = np.polyfit(np.arange(len(elo_grad_p1)), np.array(elo_grad_p1), 1)[0]
            slope_2 = np.polyfit(np.arange(len(elo_grad_p2)), np.array(elo_grad_p2), 1)[0]
            output["ELO_GRAD_LAST_"+str(k)+"_DIFF"] = slope_1-slope_2
        else:
            output["ELO_GRAD_LAST_"+str(k)+"_DIFF"] = 0

        # Stats de service
        output["P_ACE_LAST_"+str(k)+"_DIFF"] = mean(list(last_k_matches_stats[PLAYER1_ID]["p_ace"])[-k:])-mean(list(last_k_matches_stats[PLAYER2_ID]["p_ace"])[-k:])
        output["P_DF_LAST_"+str(k)+"_DIFF"] = mean(list(last_k_matches_stats[PLAYER1_ID]["p_df"])[-k:])-mean(list(last_k_matches_stats[PLAYER2_ID]["p_df"])[-k:])
        output["P_1ST_IN_LAST_"+str(k)+"_DIFF"] = mean(list(last_k_matches_stats[PLAYER1_ID]["p_1stIn"])[-k:])-mean(list(last_k_matches_stats[PLAYER2_ID]["p_1stIn"])[-k:])
        output["P_1ST_WON_LAST_"+str(k)+"_DIFF"] = mean(list(last_k_matches_stats[PLAYER1_ID]["p_1stWon"])[-k:])-mean(list(last_k_matches_stats[PLAYER2_ID]["p_1stWon"])[-k:])
        output["P_2ND_WON_LAST_"+str(k)+"_DIFF"] = mean(list(last_k_matches_stats[PLAYER1_ID]["p_2ndWon"])[-k:])-mean(list(last_k_matches_stats[PLAYER2_ID]["p_2ndWon"])[-k:])
        output["P_BP_SAVED_LAST_"+str(k)+"_DIFF"] = mean(list(last_k_matches_stats[PLAYER1_ID]["p_bpSaved"])[-k:])-mean(list(last_k_matches_stats[PLAYER2_ID]["p_bpSaved"])[-k:])

        # Stats de retour
        output["P_RPW_LAST_"+str(k)+"_DIFF"] = mean(list(last_k_matches_stats[PLAYER1_ID]["p_rpw"])[-k:])-mean(list(last_k_matches_stats[PLAYER2_ID]["p_rpw"])[-k:])
        output["P_RET_1ST_WON_LAST_"+str(k)+"_DIFF"] = mean(list(last_k_matches_stats[PLAYER1_ID]["p_ret1stWon"])[-k:])-mean(list(last_k_matches_stats[PLAYER2_ID]["p_ret1stWon"])[-k:])
        output["P_RET_2ND_WON_LAST_"+str(k)+"_DIFF"] = mean(list(last_k_matches_stats[PLAYER1_ID]["p_ret2ndWon"])[-k:])-mean(list(last_k_matches_stats[PLAYER2_ID]["p_ret2ndWon"])[-k:])
        output["P_BP_CONV_LAST_"+str(k)+"_DIFF"] = mean(list(last_k_matches_stats[PLAYER1_ID]["p_bpConv"])[-k:])-mean(list(last_k_matches_stats[PLAYER2_ID]["p_bpConv"])[-k:])
        output["P_RET_ACE_AGAINST_LAST_"+str(k)+"_DIFF"] = mean(list(last_k_matches_stats[PLAYER1_ID]["p_retAceAgainst"])[-k:])-mean(list(last_k_matches_stats[PLAYER2_ID]["p_retAceAgainst"])[-k:])

        # Autres stats
        output["P_TOTAL_PTS_WON_LAST_"+str(k)+"_DIFF"] = mean(list(last_k_matches_stats[PLAYER1_ID]["p_totalPtsWon"])[-k:])-mean(list(last_k_matches_stats[PLAYER2_ID]["p_totalPtsWon"])[-k:])
        output["DOMINANCE_RATIO_LAST_"+str(k)+"_DIFF"] = mean(list(last_k_matches_stats[PLAYER1_ID]["dominance_ratio"])[-k:])-mean(list(last_k_matches_stats[PLAYER2_ID]["dominance_ratio"])[-k:])

    return output
