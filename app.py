import pandas as pd
import numpy as np
import joblib
from flask import Flask, request, render_template

app = Flask(__name__)

match_df = pd.read_csv("match_df.csv")
model = joblib.load("ipl_model.pkl")

def get_matches_played(team, past_df):
    if past_df.empty:
        return 0
    played = past_df[(past_df["team1"] == team) | (past_df["team2"] == team)]
    return len(played)

def get_team_win_rate(team, past_df):
    if past_df.empty:
        return 0.5
    played = past_df[(past_df["team1"] == team) | (past_df["team2"] == team)]
    if len(played) == 0:
        return 0.5
    wins = (played["winner"] == team).sum()
    return wins / len(played)

def get_recent_win_rate(team, past_df, n=5):
    if past_df.empty:
        return 0.5
    played = past_df[(past_df["team1"] == team) | (past_df["team2"] == team)].sort_values("date").tail(n)
    if len(played) == 0:
        return 0.5
    wins = (played["winner"] == team).sum()
    return wins / len(played)

def get_current_streak(team, past_df):
    if past_df.empty:
        return 0
    
    played = past_df[(past_df["team1"] == team) | (past_df["team2"] == team)].sort_values("date")
    if len(played) == 0:
        return 0
    
    streak = 0
    
    for _, row in played.iloc[::-1].iterrows():
        if row["winner"] == team:
            if streak >= 0:
                streak += 1
            else:
                break
        else:
            if streak <= 0:
                streak -= 1
            else:
                break
    return streak

def get_h2h_win_rate(teamA, teamB, past_df):
    if past_df.empty:
        return 0.5
    h2h = past_df[
        ((past_df["team1"] == teamA) & (past_df["team2"] == teamB)) |
        ((past_df["team1"] == teamB) & (past_df["team2"] == teamA))
    ]
    if len(h2h) == 0:
        return 0.5
    wins = (h2h["winner"] == teamA).sum()
    return wins / len(h2h)

def get_h2h_matches(teamA, teamB, past_df):
    if past_df.empty:
        return 0
    h2h = past_df[
        ((past_df["team1"] == teamA) & (past_df["team2"] == teamB)) |
        ((past_df["team1"] == teamB) & (past_df["team2"] == teamA))
    ]
    return len(h2h)

def get_venue_win_rate(team, venue, past_df):
    if past_df.empty:
        return 0.5
    venue_matches = past_df[
        (((past_df["team1"] == team) | (past_df["team2"] == team)) &
         (past_df["venue"] == venue))
    ]
    if len(venue_matches) == 0:
        return 0.5
    wins = (venue_matches["winner"] == team).sum()
    return wins / len(venue_matches)
    

def get_team_venue_matches(team, venue, past_df):
    if past_df.empty:
        return 0
    venue_matches = past_df[
        (((past_df["team1"] == team) | (past_df["team2"] == team)) &
         (past_df["venue"] == venue))
    ]
    return len(venue_matches)

def get_chase_win_rate(team, past_df):
    if past_df.empty:
        return 0.5
    chase_matches = past_df[past_df["chasing_team"] == team]
    if len(chase_matches) == 0:
        return 0.5
    wins = (chase_matches["winner"] == team).sum()
    return wins / len(chase_matches)

def get_defend_win_rate(team, past_df):
    if past_df.empty:
        return 0.5
    defend_matches = past_df[past_df["defending_team"] == team]
    if len(defend_matches) == 0:
        return 0.5
    wins = (defend_matches["winner"] == team).sum()
    return wins / len(defend_matches)

def get_chase_matches(team, past_df):
    if past_df.empty:
        return 0
    return len(past_df[past_df["chasing_team"] == team])

def get_defend_matches(team, past_df):
    if past_df.empty:
        return 0
    return len(past_df[past_df["defending_team"] == team])

def get_venue_chase_bias(venue, past_df):
    if past_df.empty:
        return 0.5
    venue_matches = past_df[past_df["venue"] == venue]
    if len(venue_matches) == 0:
        return 0.5
    return venue_matches["chase_win"].mean()

def get_team_chase_win_rate_at_venue(team, venue, past_df):
    if past_df.empty:
        return 0.5
    matches = past_df[
        (past_df["venue"] == venue) &
        (past_df["chasing_team"] == team)
    ]
    if len(matches) == 0:
        return 0.5
    wins = (matches["winner"] == team).sum()
    return wins / len(matches)

def get_team_defend_win_rate_at_venue(team, venue, past_df):
    if past_df.empty:
        return 0.5
    matches = past_df[
        (past_df["venue"] == venue) &
        (past_df["defending_team"] == team)
    ]
    if len(matches) == 0:
        return 0.5
    wins = (matches["winner"] == team).sum()
    return wins / len(matches)

def predict_current_match_detailed(model, match_features_df):
    probs = model.predict_proba(match_features_df)[0]
    pred = int(model.predict(match_features_df)[0])

    row = match_features_df.iloc[0]
    team1 = row["team1"]
    team2 = row["team2"]

    predicted_winner = team1 if pred == 1 else team2

    return predicted_winner

def build_current_match_features(team1, team2, venue, city, toss_winner, toss_decision, match_df):
    # Decide chasing/defending teams
    if toss_decision == "bat":
        defending_team = toss_winner
        chasing_team = team2 if toss_winner == team1 else team1
    else:
        chasing_team = toss_winner
        defending_team = team2 if toss_winner == team1 else team1

    past_df = match_df.copy().sort_values("date").reset_index(drop=True)

    team1_matches = get_matches_played(team1, past_df)
    team2_matches = get_matches_played(team2, past_df)

    team1_win_rate = get_team_win_rate(team1, past_df)
    team2_win_rate = get_team_win_rate(team2, past_df)

    team1_recent3 = get_recent_win_rate(team1, past_df, 3)
    team2_recent3 = get_recent_win_rate(team2, past_df, 3)

    team1_recent5 = get_recent_win_rate(team1, past_df, 5)
    team2_recent5 = get_recent_win_rate(team2, past_df, 5)

    team1_recent10 = get_recent_win_rate(team1, past_df, 10)
    team2_recent10 = get_recent_win_rate(team2, past_df, 10)

    team1_streak = get_current_streak(team1, past_df)
    team2_streak = get_current_streak(team2, past_df)

    h2h_team1 = get_h2h_win_rate(team1, team2, past_df)
    h2h_team2 = get_h2h_win_rate(team2, team1, past_df)
    h2h_matches = get_h2h_matches(team1, team2, past_df)

    team1_venue_wr = get_venue_win_rate(team1, venue, past_df)
    team2_venue_wr = get_venue_win_rate(team2, venue, past_df)

    team1_venue_matches = get_team_venue_matches(team1, venue, past_df)
    team2_venue_matches = get_team_venue_matches(team2, venue, past_df)

    team1_chase_wr = get_chase_win_rate(team1, past_df)
    team2_chase_wr = get_chase_win_rate(team2, past_df)

    team1_defend_wr = get_defend_win_rate(team1, past_df)
    team2_defend_wr = get_defend_win_rate(team2, past_df)

    team1_chase_matches = get_chase_matches(team1, past_df)
    team2_chase_matches = get_chase_matches(team2, past_df)

    team1_defend_matches = get_defend_matches(team1, past_df)
    team2_defend_matches = get_defend_matches(team2, past_df)

    venue_chase_bias = get_venue_chase_bias(venue, past_df)

    team1_chase_venue_wr = get_team_chase_win_rate_at_venue(team1, venue, past_df)
    team2_chase_venue_wr = get_team_chase_win_rate_at_venue(team2, venue, past_df)

    team1_defend_venue_wr = get_team_defend_win_rate_at_venue(team1, venue, past_df)
    team2_defend_venue_wr = get_team_defend_win_rate_at_venue(team2, venue, past_df)

    # Difference features
    win_rate_diff = team1_win_rate - team2_win_rate
    recent3_diff = team1_recent3 - team2_recent3
    recent5_diff = team1_recent5 - team2_recent5
    recent10_diff = team1_recent10 - team2_recent10
    streak_diff = team1_streak - team2_streak
    h2h_diff = h2h_team1 - h2h_team2
    venue_win_rate_diff = team1_venue_wr - team2_venue_wr
    matches_played_diff = team1_matches - team2_matches
    venue_matches_diff = team1_venue_matches - team2_venue_matches
    chase_win_rate_diff = team1_chase_wr - team2_chase_wr
    defend_win_rate_diff = team1_defend_wr - team2_defend_wr
    chase_venue_win_rate_diff = team1_chase_venue_wr - team2_chase_venue_wr
    defend_venue_win_rate_diff = team1_defend_venue_wr - team2_defend_venue_wr

    # Toss/context features
    team1_won_toss = int(toss_winner == team1)
    team2_won_toss = int(toss_winner == team2)

    team1_is_chasing = int(chasing_team == team1)
    team2_is_chasing = int(chasing_team == team2)

    team1_is_defending = int(defending_team == team1)
    team2_is_defending = int(defending_team == team2)

    team1_context_strength = team1_chase_wr if team1_is_chasing == 1 else team1_defend_wr
    team2_context_strength = team2_chase_wr if team2_is_chasing == 1 else team2_defend_wr
    context_strength_diff = team1_context_strength - team2_context_strength

    team1_venue_context_strength = team1_chase_venue_wr if team1_is_chasing == 1 else team1_defend_venue_wr
    team2_venue_context_strength = team2_chase_venue_wr if team2_is_chasing == 1 else team2_defend_venue_wr
    venue_context_strength_diff = team1_venue_context_strength - team2_venue_context_strength

    row = {
        "team1": team1,
        "team2": team2,
        "venue": venue,
        "city": city,
        "toss_winner": toss_winner,
        "toss_decision": toss_decision,
        "chasing_team": chasing_team,
        "defending_team": defending_team,

        "team1_matches_played": team1_matches,
        "team2_matches_played": team2_matches,
        "team1_win_rate": team1_win_rate,
        "team2_win_rate": team2_win_rate,
        "team1_recent3_win_rate": team1_recent3,
        "team2_recent3_win_rate": team2_recent3,
        "team1_recent5_win_rate": team1_recent5,
        "team2_recent5_win_rate": team2_recent5,
        "team1_recent10_win_rate": team1_recent10,
        "team2_recent10_win_rate": team2_recent10,
        "team1_streak": team1_streak,
        "team2_streak": team2_streak,
        "team1_vs_team2_h2h": h2h_team1,
        "team2_vs_team1_h2h": h2h_team2,
        "h2h_matches": h2h_matches,
        "team1_venue_win_rate": team1_venue_wr,
        "team2_venue_win_rate": team2_venue_wr,
        "team1_venue_matches": team1_venue_matches,
        "team2_venue_matches": team2_venue_matches,
        "team1_chase_win_rate": team1_chase_wr,
        "team2_chase_win_rate": team2_chase_wr,
        "team1_defend_win_rate": team1_defend_wr,
        "team2_defend_win_rate": team2_defend_wr,
        "team1_chase_matches": team1_chase_matches,
        "team2_chase_matches": team2_chase_matches,
        "team1_defend_matches": team1_defend_matches,
        "team2_defend_matches": team2_defend_matches,
        "venue_chase_bias": venue_chase_bias,
        "team1_chase_venue_win_rate": team1_chase_venue_wr,
        "team2_chase_venue_win_rate": team2_chase_venue_wr,
        "team1_defend_venue_win_rate": team1_defend_venue_wr,
        "team2_defend_venue_win_rate": team2_defend_venue_wr,

        "win_rate_diff": win_rate_diff,
        "recent3_diff": recent3_diff,
        "recent5_diff": recent5_diff,
        "recent10_diff": recent10_diff,
        "streak_diff": streak_diff,
        "h2h_diff": h2h_diff,
        "venue_win_rate_diff": venue_win_rate_diff,
        "matches_played_diff": matches_played_diff,
        "venue_matches_diff": venue_matches_diff,
        "chase_win_rate_diff": chase_win_rate_diff,
        "defend_win_rate_diff": defend_win_rate_diff,
        "chase_venue_win_rate_diff": chase_venue_win_rate_diff,
        "defend_venue_win_rate_diff": defend_venue_win_rate_diff,

        "team1_won_toss": team1_won_toss,
        "team2_won_toss": team2_won_toss,
        "team1_is_chasing": team1_is_chasing,
        "team2_is_chasing": team2_is_chasing,
        "team1_is_defending": team1_is_defending,
        "team2_is_defending": team2_is_defending,
        "team1_context_strength": team1_context_strength,
        "team2_context_strength": team2_context_strength,
        "context_strength_diff": context_strength_diff,
        "team1_venue_context_strength": team1_venue_context_strength,
        "team2_venue_context_strength": team2_venue_context_strength,
        "venue_context_strength_diff": venue_context_strength_diff
    }

    return pd.DataFrame([row])

def build_current_match_features(team1, team2, venue, city, toss_winner, toss_decision, match_df):
    # Decide chasing/defending teams
    if toss_decision == "bat":
        defending_team = toss_winner
        chasing_team = team2 if toss_winner == team1 else team1
    else:
        chasing_team = toss_winner
        defending_team = team2 if toss_winner == team1 else team1

    past_df = match_df.copy().sort_values("date").reset_index(drop=True)

    team1_matches = get_matches_played(team1, past_df)
    team2_matches = get_matches_played(team2, past_df)

    team1_win_rate = get_team_win_rate(team1, past_df)
    team2_win_rate = get_team_win_rate(team2, past_df)

    team1_recent3 = get_recent_win_rate(team1, past_df, 3)
    team2_recent3 = get_recent_win_rate(team2, past_df, 3)

    team1_recent5 = get_recent_win_rate(team1, past_df, 5)
    team2_recent5 = get_recent_win_rate(team2, past_df, 5)

    team1_recent10 = get_recent_win_rate(team1, past_df, 10)
    team2_recent10 = get_recent_win_rate(team2, past_df, 10)

    team1_streak = get_current_streak(team1, past_df)
    team2_streak = get_current_streak(team2, past_df)

    h2h_team1 = get_h2h_win_rate(team1, team2, past_df)
    h2h_team2 = get_h2h_win_rate(team2, team1, past_df)
    h2h_matches = get_h2h_matches(team1, team2, past_df)

    team1_venue_wr = get_venue_win_rate(team1, venue, past_df)
    team2_venue_wr = get_venue_win_rate(team2, venue, past_df)

    team1_venue_matches = get_team_venue_matches(team1, venue, past_df)
    team2_venue_matches = get_team_venue_matches(team2, venue, past_df)

    team1_chase_wr = get_chase_win_rate(team1, past_df)
    team2_chase_wr = get_chase_win_rate(team2, past_df)

    team1_defend_wr = get_defend_win_rate(team1, past_df)
    team2_defend_wr = get_defend_win_rate(team2, past_df)

    team1_chase_matches = get_chase_matches(team1, past_df)
    team2_chase_matches = get_chase_matches(team2, past_df)

    team1_defend_matches = get_defend_matches(team1, past_df)
    team2_defend_matches = get_defend_matches(team2, past_df)

    venue_chase_bias = get_venue_chase_bias(venue, past_df)

    team1_chase_venue_wr = get_team_chase_win_rate_at_venue(team1, venue, past_df)
    team2_chase_venue_wr = get_team_chase_win_rate_at_venue(team2, venue, past_df)

    team1_defend_venue_wr = get_team_defend_win_rate_at_venue(team1, venue, past_df)
    team2_defend_venue_wr = get_team_defend_win_rate_at_venue(team2, venue, past_df)

    # Difference features
    win_rate_diff = team1_win_rate - team2_win_rate
    recent3_diff = team1_recent3 - team2_recent3
    recent5_diff = team1_recent5 - team2_recent5
    recent10_diff = team1_recent10 - team2_recent10
    streak_diff = team1_streak - team2_streak
    h2h_diff = h2h_team1 - h2h_team2
    venue_win_rate_diff = team1_venue_wr - team2_venue_wr
    matches_played_diff = team1_matches - team2_matches
    venue_matches_diff = team1_venue_matches - team2_venue_matches
    chase_win_rate_diff = team1_chase_wr - team2_chase_wr
    defend_win_rate_diff = team1_defend_wr - team2_defend_wr
    chase_venue_win_rate_diff = team1_chase_venue_wr - team2_chase_venue_wr
    defend_venue_win_rate_diff = team1_defend_venue_wr - team2_defend_venue_wr

    # Toss/context features
    team1_won_toss = int(toss_winner == team1)
    team2_won_toss = int(toss_winner == team2)

    team1_is_chasing = int(chasing_team == team1)
    team2_is_chasing = int(chasing_team == team2)

    team1_is_defending = int(defending_team == team1)
    team2_is_defending = int(defending_team == team2)

    team1_context_strength = team1_chase_wr if team1_is_chasing == 1 else team1_defend_wr
    team2_context_strength = team2_chase_wr if team2_is_chasing == 1 else team2_defend_wr
    context_strength_diff = team1_context_strength - team2_context_strength

    team1_venue_context_strength = team1_chase_venue_wr if team1_is_chasing == 1 else team1_defend_venue_wr
    team2_venue_context_strength = team2_chase_venue_wr if team2_is_chasing == 1 else team2_defend_venue_wr
    venue_context_strength_diff = team1_venue_context_strength - team2_venue_context_strength

    row = {
        "team1": team1,
        "team2": team2,
        "venue": venue,
        "city": city,
        "toss_winner": toss_winner,
        "toss_decision": toss_decision,
        "chasing_team": chasing_team,
        "defending_team": defending_team,

        "team1_matches_played": team1_matches,
        "team2_matches_played": team2_matches,
        "team1_win_rate": team1_win_rate,
        "team2_win_rate": team2_win_rate,
        "team1_recent3_win_rate": team1_recent3,
        "team2_recent3_win_rate": team2_recent3,
        "team1_recent5_win_rate": team1_recent5,
        "team2_recent5_win_rate": team2_recent5,
        "team1_recent10_win_rate": team1_recent10,
        "team2_recent10_win_rate": team2_recent10,
        "team1_streak": team1_streak,
        "team2_streak": team2_streak,
        "team1_vs_team2_h2h": h2h_team1,
        "team2_vs_team1_h2h": h2h_team2,
        "h2h_matches": h2h_matches,
        "team1_venue_win_rate": team1_venue_wr,
        "team2_venue_win_rate": team2_venue_wr,
        "team1_venue_matches": team1_venue_matches,
        "team2_venue_matches": team2_venue_matches,
        "team1_chase_win_rate": team1_chase_wr,
        "team2_chase_win_rate": team2_chase_wr,
        "team1_defend_win_rate": team1_defend_wr,
        "team2_defend_win_rate": team2_defend_wr,
        "team1_chase_matches": team1_chase_matches,
        "team2_chase_matches": team2_chase_matches,
        "team1_defend_matches": team1_defend_matches,
        "team2_defend_matches": team2_defend_matches,
        "venue_chase_bias": venue_chase_bias,
        "team1_chase_venue_win_rate": team1_chase_venue_wr,
        "team2_chase_venue_win_rate": team2_chase_venue_wr,
        "team1_defend_venue_win_rate": team1_defend_venue_wr,
        "team2_defend_venue_win_rate": team2_defend_venue_wr,

        "win_rate_diff": win_rate_diff,
        "recent3_diff": recent3_diff,
        "recent5_diff": recent5_diff,
        "recent10_diff": recent10_diff,
        "streak_diff": streak_diff,
        "h2h_diff": h2h_diff,
        "venue_win_rate_diff": venue_win_rate_diff,
        "matches_played_diff": matches_played_diff,
        "venue_matches_diff": venue_matches_diff,
        "chase_win_rate_diff": chase_win_rate_diff,
        "defend_win_rate_diff": defend_win_rate_diff,
        "chase_venue_win_rate_diff": chase_venue_win_rate_diff,
        "defend_venue_win_rate_diff": defend_venue_win_rate_diff,

        "team1_won_toss": team1_won_toss,
        "team2_won_toss": team2_won_toss,
        "team1_is_chasing": team1_is_chasing,
        "team2_is_chasing": team2_is_chasing,
        "team1_is_defending": team1_is_defending,
        "team2_is_defending": team2_is_defending,
        "team1_context_strength": team1_context_strength,
        "team2_context_strength": team2_context_strength,
        "context_strength_diff": context_strength_diff,
        "team1_venue_context_strength": team1_venue_context_strength,
        "team2_venue_context_strength": team2_venue_context_strength,
        "venue_context_strength_diff": venue_context_strength_diff
    }

    return pd.DataFrame([row])

def predict_current_match_normalized(model, teamA, teamB, venue, city, toss_winner, toss_decision, match_df):
    # Orientation 1
    match_ab = build_current_match_features(
        team1=teamA,
        team2=teamB,
        venue=venue,
        city=city,
        toss_winner=toss_winner,
        toss_decision=toss_decision,
        match_df=match_df
    )
    probs_ab = model.predict_proba(match_ab)[0]
    p_a_from_ab = probs_ab[1]
    p_b_from_ab = probs_ab[0]

    # Orientation 2
    match_ba = build_current_match_features(
        team1=teamB,
        team2=teamA,
        venue=venue,
        city=city,
        toss_winner=toss_winner,
        toss_decision=toss_decision,
        match_df=match_df
    )
    probs_ba = model.predict_proba(match_ba)[0]
    p_a_from_ba = probs_ba[0]
    p_b_from_ba = probs_ba[1]

    # final normalized probabilities
    p_a_final = (p_a_from_ab + p_a_from_ba) / 2
    p_b_final = (p_b_from_ab + p_b_from_ba) / 2

    predicted_winner = teamA if p_a_final >= p_b_final else teamB

    return predicted_winner, p_a_final, p_b_final

@app.route("/")
def Home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def prediction():
    team1 = request.form['team1']
    team2 = request.form['team2']
    toss_decision = request.form['toss_decision']
    toss_winner = request.form['toss_winner']
    venue = request.form['venue']
    city = request.form['city']
    
    result, team1_prob, team2_prob = predict_current_match_normalized(
        model=model,
        teamA=team1,
        teamB=team2,
        venue=venue,
        city=city,
        toss_winner=toss_winner,
        toss_decision=toss_decision,
        match_df=match_df
    )
    return render_template(
        "index.html",
        prediction_text="Prediction for Today's Match is:",
        prediction_value=result,
        team1_probability_text=f"{team1} Win Probability: {team1_prob * 100:.2f}%",
        team2_probability_text=f"{team2} Win Probability: {team2_prob * 100:.2f}%",
        team1_probability=team1_prob,
        team2_probability=team2_prob
    )

if __name__ == "__main__":
    app.run(debug=True)