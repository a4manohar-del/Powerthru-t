"""
PowerThru onboarding analyzer + categorizer.

Flow: 1) 1-10 survey  ->  2) ~30 min guided calibration (rest/light/moderate/hard)
      3) build a personal baseline (mean/std per feature)
      4) compare self-rating vs measured capacity, assign a category (k-means)
      5) write profile JSON the phone app stores and pushes to the band over BLE.

Run:  python onboarding_model.py            (simulated user demo)
"""
import json
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from powerthru_common import (FEATURES, N_FEAT, FS_EMG, extract_emg_features,
                              synth_emg)

SURVEY = {  # key: prompt (all answered 1-10)
    "experience":     "How experienced are you at your sport/training?",
    "fitness":        "How would you rate your current fitness?",
    "recovery":       "How quickly do you usually recover after hard sessions?",
    "sleep":          "How well do you usually sleep?",
    "hydration":      "How well do you usually hydrate?",
    "cramp_freq":     "How often do you get muscle cramps? (10 = very often)",
    "soreness":       "How sore do you usually get after hard sessions? (10 = very)",
}
CALIBRATION = [("rest", 5), ("light", 8), ("moderate", 10), ("hard", 5), ("recovery", 2)]  # minutes = 30
CATEGORIES = ["Conservative", "Standard", "High-capacity"]
SENSITIVITY = {"Conservative": 1.25, "Standard": 1.0, "High-capacity": 0.85}  # alert threshold scaling


# ------------------------------------------------------------------ survey
def validate_survey(ans):
    missing = set(SURVEY) - set(ans)
    if missing:
        raise ValueError(f"missing survey answers: {sorted(missing)}")
    for k, v in ans.items():
        if not 1 <= float(v) <= 10:
            raise ValueError(f"{k} must be 1-10, got {v}")
    return {k: float(ans[k]) for k in SURVEY}


def self_rating(ans):
    pos = [ans[k] for k in ("experience", "fitness", "recovery", "sleep", "hydration")]
    neg = [11 - ans[k] for k in ("cramp_freq", "soreness")]       # invert so higher = better
    return float(np.mean(pos + neg))


# ------------------------------------------------------------------ calibration
def epochs_to_features(stage_epochs):
    """Raw per-stage data -> (T, 8) feature matrix.
    stage_epochs: dict with 'emg' (T,fs) array, 'acc' (T,) mean|a|, 'acc_sd' (T,), 'temp' (T,), 'sweat' (T,)"""
    rows = []
    for i in range(len(stage_epochs["temp"])):
        rms, mdf, zc, wl = extract_emg_features(stage_epochs["emg"][i])
        rows.append([rms, mdf, zc, wl, stage_epochs["acc"][i], stage_epochs["acc_sd"][i],
                     stage_epochs["temp"][i], stage_epochs["sweat"][i]])
    return np.array(rows)


def build_baseline(stage_feats):
    """Baseline = resting + light work (what 'normal' looks like for THIS person)."""
    ref = np.vstack([stage_feats["rest"], stage_feats["light"]])
    return ref.mean(0), np.maximum(ref.std(0), 1e-3 * np.abs(ref.mean(0)) + 1e-6)


def capacity_metrics(stage_feats):
    i = {n: k for k, n in enumerate(FEATURES)}
    rest, hard = stage_feats["rest"].mean(0), stage_feats["hard"].mean(0)
    light = stage_feats["light"].mean(0)
    rms_reserve = np.clip((hard[i["emg_rms"]] / (rest[i["emg_rms"]] + 1e-9) - 1) / 5.0, 0, 1)
    mdf_drop = np.clip((light[i["emg_mdf"]] - hard[i["emg_mdf"]]) / (light[i["emg_mdf"]] + 1e-9), 0, 1)
    fatigue_resist = 1 - np.clip(mdf_drop / 0.3, 0, 1)             # smaller spectral shift = more resistant
    sweat_resp = np.clip((hard[i["sweat_us"]] - rest[i["sweat_us"]]) / (rest[i["sweat_us"]] + 1e-9) / 3.0, 0, 1)
    measured = 1 + 9 * float(np.mean([rms_reserve, fatigue_resist]))
    return {"measured_capacity": measured, "rms_reserve": float(rms_reserve),
            "fatigue_resistance": float(fatigue_resist), "sweat_response": float(sweat_resp)}


# ------------------------------------------------------------------ categorizer
def _cluster_vec(self_r, cramp, measured, sweat):
    return [self_r, cramp, measured, sweat * 10]


def train_categorizer(n=800, seed=0):
    """k-means on a SYNTHETIC population (latent ability drives survey + measured values).
    Swap in real onboarding records once you have a few dozen users."""
    rng = np.random.default_rng(seed)
    ability = rng.uniform(1, 10, n)
    self_r = np.clip(ability + rng.normal(0, 1.5, n), 1, 10)
    cramp = np.clip(11 - ability + rng.normal(0, 2, n), 1, 10)
    measured = np.clip(ability + rng.normal(0, 1.2, n), 1, 10)
    sweat = np.clip(0.3 + 0.04 * ability + rng.normal(0, 0.12, n), 0, 1)
    X = np.array([_cluster_vec(a, b, c, d) for a, b, c, d in zip(self_r, cramp, measured, sweat)])
    sc = StandardScaler().fit(X)
    km = KMeans(n_clusters=3, n_init=10, random_state=seed).fit(sc.transform(X))
    # name clusters by a capacity score: high self+measured, low cramps
    order = np.argsort([c[0] + c[2] - c[1] for c in sc.inverse_transform(km.cluster_centers_)])
    names = {str(int(c)): CATEGORIES[r] for r, c in enumerate(order)}
    return {"scaler_mean": sc.mean_.tolist(), "scaler_scale": sc.scale_.tolist(),
            "centroids": km.cluster_centers_.tolist(), "names": names}


def categorize(model, vec):
    z = (np.array(vec) - model["scaler_mean"]) / model["scaler_scale"]
    d = np.linalg.norm(np.array(model["centroids"]) - z, axis=1)    # nearest-centroid = easy to port to the app
    return model["names"][str(int(np.argmin(d)))]


def build_profile(survey_answers, stage_feats, categorizer):
    ans = validate_survey(survey_answers)
    mean, std = build_baseline(stage_feats)
    cap = capacity_metrics(stage_feats)
    sr = self_rating(ans)
    cat = categorize(categorizer, _cluster_vec(sr, ans["cramp_freq"], cap["measured_capacity"], cap["sweat_response"]))
    gap = sr - cap["measured_capacity"]
    flags = []
    if gap > 2.5:
        flags.append("self-rating well above measured capacity: use conservative alerts early on")
    if cap["fatigue_resistance"] < 0.3:
        flags.append("large spectral shift under load: early fatigue signature")
    return {"category": cat, "alert_sensitivity": SENSITIVITY[cat], "self_rating": round(sr, 2),
            "capacity": {k: round(v, 3) for k, v in cap.items()}, "self_vs_measured_gap": round(gap, 2),
            "flags": flags, "features": FEATURES,
            "baseline_mean": mean.tolist(), "baseline_std": std.tolist()}


# ------------------------------------------------------------------ demo
def simulate_calibration(rng, minutes=CALIBRATION, ability=6.0):
    """Fake a 30-minute session at 1 epoch/s so the whole flow can be run without hardware."""
    base_rms, base_f = rng.uniform(40, 100), rng.uniform(95, 120)
    effort = {"rest": 0.15, "light": 1.0, "moderate": 2.5, "hard": 4.5, "recovery": 0.6}
    out = {}
    for name, mins in minutes:
        T = mins * 60
        emg = np.zeros((T, FS_EMG)); acc = np.zeros(T); acc_sd = np.zeros(T)
        temp = np.zeros(T); sweat = np.zeros(T)
        for t in range(T):
            e = effort[name] * rng.uniform(0.9, 1.1)
            drift = (t / T) * (0.25 - 0.015 * ability) if name in ("moderate", "hard") else 0
            emg[t] = synth_emg(FS_EMG, FS_EMG, base_rms * e, base_f * (1 - drift), rng)
            acc[t] = 1 + 0.3 * e + rng.normal(0, 0.05); acc_sd[t] = 0.05 + 0.1 * e
            temp[t] = 33 + 0.25 * e + rng.normal(0, 0.05)
            sweat[t] = 2 + 1.0 * e + rng.normal(0, 0.15)
        out[name] = {"emg": emg, "acc": acc, "acc_sd": acc_sd, "temp": temp, "sweat": sweat}
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(1)
    cat_model = train_categorizer()
    survey = {"experience": 8, "fitness": 7, "recovery": 7, "sleep": 6, "hydration": 5,
              "cramp_freq": 6, "soreness": 5}
    print("Simulating 30 min calibration (takes a few seconds)...")
    raw = simulate_calibration(rng)
    stage_feats = {name: epochs_to_features(d) for name, d in raw.items()}
    profile = build_profile(survey, stage_feats, cat_model)
    json.dump(cat_model, open("onboarding_categorizer.json", "w"), indent=2)
    json.dump(profile, open("user_profile.json", "w"), indent=2)
    print(json.dumps({k: v for k, v in profile.items() if not k.startswith("baseline")}, indent=2))
    print("wrote user_profile.json + onboarding_categorizer.json")
