import json, numpy as np, pandas as pd
from data_gen import generate, VARS
from qc import run_qc

def prf(y, p):
    tp = (y & p).sum(); fp = (~y & p).sum(); fn = (y & ~p).sum()
    P = tp/max(tp+fp, 1); R = tp/max(tp+fn, 1)
    return dict(precision=round(P, 3), recall=round(R, 3), f1=round(2*P*R/max(P+R, 1e-9), 3))

def rule_qc(o):   # classic fixed-threshold QC: range + step + persistence + missing
    lo = np.array([-5, 0, 800, 0, 0]); hi = np.array([50, 100, 1100, 40, 100]); step = np.array([4, 30, 3, 12, 15])
    rng = ((o < lo) | (o > hi) | np.isnan(o)).any(2)
    stp = (np.abs(np.diff(o, axis=0, prepend=o[:1])) > step).any(2)
    T, S, V = o.shape; per = (pd.DataFrame(o.reshape(T, S*V)).rolling(12).std().values.reshape(T, S, V) == 0)[:, :, :4].any(2)
    return rng | stp | per

d = generate(); T, S, V = d["obs"].shape; tT = d["train_T"]
obs = d["obs"].copy(); obs[obs == -999] = np.nan
q = run_qc(obs, d["time"], d["D"], tT)
f_true = d["fault"].any(2); e_true = d["event"] & ~f_true; a_true = f_true | e_true
ours_any = q["flag"]; ours_fault = (q["cls"] == 1).any(2); rule = rule_qc(obs)
te = slice(tT, None); m = {}
m["detection_anomaly_vs_normal"] = {"ours": prf(a_true[te], ours_any[te]), "rule_based": prf(a_true[te], rule[te])}
m["fault_detection"] = {"ours": prf(f_true[te], ours_fault[te]), "rule_based": prf(f_true[te], rule[te])}
m["real_event_rows_wrongly_called_fault_pct"] = {"ours": round(100*ours_fault[te][e_true[te]].mean(), 1), "rule_based": round(100*rule[te][e_true[te]].mean(), 1)}
cells = d["fault"][te] & (q["cls"][te] == 1) & ~np.isnan(obs[te]); tr_ = d["true"][te]
m["imputation_MAE_on_corrected_cells"] = {"raw_faulty": round(float(np.abs(obs[te] - tr_)[cells].mean()), 2),
                                           "after_auto_correction": round(float(np.abs(q["corrected"][te] - tr_)[cells].mean()), 2), "cells": int(cells.sum())}
print(json.dumps(m, indent=1))
rows = []
for t in range(T):
    for s in range(S):
        r = dict(time=d["time"][t], t=t, station=d["stations"].station[s], score=round(float(q["score"][t, s]), 4),
                 label_true="fault" if f_true[t, s] else "real_event" if e_true[t, s] else "normal",
                 label_pred="fault" if ours_fault[t, s] else "real_event" if q["flag"][t, s] else "normal", rule_flag=bool(rule[t, s]),
                 explanation=q["explain"].get((t, s), ""))
        for i, v in enumerate(VARS): r[v] = obs[t, s, i]; r[v+"_corrected"] = round(float(q["corrected"][t, s, i]), 2)
        rows.append(r)
pd.DataFrame(rows).to_csv("step1_results.csv", index=False); json.dump(m, open("step1_metrics.json", "w"), indent=1)
print("rows:", len(rows), "| test faults rows:", int(f_true[te].sum()), "| test event rows:", int(e_true[te].sum()))