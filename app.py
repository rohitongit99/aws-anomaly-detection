"""Dashboard server. Run: python app.py  ->  http://127.0.0.1:5000"""
import json, threading, numpy as np, pandas as pd
from flask import Flask, jsonify, send_from_directory
from data_gen import make_stations, VARS
from live import Live

app = Flask(__name__)

def load():
    d = pd.read_csv("step1_results.csv", encoding="utf-8-sig"); st = make_stations(); names = list(st.station)
    d["explanation"] = d["explanation"].fillna("")
    piv = lambda col: d.pivot(index="t", columns="station", values=col)[names]
    lst = lambda a: [[None if np.isnan(x) else round(float(x), 2) for x in row] for row in a.values]
    code = d.assign(c=d.label_pred.map({"normal": 0, "fault": 1, "real_event": 2})).pipe(lambda x: x.pivot(index="t", columns="station", values="c")[names])
    expl = {f"{r.t},{names.index(r.station)}": r.explanation for r in d[d.explanation != ""].itertuples()}
    times = pd.to_datetime(d.drop_duplicates("t").sort_values("t").time).dt.strftime("%d %b, %H:%M").tolist()
    return dict(stations=[dict(name=n, lat=float(a), lon=float(b)) for n, a, b in zip(names, st.lat, st.lon)],
                times=times, code=code.values.astype(int).tolist(), score=piv("score").round(2).values.tolist(),
                raw={v: lst(piv(v)) for v in VARS}, corrected={v: lst(piv(v + "_corrected")) for v in VARS},
                explain=expl, metrics=json.load(open("step1_metrics.json")))
DATA = load()
print("Training the live-demo detector (a few seconds)...")
LIVE = Live(); LOCK = threading.Lock()

@app.route("/")
def index(): return send_from_directory("templates", "index.html")

@app.route("/api/data")
def data(): return jsonify(DATA)

@app.route("/api/live/start")
def live_start():
    with LOCK: LIVE.restart(); return jsonify(rows=[LIVE.prefill_row(c) for c in range(LIVE.c - 95, LIVE.c + 1)])

@app.route("/api/live/tick")
def live_tick():
    with LOCK: return jsonify(LIVE.tick())

@app.route("/api/live/glitch", methods=["POST"])
def live_glitch():
    with LOCK: return jsonify(LIVE.start_glitch())

@app.route("/api/live/stop", methods=["POST"])
def live_stop():
    with LOCK: LIVE.glitch = None; return jsonify(LIVE.glitch_info())

if __name__ == "__main__":
    app.run(debug=False, port=5000)