"""One-time patch: fewer sensor-fault alerts. Run once inside the aws_qc folder:  python apply_patch.py"""
import sys
def patch(path, pairs):
    s = open(path, encoding="utf-8").read()
    for a, b in pairs:
        if b in s: continue                       # already patched
        if a not in s: sys.exit(f"STOP: could not patch {path}. Text not found: {a[:70]!r}")
        s = s.replace(a, b, 1)
    open(path, "w", encoding="utf-8").write(s); print("patched", path)

patch("data_gen.py", [("def generate(days=8, seed=1):", "def generate(days=12, seed=1, n_faults=18):"),
                      ("for _ in range(40):", "for _ in range(n_faults):")])
patch("qc.py", [("NN = 3; Z_MIN = 3.0", "NN = 3; Z_MIN = 3.0; Z_FAULT = 4.0   # a lone station is called a fault only if clearly off (4 sigma)"),
                ("okc[t, s, v] = ok; cls[t, s, v] = 2 if ok >= 1 else 1", "okc[t, s, v] = ok; cls[t, s, v] = 2 if ok >= 1 else (1 if strength[t, s, v] >= Z_FAULT else 0)"),
                ("    cor = obs.copy()\n", "    flag = cls.any(2)\n    cor = obs.copy()\n")])
patch("live.py", [("from qc import FLOOR, NN, Z_MIN, NAME", "from qc import FLOOR, NN, Z_MIN, Z_FAULT, NAME"),
                  ("okc[s, v] = ok; cls[s, v] = 2 if ok >= 1 else 1", "okc[s, v] = ok; cls[s, v] = 2 if ok >= 1 else (1 if st[-1, s, v] >= Z_FAULT else 0)")])
patch("templates/index.html", [
    ("function beep(){if(!sound)return;", "let lastBeep=0;\nfunction beep(force){if(!sound)return;const now=Date.now();if(!force&&now-lastBeep<4000)return;lastBeep=now;"),
    ("if(sound)beep()};", "if(sound)beep(true)};"),
    ("function buildAlerts(){alerts=[];for(let s=0;s<D.stations.length;s++)for(let k=0;k<D.times.length;k++)if(D.code[k][s]>0&&!(k>0&&D.code[k-1][s]===D.code[k][s]))alerts.push({t:k,s,c:D.code[k][s]});alerts.sort((p,q)=>p.t-q.t)}",
     "const GAP=8;\nfunction isNew(k,s){const c=D.code[k][s];if(c<=0)return false;for(let j=1;j<=GAP&&k-j>=0;j++)if(D.code[k-j][s]===c)return false;return true}\nfunction buildAlerts(){alerts=[];for(let s=0;s<D.stations.length;s++)for(let k=0;k<D.times.length;k++)if(isNew(k,s))alerts.push({t:k,s,c:D.code[k][s]});alerts.sort((p,q)=>p.t-q.t)}"),
    ("D.code[k].forEach((c,s)=>{if(c>0&&(k===0||D.code[k-1][s]!==c)){", "D.code[k].forEach((c,s)=>{if(isNew(k,s)){")])
print("Done. Now run:  python run_step1.py   and then   python app.py")