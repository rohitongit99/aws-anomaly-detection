"""Live demo engine: streams clean readings one step per call, lets you break a sensor, and runs the REAL detector on every new reading."""
import numpy as np, pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from data_gen import generate, VARS
from qc import FLOOR, NN, Z_MIN, NAME
WIN, START = 40, 400
MAG = [20, 40, 25, 20]; DRIFT = [6, 25, 10, 6]
KINDS = {"erratic": "erratic readings", "stuck": "stuck (frozen) value", "drift": "slow drift", "dropout": "no data"}

def feat(obs, hour, clim, scale):
    T, S, V = obs.shape
    base = clim[:, hour, :].transpose(1, 0, 2); res = obs - base
    a = np.clip(res/scale, -40, 40); df = pd.DataFrame(obs.reshape(T, S*V))
    dev = np.clip((obs - df.rolling(9, min_periods=1).median().values.reshape(T, S, V))/scale, -40, 40)
    dz = np.clip(df.diff().values.reshape(T, S, V)/scale, -40, 40)
    flat = (df.rolling(4).std().values.reshape(T, S, V) == 0).astype(float); flat[:, :, 4] = 0
    flat[:, :, 3] *= obs[:, :, 3] > .25; flat[:, :, 1] *= (obs[:, :, 1] > 5.01) & (obs[:, :, 1] < 99.99)
    a, dev, dz = np.nan_to_num(a), np.nan_to_num(dev), np.nan_to_num(dz)
    return np.concatenate([a, dz, dev, flat], 2).reshape(T*S, -1), a, np.maximum.reduce([np.abs(a), np.abs(dev), 6*flat]), base, res, flat

class Live:
    def __init__(self):
        d = generate(days=14); self.true = d["true"]; self.time = d["time"]; self.D = d["D"]; self.names = list(d["stations"].station)
        tT = d["train_T"]; hr = self.time.hour.values; tr = self.true[:tT]
        self.clim = np.stack([np.nanmedian(tr[hr[:tT] == h], 0) for h in range(24)], 1)
        res = tr - self.clim[:, hr[:tT], :].transpose(1, 0, 2)
        self.scale = np.maximum(1.4826*np.nanmedian(np.abs(res - np.nanmedian(res, 0)), 0), FLOOR)
        X = feat(tr, hr[:tT], self.clim, self.scale)[0]
        self.iso = IsolationForest(n_estimators=200, random_state=0).fit(X); s_if = -self.iso.score_samples(X)
        keep = s_if < np.quantile(s_if, .9); self.sc = StandardScaler().fit(X[keep]); Z = self.sc.transform(X)
        self.ae = MLPRegressor(hidden_layer_sizes=(10, 4, 10), max_iter=400, random_state=0).fit(Z[keep], Z[keep])
        self.ref_if = np.sort(s_if); self.ref_ae = np.sort(((self.ae.predict(Z) - Z)**2).mean(1))
        self.thr = np.quantile(self._score(X), .8); self.nn = np.argsort(self.D, 1)[:, 1:NN+1]
        self.rng = np.random.default_rng(); self.restart()

    def _score(self, X):
        s_if = -self.iso.score_samples(X); Z = self.sc.transform(X); s_ae = ((self.ae.predict(Z) - Z)**2).mean(1)
        return (np.searchsorted(self.ref_if, s_if)/len(self.ref_if) + np.searchsorted(self.ref_ae, s_ae)/len(self.ref_ae))/2

    def restart(self): self.c = START; self.obs = self.true.copy(); self.cor = self.true.copy(); self.glitch = None

    def _row(self, c, code, score, obs, cor, expl):
        L = lambda arr: {v: [None if np.isnan(x) else round(float(x), 2) for x in arr[:, i]] for i, v in enumerate(VARS)}
        return dict(c=int(c), time=self.time[c].strftime("%d %b, %H:%M"), code=[int(x) for x in code], score=[round(float(x), 2) for x in score],
                    raw=L(obs), cor=L(cor), explain={str(k): v for k, v in expl.items()}, glitch=self.glitch_info())

    def prefill_row(self, c): S = len(self.names); return self._row(c, np.zeros(S), np.zeros(S), self.true[c], self.true[c], {})

    def glitch_info(self):
        g = self.glitch
        return dict(active=False) if not g else dict(active=True, s=g["s"], v=g["v"], name=self.names[g["s"]], var=NAME[VARS[g["v"]]].split(" (")[0],
                                                       kind=KINDS[g["kind"]], left=int(max(0, g["until"] - self.c)), c0=g["c0"])

    def start_glitch(self):
        s, v = int(self.rng.integers(len(self.names))), int(self.rng.integers(4)); kind = str(self.rng.choice(list(KINDS)))
        n = int(self.rng.integers(60, 121)); self.glitch = dict(s=s, v=v, kind=kind, c0=self.c + 1, until=self.c + n, frozen=float(self.obs[self.c, s, v]))
        return self.glitch_info()

    def _glitch_value(self, g, c):
        s, v, k = g["s"], g["v"], c - g["c0"]; tv = self.true[c, s, v]
        if g["kind"] == "stuck": return g["frozen"]
        if g["kind"] == "drift": return tv + DRIFT[v]*min(1.5, (k + 1)/20)
        if g["kind"] == "dropout": return np.nan
        return tv + self.rng.choice([-1, 1])*self.rng.uniform(.5, 1)*MAG[v]*.7

    def tick(self):
        if self.c >= len(self.time) - 2: self.restart()
        self.c += 1; c = self.c
        if self.glitch and c > self.glitch["until"]: self.glitch = None
        if self.glitch: self.obs[c, self.glitch["s"], self.glitch["v"]] = self._glitch_value(self.glitch, c)
        return self._detect(c)

    def _detect(self, c):
        lo = max(0, c - WIN + 1); ow = self.obs[lo:c+1]; hr = self.time.hour.values[lo:c+1]
        X, a, st, base, res, flat = feat(ow, hr, self.clim, self.scale); S, V = ow.shape[1:]
        sc = self._score(X).reshape(len(ow), S)[-1]; miss = np.isnan(ow[-1])
        flag = ((sc >= self.thr) & (st[-1].max(1) >= Z_MIN)) | miss.any(1)
        cls = np.zeros((S, V), int); okc = np.zeros((S, V), int)
        for s in np.where(flag)[0]:
            for v in range(V):
                if miss[s, v]: cls[s, v] = 1; continue
                if st[-1, s, v] < Z_MIN: continue
                a0 = a[-1, s, v]; ok = 0
                for n in self.nn[s]:
                    w = a[-2:, n, v]; an = w[np.argmax(np.abs(w))]
                    ok += (np.sign(an) == np.sign(a0)) and abs(an) >= max(2, .4*abs(a0))
                okc[s, v] = ok; cls[s, v] = 2 if ok >= 1 else 1
        cor = ow[-1].copy()
        for s in np.where((cls == 1).any(1))[0]:
            for v in np.where(cls[s] == 1)[0]:
                good = [n for n in self.nn[s] if not miss[n, v] and cls[n, v] != 1]
                cor[s, v] = base[-1, s, v] + ((1/self.D[s, good])*res[-1, good, v]).sum()/(1/self.D[s, good]).sum() if good else base[-1, s, v]
        cor[:, 1] = np.clip(cor[:, 1], 0, 100); cor[:, 3:] = np.clip(cor[:, 3:], 0, None); self.cor[c] = cor
        expl = {}
        for s in np.where(cls.any(1))[0]:
            v = int(np.argmax(np.where(cls[s] > 0, st[-1, s], -1))); nm = NAME[VARS[v]]
            t = f"{nm}: " + ("no valid reading. " if miss[s, v] else f"read {ow[-1, s, v]:.1f}, expected ~{base[-1, s, v]:.1f} (strongest signal {st[-1, s, v]:.1f} sigma). " + ("Value frozen for 4+ readings. " if flat[-1, s, v] else ""))
            t += (f"Sensor sent no data -> SENSOR FAULT; estimated {cor[s, v]:.1f} from nearby stations." if miss[s, v] else
                  f"Only {okc[s, v]}/{NN} nearest stations show the same change -> SENSOR FAULT; replaced with {cor[s, v]:.1f}." if cls[s, v] == 1 else
                  f"{okc[s, v]}/{NN} nearest stations show the same change -> REAL WEATHER EVENT; value kept.")
            expl[int(s)] = t
        return self._row(c, np.where((cls == 1).any(1), 1, np.where((cls == 2).any(1), 2, 0)), sc, ow[-1], cor, expl)