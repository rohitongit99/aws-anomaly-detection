"""Anomaly detection (Isolation Forest + Autoencoder) -> spatial validation -> auto-imputation -> explanation."""
import numpy as np, pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from data_gen import VARS
FLOOR = np.array([1.0, 4, 1.2, 1.2, 2.0]); NN = 3; Z_MIN = 3.0
NAME = dict(temp="Temperature (°C)", rh="Humidity (%)", pres="Pressure (hPa)", wind="Wind (m/s)", rain="Rain (mm/15min)")

def features(obs, hour, train_T):
    T, S, V = obs.shape
    clim = np.stack([np.nanmedian(obs[:train_T][hour[:train_T] == h], 0) for h in range(24)], 1)   # [S,24,V]
    base = clim[:, hour, :].transpose(1, 0, 2); res = obs - base
    scale = np.maximum(1.4826*np.nanmedian(np.abs(res[:train_T] - np.nanmedian(res[:train_T], 0)), 0), FLOOR)
    a = np.clip(res/scale, -40, 40)
    df = pd.DataFrame(obs.reshape(T, S*V))
    dev = np.clip((obs - df.rolling(9, min_periods=1).median().values.reshape(T, S, V))/scale, -40, 40)
    dz = np.clip(df.diff().values.reshape(T, S, V)/scale, -40, 40)
    flat = (df.rolling(4).std().values.reshape(T, S, V) == 0).astype(float); flat[:, :, 4] = 0
    a, dev, dz = np.nan_to_num(a), np.nan_to_num(dev), np.nan_to_num(dz)
    X = np.concatenate([a, dz, dev, flat], 2).reshape(T*S, -1)
    strength = np.maximum.reduce([np.abs(a), np.abs(dev), 6*flat])
    return X, a, strength, base, res, flat

def score_rows(X, n_train):
    tr = np.arange(n_train)
    s_if = -IsolationForest(n_estimators=200, random_state=0).fit(X[tr]).score_samples(X)
    keep = tr[s_if[tr] < np.quantile(s_if[tr], .9)]                       # AE learns only "mostly clean" rows
    Z = StandardScaler().fit(X[keep]).transform(X)
    ae = MLPRegressor(hidden_layer_sizes=(10, 4, 10), max_iter=400, random_state=0).fit(Z[keep], Z[keep])
    s_ae = ((ae.predict(Z) - Z)**2).mean(1)
    ecdf = lambda s: np.searchsorted(np.sort(s[tr]), s)/len(tr)
    return (ecdf(s_if) + ecdf(s_ae))/2

def run_qc(obs, time, D, train_T, q=.8):
    T, S, V = obs.shape; hour = time.hour.values
    X, a, strength, base, res, flat = features(obs, hour, train_T)
    score = score_rows(X, train_T*S).reshape(T, S)
    missing = np.isnan(obs)
    flag = ((score >= np.quantile(score[:train_T], q)) & (strength.max(2) >= Z_MIN)) | missing.any(2)
    nn = np.argsort(D, 1)[:, 1:NN+1]
    cls = np.zeros((T, S, V), int); okc = np.zeros((T, S, V), int)       # cls: 1=fault, 2=real event
    for t, s in zip(*np.where(flag)):
        for v in range(V):
            if missing[t, s, v]: cls[t, s, v] = 1; continue
            if strength[t, s, v] < Z_MIN: continue
            a0 = a[t, s, v]; ok = 0
            for n in nn[s]:                                               # do nearest stations see the same change?
                w = a[max(t-1, 0):t+1, n, v]; an = w[np.argmax(np.abs(w))]
                ok += (np.sign(an) == np.sign(a0)) and abs(an) >= max(2, .4*abs(a0))
            okc[t, s, v] = ok; cls[t, s, v] = 2 if ok >= 1 else 1
    cor = obs.copy()
    for t, s in zip(*np.where((cls == 1).any(2))):                        # auto-imputation: own climatology + neighbours' anomaly
        for v in np.where(cls[t, s] == 1)[0]:
            good = [n for n in nn[s] if not missing[t, n, v] and cls[t, n, v] != 1]
            if good:
                w = 1/D[s, good]; cor[t, s, v] = base[t, s, v] + (w*res[t, good, v]).sum()/w.sum()
            else:
                cor[t, s, v] = cor[t-1, s, v] if t > 0 else base[t, s, v]
    cor[:, :, 1] = np.clip(cor[:, :, 1], 0, 100); cor[:, :, 3:] = np.clip(cor[:, :, 3:], 0, None)
    expl = {}
    for t, s in zip(*np.where(cls.any(2))):
        v = int(np.argmax(np.where(cls[t, s] > 0, strength[t, s], -1))); n_ = VARS[v]
        txt = f"{NAME[n_]}: " + ("no valid reading. " if missing[t, s, v] else
              f"read {obs[t, s, v]:.1f}, expected ~{base[t, s, v]:.1f} (strongest signal {strength[t, s, v]:.1f} sigma). " +
              ("Value frozen for 4+ readings. " if flat[t, s, v] else ""))
        txt += (f"Sensor sent no data -> SENSOR FAULT; estimated {cor[t, s, v]:.1f} from nearby stations."
                if missing[t, s, v] else
                f"Only {okc[t, s, v]}/{NN} nearest stations show the same change -> SENSOR FAULT; replaced with {cor[t, s, v]:.1f}."
                if cls[t, s, v] == 1 else
                f"{okc[t, s, v]}/{NN} nearest stations show the same change -> REAL WEATHER EVENT; value kept.")
        expl[(t, s)] = txt
    return dict(score=score, flag=flag, cls=cls, corrected=cor, explain=expl)
