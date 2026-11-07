"""Synthetic AWS network (Aurangabad region) with injected sensor faults and real extreme events."""
import numpy as np, pandas as pd
VARS = ["temp", "rh", "pres", "wind", "rain"]

def make_stations(n=12, seed=0):
    r = np.random.default_rng(seed)
    return pd.DataFrame({"station": [f"AWS{i+1:02d}" for i in range(n)],
                         "lat": 19.9 + r.uniform(-.6, .6, n), "lon": 75.3 + r.uniform(-.6, .6, n)})

def dist_km(st):
    la, lo = np.radians(st.lat.values), np.radians(st.lon.values)
    a = np.sin((la[:, None]-la[None])/2)**2 + np.cos(la)[:, None]*np.cos(la)[None]*np.sin((lo[:, None]-lo[None])/2)**2
    return 6371*2*np.arcsin(np.sqrt(a))

def generate(days=8, seed=1):
    r = np.random.default_rng(seed); st = make_stations(); S = len(st); D = dist_km(st)
    T = days*96; time = pd.date_range("2026-05-10", periods=T, freq="15min")
    hr = (time.hour + time.minute/60).values
    def ar1(phi, sig):
        e = r.normal(0, sig, T); x = np.zeros(T)
        for i in range(1, T): x[i] = phi*x[i-1] + e[i]
        return x
    t_reg = 33 + 6*np.sin(2*np.pi*(hr-9)/24) + ar1(.99, .08)
    temp = t_reg[:, None] + r.normal(0, .7, S) + r.normal(0, .15, (T, S))
    rh = 55 - 2.5*(temp-33) + r.normal(0, 1.5, (T, S))
    pres = (950 + 1.2*np.cos(2*np.pi*(hr-10)/12) + ar1(.998, .05))[:, None] + r.normal(0, 1, S) + r.normal(0, .05, (T, S))
    wind = (3 + 1.5*np.sin(2*np.pi*(hr-9)/24) + ar1(.98, .25))[:, None] + r.normal(0, .4, S) + r.normal(0, .3, (T, S))
    rain = np.zeros((T, S))
    for t0 in (150, 520): rain[t0:t0+8] = np.clip(r.normal(.6, .3, (8, S)), 0, None)
    # real extreme events hit a CLUSTER of neighbouring stations (centre + 3 nearest)
    ev = np.zeros((T, S), bool)
    events = [("heatwave", 2, 130), ("cloudburst", 7, 250), ("heatwave", 5, 490), ("cloudburst", 9, 610), ("cloudburst", 1, 700)]
    for kind, c, t0 in events:
        cl = np.argsort(D[c])[:4]
        if kind == "heatwave":
            L = 32; env = np.sin(np.pi*np.arange(L)/(L-1))[:, None]
            temp[t0:t0+L, cl] += 7*env; rh[t0:t0+L, cl] -= 12*env; wind[t0:t0+L, cl] += 1
        else:
            L = 5; env = np.array([.4, 1, 1, .7, .4])[:, None]
            rain[t0:t0+L, cl] += np.array([8, 22, 28, 12, 4])[:, None]*r.uniform(.8, 1.2, len(cl))
            temp[t0:t0+L, cl] -= 5*env; rh[t0:t0+L, cl] += 25*env; pres[t0:t0+L, cl] -= 3*env; wind[t0:t0+L, cl] += 8*env
        ev[t0:t0+L, cl] = True
    true = np.stack([temp, np.clip(rh, 5, 100), pres, np.clip(wind, .2, None), rain], 2)
    # single-station sensor faults
    obs = true.copy(); fl = np.zeros(true.shape, bool)
    mag = [20, 40, 25, 20, 20]; drift = [6, 25, 10, 6]
    for _ in range(40):
        s, v = r.integers(S), r.integers(5); t0 = r.integers(20, T-60)
        kind = r.choice(["spike", "dropout"] if v == 4 else ["spike", "stuck", "drift", "dropout"])
        if kind == "spike":
            obs[t0:t0+2, s, v] += (1 if v == 4 else r.choice([-1, 1]))*mag[v]; fl[t0:t0+2, s, v] = True
        elif kind == "stuck":
            obs[t0:t0+16, s, v] = obs[t0, s, v]; fl[t0+3:t0+16, s, v] = True      # faulty once 4 identical readings seen
        elif kind == "drift":
            b = np.linspace(0, 1, 48)*drift[v]; obs[t0:t0+48, s, v] += b; fl[t0:t0+48, s, v] = b >= .3*drift[v]
        else:
            obs[t0:t0+8, s, v] = -999; fl[t0:t0+8, s, v] = True
    return dict(obs=obs, true=true, fault=fl, event=ev, time=time, stations=st, D=D, train_T=4*96)
