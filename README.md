# AI/ML-Based Anomaly Detection for Automatic Weather Stations (AWS)

An anomaly detection system for AWS data that tells a **broken sensor** apart from **real extreme weather**, replaces bad values automatically, and explains every alert in plain words.

> Status: prototype. Tested on synthetic data with injected faults and real events. Real IMD data is the next step.

## The problem
AWS stations send data every few minutes. Sensor faults, dust or network issues create wrong readings, which spoil forecasts and flood warnings. Fixed-rule quality control only checks ranges and steps, so it misses many faults and flags real events (heatwave, cloudburst) as errors.

## How it works
1. Learn the normal pattern of every station (daily cycle, typical jumps).
2. Score each reading with **Isolation Forest** and an **Autoencoder**.
3. **Neighbour check:** compare with the 3 nearest stations.
   - Only this station is odd, so it is a **sensor fault**. The value is replaced by an estimate.
   - Neighbours show the same change, so it is a **real weather event**. The value is kept.
4. Show the decision on a Flask dashboard with a map, chart and plain-language reason.

## Features
- Fault vs real-event classification using spatial validation
- Automatic correction of faulty values
- Explainable alerts ("what we saw / neighbour check / decision")
- Dashboard with station map (Leaflet + OpenStreetMap), saved replay and a **Live demo** mode
- **Sensor glitch (demo)** button: breaks a random sensor for 1 to 2 minutes so you can watch the system react

## Results (synthetic test data, unseen days)
| Metric | This system | Fixed rules |
|---|---|---|
| Finding problems (F1) | 0.91 | 0.49 |
| Real-event readings wrongly called faults | 0% | 7.5% |
| Average error of faulty readings | 14.8 to 0.18 | no correction |

Live demo: 24 of 24 random glitches were caught. The numbers come from generated data and a small number of test faults, so they are indicative only.

## Project structure
```
aws-anomaly-detection/
├── data_gen.py        # synthetic AWS network with injected faults and real events
├── qc.py              # detection, neighbour check, auto-correction, explanations
├── run_step1.py       # runs the pipeline, compares with rule-based QC, saves results
├── live.py            # live demo engine (sensor glitch)
├── app.py             # Flask dashboard server
├── templates/
│   └── index.html     # dashboard
└── requirements.txt
```

## Run it
Python 3.9 or newer.
```bash
pip install -r requirements.txt
python run_step1.py     # creates step1_results.csv and step1_metrics.json
python app.py           # open http://127.0.0.1:5000
```
The map needs an internet connection for the map tiles.

## Limitations
- Synthetic data only; real IMD data is not tested yet
- Few fault readings in the test period
- Slow drift and stuck wind or pressure can take longer to notice
- If two neighbouring stations fail together, the neighbour check can be fooled

## Roadmap
- 8th semester: real IMD / open AWS data, LSTM and GNN models, real-time alerts, paper

