import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings('ignore')

SEQ_LEN    = 48
HIDDEN     = 128
LAYERS     = 2
DROPOUT    = 0.3
TIME_FEATS = ['hour_sin','hour_cos','dow_sin','dow_cos',
              'month_sin','month_cos','is_weekend','is_night']
INPUT_SIZE = 1 + len(TIME_FEATS)   # 9
DEVICE     = torch.device('cpu')

class TrafficLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(INPUT_SIZE, HIDDEN, LAYERS,
                            batch_first=True, dropout=DROPOUT)
        self.norm = nn.LayerNorm(HIDDEN)
        self.fc   = nn.Linear(HIDDEN, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(self.norm(out[:, -1, :])).squeeze(1)

print("Loading lstm_tuned.pt ...")
checkpoint = torch.load(r'C:\Users\LENOVO\Desktop\Traffic-Pattern-Forecasting\notebooks\lstm_tuned.pt', map_location=DEVICE, weights_only=False)
model = TrafficLSTM().to(DEVICE)
model.load_state_dict(checkpoint['model_state'])
model.eval()
scalers = checkpoint['scalers']   # per-junction MinMaxScalers
print("Model loaded. Junctions in scalers:", sorted(scalers.keys()))

def add_time_features(df):
    df = df.copy()
    h   = df['DateTime'].dt.hour
    dow = df['DateTime'].dt.dayofweek
    m   = df['DateTime'].dt.month
    df['hour_sin']   = np.sin(2*np.pi*h/24)
    df['hour_cos']   = np.cos(2*np.pi*h/24)
    df['dow_sin']    = np.sin(2*np.pi*dow/7)
    df['dow_cos']    = np.cos(2*np.pi*dow/7)
    df['month_sin']  = np.sin(2*np.pi*m/12)
    df['month_cos']  = np.cos(2*np.pi*m/12)
    df['is_weekend'] = dow.isin([5,6]).astype(float)
    df['is_night']   = h.isin(list(range(0,6))+list(range(22,24))).astype(float)
    return df

train_raw = pd.read_csv(r'C:\Users\LENOVO\Desktop\Traffic-Pattern-Forecasting\data\raw\train_aWnotuB.csv', parse_dates=['DateTime'])
test_raw  = pd.read_csv(r'C:\Users\LENOVO\Desktop\Traffic-Pattern-Forecasting\data\raw\datasets_8494_11879_test_BdBKkAj.csv', parse_dates=['DateTime'])
train_raw = add_time_features(train_raw)
test_raw  = add_time_features(test_raw)

def predict_autoregressive(train_df, test_df, scalers, seq_len):
    all_ids, all_preds = [], []

    for j in sorted(test_df['Junction'].unique()):
        sc = scalers[j]

        # Seed: last seq_len rows from train for this junction
        tr_j = (train_df[train_df['Junction']==j]
                .sort_values('DateTime').reset_index(drop=True))
        seed_v  = sc.transform(tr_j['Vehicles'].values[-seq_len:].reshape(-1,1)).flatten()
        seed_tf = tr_j[TIME_FEATS].values[-seq_len:]          # (seq_len, 8)
        seed    = np.concatenate([seed_v.reshape(-1,1), seed_tf], axis=1)  # (seq_len, 9)

        te_j = (test_df[test_df['Junction']==j]
                .sort_values('DateTime').reset_index(drop=True))
        te_tf = te_j[TIME_FEATS].values   # (N_test, 8)
        te_ids = te_j['ID'].values

        window = seed.copy()   # sliding window — shape (seq_len, 9)
        preds  = []

        with torch.no_grad():
            for t in range(len(te_j)):
                x = torch.tensor(window.reshape(1, seq_len, INPUT_SIZE),
                                 dtype=torch.float32)
                pred_scaled = model(x).item()
                pred_scaled = np.clip(pred_scaled, 0, 1)   # MinMax range

                # Inverse scale → real vehicle count
                pred_real = sc.inverse_transform([[pred_scaled]])[0][0]
                preds.append(max(0, pred_real))

                # Slide window: drop oldest, append new row
                new_row = np.array([[pred_scaled] + list(te_tf[t])])  # (1, 9)
                window  = np.vstack([window[1:], new_row])            # (seq_len, 9)

        all_ids.extend(te_ids)
        all_preds.extend(preds)
        print(f"  Junction {j}: {len(preds)} predictions | "
              f"avg={np.mean(preds):.1f} | min={np.min(preds):.1f} | max={np.max(preds):.1f}")

    return pd.DataFrame({'ID': all_ids, 'Vehicles': np.round(all_preds).astype(int)})

print("\nGenerating LSTM predictions (autoregressive)...")
submission_lstm = predict_autoregressive(train_raw, test_raw, scalers, SEQ_LEN)
submission_lstm = submission_lstm.merge(test_raw[['ID','DateTime','Junction']], on='ID')
submission_lstm.sort_values(['Junction','DateTime'], inplace=True)
submission_lstm[['ID','Vehicles']].to_csv('submission_lstm.csv', index=False)
print(f"\nSaved → submission_lstm.csv ({len(submission_lstm)} rows)")
print(submission_lstm[['ID','Vehicles']].head())

xgb_sub = pd.read_csv(r'C:\Users\LENOVO\Desktop\Traffic-Pattern-Forecasting\notebooks\submission_xgb.csv')
xgb_sub = xgb_sub.merge(test_raw[['ID','DateTime','Junction']], on='ID')
xgb_sub.sort_values(['Junction','DateTime'], inplace=True)

COLORS = ['#2196F3','#4CAF50','#FF9800','#E91E63']
junctions = sorted(test_raw['Junction'].unique())

fig, axes = plt.subplots(len(junctions), 1, figsize=(18, 4*len(junctions)),
                         sharex=False)
fig.suptitle('Traffic Forecast — Jul to Nov 2017 (per Junction)',
             fontsize=15, fontweight='bold', y=1.01)

for ax, j, c in zip(axes, junctions, COLORS):
    # Last 4 weeks of train (context)
    tr_j = (train_raw[train_raw['Junction']==j]
            .sort_values('DateTime').tail(4*7*24))
    # Test predictions
    lstm_j = submission_lstm[submission_lstm['Junction']==j]
    xgb_j  = xgb_sub[xgb_sub['Junction']==j]

    ax.plot(tr_j['DateTime'],  tr_j['Vehicles'],
            color='gray', linewidth=0.8, alpha=0.6, label='Train (last 4wk)')
    ax.plot(lstm_j['DateTime'], lstm_j['Vehicles'],
            color=c, linewidth=1.2, label='LSTM Forecast')
    ax.plot(xgb_j['DateTime'],  xgb_j['Vehicles'],
            color=c, linewidth=1.2, linestyle='--', alpha=0.7, label='XGBoost Forecast')

    # Shade forecast region
    ax.axvspan(lstm_j['DateTime'].min(), lstm_j['DateTime'].max(),
               alpha=0.05, color=c)
    ax.axvline(lstm_j['DateTime'].min(), color='red', linestyle=':', linewidth=1)
    ax.set_title(f'Junction {j}', fontsize=11, fontweight='bold')
    ax.set_ylabel('Vehicles/hr')
    ax.legend(loc='upper right', fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.tick_params(axis='x', rotation=30)

plt.tight_layout()
plt.savefig('forecast_per_junction.png', dpi=130, bbox_inches='tight')
plt.show()
print("\nPlot saved → forecast_per_junction.png")

fig2, axes2 = plt.subplots(2, 2, figsize=(18, 10))
fig2.suptitle('XGBoost vs LSTM — Forecast Comparison (1 week sample per Junction)',
              fontsize=13, fontweight='bold')

for ax, j, c in zip(axes2.flatten(), junctions, COLORS):
    lstm_j = submission_lstm[submission_lstm['Junction']==j].head(7*24)
    xgb_j  = xgb_sub[xgb_sub['Junction']==j].head(7*24)

    ax.plot(lstm_j['DateTime'].values, lstm_j['Vehicles'].values,
            color=c, linewidth=1.5, label='LSTM')
    ax.plot(xgb_j['DateTime'].values,  xgb_j['Vehicles'].values,
            color='black', linewidth=1.5, linestyle='--', alpha=0.7, label='XGBoost')
    ax.set_title(f'Junction {j}', fontweight='bold')
    ax.set_ylabel('Vehicles/hr')
    ax.legend()
    ax.tick_params(axis='x', rotation=30)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))

plt.tight_layout()
plt.savefig('xgb_vs_lstm_comparison.png', dpi=130, bbox_inches='tight')
plt.show()
print("Plot saved → xgb_vs_lstm_comparison.png")

print("\n Forecast Summary")
print(f"{'Junction':<12} {'XGB avg':>10} {'LSTM avg':>10} {'XGB max':>10} {'LSTM max':>10}")
print("-"*54)
for j in junctions:
    xv = xgb_sub[xgb_sub['Junction']==j]['Vehicles']
    lv = submission_lstm[submission_lstm['Junction']==j]['Vehicles']
    print(f"J{j:<11} {xv.mean():>10.1f} {lv.mean():>10.1f} {xv.max():>10.0f} {lv.max():>10.0f}")

# ~ END ~
'''
OUTPUTS:
 Generating LSTM predictions (autoregressive)...
  Junction 1: 2952 predictions | avg=57.4 | min=25.1 | max=88.7
  Junction 2: 2952 predictions | avg=15.3 | min=6.5 | max=25.1
  Junction 3: 2952 predictions | avg=62.0 | min=11.2 | max=100.2
  Junction 4: 2952 predictions | avg=13.1 | min=4.5 | max=20.4

    ID          Vehicles
0  20170701001        71
1  20170701011        62
2  20170701021        52
3  20170701031        42
4  20170701041        35

 Forecast Summary
Junction        XGB avg   LSTM avg    XGB max   LSTM max
------------------------------------------------------
J1                 43.9       57.4         68         89
J2                 15.4       15.3         28         25
J3                 14.8       62.0         30        100
J4                  7.0       13.1         12         20

'''