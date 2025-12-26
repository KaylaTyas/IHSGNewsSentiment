import pandas as pd
import joblib
import os
import numpy as np
from sqlalchemy import create_engine, text
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("IHSG Prediction - SARIMAX Model\n")

# Config
DB_URL = os.getenv("DB_URL")
engine = create_engine(DB_URL, connect_args={"connect_timeout": 10})

# Load data
print("Loading data from database...")
query = """
SELECT date, terakhir, pembukaan, tertinggi, terendah, volume, 
       perubahan_pct, avg_sentiment, prop_pos, prop_neg
FROM ihsg_sentimen_features
ORDER BY date"""

df = pd.read_sql(query, engine)
df["date"] = pd.to_datetime(df["date"])
print(f"Loaded {len(df)} rows ({df['date'].min().date()} to {df['date'].max().date()})")
print(f"Last close: {df['terakhir'].iloc[-1]:.2f}\n")

# Rename columns
df = df.rename(columns={
    'date': 'tanggal',
    'terakhir': 'Terakhir',
    'pembukaan': 'Pembukaan',
    'tertinggi': 'Tertinggi',
    'terendah': 'Terendah',
    'volume': 'Vol.',
    'perubahan_pct': 'Perubahan%'
})

# Feature engineering
print("Engineering features...")
df['return'] = df['Terakhir'].pct_change()
df['lag1_return'] = df['return'].shift(1)
df['ma5'] = df['Terakhir'].rolling(5).mean().shift(1)
df['ma10'] = df['Terakhir'].rolling(10).mean().shift(1)
df['volatility5'] = df['return'].rolling(5).std().shift(1)

# Load model
print("Loading model...")
model = joblib.load("models/best_sarimax_model.joblib")

# Recreate scalers
print("Fitting scalers on training period...")
feature_cols = ['Pembukaan', 'Tertinggi', 'Terendah', 'Vol.', 'avg_sentiment',
                'prop_neg', 'return', 'lag1_return', 'ma5', 'ma10', 'volatility5']
target_col = ['Terakhir']

train = df[(df['tanggal'] >= "2022-01-03") & (df['tanggal'] <= "2022-12-31")].copy()
if len(train) == 0:
    print("Warning: Training period not found, using earliest 50% of data")
    train = df.iloc[:int(len(df)*0.5)].copy()

scaler_features = StandardScaler()
scaler_target = StandardScaler()
scaler_features.fit(train[feature_cols].dropna())
scaler_target.fit(train[target_col].dropna())
print(f"Scalers fitted on {len(train)} training samples\n")

# History predict
print("Generating historical predictions...")
df_historical = df[:-1].copy()
df_historical_clean = df_historical.dropna(subset=feature_cols).copy()

df_scaled = pd.DataFrame(
    scaler_features.transform(df_historical_clean[feature_cols]),
    columns=feature_cols,
    index=df_historical_clean.index
)
df_scaled['Terakhir_scaled'] = scaler_target.transform(df_historical_clean[target_col])
df_scaled['tanggal'] = df_historical_clean['tanggal'].values
df_scaled['Terakhir_actual'] = df_historical_clean['Terakhir'].values

try:
    WINDOW = min(int(model.nobs), len(df_scaled))
    df_pred = df_scaled.tail(WINDOW).copy().reset_index(drop=True)
    
    predictions_scaled = model.predict(
        start=0,
        end=len(df_pred) - 1,
        exog=df_pred[feature_cols]
    )
    predictions_actual = scaler_target.inverse_transform(
        predictions_scaled.values.reshape(-1, 1)
    ).flatten()
    
    df_pred['predicted_close'] = predictions_actual
    df_pred['actual_yesterday'] = df_pred['Terakhir_actual'].shift(1)
    df_pred['predicted_pct'] = ((df_pred['predicted_close'] - df_pred['actual_yesterday']) / 
                                 df_pred['actual_yesterday'] * 100)
    
    print(f"Generated {len(df_pred)} historical predictions")
    print("\nLast 5 historical predictions:")
    for idx in df_pred.tail(5).index:
        row = df_pred.loc[idx]
        pct = row['predicted_pct'] if not pd.isna(row['predicted_pct']) else 0
        print(f"  {row['tanggal'].date()} | Actual: {row['Terakhir_actual']:.2f} | "
              f"Predicted: {row['predicted_close']:.2f} | Change: {pct:+.2f}%")
except Exception as e:
    print(f"Error in historical predictions: {e}")
    raise

# Save history
print("\nSaving historical predictions...")
df_save = df_pred[['tanggal', 'predicted_close', 'predicted_pct']].copy()
df_save = df_save.iloc[1:].dropna(subset=['predicted_close'])

saved_count = 0
with engine.begin() as conn:
    for _, row in df_save.iterrows():
        pct_value = float(row['predicted_pct']) if not pd.isna(row['predicted_pct']) else 0.0
        conn.execute(text("""
            INSERT INTO predictions (date, predicted_close, predicted_pct)
            VALUES (:date, :close, :pct)
            ON CONFLICT (date) DO UPDATE SET
              predicted_close = EXCLUDED.predicted_close,
              predicted_pct = EXCLUDED.predicted_pct,
              created_at = now()
        """), {
            "date": row["tanggal"],
            "close": float(row["predicted_close"]),
            "pct": pct_value
        })
        saved_count += 1

print(f"Saved {saved_count} historical predictions\n")

# Predict besok
print("--- Predicting Tomorrow ---")
df_today = df.iloc[-1:].copy()
today_date = df_today['tanggal'].iloc[0]
today_close = df_today['Terakhir'].iloc[0]
tomorrow_date = today_date + pd.Timedelta(days=1)

print(f"Today: {today_date.date()} | Close: {today_close:.2f}")
print(f"Predicting: {tomorrow_date.date()}")

# Tomorrow's features
df_tomorrow = df_today.copy()
for col in feature_cols:
    if pd.isna(df_tomorrow[col].iloc[0]):
        if col in df_today.columns and not pd.isna(df_today[col].iloc[0]):
            df_tomorrow[col] = df_today[col].iloc[0]
        else:
            df_tomorrow[col] = df[col].tail(5).mean()

try:
    features_tomorrow = df_tomorrow[feature_cols].values
    features_tomorrow_scaled = scaler_features.transform(features_tomorrow)
    pred_scaled_tomorrow = model.forecast(steps=1, exog=features_tomorrow_scaled).iloc[0]
    pred_close_raw = scaler_target.inverse_transform([[pred_scaled_tomorrow]])[0][0]
    
    max_change = 0.05
    max_pred = today_close * (1 + max_change)
    min_pred = today_close * (1 - max_change)
    pred_close_tomorrow = np.clip(pred_close_raw, min_pred, max_pred)
    
    if pred_close_raw != pred_close_tomorrow:
        print(f"Note: Prediction capped from {pred_close_raw:.2f} to {pred_close_tomorrow:.2f} (±5% safety limit)")
    pred_pct_tomorrow = ((pred_close_tomorrow - today_close) / today_close) * 100
    
    print(f"\nTomorrow's Prediction:")
    print(f"  Date: {tomorrow_date.date()}")
    print(f"  Predicted Close: {pred_close_tomorrow:.2f}")
    print(f"  Change: {pred_pct_tomorrow:+.2f}% ({pred_close_tomorrow - today_close:+.2f} points)")
    
    if abs(pred_pct_tomorrow) < 0.5:
        print(f"  Volatility: Low (< 0.5%)")
    elif abs(pred_pct_tomorrow) < 1.5:
        print(f"  Volatility: Normal (0.5-1.5%)")
    else:
        print(f"  Volatility: High (> 1.5%)")
    
except Exception as e:
    print(f"Error in prediction: {e}")
    pred_close_tomorrow = today_close
    pred_pct_tomorrow = 0.0
    print("Using fallback: zero change")

# Save besok
print(f"\nSaving tomorrow's prediction...")
try:
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO predictions (date, predicted_close, predicted_pct)
            VALUES (:date, :close, :pct)
            ON CONFLICT (date) DO UPDATE SET
              predicted_close = EXCLUDED.predicted_close,
              predicted_pct = EXCLUDED.predicted_pct,
              created_at = now()
        """), {
            "date": tomorrow_date,
            "close": float(pred_close_tomorrow),
            "pct": float(pred_pct_tomorrow)
        })
    print(f"Saved prediction for {tomorrow_date.date()}")
except Exception as e:
    print(f"Database error: {e}")

print(f"\nDone! Historical: {saved_count} | Tomorrow: {pred_close_tomorrow:.2f} ({pred_pct_tomorrow:+.2f}%)")