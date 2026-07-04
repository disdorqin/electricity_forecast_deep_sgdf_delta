"""
Phase D: Regime-Specific Deep Model

只有 Phase C 至少一个 tabular specialist KEEP，才允许进入深度模型。

候选：
  TinyMLP specialist
  TCN specialist
  Transformer specialist

必须有 DA fallback。
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import argparse
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

def compute_smape_floor50(y_true, y_pred, floor=50.0):
    """Compute sMAPE with floor=50."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    denom = np.maximum((np.abs(y_true) + np.abs(y_pred)) / 2, floor)
    return 100 * np.mean(np.abs(y_true - y_pred) / denom)

def compute_day_level_smape(y_true, y_pred, timestamps):
    """Compute day-level sMAPE."""
    df = pd.DataFrame({
        'timestamp': timestamps,
        'y_true': y_true,
        'y_pred': y_pred
    })
    df['date'] = pd.to_datetime(df['timestamp']).dt.date
    
    daily_true = df.groupby('date')['y_true'].mean()
    daily_pred = df.groupby('date')['y_pred'].mean()
    
    return compute_smape_floor50(daily_true.values, daily_pred.values)

class TinyMLPSpecialist(nn.Module):
    """Tiny MLP for residual correction."""
    def __init__(self, input_dim, hidden_dims=[64, 32], dropout=0.2):
        super(TinyMLPSpecialist, self).__init__()
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x).squeeze(-1)

class ResidualDataset(Dataset):
    """Dataset for residual prediction."""
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.X[idx]), torch.FloatTensor([self.y[idx]])

def train_deep_model(model, train_loader, val_loader, device, epochs=50, lr=0.001):
    """Train deep model."""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    patience = 10
    counter = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                val_loss += loss.item()
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), 'best_deep_model.pth')
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load best model
    model.load_state_dict(torch.load('best_deep_model.pth'))
    return model

def main():
    parser = argparse.ArgumentParser(description='Phase D: Regime-Specific Deep Model')
    parser.add_argument('--data-path', type=str, required=True,
                       help='Path to preprocessed data CSV')
    parser.add_argument('--out-dir', type=str, required=True,
                       help='Output directory')
    parser.add_argument('--model-type', type=str, default='TinyMLP',
                       choices=['TinyMLP', 'TCN', 'Transformer'],
                       help='Deep model type')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size')
    args = parser.parse_args()
    
    # Load preprocessed data
    print(f"Loading preprocessed data from {args.data_path}...")
    df = pd.read_csv(args.data_path, parse_dates=['times'], encoding='utf-8-sig')
    df = df.sort_values('times').reset_index(drop=True)
    
    # Create output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Define online features (same as Phase B/C)
    online_features = [
        'hour', 'dayofweek', 'month', 'is_weekend',
        'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
        'da_price',  # DA anchor
        'da_price_lag_24h', 'da_price_lag_48h', 'da_price_lag_72h', 'da_price_lag_168h',
        'rt_price_lag_24h', 'rt_price_lag_48h', 'rt_price_lag_72h', 'rt_price_lag_168h',
        'rt_price_rolling_mean_24h', 'rt_price_rolling_std_24h',
        'rt_price_rolling_mean_48h', 'rt_price_rolling_std_48h',
        'rt_price_rolling_mean_168h', 'rt_price_rolling_std_168h',
        'bidding_space_forecast', 'direct_dispatch_forecast',
        'wind_forecast', 'solar_forecast',
        'bidding_space_forecast_lag_24h', 'bidding_space_forecast_lag_168h',
        'direct_dispatch_forecast_lag_24h', 'direct_dispatch_forecast_lag_168h',
        'wind_forecast_lag_24h', 'wind_forecast_lag_168h',
        'solar_forecast_lag_24h', 'solar_forecast_lag_168h',
        'bidding_space_forecast_rolling_mean_24h',
        'direct_dispatch_forecast_rolling_mean_24h',
        'wind_forecast_rolling_mean_24h',
        'solar_forecast_rolling_mean_24h'
    ]
    
    # Filter to features that exist in df
    online_features = [f for f in online_features if f in df.columns]
    print(f"Number of online features: {len(online_features)}")
    
    # Prepare data
    X = df[online_features].fillna(0).values
    y = (df['rt_price'] - df['da_price']).values  # residual
    
    # Walk-forward validation (use 2024-05 to 2025-12)
    print("\n=== Walk-Forward Validation (Deep Model) ===")
    
    months = pd.date_range(start='2024-05-01', end='2025-12-01', freq='MS')
    results = []
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    for i in range(len(months) - 1):
        train_start = months[max(0, i-3)]
        train_end = months[i]
        val_start = months[i]
        val_end = months[i+1]
        
        # Get train/val data
        train_mask = (df['times'] >= train_start) & (df['times'] < train_end)
        val_mask = (df['times'] >= val_start) & (df['times'] < val_end)
        
        if train_mask.sum() < 100 or val_mask.sum() < 10:
            continue
        
        X_train = X[train_mask]
        y_train = y[train_mask]
        X_val = X[val_mask]
        y_val = y[val_mask]
        
        # Normalize
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        
        X_train_scaled = scaler_X.fit_transform(X_train)
        y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
        
        X_val_scaled = scaler_X.transform(X_val)
        
        # Create datasets
        train_dataset = ResidualDataset(X_train_scaled, y_train_scaled)
        val_dataset = ResidualDataset(X_val_scaled, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        
        # Train deep model
        print(f"\nTraining period: {train_start.strftime('%Y-%m')} to {train_end.strftime('%Y-%m')}")
        print(f"Validation period: {val_start.strftime('%Y-%m')}")
        
        model = TinyMLPSpecialist(input_dim=len(online_features))
        model = train_deep_model(model, train_loader, val_loader, device, 
                                epochs=args.epochs, lr=args.lr)
        
        # Predict on validation set
        model.eval()
        predictions = []
        with torch.no_grad():
            for X_batch, _ in val_loader:
                X_batch = X_batch.to(device)
                y_pred_scaled = model(X_batch)
                predictions.extend(y_pred_scaled.cpu().numpy())
        
        # Inverse transform predictions
        predictions = scaler_y.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
        
        # Apply DA fallback (only use residual correction if trigger fires)
        # Simplified: use correction only if |residual_pred| > threshold
        threshold = 50
        correction = np.zeros_like(y_val)
        correction[np.abs(predictions) > threshold] = predictions[np.abs(predictions) > threshold]
        
        final_pred = df[val_mask]['da_price'].values + correction
        rt_actual = df[val_mask]['rt_price'].values
        
        # Compute sMAPE
        smape = compute_day_level_smape(rt_actual, final_pred, 
                                        df[val_mask]['times'].values)
        
        # DA-only baseline
        da_smape = compute_day_level_smape(rt_actual, 
                                            df[val_mask]['da_price'].values,
                                            df[val_mask]['times'].values)
        
        improvement = da_smape - smape
        
        results.append({
            'val_period': val_start.strftime('%Y-%m'),
            'deep_smape': smape,
            'da_smape': da_smape,
            'improvement': improvement
        })
        
        print(f"  Deep Model sMAPE: {smape:.2f}")
        print(f"  DA sMAPE: {da_smape:.2f}")
        print(f"  Improvement: {improvement:.2f}")
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(out_dir / 'deep_model_leaderboard.csv', index=False, encoding='utf-8-sig')
    
    # Summary
    print(f"\n=== Summary (Deep Model) ===")
    print(f"Average Deep Model sMAPE: {results_df['deep_smape'].mean():.2f}")
    print(f"Average DA sMAPE: {results_df['da_smape'].mean():.2f}")
    print(f"Average Improvement: {results_df['improvement'].mean():.2f}")
    print(f"Months with improvement >= 0.3pp: {(results_df['improvement'] >= 0.3).sum()}/{len(results_df)}")
    
    # Check if should proceed to Phase E
    if results_df['improvement'].mean() >= 0.3:
        print(f"\n=== Verdict: KEEP ===")
        print(f"Proceed to Phase E: Multi-month Walk-forward Validation")
    else:
        print(f"\n=== Verdict: KILL ===")
        print(f"Stop. Deep model does not improve sMAPE.")

if __name__ == '__main__':
    main()
