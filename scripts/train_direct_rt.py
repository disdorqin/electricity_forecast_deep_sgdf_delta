"""
Train model to directly predict RT price (not residual)
Uses all available features from preprocessed data
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import argparse
import sys
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

def smape_floor50(y_true, y_pred, floor=50.0):
    """Compute sMAPE with floor=50 for denominator."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    denom = np.maximum((np.abs(y_true) + np.abs(y_pred)) / 2, floor)
    return 100 * np.mean(np.abs(y_true - y_pred) / denom)

def day_level_smape(y_true, y_pred, timestamps):
    """
    Compute day-level sMAPE: aggregate 24h predictions per day, then compute sMAPE.
    This is the correct metric (not hourly sMAPE).
    """
    df = pd.DataFrame({
        'timestamp': timestamps,
        'y_true': y_true,
        'y_pred': y_pred
    })
    df['date'] = df['timestamp'].dt.date
    
    daily_true = df.groupby('date')['y_true'].mean()
    daily_pred = df.groupby('date')['y_pred'].mean()
    
    return smape_floor50(daily_true.values, daily_pred.values)

class TimeSeriesDataset(Dataset):
    """Time series dataset for sequence modeling."""
    def __init__(self, X, y, seq_len=168):
        self.X = X
        self.y = y
        self.seq_len = seq_len
        
    def __len__(self):
        return len(self.X) - self.seq_len
    
    def __getitem__(self, idx):
        X_seq = self.X[idx:idx+self.seq_len]
        y_target = self.y[idx+self.seq_len-1]  # Predict last step
        return torch.FloatTensor(X_seq), torch.FloatTensor([y_target])

class LSTMModel(nn.Module):
    """LSTM model for direct RT price prediction."""
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)
        # Use last output
        last_out = lstm_out[:, -1, :]
        output = self.fc(last_out)
        return output.squeeze(-1)

def train_model(model, train_loader, val_loader, device, epochs=50, lr=0.001):
    """Train model with early stopping."""
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
        for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
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
        
        print(f"Epoch {epoch+1}: train_loss={train_loss/len(train_loader):.4f}, val_loss={val_loss/len(val_loader):.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load best model
    model.load_state_dict(torch.load('best_model.pth'))
    return model

def walk_forward_backtest(df, feature_cols, target_col='rt_price', 
                         train_months=6, test_months=1, 
                         seq_len=168, batch_size=64, hidden_dim=128, 
                         num_layers=2, epochs=50, lr=0.001):
    """
    Walk-forward backtest: train on rolling window, test on future months.
    """
    # Sort by time
    df = df.sort_values('times').reset_index(drop=True)
    
    # Normalize features
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_all = df[feature_cols].values
    y_all = df[target_col].values
    times_all = df['times'].values
    
    # Fit scalers on first 6 months
    initial_end = pd.Timestamp('2022-07-01')
    initial_mask = df['times'] < initial_end
    scaler_X.fit(X_all[initial_mask])
    scaler_y.fit(y_all[initial_mask].reshape(-1, 1))
    
    # Walk-forward loop
    results = []
    current_date = pd.Timestamp('2022-07-01')
    end_date = df['times'].max()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    while current_date < end_date:
        # Define train/test periods
        train_start = current_date - pd.DateOffset(months=train_months)
        train_end = current_date
        test_start = current_date
        test_end = current_date + pd.DateOffset(months=test_months)
        
        # Get train/test data
        train_mask = (df['times'] >= train_start) & (df['times'] < train_end)
        test_mask = (df['times'] >= test_start) & (df['times'] < test_end)
        
        X_train = scaler_X.transform(X_all[train_mask])
        y_train = scaler_y.transform(y_all[train_mask].reshape(-1, 1)).flatten()
        X_test = scaler_X.transform(X_all[test_mask])
        y_test = y_all[test_mask]
        times_test = times_all[test_mask]
        
        # Create datasets
        train_dataset = TimeSeriesDataset(X_train, y_train, seq_len)
        test_dataset = TimeSeriesDataset(X_test, y_test, seq_len)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Train model
        model = LSTMModel(input_dim=len(feature_cols), hidden_dim=hidden_dim, 
                        num_layers=num_layers)
        model = train_model(model, train_loader, test_loader, device, epochs, lr)
        
        # Predict
        model.eval()
        predictions = []
        with torch.no_grad():
            for X_batch, _ in test_loader:
                X_batch = X_batch.to(device)
                y_pred = model(X_batch)
                predictions.extend(y_pred.cpu().numpy())
        
        # Inverse transform predictions
        predictions = scaler_y.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
        
        # Compute metrics
        # For test period, we need to align predictions with actuals
        # (predictions are shifted by seq_len)
        test_times = times_test[seq_len:]
        test_actual = y_test[seq_len:]
        test_pred = predictions[:len(test_actual)]
        
        # Day-level sMAPE
        smape = day_level_smape(test_actual, test_pred, pd.Series(test_times))
        
        # DA-only baseline
        da_baseline = df[test_mask]['da_price'].values[seq_len:]
        da_smape = day_level_smape(test_actual, da_baseline, pd.Series(test_times))
        
        results.append({
            'test_period': f"{test_start.strftime('%Y-%m')} to {test_end.strftime('%Y-%m')}",
            'smape': smape,
            'da_smape': da_smape,
            'improvement': da_smape - smape
        })
        
        print(f"\nTest period: {test_start.strftime('%Y-%m')} to {test_end.strftime('%Y-%m')}")
        print(f"  Model sMAPE: {smape:.2f}")
        print(f"  DA baseline sMAPE: {da_smape:.2f}")
        print(f"  Improvement: {da_smape - smape:.2f}")
        
        # Move to next period
        current_date += pd.DateOffset(months=test_months)
    
    return pd.DataFrame(results)

def main():
    parser = argparse.ArgumentParser(description='Train direct RT price prediction model')
    parser.add_argument('--data-path', type=str, 
                       default=r"D:\作业\大创_挑战杯_互联网\大学生创新创业计划\大创实现\其他资料\deep_model_for_electricity\data\preprocessed_data.csv",
                       help='Path to preprocessed data CSV')
    parser.add_argument('--target-col', type=str, default='rt_price',
                       help='Target column name')
    parser.add_argument('--seq-len', type=int, default=168,
                       help='Sequence length (hours)')
    parser.add_argument('--hidden-dim', type=int, default=128,
                       help='LSTM hidden dimension')
    parser.add_argument('--num-layers', type=int, default=2,
                       help='Number of LSTM layers')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size')
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.data_path}...")
    df = pd.read_csv(args.data_path, parse_dates=['times'], encoding='utf-8-sig')
    print(f"Data loaded: {len(df)} rows")
    
    # Define feature columns (exclude targets, timestamps, and actual values to avoid leakage)
    exclude_cols = ['times', 'rt_price', 'da_price', 'residual', 
                   'local_plant_actual', 'tie_line_load_actual', 'wind_actual', 
                   'solar_actual', 'nuclear_actual', 'self_supply_actual', 
                   'test_unit_actual', 'direct_dispatch_actual', 
                   'bidding_space_actual', 'renewable_actual']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    print(f"Number of features: {len(feature_cols)}")
    print(f"Feature columns: {feature_cols[:10]}...")  # Show first 10
    
    # Run walk-forward backtest
    print("\n=== Starting Walk-Forward Backtest ===")
    results_df = walk_forward_backtest(
        df, feature_cols, args.target_col,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        epochs=args.epochs,
        lr=args.lr
    )
    
    # Save results
    output_path = Path(args.data_path).parent / 'backtest_results.csv'
    results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\nResults saved to {output_path}")
    
    # Print summary
    print("\n=== Backtest Summary ===")
    print(f"Average sMAPE: {results_df['smape'].mean():.2f}")
    print(f"Average DA sMAPE: {results_df['da_smape'].mean():.2f}")
    print(f"Average improvement: {results_df['improvement'].mean():.2f}")
    print(f"Number of periods with improvement: {(results_df['improvement'] > 0).sum()}/{len(results_df)}")

if __name__ == '__main__':
    main()
