"""Small CLI runner to do a quick LR predict for a given symbol."""
import argparse
from src.data import fetch_history
from src.features import add_features
from src.models import split_train_test, train_lr, predict_lr, mse


def run(symbol: str = 'AAPL'):
    df = fetch_history(symbol, period='180d')
    df_feat = add_features(df)
    train, test = split_train_test(df_feat, target_col='Close', test_size=30)
    feature_cols = [c for c in df_feat.columns if c not in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
    model, scaler = train_lr(train, feature_cols)
    preds = predict_lr(model, scaler, test, feature_cols)
    print(f"MSE: {mse(test['Close'].values, preds):.4f}")
    print("Last actual:", test['Close'].iloc[-1])
    print("Last predicted:", preds[-1])


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--symbol', default='AAPL')
    args = p.parse_args()
    run(args.symbol)
