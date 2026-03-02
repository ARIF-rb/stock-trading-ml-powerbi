"""
Complete End-to-End Stock Analysis and Machine Learning Pipeline
This script performs data collection, technical indicator calculation, machine learning, 
backtesting, and creates a comprehensive DataFrame for Power BI visualization
"""
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Technical Analysis Libraries
import ta
from ta.trend import MACD, EMAIndicator, SMAIndicator, IchimokuIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands

# Machine Learning Libraries
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, confusion_matrix, classification_report,
                           roc_auc_score, roc_curve)
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE

# Set random seed for reproducibility
np.random.seed(42)

# ==================== CONFIGURATION ====================
# Stock symbols to analyze
SYMBOLS = ['META', 'USO', 'URTH']  # META (Tech), USO (Oil ETF proxy for WTI), URTH (World ETF)
START_DATE = datetime.now() - timedelta(days=5*365)  # 5 years of data
END_DATE = datetime.now()

# Thresholds for signal generation
BUY_THRESHOLD = -1.0  # Daily change < -1% = Buy signal
SELL_THRESHOLD = 1.0  # Daily change > 1% = Sell signal

# ==================== DATA COLLECTION ====================
def fetch_stock_data(symbol, start, end):
    """Fetch historical stock data from Yahoo Finance"""
    print(f"Fetching data for {symbol}...")
    stock = yf.Ticker(symbol)
    data = stock.history(start=start, end=end)
    
    # Add symbol column
    data['Symbol'] = symbol
    
    # Ensure we have all required columns
    data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
    
    # Calculate additional price metrics
    data['Highest_Price_Period'] = data['High'].rolling(window=20).max()
    data['Lowest_Price_Period'] = data['Low'].rolling(window=20).min()
    data['Price_Range'] = data['High'] - data['Low']
    data['Typical_Price'] = (data['High'] + data['Low'] + data['Close']) / 3
    
    return data

# ==================== TECHNICAL INDICATORS ====================
def calculate_indicators(df):
    """Calculate all technical indicators"""
    print("Calculating technical indicators...")
    
    # RSI
    df['RSI'] = RSIIndicator(close=df['Close'], window=14).rsi()
    
    # MACD
    macd = MACD(close=df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Histogram'] = macd.macd_diff()
    
    # Moving Averages
    df['SMA_20'] = SMAIndicator(close=df['Close'], window=20).sma_indicator()
    df['SMA_50'] = SMAIndicator(close=df['Close'], window=50).sma_indicator()
    df['EMA_12'] = EMAIndicator(close=df['Close'], window=12).ema_indicator()
    df['EMA_26'] = EMAIndicator(close=df['Close'], window=26).ema_indicator()
    
    # Stochastic Oscillator
    stoch = StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close'])
    df['Stoch_K'] = stoch.stoch()
    df['Stoch_D'] = stoch.stoch_signal()
    
    # Bollinger Bands
    bb = BollingerBands(close=df['Close'])
    df['BB_Upper'] = bb.bollinger_hband()
    df['BB_Middle'] = bb.bollinger_mavg()
    df['BB_Lower'] = bb.bollinger_lband()
    df['BB_Width'] = bb.bollinger_wband()
    df['BB_Percent'] = bb.bollinger_pband()
    
    # Ichimoku Cloud
    ichimoku = IchimokuIndicator(high=df['High'], low=df['Low'])
    df['Ichimoku_Tenkan'] = ichimoku.ichimoku_conversion_line()
    df['Ichimoku_Kijun'] = ichimoku.ichimoku_base_line()
    df['Ichimoku_Senkou_A'] = ichimoku.ichimoku_a()
    df['Ichimoku_Senkou_B'] = ichimoku.ichimoku_b()
    
    # Standard Deviation
    df['StdDev'] = df['Close'].rolling(window=20).std()
    
    # Volume indicators
    df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
    
    # Price change metrics
    df['Daily_Return'] = df['Close'].pct_change()
    df['Daily_Change_Percent'] = ((df['Open'] - df['Open'].shift(1)) / df['Open'].shift(1)) * 100
    
    # Volatility
    df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])
    
    # Additional features
    df['High_Low_Ratio'] = df['High'] / df['Low']
    df['Close_Open_Ratio'] = df['Close'] / df['Open']
    
    return df

# ==================== TARGET VARIABLE ====================
def create_target_variable(df):
    """Create target variable based on daily change thresholds"""
    conditions = [
        df['Daily_Change_Percent'] > SELL_THRESHOLD,
        df['Daily_Change_Percent'] < BUY_THRESHOLD
    ]
    choices = [1, 2]  # 1: Sell, 2: Buy
    df['Target'] = np.select(conditions, choices, default=0)  # 0: Hold
    
    # Create target names for clarity
    df['Target_Label'] = df['Target'].map({0: 'Hold', 1: 'Sell', 2: 'Buy'})
    
    return df

# ==================== DATA PREPARATION ====================
def prepare_ml_data(df):
    """Prepare data for machine learning"""
    # Drop rows with NaN values
    df = df.dropna()
    
    # Feature columns (all indicators)
    feature_cols = ['RSI', 'MACD', 'MACD_Signal', 'MACD_Histogram', 
                   'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26',
                   'Stoch_K', 'Stoch_D', 'BB_Upper', 'BB_Middle', 
                   'BB_Lower', 'BB_Width', 'BB_Percent',
                   'Ichimoku_Tenkan', 'Ichimoku_Kijun', 
                   'Ichimoku_Senkou_A', 'Ichimoku_Senkou_B',
                   'StdDev', 'Volume_Ratio', 'ATR',
                   'High_Low_Ratio', 'Close_Open_Ratio']
    
    X = df[feature_cols]
    y = df['Target']
    
    return X, y, feature_cols

# ==================== MACHINE LEARNING MODELS ====================
def train_models(X_train, X_test, y_train, y_test, symbol):
    """Train multiple ML models and return results"""
    print(f"\nTraining models for {symbol}...")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Handle class imbalance with SMOTE
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
    
    # Initialize models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'AdaBoost': AdaBoostClassifier(random_state=42)
    }
    
    results = {}
    best_model = None
    best_accuracy = 0
    
    for name, model in models.items():
        # Train model
        model.fit(X_train_balanced, y_train_balanced)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Store results
        results[name] = {
            'model': model,
            'predictions': y_pred,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            
        print(f"{name}: Accuracy={accuracy:.3f}, F1={f1:.3f}")
    
    # Create ensemble model
    ensemble = VotingClassifier(
        estimators=[(name, model) for name, model in models.items()],
        voting='soft'
    )
    ensemble.fit(X_train_balanced, y_train_balanced)
    y_pred_ensemble = ensemble.predict(X_test_scaled)
    
    results['Ensemble'] = {
        'model': ensemble,
        'predictions': y_pred_ensemble,
        'accuracy': accuracy_score(y_test, y_pred_ensemble),
        'precision': precision_score(y_test, y_pred_ensemble, average='weighted', zero_division=0),
        'recall': recall_score(y_test, y_pred_ensemble, average='weighted'),
        'f1_score': f1_score(y_test, y_pred_ensemble, average='weighted'),
        'confusion_matrix': confusion_matrix(y_test, y_pred_ensemble)
    }
    
    return results, scaler, best_model

# ==================== BACKTESTING ====================
def backtest_strategy(df, predictions, initial_capital=100):
    """Perform backtesting of the trading strategy"""
    print("Running backtesting...")
    
    # Initialize portfolio
    portfolio = {
        'cash': initial_capital,
        'holdings': 0,
        'portfolio_value': [],
        'dates': [],
        'actions': [],
        'returns': []
    }
    
    df_backtest = df.copy()
    df_backtest['Predicted_Signal'] = predictions
    
    for idx, row in df_backtest.iterrows():
        current_price = row['Close']
        signal = row['Predicted_Signal']
        
        # Calculate current portfolio value
        current_value = portfolio['cash'] + (portfolio['holdings'] * current_price)
        portfolio['portfolio_value'].append(current_value)
        portfolio['dates'].append(idx)
        
        # Trading logic
        action = 'Hold'
        if signal == 2 and portfolio['cash'] > 0:  # Buy signal
            shares_to_buy = portfolio['cash'] / current_price
            portfolio['holdings'] += shares_to_buy
            portfolio['cash'] = 0
            action = 'Buy'
        elif signal == 1 and portfolio['holdings'] > 0:  # Sell signal
            portfolio['cash'] = portfolio['holdings'] * current_price
            portfolio['holdings'] = 0
            action = 'Sell'
            
        portfolio['actions'].append(action)
        
        # Calculate returns
        if len(portfolio['portfolio_value']) > 1:
            daily_return = (portfolio['portfolio_value'][-1] - portfolio['portfolio_value'][-2]) / portfolio['portfolio_value'][-2]
            portfolio['returns'].append(daily_return)
        else:
            portfolio['returns'].append(0)
    
    # Calculate performance metrics
    total_return = (portfolio['portfolio_value'][-1] - initial_capital) / initial_capital
    sharpe_ratio = np.mean(portfolio['returns']) / (np.std(portfolio['returns']) + 1e-10) * np.sqrt(252)
    max_drawdown = calculate_max_drawdown(portfolio['portfolio_value'])
    
    return {
        'total_return': total_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'final_value': portfolio['portfolio_value'][-1],
        'portfolio_values': portfolio['portfolio_value'],
        'dates': portfolio['dates'],
        'actions': portfolio['actions']
    }

def calculate_max_drawdown(portfolio_values):
    """Calculate maximum drawdown"""
    peak = portfolio_values[0]
    max_dd = 0
    
    for value in portfolio_values:
        if value > peak:
            peak = value
        dd = (peak - value) / peak
        if dd > max_dd:
            max_dd = dd
            
    return max_dd

# ==================== MAIN PIPELINE ====================
def create_powerbi_dataframe(all_data, all_results, all_backtest):
    """Create comprehensive DataFrame for Power BI"""
    print("\nCreating Power BI output DataFrame...")
    
    powerbi_data = []
    
    for symbol in all_data.keys():
        df = all_data[symbol]
        results = all_results[symbol]
        backtest = all_backtest[symbol]
        
        # Get the best model's predictions
        best_model_name = max(results.keys(), key=lambda k: results[k]['accuracy'])
        best_predictions = results[best_model_name]['predictions']
        
        # Create base dataframe
        df_export = df.copy()
        df_export['Symbol'] = symbol
        df_export['Predicted_Signal'] = np.nan
        df_export.iloc[-len(best_predictions):, df_export.columns.get_loc('Predicted_Signal')] = best_predictions
        
        # Add model performance metrics
        for model_name, model_results in results.items():
            df_export[f'{model_name}_Accuracy'] = model_results['accuracy']
            df_export[f'{model_name}_Precision'] = model_results['precision']
            df_export[f'{model_name}_Recall'] = model_results['recall']
            df_export[f'{model_name}_F1'] = model_results['f1_score']
            
            # Flatten confusion matrix
            cm = model_results['confusion_matrix']
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    df_export[f'{model_name}_CM_{i}_{j}'] = cm[i, j]
        
        # Add backtesting results
        df_export['Backtest_Total_Return'] = backtest['total_return']
        df_export['Backtest_Sharpe_Ratio'] = backtest['sharpe_ratio']
        df_export['Backtest_Max_Drawdown'] = backtest['max_drawdown']
        df_export['Backtest_Final_Value'] = backtest['final_value']
        
        # Add portfolio values to matching dates
        portfolio_df = pd.DataFrame({
            'Date': backtest['dates'],
            'Portfolio_Value': backtest['portfolio_values'],
            'Trading_Action': backtest['actions']
        })
        portfolio_df.set_index('Date', inplace=True)
        
        # Merge portfolio values
        df_export = df_export.merge(portfolio_df[['Portfolio_Value', 'Trading_Action']], 
                                   left_index=True, right_index=True, how='left')
        
        # Add date components for Power BI time intelligence
        df_export.reset_index(inplace=True)
        df_export['Date'] = pd.to_datetime(df_export['Date'])
        df_export['Year'] = df_export['Date'].dt.year
        df_export['Month'] = df_export['Date'].dt.month
        df_export['Quarter'] = df_export['Date'].dt.quarter
        df_export['Week'] = df_export['Date'].dt.isocalendar().week
        df_export['DayOfWeek'] = df_export['Date'].dt.dayofweek
        df_export['DayName'] = df_export['Date'].dt.day_name()
        
        powerbi_data.append(df_export)
    
    # Combine all stocks
    final_df = pd.concat(powerbi_data, ignore_index=True)
    
    # Add additional calculated fields for Power BI
    final_df['Signal_Accuracy'] = (final_df['Target'] == final_df['Predicted_Signal']).astype(int)
    final_df['Profit_Loss'] = final_df['Portfolio_Value'].diff()
    final_df['Cumulative_Return'] = (final_df.groupby('Symbol')['Portfolio_Value']
                                      .transform(lambda x: (x / x.iloc[0] - 1) * 100))
    
    return final_df

# ==================== MAIN EXECUTION ====================
def main():
    """Main execution function"""
    print("Starting complete stock analysis pipeline...")
    print("="*50)
    
    all_data = {}
    all_results = {}
    all_backtest = {}
    
    for symbol in SYMBOLS:
        print(f"\n{'='*20} Processing {symbol} {'='*20}")
        
        # Fetch data
        df = fetch_stock_data(symbol, START_DATE, END_DATE)
        
        # Calculate indicators
        df = calculate_indicators(df)
        
        # Create target variable
        df = create_target_variable(df)
        
        # Prepare ML data
        X, y, feature_cols = prepare_ml_data(df)
        
        # Split data (80/20)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train models
        results, scaler, best_model = train_models(X_train, X_test, y_train, y_test, symbol)
        
        # Get predictions for entire dataset for backtesting
        X_scaled = scaler.transform(X)
        full_predictions = best_model.predict(X_scaled)
        
        # Backtest
        backtest_results = backtest_strategy(df.loc[X.index], full_predictions)
        
        # Store results
        all_data[symbol] = df.loc[X.index]
        all_results[symbol] = results
        all_backtest[symbol] = backtest_results
        
        # Print summary
        print(f"\nSummary for {symbol}:")
        print(f"  Total Return: {backtest_results['total_return']*100:.2f}%")
        print(f"  Sharpe Ratio: {backtest_results['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown: {backtest_results['max_drawdown']*100:.2f}%")
    
    # Create Power BI DataFrame
    powerbi_df = create_powerbi_dataframe(all_data, all_results, all_backtest)
    
    # Save to CSV for Power BI
    output_file = 'stock_analysis_powerbi_output.csv'
    powerbi_df.to_csv(output_file, index=False)
    print(f"\n{'='*50}")
    print(f"Power BI data saved to: {output_file}")
    print(f"Total rows: {len(powerbi_df)}")
    print(f"Total columns: {len(powerbi_df.columns)}")
    
    # Display sample of the output
    print("\nSample of Power BI DataFrame:")
    print(powerbi_df[['Date', 'Symbol', 'Close', 'RSI', 'Target', 
                      'Predicted_Signal', 'Portfolio_Value', 'Cumulative_Return']].head(10))
    
    # Print column categories for Power BI setup
    print("\n" + "="*50)
    print("POWER BI COLUMN CATEGORIES:")
    print("-"*30)
    
    price_columns = [col for col in powerbi_df.columns if any(x in col for x in ['Open', 'High', 'Low', 'Close', 'Price'])]
    indicator_columns = [col for col in powerbi_df.columns if any(x in col for x in ['RSI', 'MACD', 'BB_', 'Stoch', 'SMA', 'EMA', 'Ichimoku', 'ATR'])]
    ml_columns = [col for col in powerbi_df.columns if any(x in col for x in ['Accuracy', 'Precision', 'Recall', 'F1', 'CM_'])]
    backtest_columns = [col for col in powerbi_df.columns if 'Backtest' in col or 'Portfolio' in col]
    
    print(f"Price Columns ({len(price_columns)}): {price_columns[:5]}...")
    print(f"Indicator Columns ({len(indicator_columns)}): {indicator_columns[:5]}...")
    print(f"ML Metrics Columns ({len(ml_columns)}): {ml_columns[:5]}...")
    print(f"Backtest Columns ({len(backtest_columns)}): {backtest_columns[:5]}...")
    
    return powerbi_df

# Run the pipeline
if __name__ == "__main__":
    final_dataframe = main()
    
    print("\n" + "="*50)
    print("PIPELINE COMPLETE!")
    print("="*50)
    print("\nYou can now import 'stock_analysis_powerbi_output.csv' into Power BI")
    print("All calculations are complete - just visualize in Power BI!")
