import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import zscore
from itertools import combinations
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------------
# 1. 데이터 수집
# ----------------------------------

# IXIC 데이터 수집 (종가/거래량)
ixic = yf.download("^IXIC", start="2024-01-01", end="2025-01-01", interval="1d")[['Close', 'Volume']]
ixic.rename(columns={'Close': 'IXIC_Close', 'Volume': 'IXIC_Volume'}, inplace=True)

# 기술적 지표 계산
bb = ta.bbands(close=ixic['IXIC_Close'], length=20, append=False)
macd = ta.macd(ixic['IXIC_Close'], append=False)

# 외부 지표 목록
tickers = {
    '^GSPC': 'S&P500',
    '^DJI': 'DOWJONES',
    '^VIX': 'VIX',
    'DX-Y.NYB': 'DOLLAR_INDEX',
    'CL=F': 'WTI',
    'GC=F': 'GOLD',
    '^TNX': 'US10Y',
    '^SOX': 'SEMI_INDEX',
    'SPY': 'SPY_ETF',
    'DIA': 'DIA_ETF',
    '^RUT': 'RUSSELL_2000',
    '^IRX': 'US3M',
    '^TYX': 'US30Y',
    '^FVX': 'US5Y',
    'LQD': 'INVESTMENT_GRADE_BOND',
    'TIP': 'TIPS_ETF',
    '^XAU': 'GOLD_MINER_INDEX'
}

# 외부 지표 수집
failed_tickers = []
for ticker, name in tickers.items():
    try:
        data = yf.download(ticker, start="2024-01-01", end="2025-01-01", interval='1d')
        ixic[name] = data['Close']
    except Exception as e:
        failed_tickers.append(name)
        print(f"Error downloading {name}: {e}")

if failed_tickers:
    print(f"Failed to download data for: {failed_tickers}")

# 기술적 지표 병합
df = pd.concat([ixic, bb, macd], axis=1)
df = df.loc[:, ~df.columns.duplicated()]  # 중복 제거

# ----------------------------------
# 2. EDA: 결측치 및 이상치 처리
# ----------------------------------

# 결측치 처리
df.interpolate(method='time', inplace=True)
df.fillna(method='ffill', inplace=True)
df.fillna(method='bfill', inplace=True)

# 이상치 제거 (Z-score 방식)
numeric_cols = df.select_dtypes(include='number').columns
z_scores = df[numeric_cols].apply(zscore)
df = df[(z_scores.abs() <= 3).all(axis=1)]

# ----------------------------------
# 3. 피처 중요도 분석 및 선택
# ----------------------------------

# 상관계수 기반 피처 선택
target_column = 'IXIC_Close'
correlation = df.corr()[target_column].drop(target_column).sort_values(ascending=False)
top_10_features = correlation.abs().head(10).index.tolist()

# 상관관계 Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df[top_10_features + [target_column]].corr(), annot=True, cmap='coolwarm')
plt.title("상위 피처 상관관계 Heatmap")
plt.tight_layout()
plt.show()

# ----------------------------------
# 4. 모델링: 최적 피처 조합 탐색 및 평가
# ----------------------------------

# 피처 조합 및 모델들
model_dict = {
    'Linear': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=0.1),
    'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5)
}

best_result = {'model': None, 'features': None, 'r2': -np.inf}

tscv = TimeSeriesSplit(n_splits=5)

for combo in tqdm(list(combinations(top_10_features, 4)), desc="Searching"):
    for model_name, model in model_dict.items():
        for train_index, test_index in tscv.split(df):
            X_train, X_test = df.iloc[train_index][list(combo)], df.iloc[test_index][list(combo)]
            y_train, y_test = df.iloc[train_index][target_column], df.iloc[test_index][target_column]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)

            if r2 > best_result['r2']:
                best_result.update({
                    'model': model_name,
                    'features': combo,
                    'r2': r2
                })
                
# 최종 결과 출력
print("\n✅ Best R² Score:", round(best_result['r2'], 4))
print("✅ Best Model:", best_result['model'])
print("✅ Best Feature Set:", best_result['features'])

# ----------------------------------
# 5. 최종 모델 학습 및 예측 시각화
# ----------------------------------

final_model = model_dict[best_result['model']]
X = df[list(best_result['features'])]
y = df[target_column]

for train_index, test_index in tscv.split(df):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

final_model.fit(X_train, y_train)
y_pred = final_model.predict(X_test)

results = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred
}, index=y_test.index)

# 시각화
plt.figure(figsize=(14, 6))
plt.plot(results.index, results['Actual'], label='Actual', linewidth=2)
plt.plot(results.index, results['Predicted'], label='Predicted', linestyle='--')
plt.title("Actual vs Predicted IXIC_Close")
plt.xlabel("Date")
plt.ylabel("IXIC_Close")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 잔차 분석
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True)
plt.title("Residuals Distribution")
plt.show()