import joblib
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import numpy as np

# 前処理済みのデータをロード
X, y = joblib.load('hog_features_labels.pkl')

# データをトレーニングセットとテストセットに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# グリッドサーチのためのパラメータ候補を設定（範囲をさらに広げる）
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100, 500, 1000, 5000],
    'gamma': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
    'kernel': ['rbf']
}

# SVMモデルを初期化
svm = SVC()

# グリッドサーチを実行
print("Starting Grid Search...")
grid_search = GridSearchCV(svm, param_grid, cv=10, verbose=10, return_train_score=True)
grid_search.fit(X_train, y_train)

# 最適なパラメータを表示
print("Best parameters set found on development set:")
print(grid_search.best_params_)

# 最適なモデルを保存
joblib.dump(grid_search.best_estimator_, 'svm_model.pkl')
print("Model saved as svm_model.pkl")

# テストセットでの予測
y_pred = grid_search.predict(X_test)

# 分類レポートを表示
print("Classification Report:")
print(classification_report(y_test, y_pred))

# 精度を表示
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# 学習曲線をプロットして保存
results = grid_search.cv_results_

# 各パラメータに対するスコアを取得
mean_train_scores = results['mean_train_score']
mean_test_scores = results['mean_test_score']
params = results['params']

# Cとgammaのペアに対する学習曲線のプロット
plt.figure(figsize=(12, 6))
for gamma in param_grid['gamma']:
    gamma_indices = [i for i, param in enumerate(params) if param['gamma'] == gamma]
    C_vals = [params[i]['C'] for i in gamma_indices]
    train_scores = [mean_train_scores[i] for i in gamma_indices]
    test_scores = [mean_test_scores[i] for i in gamma_indices]
    plt.plot(C_vals, train_scores, label=f'Train Score (gamma={gamma})', marker='o')
    plt.plot(C_vals, test_scores, label=f'Validation Score (gamma={gamma})', marker='x')

plt.xlabel('C (log scale)')
plt.ylabel('Score')
plt.xscale('log')
plt.title('Learning Curve (RBF Kernel)')
plt.legend()
plt.savefig('learning_curve_rbf.png')
plt.show()
