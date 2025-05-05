# 🧠 ML Олимпиада: Полный Справочник по Базовым Методам

## 🔬 Примеры применений по задачам

### 📷 Задача: извлечение признаков из изображений
**Метод:** PCA на flattened array изображений
```python
from sklearn.decomposition import PCA

X_images = images.reshape(images.shape[0], -1)
X_pca = PCA(n_components=50).fit_transform(X_images)
```

### 📚 Задача: классификация темы предложения из статьи
**Метод:** TF-IDF/CountVectorizer + LogisticRegression
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

texts = df["sentence"]
tfidf = TfidfVectorizer(max_features=500)
X = tfidf.fit_transform(texts)
model = LogisticRegression()
model.fit(X, y)
```

### 📈 Задача: прогнозирование числа публикаций
**Метод:** groupby + признаки по дате + LGBMRegressor
```python
import pandas as pd
import lightgbm as lgb

# Добавление признаков по дате
df["date"] = pd.to_datetime(df["date"])
df["month"] = df["date"].dt.month
df["year"] = df["date"].dt.year

# Groupby признаки
agg = df.groupby("category")["num_papers"].agg(["mean", "std"])
df = df.merge(agg, on="category", how="left")

X = df[["month", "year", "mean", "std"]]
y = df["num_papers"]

model = lgb.LGBMRegressor()
model.fit(X, y)
```

### 🐶 Задача: визуализация корги (кластеризация, PCA)
**Метод:** PCA + визуализация
```python
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

X_corgis = images.reshape(images.shape[0], -1)
X_pca = PCA(n_components=2).fit_transform(X_corgis)
plt.scatter(X_pca[:, 0], X_pca[:, 1])
plt.title("Corgi Clusters")
plt.show()
```

### 🔄 Задача: поворот MNIST
**Метод:** PCA на изображениях + классификатор (например, GradientBoosting)
```python
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.decomposition import PCA

X_mnist = images.reshape(images.shape[0], -1)
X_pca = PCA(n_components=40).fit_transform(X_mnist)
model = GradientBoostingClassifier()
model.fit(X_pca, y)
```

### 💬 Задача: матчинг врачебного заключения и саммари
**Метод:** TF-IDF + cosine similarity
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

vec = TfidfVectorizer(max_features=500)
X = vec.fit_transform(df["report"])
Y = vec.transform(df["summary"])

similarities = cosine_similarity(X, Y)
preds = similarities.argmax(axis=1)
```

## 🧩 Извлечение признаков

### TF-IDF (для текста)
```python
from sklearn.feature_extraction.text import TfidfVectorizer
texts = ["AI is the future", "Machine learning is a subset of AI"]
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(texts)
```

### CountVectorizer (простой bag-of-words)
```python
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)
```

### PCA (для снижения размерности)
```python
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X.toarray())
```

### GroupBy признаки
```python
df = pd.DataFrame({"category": ["cs", "cs", "physics"], "length": [100, 150, 200]})
grouped = df.groupby("category")["length"].agg(["mean", "std"])
df = df.merge(grouped, on="category", how="left")
```

### Извлечение из изображений (flatten)
```python
X_images = images.reshape(images.shape[0], -1)
X_pca = PCA(n_components=50).fit_transform(X_images)
```

## 📥 Работа с pandas и NumPy

### Импорт и базовые операции
```python
df = pd.read_csv("data.csv")
df["mean"] = df.mean(axis=1)
df["sum"] = df.sum(axis=1)
```

### NumPy
```python
import numpy as np
np.mean(arr), np.std(arr), np.argmax(arr), np.argsort(arr), np.unique(arr)
```

## 🤖 Модели

### Linear Regression
```python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
```

### Logistic Regression
```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=300)
model.fit(X_train, y_train)
```

### LightGBM
```python
import lightgbm as lgb
model = lgb.LGBMClassifier()
model.fit(X_train, y_train)
```

## ✅ Валидация

### Train/Test Split
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```

### KFold
```python
from sklearn.model_selection import KFold
kf = KFold(n_splits=5)
for train_idx, test_idx in kf.split(X):
    model.fit(X[train_idx], y[train_idx])
    preds = model.predict(X[test_idx])
```

## 📤 Сохранение сабмита
```python
submission = pd.DataFrame({"id": df["id"], "label": predictions})
submission.to_csv("submission.csv", index=False)
```

## 🏁 Финал: Используй минимум — побеждай максимумом 💪
