# 🧠 ML Олимпиада: Полный Справочник по Базовым Методам

## 🔬 Примеры применений по задачам

### 📷 Задача: извлечение признаков из изображений
**Метод:** PCA на flattened array изображений
```python
X_images = images.reshape(images.shape[0], -1)
X_pca = PCA(n_components=50).fit_transform(X_images)
```

### 📚 Задача: классификация темы предложения из статьи
**Метод:** TF-IDF/CountVectorizer + LogisticRegression
```python
texts = df["sentence"]
tfidf = TfidfVectorizer(max_features=500)
X = tfidf.fit_transform(texts)
model = LogisticRegression()
model.fit(X, y)
```

### 📈 Задача: прогнозирование числа публикаций
**Метод:** groupby + признаки по дате + LGBMRegressor
```python
df["date"] = pd.to_datetime(df["date"])
df["month"] = df["date"].dt.month
df["year"] = df["date"].dt.year

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
X_corgis = images.reshape(images.shape[0], -1)
X_pca = PCA(n_components=2).fit_transform(X_corgis)
plt.scatter(X_pca[:, 0], X_pca[:, 1])
plt.title("Corgi Clusters")
```

### 🔄 Задача: поворот MNIST
**Метод:** PCA на изображениях + классификатор (например, GradientBoosting)
```python
X_mnist = images.reshape(images.shape[0], -1)
X_pca = PCA(n_components=40).fit_transform(X_mnist)
model = GradientBoostingClassifier()
model.fit(X_pca, y)
```

### 💬 Задача: матчинг врачебного заключения и саммари
**Метод:** TF-IDF + cosine similarity
```python
from sklearn.metrics.pairwise import cosine_similarity

vec = TfidfVectorizer(max_features=500)
X = vec.fit_transform(df["report"])
Y = vec.transform(df["summary"])

similarities = cosine_similarity(X, Y)
preds = similarities.argmax(axis=1)
```

_Далее — основные методы и сниппеты:_

## 🧩 Извлечение признаков

### 1. TF-IDF (для текста)
```python
from sklearn.feature_extraction.text import TfidfVectorizer

texts = ["AI is the future", "Machine learning is a subset of AI"]
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(texts)
```

### 2. CountVectorizer (простой bag-of-words)
```python
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)
```

### 3. PCA (снижение размерности: для картинок, TF-IDF и т.п.)
```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X.toarray())
```

### 4. GroupBy в pandas (для агрегированных признаков)
```python
import pandas as pd

df = pd.DataFrame({
    "category": ["cs", "cs", "physics"],
    "length": [100, 150, 200]
})
grouped = df.groupby("category")["length"].agg(["mean", "std"])
df = df.merge(grouped, on="category", how="left")
```

### 5. Извлечение признаков из изображений (flatten + PCA)
```python
import numpy as np
X_images = images.reshape(images.shape[0], -1)  # (n_samples, height*width)
X_pca = PCA(n_components=50).fit_transform(X_images)
```

... (оставшаяся часть без изменений)
