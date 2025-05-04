# ✅ MACHINE LEARNING OLYMPIAD CHEATSHEET (with code snippets, beginner-friendly)

### 🌐 USEFUL RESOURCES DURING COMPETITION (EXTENDED)
| Название | Ссылка | Когда использовать / Что искать |
|----------|--------|-------------------------------|
| Python Docs | https://docs.python.org/3/ | Любой стандартный модуль, синтаксис, типы (например: `str.split`, `itertools`, `zip`) |
| NumPy | https://numpy.org/doc/ | Векторные операции, линалг, reshape, broadcasting (например: `np.mean`, `np.linalg.inv`) |
| Pandas | https://pandas.pydata.org/docs/ | Работа с таблицами: фильтрация, группировка, обработка datetime (например: `df.groupby`, `df.merge`) |
| SciPy | https://docs.scipy.org/doc/scipy/ | Статистика (`scipy.stats`), интерполяция, матричное разложение (например: `ttest_ind`, `sparse`) |
| Matplotlib | https://matplotlib.org/stable/contents.html | Построение графиков: `plt.plot`, `plt.hist`, `plt.imshow` и настройка отображения |
| Seaborn | https://seaborn.pydata.org/ | Статистическая визуализация: `sns.heatmap`, `sns.boxplot`, `sns.pairplot` |
| Scikit-learn | https://scikit-learn.org/stable/ | Все ML модели, метрики (`accuracy_score`, `f1_score`), пайплайны, трансформеры |
| XGBoost | https://xgboost.readthedocs.io/en/stable/ | Все про `XGBClassifier`, DMatrix, `cv`, параметры `eta`, `max_depth`, `early_stopping_rounds` |
| LightGBM | https://lightgbm.readthedocs.io/en/stable/ | Быстрая альтернатива XGBoost, часто лучше на больших данных или с множеством категорий |
| CatBoost | https://catboost.ai/en/docs/ | Бустинг без кодирования категорий: см. параметры `cat_features`, `loss_function` |
| PyTorch | https://pytorch.org/docs/stable/index.html | Все про `nn.Module`, `optim`, `autograd`, `torch.utils.data`, `device` |
| TensorFlow | https://www.tensorflow.org/api_docs | Альтернатива PyTorch, см. `tf.keras.models`, `tf.data`, `@tf.function`, TPU support |
| Kaggle Docs | https://www.kaggle.com/docs | Как загружать файлы (`/kaggle/input`), как сабмитить (`submission.csv`) и как писать notebook |
| Towards Data Science | https://towardsdatascience.com/ | Как реализовать идею/проект (часто используют для NLP, CV, feature engineering) |
| Analytics Vidhya | https://www.analyticsvidhya.com/ | Задачи по structured data, объяснения по метрикам, pipeline, EDA |
| Distill.pub | https://distill.pub/ | Интерактивное объяснение attention, градиентов, backpropagation — полезно для визуализации концепций |
| ML Cheatsheet | https://ml-cheatsheet.readthedocs.io/ | Формулы по классификаторам, регрессии, loss-функции, пайплайн |
| Sebastian Raschka Blog | https://sebastianraschka.com/ | Продвинутая ML теория + код: softmax, crossval, multinomialNB и многое другое |
| KDnuggets | https://www.kdnuggets.com/ | Быстрые новости, тенденции в индустрии, разбор архитектур и кейсов |
| GitHub | https://github.com/ | Найти пример реализации: `resnet pytorch`, `bert classifier`, `xgboost cv` |
| Kaggle | https://www.kaggle.com/ | Поиск kernel с похожей задачей, baseline решения, обсуждение структуры submission и leaderboard tips |

> 📌 Tip: Используйте CTRL+F на этих сайтах для быстрого поиска нужной функции или примера. Например: `site:scikit-learn.org train_test_split` в Google.

...

# ✅ MACHINE LEARNING OLYMPIAD CHEATSHEET (with code snippets, beginner-friendly)

---

### ⚙️ Additional Preprocessing Techniques (For Any Case)

```python
from sklearn.preprocessing import MinMaxScaler, RobustScaler, LabelEncoder, PowerTransformer
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif

# Missing values
imputer = SimpleImputer(strategy='mean')  # Or 'median', 'most_frequent'
X_imputed = imputer.fit_transform(X)

# Scaling alternatives
X_minmax = MinMaxScaler().fit_transform(X)
X_robust = RobustScaler().fit_transform(X)  # Works better with outliers

# Encode target if it's not numeric
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Gaussianize skewed features
X_powered = PowerTransformer().fit_transform(X)  # Box-Cox or Yeo-Johnson

# Feature selection
X_kbest = SelectKBest(f_classif, k=10).fit_transform(X, y)  # Use with caution
```

---

### 🧠 Alternative Models for Tabular Data

```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Try multiple classifiers quickly
models = [
    LogisticRegression(max_iter=1000),
    RandomForestClassifier(n_estimators=100),
    GradientBoostingClassifier(),
    LGBMClassifier(),
    CatBoostClassifier(verbose=0)
]

for model in models:
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    print(type(model).__name__, f1_score(y_valid, preds))
```

---

### 🔁 Model Ensembling & Stacking (Classic Way)

```python
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# Hard/soft voting
voting = VotingClassifier([
    ('lr', LogisticRegression(max_iter=1000)),
    ('rf', RandomForestClassifier()),
    ('svc', SVC(probability=True))
], voting='soft')

voting.fit(X_train, y_train)
print('Voting F1:', f1_score(y_valid, voting.predict(X_valid)))

# Stacking (uses meta-model)
stack = StackingClassifier(
    estimators=[('rf', RandomForestClassifier()), ('nb', GaussianNB())],
    final_estimator=LogisticRegression()
)
stack.fit(X_train, y_train)
```

---

### 🗂 Use These Depending On:
- **Heavy outliers?** → `RobustScaler`, `IsolationForest`, `PowerTransformer`
- **Mixed dtypes?** → `CatBoost`, `ColumnTransformer`, `OneHot + Num`
- **Small data?** → `TF-IDF`, `LogisticRegression`, no deep models
- **Many classes?** → use `objective='multi:softprob'` or `multi_class='multinomial'`
- **Time-series?** → sort by time, no shuffling, use lag features, `TimeSeriesSplit`
- **Sparse inputs?** → `SGDClassifier`, `MultinomialNB`
- **Slow training?** → `hist` tree method (XGBoost), `max_bins`, or downsampling


### 📌 1. DATA PREPROCESSING PIPELINE (EXPLAINED)
```python
# Step-by-step data preparation for structured/tabular data
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE

# 1. Load dataset
df = pd.read_csv('data.csv')             # Your dataset must contain a 'target' column
X = df.drop('target', axis=1)            # Features only
y = df['target']                         # Labels

# 2. Remove multivariate outliers
iso = IsolationForest(contamination=0.01, random_state=42)
outlier_mask = iso.fit_predict(X) == 1
X, y = X[outlier_mask], y[outlier_mask]  # Keep normal rows only

# 3. Encode categorical features
cat_cols = X.select_dtypes(['object', 'category']).columns
ohe = OneHotEncoder(sparse=False, drop='first')
X_cat = pd.DataFrame(ohe.fit_transform(X[cat_cols]), index=X.index)
X = pd.concat([X.drop(cat_cols, axis=1), X_cat], axis=1)

# 4. Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Optionally reduce dimensions (e.g., to speed up training)
pca = PCA(n_components=0.95)  # Keeps 95% of variance
X_reduced = pca.fit_transform(X_scaled)

# 6. Handle class imbalance (e.g., in binary classification)
smote = SMOTE(random_state=42)
X_bal, y_bal = smote.fit_resample(X_reduced, y)
```

### 📌 2. HYPERPARAMETER TUNING FOR XGBOOST (EXPLAINED)

> 🎯 **Цель:** найти наилучшие параметры модели XGBoost, чтобы улучшить её обобщающую способность.
> 📦 **Когда использовать:** при наличии большого количества данных или признаков, когда стандартные параметры не дают нужной метрики.
> 🧠 **Почему Optuna:** он быстрее и умнее, чем перебор (grid search), т.к. использует байесовскую оптимизацию.

🧩 **Типовые гиперпараметры:**
- `eta` — learning rate: влияет на скорость и точность обучения
- `max_depth` — глубина дерева: влияет на переобучение
- `subsample` — доля примеров для каждого дерева: регулирует шум
- `colsample_bytree` — доля признаков: снижает корреляцию деревьев
- `min_child_weight` — минимальная сумма веса узла

💡 **Как адаптировать:**
- Можно поменять `eval_metric` на `logloss`, `error`, `rmse` для регрессии
- Для multi-class: `objective='multi:softprob', num_class=N`

```python
# Use Optuna to find optimal model parameters
import optuna
import xgboost as xgb
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X_bal, y_bal, test_size=0.2, random_state=42)

def objective(trial):
    # Define hyperparameter search space
    param = {
        'verbosity': 0,
        'objective': 'binary:logistic',
        'tree_method': 'hist',
        'eval_metric': 'auc',
        'eta': trial.suggest_loguniform('eta', 1e-3, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
    }
    dtrain = xgb.DMatrix(X_train, label=y_train)
    cv = xgb.cv(param, dtrain, nfold=3, seed=42, num_boost_round=100, early_stopping_rounds=10)
    return cv['test-auc-mean'].max()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=30)

# Train final model with best parameters
best_params = study.best_params
model = xgb.XGBClassifier(**best_params, use_label_encoder=False)
model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], early_stopping_rounds=10, verbose=False)
```

### 📌 3. TEXT CLASSIFICATION PIPELINES (WITH OPTIONS)

> 🎯 **Цель:** классифицировать текст (например: спам, тональность, категорию).
> 📘 **Когда использовать:** когда признаки — текстовые строки, и их нужно превратить в вектор для подачи в модель.

🧩 **Два популярных подхода:**
- A: TF-IDF (легко, быстро, интерпретируемо)
- B: BERT embedding (глубокое представление смысла текста)

💡 **TF-IDF подойдёт для:** коротких текстов, когда важны ключевые слова
💡 **BERT embeddings подойдёт для:** длинных, сложных фраз; захватывает контекст лучше

🛠 **Альтернативы:** использовать другие модели HuggingFace или fine-tune напрямую (например: DistilBERT + классификатор)
```python
# Approach A: TF-IDF + Logistic Regression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

texts = [...]   # List of raw strings
labels = [...]  # Corresponding labels

tfidf_lr = Pipeline([
    ('tfidf', TfidfVectorizer(lowercase=True, stop_words='english', ngram_range=(1,2), max_df=0.8, min_df=5)),
    ('clf', LogisticRegression(max_iter=1000))
])
tfidf_lr.fit(texts, labels)
```

```python
# Approach B: BERT Embeddings + Random Forest
from transformers import AutoTokenizer, AutoModel
from sklearn.ensemble import RandomForestClassifier
import torch
import numpy as np

# Load pretrained tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
bert = AutoModel.from_pretrained('bert-base-uncased').eval()

def get_cls_emb(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        out = bert(**inputs)
    return out.last_hidden_state[:,0].squeeze().numpy()

X_emb = np.vstack([get_cls_emb(t) for t in texts])
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_emb, labels)
```

### ✅ HUGGINGFACE MODELS USABLE ON KAGGLE WITHOUT AUTH
These models are small/medium and commonly cached on Kaggle:
- 'bert-base-uncased'
- 'distilbert-base-uncased'
- 'roberta-base'
- 'distilroberta-base'
- 'albert-base-v2'
- 'sentence-transformers/all-MiniLM-L6-v2'
- 'google/electra-small-discriminator'
- 'cardiffnlp/twitter-roberta-base-sentiment'

```python
from transformers import AutoTokenizer, AutoModel
model = AutoModel.from_pretrained('distilbert-base-uncased')
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
```

### 📌 4. CROSS-VALIDATION PIPELINE (EXPLAINED)

> 🎯 **Цель:** получить честную метрику качества модели за счёт её тестирования на нескольких частях данных.
> 🧪 **Когда использовать:** почти всегда, особенно при небольшой выборке или для оценки стабильности модели.

🧩 **Почему StratifiedKFold:** он сохраняет пропорции классов в каждом сплите (важно при дисбалансе классов).

💡 **Советы:**
- Используй `GroupKFold`, если есть связанные группы (например, user_id)
- Для time-series: `TimeSeriesSplit`, чтобы избежать утечек будущего
- Метрики можно менять: `roc_auc_score`, `log_loss`, `accuracy_score`
```python
# Use StratifiedKFold to evaluate model performance reliably
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
all_scores = []

for train_idx, val_idx in cv.split(X_bal, y_bal):
    X_tr, X_val = X_bal[train_idx], X_bal[val_idx]
    y_tr, y_val = y_bal[train_idx], y_bal[val_idx]

    model = XGBClassifier(**best_params)
    model.fit(X_tr, y_tr)
    preds = model.predict(X_val)
    score = f1_score(y_val, preds)
    all_scores.append(score)

print("Mean CV F1:", np.mean(all_scores))
```

### 📌 5. COMPUTER VISION (CV) SNIPPETS (FOR CLASSIFICATION)

> 🎯 **Цель:** классификация изображений при помощи сверточных нейросетей.
> 📘 **Когда использовать:** при задачах типа распознавания объектов, классов, признаков на изображении.

🧩 **Пример пайплайна:**
1. Torchvision загружает предобученную модель (ResNet, EfficientNet)
2. ImageFolder — загружает кастомный датасет с метками по папкам
3. Albumentations — делает аугментации, чтобы улучшить обобщающую способность

💡 **Что можно менять:**
- Слой `model.fc` можно адаптировать под своё количество классов
- Аугментации: можно добавить `Blur`, `RandomCrop`, `ShiftScaleRotate`
- Лосс: можно заменить на `nn.BCEWithLogitsLoss()` для multilabel
```python
# Torchvision pretrained model + custom classifier
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch import nn, optim

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_ds = ImageFolder("/path/train", transform=transform)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

model = torchvision.models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, len(train_ds.classes))  # Set to num_classes

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Train loop (simple version)
for epoch in range(5):
    model.train()
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
```

```python
# Albumentations for image augmentation
import albumentations as A
from albumentations.pytorch import ToTensorV2

transform = A.Compose([
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Normalize(),
    ToTensorV2()
])
```

### 🔁 FULL TRAINING PIPELINE TEMPLATE (START TO FINISH)
```python
# 1. Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

# 2. Load dataset
df = pd.read_csv("your_data.csv")
X = df.drop("target", axis=1)
y = df["target"]

# 3. Clean and preprocess data
numeric_cols = X.select_dtypes(include=np.number).columns
cat_cols = X.select_dtypes(include='object').columns

num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

cat_pipe = Pipeline([
    ("encoder", OneHotEncoder(drop='first', sparse=False))
])

X_num = pd.DataFrame(num_pipe.fit_transform(X[numeric_cols]), columns=numeric_cols)
X_cat = pd.DataFrame(cat_pipe.fit_transform(X[cat_cols]), index=X.index)

X_final = pd.concat([X_num, X_cat], axis=1)

# 4. Dimensionality reduction (optional)
#from sklearn.decomposition import PCA
#pca = PCA(n_components=0.95)
#X_final = pca.fit_transform(X_final)

# 5. Balance classes (if needed)
sm = SMOTE(random_state=42)
X_bal, y_bal = sm.fit_resample(X_final, y)

# 6. Split train/validation
X_train, X_valid, y_train, y_valid = train_test_split(X_bal, y_bal, test_size=0.2, stratify=y_bal, random_state=42)

# 7. Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 8. Validate
preds = model.predict(X_valid)
print("F1 Score:", f1_score(y_valid, preds))

# 9. (Optional) Cross-validation
cv = StratifiedKFold(n_splits=5)
scores = []
for train_idx, val_idx in cv.split(X_bal, y_bal):
    model.fit(X_bal[train_idx], y_bal[train_idx])
    preds = model.predict(X_bal[val_idx])
    scores.append(f1_score(y_bal[val_idx], preds))

print("CV F1 avg:", np.mean(scores))
```

# End of Cheatsheet
