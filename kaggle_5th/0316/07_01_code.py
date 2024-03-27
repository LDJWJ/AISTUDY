# 데이터 불러오기
import pandas as pd
train = pd.read_csv("/kaggle/input/the-5th-kaggle-competition-challenge-with-kakr/train.csv")
test = pd.read_csv("/kaggle/input/the-5th-kaggle-competition-challenge-with-kakr/test.csv")

# 데이터 크기
train.shape, test.shape

# train 데이터 샘플
train.head()

# 타겟 비율 확인 (0.5기준)
(train['toxicity'] >= 0.5).astype(int).value_counts()

# comment_text만 살펴보기
for text in train['comment_text'][:7]:
    print(text)
    print("\n")  # 텍스트 사이에 공백 줄 추가

# test데이터 샘플 확인
test.head()

from sklearn.model_selection import train_test_split

# 타겟 인코딩(0 또는 1)
y = (train['toxicity'] >= 0.5).astype(int)

# 검증데이터 분리
X_train, X_test, y_train, y_test = train_test_split(train['comment_text'], y, test_size=0.2, random_state=0)
X_train.shape, X_test.shape, y_train.shape, y_test.shape


# TF-IDF 벡터라이저를 사용하여 텍스트 데이터 벡터화
from sklearn.feature_extraction.text import TfidfVectorizer

# use_idf 파라미터를 False로 설정하여 TfidfVectorizer 초기화
vectorizer_tf_only = TfidfVectorizer(max_features=10000, 
                                     ngram_range=(1, 2),
                                     use_idf=False)

# 훈련 데이터에 fit_transform을 적용하여 TF-IDF 벡터로 변환
X_train_tfidf = vectorizer_tf_only.fit_transform(X_train)

# 테스트 데이터에 transform을 적용하여 TF-IDF 벡터로 변환
X_test_tfidf = vectorizer_tf_only.transform(X_test)


# 변환된 훈련 및 테스트 데이터의 크기 확인
print("Transformed train data shape:", X_train_tfidf.shape)
print("Transformed test data shape:", X_test_tfidf.shape)

## 모델 학습 및 평가
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# 로지스틱 회귀
model = LogisticRegression(random_state=0)
model.fit(X_train_tfidf, y_train)

# 테스트 데이터에 대한 예측 수행
y_pred = model.predict_proba(X_test_tfidf)[:, 1]

# ROC AUC 점수 계산
roc_auc = roc_auc_score(y_test, y_pred)
roc_auc

## Test 데이터 예측
# TF-IDF 
test_tfidf = vectorizer_tf_only.transform(test['comment_text'])

# 예측
pred = model.predict_proba(test_tfidf)

# 제출용 데이터프레임
sample_submissions = pd.read_csv("/kaggle/input/the-5th-kaggle-competition-challenge-with-kakr/sample_submissions.csv")
sample_submissions['prediction'] = pred[:, 1]
sample_submissions

# submission 파일 생성 -> 오른쪽에 있는 Submit버튼 클릭
sample_submissions.to_csv("submission.csv", index=False)