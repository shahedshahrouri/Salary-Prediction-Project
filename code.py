import pandas as pd
import requests
from pyquery import PyQuery as pq
import threading, json, multiprocessing

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression

df = pd.read_csv("ds_salaries.csv")
print(df.columns)

possible_salary_cols = [
    "salary_in_usd","salary","salary_usd","Salary","Salary_in_USD",
    "unconverted_salary","salary_in_local_currency"
]

salary_col = next((c for c in possible_salary_cols if c in df.columns), None)
if salary_col is None:
    raise ValueError("No salary column found.")

df.drop_duplicates(inplace=True)
df[salary_col] = df[salary_col].fillna(df[salary_col].mean())
df["salary_vs_average"] = df[salary_col] - df[salary_col].mean()

numeric_cols = [c for c in ["remote_ratio","work_year"] if c in df.columns]
categorical_cols = [c for c in ["experience_level","employment_type","company_location","company_size"] if c in df.columns]

X = df[numeric_cols + categorical_cols]
y = df[salary_col]

num_pipe = Pipeline([("impute", SimpleImputer(strategy="mean")), ("scale", StandardScaler())])
cat_pipe = Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore"))])

preprocess = ColumnTransformer([
    ("num", num_pipe, numeric_cols),
    ("cat", cat_pipe, categorical_cols)
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Pipeline([("prep", preprocess), ("reg", LinearRegression())])
model.fit(X_train, y_train)
print(model.score(X_test, y_test))

sample_cols = [c for c in ["job_title","company_location",salary_col] if c in df.columns]
sample = df[sample_cols].head(10).to_dict(orient="records")

with open("salaries_sample.json", "w", encoding="utf-8") as f:
    json.dump(sample, f, indent=4)

with open("salaries_sample.json", "r", encoding="utf-8") as f:
    restored = json.load(f)

urls = [
    "https://www.exchangerate-api.com/",
    "https://www.xe.com/currencyconverter/",
    "https://www.investing.com/currencies/usd-jod"
]

scraped = []
def fetch(url):
    try:
        r = requests.get(url, timeout=10)
        scraped.append({"url": url, "title": pq(r.text)("title").text()})
    except:
        pass

threads = [threading.Thread(target=fetch, args=(u,)) for u in urls]
for t in threads: t.start()
for t in threads: t.join()

with open("scraped_titles.json", "w", encoding="utf-8") as f:
    json.dump(scraped, f, indent=4)

def avg(): print(df[salary_col].mean())
def mx(): print(df[salary_col].max())
def mn(): print(df[salary_col].min())

if __name__ == "__main__":
    for f in [avg, mx, mn]:
        p = multiprocessing.Process(target=f)
        p.start()
        p.join()
