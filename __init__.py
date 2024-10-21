import datasets
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import classification_report
from rich import inspect
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
import logging
from rich.logging import RichHandler
from rich.traceback import install
import click

# Logging
FORMAT = "%(message)s"
logging.basicConfig(
    level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler(rich_tracebacks=True, tracebacks_suppress=[click])]
)
log = logging.getLogger("rich")
install(show_locals=True)

# upload dataset
# can choose any dataset from here: https://huggingface.co/datasets
download_link = 'fancyzhx/yelp_polarity'
dataset = datasets.load_dataset(download_link)


# дата для тестирования
train_ds = dataset['train']
train_df = train_ds.to_pandas()


# vectorize
seed = 8
testing_set = train_df.sample(350_000, random_state=seed)
value_counts = testing_set['label'].value_counts()


vectorized = TfidfVectorizer()
vectorized_data = vectorized.fit_transform(testing_set['text'])
transformed_data = vectorized.transform(testing_set['text'])

# classifier
classifier = LogisticRegression()
X = vectorized_data
y_true = testing_set['label']
classifier.fit(X, y_true)

y_pred = classifier.predict(transformed_data)

print(classification_report(y_true=y_true, y_pred=y_pred, digits=4))
