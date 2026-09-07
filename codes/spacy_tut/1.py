import pandas as pd

df = (pd.read_csv("./data/Questions.csv", nrows=1_000_000,encoding="ISO-8859-1", usecols=['Title', 'Id']))
titles = [_ for _ in df ['Title']]

def has_go_lang(text):
    return " go " in text

g = (title for title in titles if has_go_lang(title))

[next(g) for i in range(2)]


import spacy

# python -m spacy download en_core_web_sm

nlp = spacy.load("en_core_web_sm")
a= (nlp("my name is prasad"))

from spacy import displacy

displacy.render(a)

spacy.explain("poss")

for t in nlp("Where does Console.WriteLine go in ASP.NET?"):
    print(t, t.pos_, t.dep_)


titles = [_ for _ in df.loc [lambda d: d['Title'].str. lower().str.contains ("go")]['Title']]


def has_go_lang(doc):
    # doc = nlp(text)
    for t in doc:
        if t.lower_ in ["go", "golang"]:
            if t.pos_ != 'VERB':
                if t.dep_ != 'csubj':
                    return True
    return False

g = (doc for doc in nlp.pipe(titles) if has_go_lang(doc))

[next(g) for i in range(5)]

displacy.render(nlp("Removing all event handlers in one go"))


# 

df_tags = pd. read_csv ("./data/Tags.csv")
go_ids = df_tags.loc[lambda d: d['Tag'] == 'go']['Id']

def has_go_token (doc):
    for t in doc:
        if t.lower_ in ['go', 'golang']:
            return True
    return False

all_go_sentences = df.loc[lambda d: d['Id'].isin(go_ids)]['Title'].tolist()
detectable = [d.text for d in nlp.pipe(all_go_sentences) if has_go_token(d)]

non_detectable =  (df.loc[lambda d: ~d['Id'].isin(go_ids)].loc[lambda d: d['Title'].str.lower().str.contains("go")]['Title'].tolist())

non_detectable = [d.text for d in nlp.pipe(non_detectable) if has_go_token(d)]

len(all_go_sentences), len(detectable), len(non_detectable)



model_name = "en_core_web_sm"
model = spacy.load (model_name, disable=["ner"])

def has_go_token (doc):
    for t in doc:
        if t.lower_ in ["go", "golang"]:
            if t.pos_ != "VERB":
                if t.dep_ == "pobj":
                    return True
    return False

method= "not-verb-but-pobj"
correct = sum(has_go_token(doc) for doc in model.pipe(detectable))
wrong = sum(has_go_token(doc) for doc in model.pipe(non_detectable))
precision =  correct/(correct + wrong)
recall  = correct/len(detectable)
accuracy = (correct + len(non_detectable) -  wrong)/(len (detectable) + len(non_detectable))
f"{precision}, {recall}, {accuracy}, {model_name}, {method}" # this is logged


