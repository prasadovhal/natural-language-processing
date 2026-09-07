import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from seaborn import color_palette
import os
import json
from urllib.parse import urlparse
import re
import plotly.express as px
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import TensorDataset, DataLoader,Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR
from sklearn import model_selection, metrics, preprocessing
from catboost import CatBoostClassifier, Pool
import transformers
from transformers import AdamW, get_linear_schedule_with_warmup, DebertaTokenizer, BertTokenizer,BertModel

plt.style.use("ggplot")
col_pal = color_palette()
pd.set_option('display.max_columns', 50)


train = pd.read_csv(f"./train.tsv", sep = "\t")
test = pd.read_csv(f"./test.tsv", sep = "\t")

train.loc[train.loc[:,"alchemy_category_score"]=="?","alchemy_category_score"] = np.nan
train["alchemy_category_score"] = train["alchemy_category_score"].astype("float64")
test.loc[test.loc[:,"alchemy_category_score"]=="?","alchemy_category_score"] = np.nan
test["alchemy_category_score"] = test["alchemy_category_score"].astype("float64")

train.loc[train.loc[:,"is_news"]=="?","is_news"] = np.nan
train["is_news"] = train["is_news"].astype("float64")
test.loc[test.loc[:,"is_news"]=="?","is_news"] = np.nan
test["is_news"] = test["is_news"].astype("float64")


# extract website
train.loc[:,"website"] = train.loc[:,"url"].apply(urlparse).apply(lambda x: x[1].replace('www.', '').replace('.com', ''))
series = pd.value_counts(train.website)
mask = (series/series.sum() * 100).lt(0.4)
train['website'] = np.where(train['website'].isin(series[mask].index),'rare',train['website'])


test.loc[:,"website"] = test.loc[:,"url"].apply(urlparse).apply(lambda x: x[1].replace('www.', '').replace('.com', ''))
series = pd.value_counts(test.website)
mask = (series/series.sum() * 100).lt(0.4)
test['website'] = np.where(test['website'].isin(series[mask].index),'rare',test['website'])


# extract year

train.loc[:,"year"] = train.loc[:,"url"]\
                                    .apply(urlparse)\
                                    .apply(lambda x:"/".join(x[2:]))\
                                    .apply(lambda x:re.findall(r"\d{4}", x))\
                                    .apply(lambda x: int(x[0]) if len(x) > 0 else np.nan)\
                                    .apply(lambda x: x if x>1990 and x<2050 else np.nan )
                                    
train["YearPresent"] = train["year"].isnull().apply(lambda x: 0 if x == True else 1)

print("Only {:.2f} % of training data have year information".format((train.shape[0] - train.year.isnull().sum()) / train.shape[0]* 100))

train["YearPresent"].map({0:"year_absent",1:"year_present"}).value_counts().plot(kind = "bar",
                                    figsize = (7,5),
                                    color = col_pal[1],
                                    fontsize = 15,
                                    rot = 60,
                                    title = "Count of records have year values")


test.loc[:,"year"] = test.loc[:,"url"]\
                                    .apply(urlparse)\
                                    .apply(lambda x:"/".join(x[2:]))\
                                    .apply(lambda x:re.findall(r"\d{4}", x))\
                                    .apply(lambda x: int(x[0]) if len(x) > 0 else np.nan)\
                                    .apply(lambda x: x if x>1990 and x<2050 else np.nan )
test["YearPresent"] = test["year"].isnull().apply(lambda x: 0 if x == True else 1)
print("Only {:.2f} % of training data have year information".format((test.shape[0] - test.year.isnull().sum()) / test.shape[0]* 100))


test["YearPresent"].map({0:"year_absent",1:"year_present"}).value_counts().plot(kind = "bar",
                                    figsize = (7,5),
                                    color = col_pal[1],
                                    fontsize = 15,
                                    rot = 60,
                                    title = "Count of records have year values")

# Extract Website_type

yearList = list(train["year"].unique())

map_dic = {}
for year in yearList:
    if pd.isnull(year) != True:
        map_dic[str(int(year))] = "Year"
        
train.loc[:,"website_type"] = train.loc[:,"url"]\
                                    .apply(urlparse)\
                                    .apply(lambda x:x[2].split("/")[1])\
                                    .apply(lambda x: map_dic[x] if x in map_dic.keys() else x)
                                    
series = pd.value_counts(train.website_type)
mask = (series/series.sum() * 100).lt(0.4)
train['website_type'] = np.where(train['website_type'].isin(series[mask].index),'rare',train['website_type'])
train["website_type"].value_counts().to_frame().style.background_gradient(axis=0)  


yearList = list(test["year"].unique())
map_dic = {}
for year in yearList:
    if pd.isnull(year) != True:
        map_dic[str(int(year))] = "Year"
test.loc[:,"website_type"] = test.loc[:,"url"]\
                                    .apply(urlparse)\
                                    .apply(lambda x:x[2].split("/")[1])\
                                    .apply(lambda x: map_dic[x] if x in map_dic.keys() else x)
series = pd.value_counts(test.website_type)
mask = (series/series.sum() * 100).lt(0.4)
# To replace df['column'] use np.where I.e 
test['website_type'] = np.where(test['website_type'].isin(series[mask].index),'rare',test['website_type'])
test["website_type"].value_counts().to_frame().style.background_gradient(axis=0)  


#Extract Website_sub_type

yearList = list(train["year"].unique())
map_dic = {}
for year in yearList:
    if pd.isnull(year) != True:
        map_dic[str(int(year))] = "Year"

train.loc[:,"website_sub_type"] = train.loc[:,"url"]\
                                    .apply(urlparse)\
                                    .apply(lambda x: x[2].split("/")[2] if ((len(x) > 2) and len(x[2].split("/"))>2) else "?")\
                                    .apply(lambda x: map_dic[x] if x in map_dic.keys() else x)
                                    
series = pd.value_counts(train.website_sub_type)
mask = (series/series.sum() * 100).lt(0.4)
# To replace df['column'] use np.where I.e 
train['website_sub_type'] = np.where(train['website_sub_type'].isin(series[mask].index),'rare',train['website_sub_type'])
train["website_sub_type"].value_counts().to_frame().style.background_gradient(axis=0)  


yearList = list(test["year"].unique())
map_dic = {}
for year in yearList:
    if pd.isnull(year) != True:
        map_dic[str(int(year))] = "Year"

test.loc[:,"website_sub_type"] = test.loc[:,"url"]\
                                    .apply(urlparse)\
                                    .apply(lambda x: x[2].split("/")[2] if ((len(x) > 2) and len(x[2].split("/"))>2) else "?")\
                                    .apply(lambda x: map_dic[x] if x in map_dic.keys() else x)
series = pd.value_counts(test.website_sub_type)
mask = (series/series.sum() * 100).lt(0.4)
# To replace df['column'] use np.where I.e 
test['website_sub_type'] = np.where(test['website_sub_type'].isin(series[mask].index),'rare',test['website_sub_type'])
test["website_sub_type"].value_counts().to_frame().style.background_gradient(axis=0)  


# Extract Domain

train.loc[:,"domain"] = train.loc[:,"url"]\
                                    .apply(urlparse)\
                                    .apply(lambda x: x[1].split(".")[-1])

series = pd.value_counts(train.domain)
mask = (series/series.sum() * 100).lt(0.4)
# To replace df['column'] use np.where I.e 
train['domain'] = np.where(train['domain'].isin(series[mask].index),'rare',train['domain'])
train["domain"].value_counts().to_frame().style.background_gradient(axis=0)  


test.loc[:,"domain"] = test.loc[:,"url"]\
                                    .apply(urlparse)\
                                    .apply(lambda x: x[1].split(".")[-1])

series = pd.value_counts(test.domain)
mask = (series/series.sum() * 100).lt(0.4)
# To replace df['column'] use np.where I.e 
test['domain'] = np.where(test['domain'].isin(series[mask].index),'rare',test['domain'])
test["domain"].value_counts().to_frame().style.background_gradient(axis=0)  


# boilerplate

def extract_title_and_body(data):
    boilerplatedf = data["boilerplate"].apply(json.loads)
    boilerplatedf = pd.DataFrame(boilerplatedf.tolist())
    data["boilerplate_title"] = boilerplatedf["title"].copy()
    data["boilerplate_body"] = boilerplatedf["body"].copy()
    data["boilerplate_title_length"] = data["boilerplate_title"].apply(lambda x: len(str(x)))
    data["boilerplate_body_length"] = data["boilerplate_body"].apply(lambda x: len(str(x)))
    data["boilerplate_body_to_title_length_ratio"] = data["boilerplate_body_length"]  / (data["boilerplate_title_length"] + 10)
    data["boilerplate_title"] = data["boilerplate_title"].fillna('')
    data["boilerplate_body"] = data["boilerplate_body"].fillna('')
    data = data.drop(columns = ["boilerplate"])
    del boilerplatedf
    return data

train = extract_title_and_body(train)
test = extract_title_and_body(test)

sub_train = train.copy()
sub_train["label"] = sub_train["label"].map({0:"Ephemeral", 1:"Evergreen"})
sub_train.groupby("label")["boilerplate_title_length"]\
    .mean()\
    .plot(kind = "barh",
        figsize = (7,5),
        color = col_pal[1],
        fontsize = 15,
        title = "Average Boilerplate title Length of Evergreen vs Non Evergreen Labels")
del sub_train

sub_train = train.copy()
sub_train["label"] = sub_train["label"].map({0:"Ephemeral", 1:"Evergreen"})
sub_train.groupby("label")["boilerplate_body_length"]\
    .mean()\
    .plot(kind = "barh",
        figsize = (7,5),
        color = col_pal[1],
        fontsize = 15,
        title = "Average Boilerplate Body Length of Evergreen vs Non Evergreen Labels")
del sub_train


import torch
from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import pipeline

task='sentiment'
MODEL = f"cardiffnlp/twitter-roberta-base-{task}"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

from transformers import pipeline

sentiment_classifier = pipeline(task= "sentiment-analysis", 
                                model=model, 
                                tokenizer=tokenizer)

train.loc[:,"boilerplate_title_sentiment"]  = train.loc[:,"boilerplate_title"].apply(sentiment_classifier)\
                                                        .apply(lambda x: x[0]["label"])\
                                                        .map({"LABEL_0":"Negative",
                                                        "LABEL_1":"Neutral",
                                                        "LABEL_2":"Positive"}).values
                                                    
test.loc[:,"boilerplate_title_sentiment"]  = test.loc[:,"boilerplate_title"].apply(sentiment_classifier)\
                                                        .apply(lambda x:x[0]["label"])\
                                                        .map({"LABEL_0":"Negative",
                                                        "LABEL_1":"Neutral",
                                                        "LABEL_2":"Positive"}).values
                                                        

train.loc[:,"boilerplate_title_sentiment_score"]  = train.loc[:,"boilerplate_title"].apply(sentiment_classifier)\
                                                        .apply(lambda x:x[0]["score"])\

train.loc[train["boilerplate_title_sentiment"] == "Neutral","boilerplate_title_sentiment_score"] = 0
train.loc[train["boilerplate_title_sentiment"] == "Negative","boilerplate_title_sentiment_score"] = -1 * train.loc[train["boilerplate_title_sentiment"] == "Negative","boilerplate_title_sentiment_score"]

print(train.query("boilerplate_title_sentiment == 'Neutral'").boilerplate_title_sentiment_score.min())
print(train.query("boilerplate_title_sentiment == 'Negative'").boilerplate_title_sentiment_score.max())
print(train.query("boilerplate_title_sentiment == 'Positive'").boilerplate_title_sentiment_score.min())

sub_train = train.copy()
sub_train["label"] = sub_train["label"].map({0:"Ephemeral", 1:"Evergreen"})
fig, ax = plt.subplots(1,1, figsize = (10,7))
sub_train.groupby("label")["boilerplate_title_sentiment"]\
        .value_counts()\
        .unstack()\
        .plot(kind = "bar",
            stacked=True,
            ax = ax,
            fontsize = 15,
            title = "Distribution of Boilerplate Title Sentiment of Evergreen vs Non Evergreen Labels",
            )
ax.legend(bbox_to_anchor=(1.15, 1.0))
del sub_train

# alchemy_category

fig, ax = plt.subplots(2,1, figsize = (15,15))

train.alchemy_category.value_counts(ascending = True)\
                        .plot(kind = "bar",
                                color = col_pal[1],
                                ax = ax[0],
                                rot =60,)
ax[0].set_title("Alchemy Category CountPlot", fontsize=25)
ax[0].set_xlabel('alchemy_category', fontsize=25)
ax[0].set_ylabel('Count', fontsize=25)
ax[0].tick_params(axis='both', which='major', labelsize=15)


train.groupby("alchemy_category")["label"]\
    .mean()\
    .sort_values()\
    .plot(kind = "bar",
        color = col_pal[1],
        ax= ax[1],
        rot =60,
        title = "Alchemy Category VS Evergreen rate")
ax[1].set_title("Alchemy Category VS Evergreen rate", fontsize=25)
ax[1].set_xlabel('alchemy_category', fontsize=25)
ax[1].set_ylabel('Evergreen rate', fontsize=25)
ax[1].tick_params(axis='both', which='major', labelsize=15)

plt.suptitle("Alchemy Category Count Plot and Comparison with Evergreen rate",fontsize=30 )
plt.tight_layout()
plt.show()

# What categories are the least common but have a high evergreen chances¶

_temp = train.groupby(["alchemy_category","label"])["urlid"]\
                .count()\
                .unstack()\
                .fillna(0)\
                .reset_index()
                
_temp["total_count"] = _temp[0] + _temp[1]
_temp["Ephemeral_percent"] = _temp[0] / _temp["total_count"]
_temp["Evergreen_percent"] = _temp[1] / _temp["total_count"]
px.scatter(x = "Evergreen_percent",
        y = "total_count",
        data_frame = _temp,
        hover_data = ["alchemy_category"])

# alchemy_category_score

temp = train.copy()
temp.label = temp.label.map({0:"Ephemeral" , 1:"Evergreen"})
temp.groupby("label")["alchemy_category_score"]\
        .mean()\
        .plot(kind = "bar",
        figsize = (7,5),
        color = col_pal[1],
        fontsize = 15,
        rot = 60,
        title = "Count of records have year values")
        
        
columns_To_Aggregate = ["label",
                        "boilerplate_title_sentiment_score",
                        "is_news",
                        "spelling_errors_ratio",
                        "boilerplate_title_length",
                        "boilerplate_body_length",
                        "boilerplate_body_to_title_length_ratio"]

train.groupby("year")[columns_To_Aggregate].mean()

# 4. What websites are the least common but have a high evergreen chances¶

_temp = train.query("website != 'rare'") \
                .groupby(["website","label"])["urlid"]\
                .count()\
                .unstack()\
                .fillna(0)\
                .reset_index()
_temp["total_count"] = _temp[0] + _temp[1]
_temp["Ephemeral_percent"] = _temp[0] / _temp["total_count"]
_temp["Evergreen_percent"] = _temp[1] / _temp["total_count"]
px.scatter(x = "Evergreen_percent",
           y = "total_count",
           data_frame = _temp,
           hover_data = ["website"])


# 5. line plots over years comparing boilerplate_title_sentiment_score and spelling_errors_ratio with evergreen rate

cols = ["boilerplate_title_sentiment_score",
        "spelling_errors_ratio",
]

for col in cols:
    fig , ax = plt.subplots(1,1, figsize = (15,7))
    train.rename(columns = {"label":"IsEvergreen"})\
                                .groupby("year")[["IsEvergreen",col]]\
                                .mean()\
                                .plot(ax = ax)
    ax.set_title(f" {col} and Evergreen rate distribution over years",fontdict={'fontsize': 20})
    ax.legend(bbox_to_anchor=(1, 1.0), prop={'size': 15})
    ax.set_xlabel("Year", fontdict={'fontsize': 20})
    ax.set_ylabel("Rate",fontdict={'fontsize': 20})
    ax.xaxis.set_tick_params(labelsize=15)
    ax.yaxis.set_tick_params(labelsize=15)
    plt.show()
    
# For all the alchemy_category what is their boilerplate_title_sentiment_score distribution over years

_temp = train.query("alchemy_category != '?'")\
                            .groupby(["year","alchemy_category"])[["boilerplate_title_sentiment_score"]]\
                            .mean()\
                            .unstack()
_temp.columns = [col[1] for col in _temp.columns.values]
px.scatter(_temp)

# 7. Does the urls with year information have more/less/neutral chance making a webpage evergreen¶

fig, ax = plt.subplots(1,1, figsize = (7,5))
train.groupby("YearPresent")["label"]\
    .mean()\
    .sort_values()\
    .plot(kind = "bar",
        color = col_pal[1],
        ax= ax,
        rot =60,
        title = "YearPresent VS Evergreen rate")
ax.set_title("YearPresent VS Evergreen rate", fontsize=20)
ax.set_xlabel('YearPresent', fontsize=15)
ax.set_ylabel('Evergreen rate', fontsize=15)
ax.tick_params(axis='both', which='major', labelsize=10)

plt.tight_layout()
plt.show()



# catboost model

columnsToDrop = ["url",
                "urlid",
                "year"]
train = train.drop(columns = columnsToDrop)
test = test.drop(columns = columnsToDrop)

train["boilerplate_title"] = train["boilerplate_title"].fillna("")
train["boilerplate_body"] = train["boilerplate_body"].fillna("")
test["boilerplate_title"] = test["boilerplate_title"].fillna("")
test["boilerplate_body"] = test["boilerplate_body"].fillna("")

trainCB = train.drop(columns = ["boilerplate_title","boilerplate_body"])
testCB = test.drop(columns = ["boilerplate_title","boilerplate_body"])

trainCB["isTrain"] = True
testCB["isTrain"] = False
combined = pd.concat([trainCB,testCB], axis = 0)

cat_features = [ 'alchemy_category',
                'hasDomainLink',
                'is_news',
                'lengthyLinkDomain',
                'news_front_page',
                'website',
                'website_type',
                'website_sub_type',
                'domain',
                'YearPresent',
                'boilerplate_title_sentiment',
                ] 
# cat_features.extend(cat_feats_new)

numerical_features = [ 
                        'boilerplate_title_length',
                        'boilerplate_body_length',
                        'numberOfLinks',
                        'linkwordscore',
                        'non_markup_alphanum_characters',
                        'numwords_in_url',
                        'alchemy_category_score',
                        'avglinksize',
                        'commonlinkratio_1',
                        'commonlinkratio_2',
                        'commonlinkratio_3',
                        'commonlinkratio_4',
                        'compression_ratio',
                        'embed_ratio',
                        'frameTagRatio',
                        'html_ratio',
                        'image_ratio',
                        'parametrizedLinkRatio',
                        'spelling_errors_ratio',
                        'boilerplate_body_to_title_length_ratio',
                        'boilerplate_title_sentiment_score'] 
# numerical_features.extend(num_feats_new)


for feat in cat_features:
    enc = preprocessing.LabelEncoder()
    temp_col = combined[feat].fillna("NONE").astype(str).values
    combined.loc[:, feat] = enc.fit_transform(temp_col)

combined = combined.fillna(0)
xtrain = combined.loc[combined["isTrain"] == True,numerical_features +cat_features + ["label"] ]
xtest = combined.loc[combined["isTrain"] == False,numerical_features +cat_features ]

print(xtrain.shape , xtest.shape)

num_folds = 8
kf = model_selection.StratifiedKFold(n_splits=num_folds)

# fill the new kfold column
for f, (t_, v_) in enumerate(kf.split(X=xtrain, y=xtrain["label"])):
    xtrain.loc[v_, 'kfold'] = f

val_auc = 0
y_test_pred = 0
for fold in xtrain.kfold.unique():

        x_train = xtrain[xtrain.kfold != fold].reset_index(drop=True)
        x_valid = xtrain[xtrain.kfold == fold].reset_index(drop=True)
        # drop the label column from dataframe and convert it to
        # a numpy array by using .values.
        # target is label column in the dataframe
        features = x_train.columns
        y_train = x_train["label"].values
        x_train =x_train.drop(columns = ["label","kfold"], axis=1).values
        # similarly, for validation, we have
        y_valid = x_valid["label"].values
        x_valid = x_valid.drop(columns = ["label","kfold"], axis=1).values

        _train = Pool(x_train, label=y_train)
        _valid = Pool(x_valid, label=y_valid)

        clf = CatBoostClassifier()
        clf.fit(
                _train,
                eval_set=_valid,
                use_best_model=True,
                verbose=200,
                #plot=True
        )
            
        # create predictions for validation samples
        preds = clf.predict_proba(x_valid)[:,1]
        # calculate & print accuracy
        auc = metrics.roc_auc_score(y_valid, preds)
        val_auc += auc
        print(f"OOF={fold}, AUC={auc}")
        
        y_test_pred = y_test_pred +  clf.predict_proba(xtest)[:,1]

print(f"Mean Val AUC={val_auc / num_folds}")


cb_pred =  y_test_pred / num_folds
# ss["label"] = cb_pred


# BERT


train["boilerplate_title"] = train["boilerplate_title"].fillna("")
train["boilerplate_body"] = train["boilerplate_body"].fillna("")
test["boilerplate_title"] = test["boilerplate_title"].fillna("")
test["boilerplate_body"] = test["boilerplate_body"].fillna("")
train["text"] = train["boilerplate_title"] +". " + train["boilerplate_body"]
test["text"] = test["boilerplate_title"] +". " + test["boilerplate_body"]

train["text"] = train["text"].str.lower()
test["text"] = test["text"].str.lower()

# numerical_features.extend(num_feats_new)
text_features = ["text"]

xtrain =   train[text_features + ["label"]]
xtest = test[text_features]


class BERTDataset:
    def __init__(self, text, target):
        self.text = text
        self.target = target
        self.tokenizer = config["TOKENIZER"]
        self.max_len = config["MAX_LEN"]

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        text = str(self.text[item])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
#             pad_to_max_length=True,
            truncation=True,
            padding='max_length'
        )
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        if("token_type_ids" in inputs.keys()):
            token_type_ids = inputs["token_type_ids"]
            return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "targets": torch.tensor(self.target[item], dtype=torch.float),
        }
        else:
            return {
                "ids": torch.tensor(ids, dtype=torch.long),
                "mask": torch.tensor(mask, dtype=torch.long),
                "targets": torch.tensor(self.target[item], dtype=torch.float),
            }

class BERT_Test_Dataset:
    def __init__(self, text):
        self.text = text
        self.tokenizer = config["TOKENIZER"]
        self.max_len = config["MAX_LEN"]

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        text = str(self.text[item])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
#             pad_to_max_length=True,
            truncation=True,
            padding='max_length'
        )
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        if("token_type_ids" in inputs.keys()):
            token_type_ids = inputs["token_type_ids"]
            return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
        }
        else:
            return {
                "ids": torch.tensor(ids, dtype=torch.long),
                "mask": torch.tensor(mask, dtype=torch.long),
            }
            
from transformers import BertModel

class TextClassification(nn.Module):

    def __init__(self):
        super(TextClassification, self).__init__()
        self.bert = transformers.BertModel.from_pretrained(pretrained_model_name_or_path =config["model_name"],
                                                                                        return_dict=False,
                                                                                        )
        self.bert_drop = nn.Dropout(0.3)
        self.out = nn.Linear(768, 1)

    def forward(self, ids, mask, token_type_ids):
        _, o2 = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)
        bo = self.bert_drop(o2)
        output = self.out(bo)
        return output

import torch
import torch.nn as nn
from tqdm.notebook import tqdm


def loss_fn(outputs, targets):
    return nn.BCEWithLogitsLoss()(outputs, targets.view(-1, 1))


def train_fn(data_loader, model, optimizer, device, scheduler, epoch,fold):
    model.train()
    loss_train_total = 0
    
    progress_bar = tqdm(enumerate(data_loader), 
                        total=len(data_loader),
                        desc='OOF {:1d} Epoch {:1d}'.format(int(fold) , epoch), 
                        leave=False, 
                        disable=False)

    for bi, d in  progress_bar:
        ids = d["ids"]
        mask = d["mask"]
        targets = d["targets"]

        ids = ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.float)

        optimizer.zero_grad()
        if("token_type_ids" in d.keys()):
            token_type_ids = d["token_type_ids"].to(device, dtype=torch.long)
            outputs = model(ids=ids, mask=mask, token_type_ids=token_type_ids)
        else:
            outputs = model(ids=ids, mask=mask)

        loss = loss_fn(outputs, targets)
        loss_train_total +=loss.item()
        loss.backward()
        optimizer.step()
        scheduler.step()
        progress_bar.set_postfix({'training_loss': '{:.5f}'.format(loss.item()/len(targets))})
    loss_train_avg = loss_train_total/len(data_loader)
    return loss_train_avg 


def eval_fn(data_loader, model, device):
    model.eval()
    fin_targets = []
    fin_outputs = []
    loss_total = 0
    with torch.no_grad():
        for bi, d in tqdm(enumerate(data_loader), total=len(data_loader)):
            ids = d["ids"]
            mask = d["mask"]
            targets = d["targets"]

            ids = ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            targets = targets.to(device, dtype=torch.float)
            if("token_type_ids" in d.keys()):
                token_type_ids = d["token_type_ids"].to(device, dtype=torch.long)
                outputs = model(ids=ids, mask=mask, token_type_ids=token_type_ids)
            else:
                outputs = model(ids=ids, mask=mask)
            
                loss_total = loss_total + loss_fn(outputs, targets).item()
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
            loss_total = loss_total / len(data_loader)
    return loss_total,fin_outputs, fin_targets

def eval_test(data_loader, model, device):
    model.eval()
    fin_outputs = []
    progress_bar = tqdm(enumerate(data_loader), 
                        total=len(data_loader),
                        desc='Generating Test Output'.format(epoch), 
                        leave=False, 
                        disable=False)
    with torch.no_grad():
        for bi, d in progress_bar:
            ids = d["ids"]
            mask = d["mask"]
            ids = ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            if("token_type_ids" in d.keys()):
                token_type_ids = d["token_type_ids"].to(device, dtype=torch.long)
                outputs = model(ids=ids, mask=mask, token_type_ids=token_type_ids)
            else:
                outputs = model(ids=ids, mask=mask)
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
    return fin_outputs


config = { 
    "model_name" :"bert-base-uncased",
    "TOKENIZER"  :BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True,),
    "MAX_LEN" : 512,
    "TRAIN_BATCH_SIZE" : 8,
    "EPOCHS" : 2,
    "DEVICE" : "cuda",
    "MODEL_PATH":"model.pth"
}


from sklearn.model_selection import train_test_split

def get_data_loaders(x_train, x_valid):
    # x_train , x_valid = train_test_split(train, test_size=0.1,random_state=2020)
    train_dataset = BERTDataset(text=x_train.text.values, target=x_train.label.values)
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size = config["TRAIN_BATCH_SIZE"],shuffle=True)
    valid_dataset = BERTDataset(text=x_valid.text.values, target=x_valid.label.values)
    valid_loader = torch.utils.data.DataLoader(valid_dataset,batch_size = config["TRAIN_BATCH_SIZE"],shuffle=True)
    return train_loader , valid_loader

def get_test_data_loaders(x_test):
    # x_train , x_valid = train_test_split(train, test_size=0.1,random_state=2020)
    test_dataset = BERT_Test_Dataset(text=x_test.text.values)
    test_loader = torch.utils.data.DataLoader(test_dataset,batch_size = config["TRAIN_BATCH_SIZE"],shuffle=False)
    return test_loader 


device = torch.device(config["DEVICE"])
num_folds = 8
kf = model_selection.StratifiedKFold(n_splits=num_folds)

# fill the new kfold column
for f, (t_, v_) in enumerate(kf.split(X=xtrain, y=xtrain["label"])):
    xtrain.loc[v_, 'kfold'] = f

val_auc = 0
y_test_pred = []
for fold in xtrain.kfold.unique():

    model = TextClassification()
    model = nn.DataParallel(model)
    model.to(device)
    x_train = xtrain.loc[xtrain.kfold != fold,:].reset_index(drop=True)
    x_valid = xtrain.loc[xtrain.kfold == fold,:].reset_index(drop=True)

    train_data_loader , valid_data_loader = get_data_loaders(x_train,x_valid)

    num_train_steps = int(len(x_train) / config["TRAIN_BATCH_SIZE"] * config["EPOCHS"])
    optimizer = AdamW(model.parameters(), lr=2e-5)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
    )

    for epoch in tqdm(range(config["EPOCHS"])):

        loss_train_avg = train_fn(train_data_loader, model, optimizer, device, scheduler,epoch,fold)
        tqdm.write('\nEpoch {epoch}')
        tqdm.write(f'Training loss: {loss_train_avg}')
        val_loss, outputs, targets = eval_fn(valid_data_loader, model, device)
        auc = metrics.roc_auc_score(targets, outputs)
        tqdm.write(f'Validation Loss: {val_loss}')
        tqdm.write(f'AUC : {auc}')        

    val_auc = val_auc + auc
    test_data_loader = get_test_data_loaders(xtest)
    outputs = eval_test(test_data_loader, model, device)
    y_test_pred.append(outputs)
    tqdm.write(f"OOF -- {fold} ROC AUC Score = {auc}") 

val_auc = val_auc / num_folds
print(f"Total Val AUC -- {val_auc}")


# Logistic regression

from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn.linear_model as lm
from sklearn import model_selection

print("loading data..")
traindata = list(xtrain.text.values)
testdata = list(xtest.text.values)
y = xtrain.label.values

tfv = TfidfVectorizer(min_df=3,  
                    max_features=None, 
                    strip_accents='unicode',  
                    analyzer='word',
                    token_pattern=r'\w{1,}',
                    ngram_range=(1, 2), 
                    use_idf=1,
                    smooth_idf=1,
                    sublinear_tf=1)

rd = lm.LogisticRegression(solver = 'liblinear', 
                           penalty='l2', 
                           dual=True, 
                           tol=0.0001, 
                           C=1, 
                           fit_intercept=True, 
                           intercept_scaling=1.0, 
                           class_weight=None, 
                           random_state=None)

X_all = traindata + testdata
lentrain = len(traindata)

tfv.fit(traindata)
X_all = tfv.transform(X_all)

X = X_all[:lentrain]
X_test = X_all[lentrain:]

print("20 Fold CV Score: ", np.mean(model_selection.cross_val_score(rd, X, y, cv=20, scoring='roc_auc')))

rd.fit(X,y)
lr_pred = rd.predict_proba(X_test)[:,1]
