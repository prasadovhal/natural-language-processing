import spacy
import pandas as pd
from spacy import displacy

nlp = spacy.load('en_core_web_sm')

def has_go_token(doc):
    for t in doc:
        if t.lower_ in ['go','golang', 'python', 'ruby', 'objective-c']:
            if t.pos_ != 'VERB':
                return True
    return False

doc = nlp("i like to program in python")
has_go_token(doc)

doc = nlp("i am an iOS dev and I like to code in objective-c as well as go/golang")
has_go_token(doc)

pattern_c = [{'LOWER': 'objective'}, {'IS_PUNCT': True}, {'LOWER': 'c'}]
golang_pattern1 = [{'LOWER': 'golang'}]
golang_pattern2 = [{'LOWER': 'go', 'POS': {'NOT_IN': ['VERB']}}]

from spacy.matcher import Matcher

matcher = Matcher(nlp.vocab)
matcher.add("OBJ_C",[pattern_c])
matcher.add("GOLANG_LANG1",[golang_pattern1])
matcher.add("GOLANG_LANG2",[golang_pattern2])


for match_id, start, end in matcher(doc):
    print(doc[start: end])
    
[(t, t.pos_) for t in doc]

#################################################

obj_c_pattern1 = [{'LOWER': 'objective'},{'IS_PUNCT': True, 'OP': '?'},{'LOWER': 'c'}]
obj_c_pattern2 = [{'LOWER': 'objectivec'}]
golang_pattern1 = [{'LOWER': 'golang'}]
golang_pattern2 = [{'LOWER': 'go','POS': {'NOT_IN': ['VERB']}}]
python_pattern = [{'LOWER': 'python'}]
ruby_pattern = [{'LOWER': 'ruby'}]
js_pattern = [{'LOWER': {'IN': ['js', 'javascript']}}]

matcher = Matcher (nlp. vocab, validate=True)
matcher.add("OBJ_C_LANG1", [obj_c_pattern1])
matcher.add("OBJ_C_LANG2", [obj_c_pattern2])
matcher.add("PYTHON LANG", [python_pattern])
matcher.add("GO_LANG1", [golang_pattern1])
matcher.add("GO_LANG2", [golang_pattern2])
matcher.add("JS_LANG", [js_pattern])
matcher.add("RUBY_LANG", [ruby_pattern])


df = (pd.read_csv("./data/Questions.csv", nrows=1_000_000,encoding="ISO-8859-1", usecols=['Title', 'Id']))
titles = (_ for _ in df ['Title'] if "python" in _.lower())

for i in range(100):
    doc=nlp(next(titles))
    if len(matcher(doc)) == 0:
        print(doc)

