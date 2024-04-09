import spacy
from spacy.matcher import Matcher

nlp = spacy.load('en_core_web_sm')
doc = nlp('2018 FIFA World Cup : France Won!!! ')

pattern = [{'IS_DIGIT':True},{'LOWER':'fifa'}, {'LOWER':'world' },{'LOWER':'cup'}]
matcher2 = Matcher(nlp.vocab)
matcher2.add('fifa_pattern',[pattern])
matcher2 = matcher2(doc)

for match_id,start,end in matcher2:
    matched_span = doc[start:end]
    print(matched_span)