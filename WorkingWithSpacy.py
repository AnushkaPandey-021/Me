import spacy


nlp = spacy.load('en_core_web_sm')
doc = nlp('Tony gave three $ to Sam, Mari gave 600 ₹ to Ana')

print("Tokens in specific index", doc[3])

print ("Tokens between indices", doc[3:6])

print("Perform POS :\n")
for token in doc:
    print(token.i, "|", token.text, "|", token.pos, "|", token.pos_)

print("Entities in Doc: \n") 
for ent in doc.ents:
    print(ent.text, "|", ent.label,"|" ,ent.label_)
  
print("All Entities: \n")  
print(nlp.get_pipe('ner').labels)