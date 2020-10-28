import random
import spacy
from tqdm import tqdm


# training data
product_train_data = [('what is the price of polo?', {'entities': [(21, 25, 'PrdName')]}), ('what is the price of ball?', {'entities': [(21, 25, 'PrdName')]}), ('what is the price of jegging?', {'entities': [(21, 28, 'PrdName')]}), ('what is the price of t-shirt?', {'entities': [(21, 28, 'PrdName')]}), ('what is the price of jeans?', {'entities': [(21, 26, 'PrdName')]}), ('what is the price of bat?', {'entities': [(21, 24, 'PrdName')]}), ('what is the price of shirt?', {'entities': [(21, 26, 'PrdName')]}), ('what is the price of bag?', {'entities': [(21, 24, 'PrdName')]}), ('what is the price of cup?', {'entities': [(21, 24, 'PrdName')]}), ('what is the price of jug?', {'entities': [(21, 24, 'PrdName')]}), ('what is the price of plate?', {'entities': [(21, 26, 'PrdName')]}), ('what is the price of glass?', {'entities': [(21, 26, 'PrdName')]}), ('what is the price of moniter?', {'entities': [(21, 28, 'PrdName')]}), ('what is the price of desktop?', {'entities': [(21, 28, 'PrdName')]}), ('what is the price of bottle?', {'entities': [(21, 27, 'PrdName')]}), ('what is the price of mouse?', {'entities': [(21, 26, 'PrdName')]}), ('what is the price of keyboad?', {'entities': [(21, 28, 'PrdName')]}), ('what is the price of chair?', {'entities': [(21, 26, 'PrdName')]}), ('what is the price of table?', {'entities': [(21, 26, 'PrdName')]}), ('what is the price of watch?', {'entities': [(21, 26, 'PrdName')]})]
print("Data initialization completed")

def ner_model_training(data, n_iter):
    # loading blank model
    nlp = spacy.blank('en')  
    print("Created blank 'en' model")

    # As there is nothing is the model, pipline needs to be created
    ner = nlp.create_pipe('ner')
    nlp.add_pipe(ner, last=True)

    # training the model
    for _, record in product_train_data:
        for ent in record.get('entities'):
            ner.add_label(ent[2])

    # diabling other pipes while training NER
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']

    print("NER training started")
    losses_out={}
    with nlp.disable_pipes(*other_pipes):  # only train NER
        optimizer = nlp.begin_training()
        for itn in range(n_iter):
            random.shuffle(product_train_data)
            losses = {}
            for text, annotations in tqdm(product_train_data):
                nlp.update(
                    [text],  
                    [annotations],  
                    drop=0.5, # dropout - make it harder to memorise data
                    sgd=optimizer,  # callable to update weights
                    losses=losses)
            losses_out[itn]=losses
    print("training completed")
    return nlp, losses_out


########################################
##### Train and save the model #########
########################################
nlp, losses_out = ner_model_training(product_train_data, n_iter=10) # train

modelfile = 'bin/ner_model'
nlp.to_disk(modelfile) # save

########################################
####### Loading the trained Model ######
########################################
try:
    nlp = spacy.load('bin/ner_model')
except:
    print("Error: model not found")

########################################
############ Testing model #############
########################################
test_text = "what is the price of keyboad?"
doc = nlp(test_text)

for ent in doc.ents:
    print("Output***  \n Product found:{prd} \n starting:{start} \n ending:{end} \n Label:{label}".format(
        prd=ent.text
        , start=ent.start_char
        , end=ent.end_char
        , label=ent.label_
        )
    )
########################################