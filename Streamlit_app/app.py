# Core Pkgs
import streamlit as st 
# NLP Pkgs
import spacy_streamlit
import spacy
import os
from PIL import Image

def main():
	"""
	Spacy-Streamlit App for Tokenizing and NER.
	"""
	nlp = spacy.load('en')

	# Creating a title for the app
	st.title("Spacy-Streamlit App")

	# Creating a menu bar
	menu = ["Home","NER","Similarity","Spacy_models"]
	choice = st.sidebar.selectbox("Menu",menu)

	# Conditional loops to perform the output of menu selection
	if choice == "Home":
		st.subheader("Tokenization")
		raw_text = st.text_area("Your Text","Enter Text Here") # to dispaly a input text box
		docx = nlp(raw_text)
		if st.button("Tokenize"):
			spacy_streamlit.visualize_tokens(docx,attrs=['text','pos_','dep_','ent_type_'])

	elif choice == "NER":
		st.subheader("Named Entity Recognition")
		raw_text = st.text_area("Your Text","Enter Text Here") # to dispaly a input text box
		docx = nlp(raw_text)
		if st.button("Get NER"):
			# using spacy's NER model
			spacy_streamlit.visualize_ner(docx,labels=nlp.get_pipe('ner').labels)

	elif choice=="Spacy_models":
		st.subheader("Spacy models")
		models = ["en_core_web_sm", "en_core_web_md"]
		default_text = "Sundar Pichai is the CEO of Google."
		spacy_streamlit.visualize(models, default_text)

        
if __name__ == '__main__':
	main()