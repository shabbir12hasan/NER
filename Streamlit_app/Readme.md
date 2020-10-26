### This POC is targetting Streamlit library and it's easy interaction with spacy.

Streamlit is a tool designed to ease the life of Data Scientist. It is used to convert data into app in a very quick time, without any front-end coding.

Interactive apps could be created using streamlit which can be hosted over the network and an easy access could be provided to the users.

### To install streamlit library:
` pip install streamlit `

Detailed documentaion and description of this library is clearly provided by the streamlit; https://www.streamlit.io/.

### Spacy Integration:
import spacy_streamlit

This library will provide the visualizing capabilities for spaCy models.
More details on this library are available here: https://spacy.io/universe/project/spacy-streamlit. 

### Code preview:
The sidebar could be designed using sidebar feature.
You just have to define a list of items which needs to be displayed in the menu and pass that inside the function.

` menu = ["Home","NER","Similarity","Spacy_models"] `
` choice = st.sidebar.selectbox("Menu",menu) `

Now, for each menu option, we can define the rules inside the conditional loops. Selecting what needs to be performed.

### Launching the app:
Locally this app could be launched using the following command:
` streamlit run app.py `
Here app.py is the name of the app.
