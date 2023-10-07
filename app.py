import streamlit as st
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
import torch

#@st.cache(allow_output_mutation=True)
#@st.cache_data()

#def get_model():
#    tokenizer = BertTokenizer.from_pretrained("Recognai/bert-base-spanish-wwm-cased-xnli")
#    model = BertForSequenceClassification.from_pretrained("alechain/bert-finetunned-no-time")
#    return tokenizer,model
@st.cache_resource()
def get_model():
    tokenizer = BertTokenizer.from_pretrained("Recognai/bert-base-spanish-wwm-cased-xnli")
    model = BertForSequenceClassification.from_pretrained("alechain/bert-finetunned-no-time")
    return tokenizer, model

tokenizer, model = get_model()


# Interfaz de usuario
st.title("Predicción de artículos de Página 12 (Validacion Temporal)")
st.markdown(
    "### Predicción de la sección de Página 12 a la que pertenece un artículo mediante un modelo BERT fine-tunned considerando el Horizonte Temporal en el entrenamiento."
)
st.markdown(
 ''' Elegir una noticia de alguna de las siguientes secciones y copiar el texto del artículo:
 - https://www.pagina12.com.ar/secciones/el-pais
 - https://www.pagina12.com.ar/secciones/economia
 - https://www.pagina12.com.ar/secciones/sociedad
 - https://www.pagina12.com.ar/secciones/el-mundo '''
)

user_input = st.text_area('Entrar texto del artículo a analizar')



button = st.button("Predecir")

label_texts=['Economía', 'El Mundo', 'Sociedad', 'El País']
MY_DATASET_MAX_TOKENS=512

if user_input and button :
    encoded_dict = tokenizer.encode_plus(
                    user_input,                      # Frase a codificar.
                    add_special_tokens = True, # Agregar '[CLS]' y '[SEP]'
                    max_length = MY_DATASET_MAX_TOKENS,  # Llenar con el token PAD a frases cortas, o truncar frases mas  largas que MAX_TOKENS
                    padding='max_length',
                    truncation=True,
                    return_token_type_ids=True,
                    return_attention_mask = True,   # Construir las attn. masks.
                    return_tensors = 'pt',     # retornar pytorch tensors.
                )
    # test_sample
    output = model(**encoded_dict)
   # st.write("Logits: ",output.logits)
    # Crear una tabla para mostrar los logits y sus etiquetas
    logits_table = {'Sección': label_texts, 'Logit': output.logits.squeeze().tolist()}
    st.table(logits_table)

    y_pred = label_texts[output.logits.argmax(dim=1).item()]
  #  st.write("Clase Predicha (más probable): ",y_pred)
      # Dar formato al texto de la predicción
    st.markdown(f"<p style='color: #FF0000; font-size: 22px;'>Clase Predicha (más probable): {y_pred}</p>", unsafe_allow_html=True)

    


