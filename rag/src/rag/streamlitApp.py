

#%%
import requests
import streamlit as st
import time
import os
import dotenv
dotenv.load_dotenv()


URI_API = os.getenv("URI_API")

st.title("chat sobre conceitos de Machine Learning")


#%%
query = st.text_area("Pergunte ao chat sobre conceitos de machine learning:",height=150)


def transformar_em_stream(texto):
    for palavra in texto.split(" "):
        yield palavra + " "
        time.sleep(0.10)



if st.button("Processar Texto"):
  if query.strip() == "":
    st.warning("Por favor, digite algum texto antes de enviar.")
  else:
    url = URI_API  # Substitua pela sua URL

    # Define que estamos enviando e aceitando texto puro
    headers = {"Content-Type": "text/plain", "Accept": "text/plain"}

    try:
      # 2. Envia o texto bruto usando o parâmetro 'data'
      # .encode('utf-8') garante que acentos e caracteres especiais não quebrem
      
      resp = requests.post(url,
      json={"query": query},)
      
      print( resp.json().get("response"))
        
      if resp.status_code == 200:
        st.success("Texto processado com sucesso!")

        # 3. Captura e exibe o texto de retorno
        texto_saida = resp.json().get("response")

        st.subheader("Resposta")
        st.write_stream(transformar_em_stream(texto_saida))

      else:
        st.error(f"Erro na API ({resp.status_code}): {resp.text}")

    except requests.exceptions.RequestException as e:
      st.error(f"Falha ao conectar com a API: {e}")

