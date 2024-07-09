import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers


def getresponse(input_text,no_words,blog_style):
    llm=CTransformers(model='llama-2-7b-chat.ggmlv3.q8_0.bin',model_type='llama',config= {'max_new_tokens':256,'temperature':0.01})

    template="""
    write a blog for {blog_style} within a total number of {no_words} words for the topic {input_text}.
    """

    prompt=PromptTemplate(input_variables=['blog_style','no_words','input_text'],template=template)
    response=llm(prompt.format(blog_style=blog_style,no_words=no_words,input_text=input_text))
    print(response)

st.set_page_config(page_title="generate blogs",layout="centered",initial_sidebar_state='collapsed')
st.header("Generate blogs using AI")
input_text=st.text_input("Enter blog topic")
col1,col2=st.columns([5,5])
with col1:
    no_words=st.text_input("Enter number of words")
with col2:
    blog_style=st.selectbox("Blog for who?",('Researchers','Scientists','Common People'),index=0)

submit=st.button('Generate')
if submit:
    st.write(getresponse(input_text,no_words,blog_style))