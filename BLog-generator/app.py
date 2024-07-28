from langchain_huggingface import HuggingFaceEndpoint

import os


import streamlit as st

from langchain import PromptTemplate, LLMChain

def llmfunc(topic,platform):
#   topic= "football"
# platform="facebook"
    os.environ['HF_TOKEN']=key

    repo_id="mistralai/Mistral-7B-Instruct-v0.2"        
    llm=HuggingFaceEndpoint(repo_id=repo_id,max_length=128,temperature=0.7,token=key)
    template=""" Write a blog on the {topic} to be published on {platform} 
    """
    prompt=PromptTemplate(template=template,input_variables=['topic'])
    model=LLMChain(llm=llm,prompt=prompt)

    result = model.run({"topic": topic,"platform":platform})
    return result



st.set_page_config(page_title="Generate Blogs",
                    page_icon='🤖',
                    layout='centered',
                    initial_sidebar_state='collapsed')

st.header("Generate Blogs 🤖")


## creating to more columns for additonal 2 fields

col1,col2=st.columns([5,5])

with col1:
    topic=st.text_input('Enter blog topic')
with col2:
    platform=st.selectbox('Choose the social media platform',
                            ('Instagram','Facebook','Twitter'),index=0)
    
submit=st.button("Generate")

## Final response
if submit:
    st.write(llmfunc(topic,platform))
