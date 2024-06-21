import outlines
import outlines.generate
import os
from groq import Groq
import outlines.generate.text
from outlines import models
from pydantic import BaseModel
from dataclasses import dataclass,asdict
from outlines.samplers import greedy
import json
from typing import List
from enum import Enum
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain.retrievers.tavily_search_api import TavilySearchAPIRetriever
from pre import setup_local_rag,read_config
import requests
config = read_config()

TAVILY_API_KEY = config["DEFAULT"]["TAVILY_API_KEY"]
GROQ_API_KEY = config["DEFAULT"]["GROQ_API_KEY"]
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY
#search_retriever = TavilySearchAPIRetriever(k=3)
def search(arg,search_retriever):
    try:
        res = search_retriever.invoke(arg)
        return res[0].page_content
    except requests.exceptions.HTTPError as e:
        return "No search result Finish agent"
def retrieve(arg,rag_retriever):
    docs = rag_retriever.invoke(arg)
    res = ""
    mx_iter = 3
    for d in docs:
        if mx_iter >0:
            res += d.page_content
            mx_iter-=1

    return res 

def create_QAprompt(question,options,hint):

    # optionsをJSON文字列に変換
    options_str = json.dumps(options, ensure_ascii=False)

    prompt = f"""
    You are an LLM (Language Learning Model) preparing for the national medical licensing examination. 
    You will be given a question and multiple options. Please select the correct answer from the choices [A, B, C, D].

    ---

    Your task: 
    0. Read Hints(note this hints may be not relevant)
    1. Analyze the provided question.
    2. Evaluate each option.
    3. you can use search engine such as Google.
    4. output search keywords.
    5. Choose the correct answer from [A, B, C, D].
    6. output answer and reason why you choose.

    ---
    Hints :{hint}
    Question: {question}
    Options: {options_str}
    """
    return prompt
def generate_hint_prompt(question,options):

    # optionsをJSON文字列に変換
    options_str = json.dumps(options, ensure_ascii=False)

    prompt = f"""
    You are an LLM (Language Learning Model) preparing for the national medical licensing examination. 
    You will be given a question and multiple options. Please output search keywords to solve quesitons

    ---

    Your task: 
    0. Read Hints(note this hints may be not relevant)
    1. Analyze the provided question.
    2. Evaluate each option.
    3. you can use search engine such as Google.
    4. output search keywords

    ---
    Question: {question}
    Options: {options_str}
    """
    return prompt
class Label(str, Enum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"

def label_output(answer):
    if answer == Label.A:
        return "A"
    elif answer == Label.B:
        return "B"
    elif answer == Label.C:
        return "C"
    elif answer == Label.D:
        return "D"

@dataclass
class AnswerFormat(BaseModel):
    reason : str #why you choose the answer
    #web_search_keywords:List[str]
    answer :Label

@dataclass
class HintsFormat(BaseModel):
    web_search_keywords:List[str]

def hint_output(model,question,options):
    prompt = generate_hint_prompt(question,options)
    client = Groq(
        api_key=os.environ.get("GROQ_API_KEY"),
    )

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="llama3-70b-8192",
    )
    output = chat_completion.choices[0].message.content


    generator = outlines.generate.json(model, HintsFormat, sampler=greedy())
    search_words = generator(output)
    # dataclassを辞書に変換
    search_dict = asdict(search_words)
    hint = search(search_dict["web_search_keywords"][0])

    return hint

def hint_output_with_rag(retriever,options):
    # optionsをJSON文字列に変換
    options_str = json.dumps(options, ensure_ascii=False)
    hint = retrieve(options_str,retriever)
    return hint


def QA_output(model,question,options,hints):
    prompt = create_QAprompt(question,options,hints)

    client = Groq(
        api_key=os.environ.get("GROQ_API_KEY"),
    )

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="llama3-70b-8192",
    )

    output = chat_completion.choices[0].message.content


    generator = outlines.generate.json(model, AnswerFormat, sampler=greedy())
    answer = generator(output)
    # dataclassを辞書に変換
    answer_dict = asdict(answer)
    answer_dict["output"] = output
    answer_dict["hint"] = hints
    answer_dict["answer"] = label_output(answer_dict["answer"])

    return answer_dict 

def const_phi_model():
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct", trust_remote_code=True,)
    llm = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct", trust_remote_code=True)
    model = models.Transformers(llm, tokenizer)
    return model

def const_llama_model():
    return models.transformers("meta-llama/Meta-Llama-3-8B")

def test():
    question = """A 19-year-old man is brought to the physician by his mother because she is worried about his strange behavior. Over the past 3 years, he has been dressing all in black and wears a long black coat, even in the summer. The mother reports that her son has always had difficulties fitting in. He does not have any friends and spends most of his time in his room playing online games. He is anxious at college because he is uncomfortable around other people, and his academic performance is poor. Rather than doing his coursework, he spends most of his time reading up on paranormal phenomena, especially demons. He says that he has never seen any demons, but sometimes there are signs of their presence. For example, a breeze in an enclosed room is likely the “breath of a demon”. Mental status examination shows laborious speech. The patient avoids eye contact. Which of the following is the most likely diagnosis?
    """
    options = {'A': 'Social anxiety disorder', 'B': 'Avoidant personality disorder', 'C': 'Schizotypal personality disorder', 'D': 'Schizophrenia'}
    hints = """Schizophrenia is a serious mental health condition that affects how people think, feel and behave. It may result in a mix of hallucinations, delusions, and disorganized thinking and behavior. Hallucinations involve seeing things or hearing voices that aren't observed by others. Delusions involve firm beliefs about things that are not true. People with schizophrenia can seem to lose touch with reality, which can make daily living very hard.
    People with schizophrenia need lifelong treatment. This includes medicine, talk therapy and help in learning how to manage daily life activities."""
    model = const_phi_model()
    ans_dict =QA_output(model,question,options,hints)
    print(ans_dict["answer"])

def test2():
    question = """A 19-year-old man is brought to the physician by his mother because she is worried about his strange behavior. Over the past 3 years, he has been dressing all in black and wears a long black coat, even in the summer. The mother reports that her son has always had difficulties fitting in. He does not have any friends and spends most of his time in his room playing online games. He is anxious at college because he is uncomfortable around other people, and his academic performance is poor. Rather than doing his coursework, he spends most of his time reading up on paranormal phenomena, especially demons. He says that he has never seen any demons, but sometimes there are signs of their presence. For example, a breeze in an enclosed room is likely the “breath of a demon”. Mental status examination shows laborious speech. The patient avoids eye contact. Which of the following is the most likely diagnosis?
    """
    options = {'A': 'Social anxiety disorder', 'B': 'Avoidant personality disorder', 'C': 'Schizotypal personality disorder', 'D': 'Schizophrenia'}
    hints = """Schizophrenia is a serious mental health condition that affects how people think, feel and behave. It may result in a mix of hallucinations, delusions, and disorganized thinking and behavior. Hallucinations involve seeing things or hearing voices that aren't observed by others. Delusions involve firm beliefs about things that are not true. People with schizophrenia can seem to lose touch with reality, which can make daily living very hard.
    People with schizophrenia need lifelong treatment. This includes medicine, talk therapy and help in learning how to manage daily life activities."""
    model = const_phi_model()
    search_dict =hint_output(model,question,options)
    print(search_dict)

def rag_test():
    retriever = setup_local_rag()
    question = """A 19-year-old man is brought to the physician by his mother because she is worried about his strange behavior. Over the past 3 years, he has been dressing all in black and wears a long black coat, even in the summer. The mother reports that her son has always had difficulties fitting in. He does not have any friends and spends most of his time in his room playing online games. He is anxious at college because he is uncomfortable around other people, and his academic performance is poor. Rather than doing his coursework, he spends most of his time reading up on paranormal phenomena, especially demons. He says that he has never seen any demons, but sometimes there are signs of their presence. For example, a breeze in an enclosed room is likely the “breath of a demon”. Mental status examination shows laborious speech. The patient avoids eye contact. Which of the following is the most likely diagnosis?
    """
    options = {'A': 'Social anxiety disorder', 'B': 'Avoidant personality disorder', 'C': 'Schizotypal personality disorder', 'D': 'Schizophrenia'}
    hint = hint_output_with_rag(retriever,options)
    print("rag_text")
    print(hint)

if __name__ == "__main__":
    #test()
    #test2()
    rag_test()