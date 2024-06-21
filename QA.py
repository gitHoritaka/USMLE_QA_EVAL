import json
from exp import QA_output,const_llama_model,hint_output_with_rag
from pre import setup_local_rag,read_config


def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = []
        for line in file:
            try:
                json_data = json.loads(line)
                data.append(process_data(json_data))
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
                continue
    return data

def process_data(entry):
    data = {}
    try:
        question = entry.get('question', 'N/A')
        answer = entry.get('answer', 'N/A')
        options = entry.get('options', {})
        meta_info = entry.get('meta_info', 'N/A')
        answer_idx = entry.get('answer_idx', 'N/A')
        metamap_phrases = entry.get('metamap_phrases', [])

        data = {
        "Question": question,
        "Answer": answer,
        "Options": options,
        "Meta Info": meta_info,
        "Answer Index": answer_idx,
        "MetaMap Phrases": metamap_phrases
        }


    except AttributeError as e:
        print(f"error :{e}")
    return data

def qa_eval(qa_output_func,model,question,options,hints,answer_index,show = False):
    if show:
        print(question,options)
    ans_json = qa_output_func(model,question,options,hints)
    flag= answer_index == ans_json["answer"]
    if show:
        print(flag,ans_json)
    return flag

# メイン関数
if __name__ == "__main__":
    print("loading json file")
    config = read_config()
    json_file_path = config["DEFAULT"] ["JSON_FILE_PATH"]
    data = load_json(json_file_path)
    print("finished loading")
    score = 0
    model = const_llama_model()
    retriever = setup_local_rag(config)
    for id,entry in enumerate(data):
        question = entry["Question"]
        options = entry["Options"]
        hint = hint_output_with_rag(retriever,options) 
        score += qa_eval(QA_output,model,question,options,hint,entry["Answer Index"],True)
        print("-------------------------------------------------------------------")
        print(f"socre updated current id = {id}")
        print(f"new score = {score / (id+1)}")

    print(f"pretrained llm score :{score/len(data)}")
        
    
