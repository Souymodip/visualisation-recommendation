import os
import subprocess
import ollama

def run_llama3(query):
    try:
        print(f'Query:\n --------------------------\n')
        response = ollama.chat(model="llama3.1", messages=[{'role': 'user', 'content': query}])
        print(f'Response:\n -------------\n {response['message']['content']} \n ==============================\n')
        return response['message']['content']
    except Exception as e:
        return f"An error occurred: {e}"


def test():
    with open("/Users/priyankachakraborti/GIT/visualisation-recommendation/network_decision.py", "r") as file:
        file_contents = file.read()
    q = f"\"Summarize this file:\n {file_contents} \""

    print(f'Query:\n -------------\n {q} \n -------------\n')
    print(run_llama3(q))


if __name__ == '__main__':
    test()
