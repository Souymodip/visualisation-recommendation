import os
import subprocess
import ollama
from aux import print_yellow, print_blue

def query_print(q):
    print_blue(f'============================================================\n| {chr(0x1F5E3)} ')
    string = f'{q[:100]}... {chr(0x1F971)}{chr(0x1F971)}{chr(0x1F971)}'
    string = '|\t' + '\n|\t'.join(string.split('\n'))
    print_blue(string)
    print_blue(f'============================================================\n| {chr(0x1F575)} ...')

def response_print(r):
    print_yellow('============================================================')
    string = break_text_into_lines(r, 100)
    string = '|\t' + '\n|\t'.join(string.split('\n'))
    print_yellow(string + f' {chr(0x1F575)}')
    print_yellow('============================================================')

def break_text_into_lines(text: str, max_length: int) -> list[str]:
    lines = text.split('\n')
    new_lines = []
    for line in lines:
        if len(line) <= max_length:
            new_lines.append(line)
        else:
            n = len(line) // max_length +1
            for i in range(n):
                new_lines.append(line[i*max_length:(i+1)*max_length]+ ('' if i == n-1 or line[(i+1)*max_length:(i+2)*max_length][0] == ' ' else '-'))
    return '\n'.join(new_lines)

def run_llama3(query):
    try:
        query_print(query)
        response = ollama.chat(model="llama3.1", messages=[{'role': 'user', 'content': query}])
        response_print(response['message']['content'])
        return response['message']['content']
    except Exception as e:
        return f"An error occurred: {e}"
    

def test():
    with open("/Users/priyankachakraborti/GIT/visualisation-recommendation/main.py", "r") as file:
        file_contents = file.read()
    q = f"\"Summarize this file:\n{file_contents} \""
    run_llama3(q)

if __name__ == '__main__':
    test()
