'''
Create a json file from book.text with the following format:

[    
    {
        "instruction": "Write a limerick about a    
                        pelican.”,
        "input": "",
        "output": "There once was a pelican so fine,
                   \nHis beak was as colorful as 
                   sunshine,\nHe would fish all day,\nIn 
                   a very unique way,\nThis pelican was 
                   truly divine!\n\n\n"
    },
    {
        "instruction": "Identify the odd one out from 
                        the group.",
        "input": "Carrot, Apple, Banana, Grape",
        "output": "Carrot\n\n"
    },
]

'''

import json
import re

# in every instruction field of the json file add a paragraph of the text file that starts with '##'
def create_dataset(filename: str = 'data/book/book'):
    with open(filename + '.md', 'r') as f:
        texts = f.read()
    texts = texts.split('## ') # split text file into paragraphs
    texts = [t.strip() for t in texts] # remove leading and trailing whitespaces
    texts = [t for t in texts if t != ''] # remove empty paragraphs
    texts = [t for t in texts if t != '\n'] # remove empty paragraphs

    # save text title for each text that is the following sentence after ##
    titles = []
    text = []
    for t in texts:
        # title is the sentence after ## and before \n
        title = t.split('\n')[0].strip()
        title = re.sub(r'#{1,2}|\{.*\}', '', title)
        title = title.strip()
        titles.append('Explain ' + title)
        # remove title from text and remove leading and trailing whitespaces and # {*}
        t = t.replace(title, '').strip()
        t = re.sub(r'#{1,2}|\{.*\}', '', t)
        t = t.replace('\n\n', '', 1).strip()
        # remove empty lines
        t = '\n'.join([line for line in t.split('\n') if line != ''])
        # remove [] from text without removing the text inside
        t = re.sub(r'\[.*\]', '', t)
        t = t.replace(':::', '')
        text.append(t)

    # create json file
    data = []
    for i in range(len(text)):
        if i % 2 == 0:
            # add instruction as title[i] and output as text[i]
            data.append({'instruction': titles[i], 'input': '', 'output': text[i]})
        else:
            data[-1]['output'] = text[i]
    with open(filename + '.json', 'w') as f:
        json.dump(data, f, indent=4)

import json
if __name__ == '__main__':
    from jsonargparse import CLI
    CLI(create_dataset)
