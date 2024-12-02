# from dataset.json file split objects that have the \output\ field with a sequence length > 250
# into multiple objects with a sequence length of 249 or less
# and save the new objects into a new json file
import json
import os

def main(filename: str = 'data/book/book'):
    # load json file
    with open(filename + '.json') as f:
        data = json.load(f)

        # create new json file
        new_data = []

        # iterate through objects in json file
        for obj in data:
            # if the object has a number of tokens > 250
            if len(obj['output'].split()) > 250:
                # split the object after .
                objects = obj['output'].split(".")
                # add . to each object
                for i in range(len(objects)):
                    objects[i] += "."
                
                # create a new objects list with the join of the objects with a sequence length of max 249
                new_objects = []
                new_object = ""
                for object in objects:
                    if len(new_object.split()) + len(object.split()) < 250:
                        new_object += object
                    else:
                        new_objects.append(new_object)
                        new_object = object
                
                # create new objects with the same fields as the original object
                for i in range(len(new_objects)):
                    new_obj = {}
                    new_obj['input'] = ""
                    new_obj['output'] = new_objects[i]
                    new_obj['instruction'] = obj['instruction'] + " - Part " + str(i+1)
                    new_data.append(new_obj)

            # if the object has a sequence length <= 250
            else:
                new_data.append(obj)

        # save new json file
        with open(filename + '_dataset.json', 'w') as outfile:
            json.dump(new_data, outfile)

import json
if __name__ == '__main__':
    from jsonargparse import CLI
    CLI(main)