# for every json file in the directory verified_results, create a result csv file
import json
import csv
import os

def main(csv_file: str = "data/all_results.csv") -> None:
    # create one csv file containing results from every json file
    columns = ['id', 'mode', 'model', 'quantized', 'dataset', 'temperature', 'finetuning_dataset', 'finetuning_quantized', 'lr', 'bs', 'mbs', 'mi', 'correct_responses', 'functional_programming', 'algorithms', 'fundations', 'abstract_machines', 'memory_management', 'names_and_the_environment', 'describing_a_programming_language', 'object_oriented_paradigm', 'control_structure', 'structuring_data', 'programming_languages', 'num_functional_programming', 'num_algorithms', 'num_fundations', 'num_abstract_machines', 'num_memory_management', 'num_names_and_the_environment', 'num_describing_a_programming_language', 'num_object_oriented_paradigm', 'num_control_structure', 'num_structuring_data', 'num_programming_languages', 'questions_number']
    with open(csv_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(columns)

    modes = ['pretrained', 'finetuned']
    models = ['Llama-2-7b-chat-hf', 'Llama-2-13b-chat-hf', 'Llama-2-70b-chat-hf']
    temperatures = ["0"]
    quantized = ['base', 'quantized']
    f_quantized = ['f_base', 'f_quantized']
    datasets = ['MCQs_PL_all']
    finetuning_datasets=['book_3chapters_dataset', 'book_dataset', 'book_1chapter_dataset']
    LRs = ['0.001', '0.0001']
    BSs = ['16', '32', '64', '128']
    MBSs = ['2']
    MIs = ['100', '500', '1000', '1500', '2000']

    count = 0
    for mode in modes:
        for m in models:
            for q in quantized:
                for ds in datasets:
                    for t in temperatures:
                        if mode == 'pretrained':
                            results_path = "data/verified_results/" + mode + "/temperature" + t + "/" + m + "/" + q + "/" + ds + ".json"
                            print("results_path: ", results_path)
                            # if the json file exists, add a row in the csv file
                            if os.path.exists(results_path):
                                add_row(count, csv_file, results_path, mode, m, q, ds, t)
                                count += 1
                        else:
                            for f_dataset in finetuning_datasets:
                                for f_q in f_quantized:
                                    for lr in LRs:
                                        for bs in BSs:
                                            for mbs in MBSs:
                                                for mi in MIs:
                                                    results_path = "data/verified_results/" + mode + "/temperature" + t + "/" + f_dataset + "/" + f_q + "/" + m + "/" + q + "/" + ds + "/lit_model_lora_finetuned-lr" + lr + "-bs" + bs + "-mbs" + mbs + "-mi" + mi + ".json"
                                                    print("results_path: ", results_path)
                                                    # if the json file exists, add a row in the csv file
                                                    if os.path.exists(results_path):
                                                        add_row(count, csv_file, results_path, mode, m, q, ds, t, f_dataset, f_q, lr, bs, mbs, mi)
                                                        count += 1

def add_row(count, csv_file, json_file, mode, model, quantized, dataset, t, f_dataset=None, f_q=None, lr=None, bs=None, mbs=None, mi=None):
    # if the json file exists, add a row in the csv file
    if os.path.exists(json_file):
        with open(json_file, 'r') as f:
            data = json.load(f, strict=False)
            # get last element of array in json file
            last = data[-1]
            # get correct responses from last element that is in form of:
            """ {
                "correct_responses": "3",
                "questions_number": "25"
            } """
            correct_responses = last['correct_responses']
            functional_programming = last['functional_programming']
            algorithms = last['algorithms']
            fundations = last['fundations']
            abstract_machines = last['abstract_machines']
            memory_management = last['memory_management']
            names_and_the_environment = last['names_and_the_environment']
            describing_a_programming_language = last['describing_a_programming_language']
            object_oriented_paradigm = last['object_oriented_paradigm']
            control_structure = last['control_structure']
            structuring_data = last['structuring_data']
            programming_languages = last['programming_languages']
            questions_number = last['questions_number']
            num_functional_programming = last['num_functional_programming']
            num_algorithms = last['num_algorithms']
            num_fundations = last['num_fundations']
            num_abstract_machines = last['num_abstract_machines']
            num_memory_management = last['num_memory_management']
            num_names_and_the_environment = last['num_names_and_the_environment']
            num_describing_a_programming_language = last['num_describing_a_programming_language']
            num_object_oriented_paradigm = last['num_object_oriented_paradigm']
            num_control_structure = last['num_control_structure']
            num_structuring_data = last['num_structuring_data']
            num_programming_languages = last['num_programming_languages']
            # add a line in the csv file with the correct responses string
            with open(csv_file, 'a') as f:
                writer = csv.writer(f)
                writer.writerow([count, mode, model, quantized, dataset, t, f_dataset, f_q, lr, bs, mbs, mi, correct_responses, functional_programming, algorithms, fundations, abstract_machines, memory_management, names_and_the_environment, describing_a_programming_language, object_oriented_paradigm, control_structure, structuring_data, programming_languages, num_functional_programming, num_algorithms, num_fundations, num_abstract_machines, num_memory_management, num_names_and_the_environment, num_describing_a_programming_language, num_object_oriented_paradigm, num_control_structure, num_structuring_data, num_programming_languages, questions_number])

if __name__ == "__main__":
    main()