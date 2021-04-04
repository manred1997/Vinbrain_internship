import json
import re

from multiprocessing import cpu_count, Pool

def chunk(list_text, chunk_len):
    for i in range(0, len(list_text), chunk_len):
        yield list_text[i: i + chunk_len]

def get_data(acronym, expansion, line):
    if re.search(expansion, line):    
        start_char_idx = list(re.finditer(expansion, line))[0].span()[0]
        len_acronym = len(acronym)
        line = re.sub(expansion, acronym, line)
        line = line.split("\n")[0]
        if len(line) > 10:
        # print(line)
            return {
                "text": line,
                "start_char_idx": start_char_idx,
                "length_acronym": len_acronym,
                "expansion": expansion
            }
    

def build_dataset(data_info):
    tmp = []
    for line in data_info['lines']:
        sample = get_data(data_info['acronym'], data_info['expansion'], line)
        if not sample: continue
        else: tmp.append(sample)
    return tmp
    

if __name__ == "__main__":
    with open("./final_long_dict.json", "r", encoding="UTF-8") as f:
        diction = json.load(f)

    print(len(diction))
    with open("./cxr.txt", "r", encoding="UTF-8") as f:
        data = []
        for line in f.readlines():
            list_line = line.split("\n")[0]
            data.append(list_line)
    
    # print(len(data))

    procs = 4#cpu_count()
    # print(procs)

    
    num_line_per_proc = len(data) // procs
    # print(num_line_per_proc)
    for idx, (acronym, expansions) in enumerate(diction.items()):
        if 97<idx <102:
            dataset = []
            print(f"=================={acronym}==================")
            for expansion in expansions:
                print(f"=================={expansion}==================")
                chunked_lists = list(chunk(data, num_line_per_proc))
                data_info = []
                for acr_exp, chunked_list in zip([(acronym, expansion)]*num_line_per_proc, chunked_lists):

                    chunk_info = {
                        'lines': chunked_list,
                        'acronym': acr_exp[0],
                        'expansion': acr_exp[1]
                    }

                    data_info.append(chunk_info)
                    
                pool = Pool(processes=procs)
                dataset.extend(pool.map(build_dataset, data_info))

            with open(f"./data_cxr_ad/train_{acronym}.json", "w", encoding="UTF-8") as f:
                json.dump(dataset, f)
            del dataset
        if acronym == "mvl": break
    print("Done create dataset")