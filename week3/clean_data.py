import re
import pprint
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="./data/sentences_800k_subtract_drill_1.txt", help="Input file")
    parser.add_argument("--output_file", type=str, default="./data/cleaned_data.txt", help="Output file")
    args = parser.parse_args()
    # pp = pprint.PrettyPrinter(indent=4)
    samples = []
    with open(args.input_file, "r", encoding="utf8") as f:
        for line in f.readlines():
            line = re.sub("\\n", " ", line)
            line = re.sub("\d+\.\d+", " ", line)
            line = re.sub("\w+\d+\w+", " ", line)
            line = re.sub("\d", "", line)
            line = re.sub(":", ".", line)
            line = re.sub("mm+", " ", line)
            line = re.sub("cm", " ", line)
            line = re.sub("(\w+\/)+\w+", " " ,line)
            line = re.sub("[-<>\(\)\+\/#@%!&\*·]", " ", line)
            line = re.sub("xx+", " ", line)
            line = re.sub("(x\sx)+", " ", line)
            line = re.sub("(x\s,)+", " ", line)
            line = re.sub("^\.", "", line)
            line = re.sub("^\s+", "", line)
            line = re.sub("^\s+\.+", "", line)
            line = re.sub("\.+", ".", line)
            line = re.sub(",\.", ".", line)
            line = re.sub(",+", ",", line)
            line = re.sub(",\s+,", ",", line)
                        
            line = re.sub("\s\s+", " ", line)
            line = re.sub("\s+\.+", ".", line)  
            line = re.sub(",+", ",", line) 
            line = re.sub("\.+", ".", line)
            line = re.sub("(x\s,)+", " ", line)

            samples.append(line.lower())
        
    # pprint.pprint(samples)
    with open(args.output_file, "w", encoding="utf8") as f:
        f.write("\n".join(samples))