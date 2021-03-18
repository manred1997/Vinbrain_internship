import re
import pprint
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="./english.txt", help="Input file")
    parser.add_argument("--output_file", type=str, default="./english_clean.txt", help="Output file")
    args = parser.parse_args()
    # pp = pprint.PrettyPrinter(indent=4)
    samples = []
    with open(args.input_file, "r", encoding="utf8") as f:
        for line in f.readlines():
            try:
                line = line.split("=")[1]
                line = re.sub("\\n", "", line)
                line = re.sub("[\(\)]", "", line)
                line = re.sub("[^\w\s]", "", line)
                # line = re.sub("\{\d\}", "x", line)
                if not line: continue
                else: samples.append(line)
            except: continue
    pprint.pprint(samples)
    with open(args.output_file, "w", encoding="utf8") as f:
        f.write("\n".join(samples))