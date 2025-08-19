import argparse

split = lambda x : x.split(",")
get_ultimate = lambda x : float(split(x)[-1])
get_penultimate = lambda x : float(split(x)[-2])
get_average = lambda x : (get_ultimate(x) + get_penultimate(x))/2

def main(args):
    with open(args.file, "r") as file:
        lines = file.readlines()
        line_idxs = range(len(lines))
        if "cat" == args.type:
            best = max(line_idxs, key=lambda idx : get_ultimate(lines[idx]))
            print(f"V: E={best} : {get_ultimate(lines[best])}")
        elif "va" == args.type:
            best_v = max(line_idxs, key=lambda idx : get_penultimate(lines[idx]))
            best_a = max(line_idxs, key=lambda idx : get_ultimate(lines[idx]))
            best_avg = max(line_idxs, key=lambda idx : get_average(lines[idx]))
            print(f"V: E={best_v} : {get_penultimate(lines[best_v])}")
            print(f"A: E={best_a} : {get_ultimate(lines[best_a])}")
            print(f"AVG: E={best_avg} : V={get_penultimate(lines[best_avg])} : A={get_ultimate(lines[best_avg])} : AVG={get_average(lines[best_avg])}")
            
            

if "__main__" == __name__:
    parser = argparse.ArgumentParser("values")
    parser.add_argument("--type", type=str, choices = ["va", "cat"], default="va")
    parser.add_argument("--file", type=str)

    args = parser.parse_args()

    main(args)

