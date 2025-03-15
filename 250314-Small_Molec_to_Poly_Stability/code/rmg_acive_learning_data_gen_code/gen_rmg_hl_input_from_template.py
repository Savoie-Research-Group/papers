import os
import argparse


this_script_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(this_script_dir)


def main():
    parser = argparse.ArgumentParser("Generate RMG half life input file from input template")
    parser.add_argument("--template_path", type=str, required=True, help="Path to template file")
    parser.add_argument("--output_path", type=str, required=True, help="Path to output file")
    parser.add_argument("--smi", type=str, required=True, help="SMILES string")
    parser.add_argument("--keyword", type=str, default="<smi>", help="Keyword to replace")
    args = parser.parse_args()

    template_path = args.template_path
    output_path = args.output_path
    smi = args.smi
    keyword = args.keyword
    
    with open(template_path, "r") as f:
        template = f.read()
        f.close()
        
    template = template.replace(keyword, smi)
    
    with open(output_path, "w") as f:
        f.write(template)
        f.close()


if __name__ == "__main__":
    main()
