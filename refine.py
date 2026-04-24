import argparse
import os
from modeller import *
from modeller import *
from modeller.automodel import *
import argparse
import sys
sys.stdout = open(os.devnull, 'w')
sys.stderr = open(os.devnull, 'w')
'''
usage:
python ./evaluate_model.py ./123/1ake.pdb ./123/1ake.fasta 
'''
parser = argparse.ArgumentParser(description="Run Modeller with external parameters.")
parser.add_argument("template", help="The name of the template structure")
parser.add_argument("fasta_file", help="The sequence of target pdb")
args = parser.parse_args()
allowed_extensions = ['.ali', '.pdb', '.fasta']

template = args.template.split("/")[-1].split(".")[0]
seq_name = args.fasta_file.split("/")[-1].split(".")[0]

sequence = ''
with open(args.fasta_file, "r") as f:
    lines = f.readlines()
    sequence = ''.join([line.strip() for line in lines[1:]])


folder = os.path.dirname(args.fasta_file)
os.chdir(folder)

with open(f"{seq_name}.ali", "w") as file:
    file.write(f">{seq_name}\n")
    file.write(f"sequence:{seq_name}:::::::0.00: 0.00\n")
    file.write(sequence)

env = environ()
aln = alignment(env)
mdl = model(env, file=f'{template}', model_segment=('FIRST:A', 'LAST:A'))
aln.append_model(mdl, align_codes=f'{template}', atom_files=f'{template}.pdb')
aln.append(file=f'{seq_name}.ali', align_codes=f'{seq_name}')
aln.align2d()
aln.write(file=f'{seq_name}-{template}.ali', alignment_format='PIR')
aln.write(file=f'{seq_name}-{template}.pap', alignment_format='PAP')

a = automodel(env, alnfile=f'{seq_name}-{template}.ali',
              knowns=f'{template}', sequence=f'{seq_name}',
              assess_methods=(assess.DOPE,
                              assess.GA341))
a.starting_model = 1
a.ending_model = 1
a.make()

def delete_files_except(target_folder, allowed_extensions):
    try:
        for root, dirs, files in os.walk(target_folder):
            for file in files:
                file_path = os.path.join(root, file)
                if not any(file.endswith(ext) for ext in allowed_extensions):
                    os.remove(file_path)
    except Exception as e:
        print(f"error: {str(e)}")


delete_files_except(folder, allowed_extensions)