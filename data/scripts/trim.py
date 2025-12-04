# code from ChatGPT
import argparse
from Bio import SeqIO

# corresponds to SARS-CoV-2 spike for Wuhan/1 alignment
DEFAULT_BEGIN = 21563
DEFAULT_END = 25384
# DEFAULT_END = 23617

def trim_alignment(input_file, output_file, begin, end):
    with open(input_file, "r") as infile, open(output_file, "w") as outfile:
        for record in SeqIO.parse(infile, "fasta"):
            trimmed_sequence = record.seq[begin-1:end]  # Convert to 0-based indexing
            record.seq = trimmed_sequence
            SeqIO.write(record, outfile, "fasta")

def main():
    parser = argparse.ArgumentParser(description="Trim an alignment to specified positions.")
    parser.add_argument("--input-alignment", type=str, required=True, help="Path to the input alignment in FASTA format.")
    parser.add_argument("--output-alignment", type=str, default="data/trimmed.fasta", help="Path to save the trimmed alignment.")
    parser.add_argument("--begin", type=int, default=DEFAULT_BEGIN, help="Start position (1-based, inclusive).")
    parser.add_argument("--end", type=int, default=DEFAULT_END, help="End position (1-based, inclusive).")

    args = parser.parse_args()

    print(f"Trimming alignment from positions {args.begin} to {args.end}...")
    trim_alignment(args.input_alignment, args.output_alignment, args.begin, args.end)
    print(f"Trimmed alignment saved to {args.output_alignment}")

if __name__ == "__main__":
    main()
