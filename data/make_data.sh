dataset="ncov/open/global"
gene="nuc"

all_data="./all_data"
tree_file="all_auspice.json"
root_file="all_auspice_root-sequence.json"
aligned_file="all_aligned.fasta"
metadata_file="all_metadata.tsv"
branches_file="all_branches.tsv"
spike_file="all_spike.fasta"

echo "https://nextstrain.org/"$dataset
echo "https://data.nextstrain.org/files/"$dataset"/metadata.tsv.xz"

echo "downloading data from nextstrain..."
nextstrain remote download https://nextstrain.org/$dataset all_data/all_auspice.json

# echo "downloading other_metadata from nextstrain"
# wget https://data.nextstrain.org/files/$dataset/metadata.tsv.xz -O all_data/all_other_metadata.tsv.xz
# xz --decompress all_data/all_other_metadata.tsv.xz

echo "aligning sequences..."
python scripts/alignment.py --tree $all_data"/"$tree_file --root $all_data"/"$root_file --output $all_data"/"$aligned_file --gene $gene

echo "making metadata file..."
python scripts/metadata.py --tree $all_data"/"$tree_file --output $all_data"/"$metadata_file

echo "making tree metadata file..."
python scripts/branches.py --tree $all_data"/"$tree_file --output $all_data"/"$branches_file

echo "making file that contains spike sequences..."
python scripts/trim.py --input-alignment $all_data"/"$aligned_file --output-alignment $all_data"/"$spike_file

augur filter --metadata $all_data"/"$metadata_file --sequences $all_data"/"$spike_file --min-date 2022-06-04 --max-date 2025-06-04 --output-sequences ./training_spike.fasta

augur filter --metadata $all_data"/"$metadata_file --sequences $all_data"/"$spike_file --min-date 2021-06-04 --max-date 2022-06-04 --output-sequences ./valid_spike.fasta

augur filter --metadata $all_data"/"$metadata_file --sequences $all_data"/"$spike_file --min-date 2020-06-04 --max-date 2021-06-04 --output-sequences ./test_spike.fasta
