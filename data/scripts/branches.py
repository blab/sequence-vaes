
"""
Extract branch structure from Nextstrain auspice.json tree file.
Creates a TSV file with parent-child relationships for phylogenetic visualization.
"""
import argparse
import json
import pandas as pd
from augur.utils import json_to_tree
from urllib.parse import urlparse
import requests

def load_json(source):
    """Load JSON data from a URL or local file."""
    parsed = urlparse(source)
    if parsed.scheme in ('http', 'https'):
        return requests.get(source, headers={"accept": "application/json"}).json()
    else:
        with open(source, 'r') as f:
            return json.load(f)

def extract_branches(tree):
    """Extract parent-child relationships from phylogenetic tree."""
    branches = []
    
    # Traverse all nodes in the tree
    for node in tree.find_clades():
        parent_name = node.name
        # Strip hCoV-19/ prefix to match alignment.py behavior
        if parent_name and parent_name.startswith('hCoV-19/'):
            parent_name = parent_name.removeprefix('hCoV-19/')
        
        # Get all direct children
        for child in node.clades:
            child_name = child.name
            # Strip hCoV-19/ prefix to match alignment.py behavior
            if child_name and child_name.startswith('hCoV-19/'):
                child_name = child_name.removeprefix('hCoV-19/')
                
            branches.append({
                'parent': parent_name,
                'child': child_name
            })
    
    return branches

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--tree", required=True,
                        help="URL or local path for the tree.json file")
    parser.add_argument("--output", type=str, default="branches.tsv", 
                        help="Output TSV file for branch relationships")

    args = parser.parse_args()

    # Load tree JSON
    tree_json = load_json(args.tree)
    
    # Convert tree JSON to Bio.phylo format
    tree = json_to_tree(tree_json)
    
    # Extract branches
    branches = extract_branches(tree)
    
    # Create DataFrame and save
    branches_df = pd.DataFrame(branches)
    branches_df.to_csv(args.output, sep='\t', index=False)
    
    print(f"Extracted {len(branches)} branches to {args.output}")

