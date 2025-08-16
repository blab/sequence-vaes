
# code from Trevor Bedford and Katie Kistler
# MIT license
import argparse
import json
import os
import requests
import Bio.Phylo
from tqdm import tqdm
from urllib.parse import urlparse

def json_to_tree(json_dict, root=True):
    """Returns a Bio.Phylo tree corresponding to the given JSON dictionary exported
    by `tree_to_json`.
    Assigns links back to parent nodes for the root of the tree.
    """
    # Check for v2 JSON which has combined metadata and tree data.
    if root and "meta" in json_dict and "tree" in json_dict:
        json_dict = json_dict["tree"]

    node = Bio.Phylo.Newick.Clade()

    # v1 and v2 JSONs use different keys for strain names.
    if "name" in json_dict:
        node.name = json_dict["name"]
    else:
        node.name = json_dict["strain"]

    if "children" in json_dict:
        # Recursively add children to the current node.
        node.clades = [json_to_tree(child, root=False) for child in json_dict["children"]]

    # Assign all non-children attributes.
    for attr, value in json_dict.items():
        if attr != "children":
            setattr(node, attr, value)

    # Only v1 JSONs support a single `attr` attribute.
    if hasattr(node, "attr"):
        node.numdate = node.attr.get("num_date")
        node.branch_length = node.attr.get("div")

        if "translations" in node.attr:
            node.translations = node.attr["translations"]
    elif hasattr(node, "node_attrs"):
        node.branch_length = node.node_attrs.get("div")

    if root:
        node = annotate_parents_for_tree(node)

    return node

def annotate_parents_for_tree(tree):
    """Annotate each node in the given tree with its parent."""
    tree.root.parent = None
    for node in tree.find_clades(order="level"):
        for child in node.clades:
            child.parent = node
    return tree

def load_json(source):
    """Load JSON data from a URL or local file."""
    parsed = urlparse(source)
    if parsed.scheme in ('http', 'https'):
        return requests.get(source, headers={"accept": "application/json"}).json()
    else:
        with open(source, 'r') as f:
            return json.load(f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and simplify Auspice JSON as metadata TSV")
    parser.add_argument("--tree", required=True, help="URL or local path for the tree.json file")
    parser.add_argument("--output", type=str, default="metadata.tsv", help="Output TSV file")

    args = parser.parse_args()

    # Load tree JSON from either URL or local file
    tree_json = load_json(args.tree)
    tree = json_to_tree(tree_json)

    data = []
    include_clade = False
    include_mutations = False

    nodes = list(tree.find_clades(order="postorder"))
    for n in tqdm(nodes, desc="Processing nodes", unit="node"):
        node_elements = {"name": n.name.removeprefix('hCoV-19/')}
        node_elements["parent"] = n.parent.name.removeprefix('hCoV-19/') if n.parent else None

        if hasattr(n, 'node_attrs'):
            if 'clade_membership' in n.node_attrs and 'value' in n.node_attrs["clade_membership"]:
                node_elements["clade_membership"] = n.node_attrs["clade_membership"]["value"]
                include_clade = True
            if 'S1_mutations' in n.node_attrs and 'value' in n.node_attrs["S1_mutations"]:
                node_elements["S1_mutations"] = n.node_attrs["S1_mutations"]["value"]
                include_mutations = True

            # Aayush
            node_elements["date"] = n.node_attrs["num_date"]["value"]

        data.append(node_elements)

    # Determine column headers dynamically
    headers = ["name", "parent", "date"]
    if include_clade:
        headers.append("clade_membership")
    if include_mutations:
        headers.append("S1_mutations")

    with open(args.output, 'w', encoding='utf-8') as handle:
        print(*headers, sep='\t', file=handle)
        for elem in data:
            row = [elem.get(header, "") for header in headers]
            print(*row, sep='\t', file=handle)

