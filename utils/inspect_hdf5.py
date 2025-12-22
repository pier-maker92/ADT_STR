import os
import sys
import argparse
import h5py


def build_group_tree(group):
    """
    Construct a tree representation of the HDF5 hierarchy starting at `group`.

    Returns a dict with keys:
      - name: absolute HDF5 path of the group
      - datasets: list of dataset names directly under this group
      - children: list of child group dicts (same structure)
      - total: total number of datasets under this group including all descendants
    """
    node = {
        "name": group.name,
        "datasets": [],
        "children": [],
        "total": 0,
    }

    # Separate immediate datasets and child groups
    for key, item in group.items():
        if isinstance(item, h5py.Dataset):
            node["datasets"].append(key)
        elif isinstance(item, h5py.Group):
            child_node = build_group_tree(item)
            node["children"].append(child_node)
            node["total"] += child_node["total"]

    node["total"] += len(node["datasets"])
    return node


def print_group_tree(node, indent=""):
    """
    Pretty-print the group tree. For each group, show aggregated dataset count
    including all datasets in its subgroups. Does not list individual items.
    """
    print(f"{indent}Group: {node['name']} (items: {node['total']})")
    for child in node["children"]:
        print_group_tree(child, indent + "  ")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Inspect an HDF5 file: list all groups/subgroups and print, for each, "
            "the total number of datasets it contains including all subgroups."
        )
    )
    parser.add_argument("hdf5_path", help="Path to the HDF5 file to inspect")
    args = parser.parse_args()

    if not os.path.isfile(args.hdf5_path):
        print(f"Error: file not found: {args.hdf5_path}", file=sys.stderr)
        sys.exit(1)

    try:
        with h5py.File(args.hdf5_path, "r") as f:
            root_group = f["/"]
            tree = build_group_tree(root_group)
            print_group_tree(tree)
    except OSError as exc:
        print(f"Failed to open HDF5 file: {exc}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
