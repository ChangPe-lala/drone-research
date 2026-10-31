import pandas as pd
import networkx as nx
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.multihub.build_route_graph import build_route_graph
from config.paths import OUTPUT_DIR

def run_g08_audit():
    # 1. Run G08
    G, edges_df = build_route_graph()
    
    # 2. Audit Topology using historical mapping
    mapping_path = OUTPUT_DIR / "data" / "multihub" / "trajectory_node_mapping.csv"
    if not mapping_path.exists():
        print(f"Mapping file not found: {mapping_path}")
        return
        
    mapping_df = pd.read_csv(mapping_path)
    
    od_counts = {"0_paths": 0, "1_path": 0, ">=2_paths": 0}
    total_requests = len(mapping_df)
    
    print("\n" + "="*60)
    print("OD CONNECTIVITY AUDIT (Historical Flights)")
    print("="*60)
    
    for i, row in mapping_df.iterrows():
        node_seq_str = row["ordered_node_sequence"]
        if pd.isna(node_seq_str):
            continue
        nodes = str(node_seq_str).split(",")
        if len(nodes) < 2:
            continue
            
        source = nodes[0]
        target = nodes[-1]
        
        if source not in G or target not in G:
            od_counts["0_paths"] += 1
            continue
            
        try:
            import itertools
            path_generator = nx.shortest_simple_paths(G, source, target)
            paths = list(itertools.islice(path_generator, 2))
                    
            if len(paths) == 0:
                od_counts["0_paths"] += 1
            elif len(paths) == 1:
                od_counts["1_path"] += 1
            else:
                od_counts[">=2_paths"] += 1
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            od_counts["0_paths"] += 1
            
    print(f"Total OD Requests (from historical flights): {total_requests}")
    print(f"  - 0 paths (Disconnected): {od_counts['0_paths']}")
    print(f"  - 1 path (No choice)    : {od_counts['1_path']}")
    print(f"  - >=2 paths (Routable)  : {od_counts['>=2_paths']}")
    print("="*60)
    
if __name__ == "__main__":
    run_g08_audit()
