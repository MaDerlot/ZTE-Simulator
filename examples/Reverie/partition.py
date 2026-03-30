#!/usr/bin/env python3
"""
Spatial partition algorithm for MPI-based parallel simulation.

Partitions the Leaf-Spine topology nodes to MPI ranks based on Pod locality.
Exploits the traffic locality of LLM training:
  - TP/EP flows stay within a Pod (same ToR group)   -> minimal cross-rank traffic
  - DP flows cross Pods                               -> boundary between ranks
  - Spine switches have links to all Pods             -> each rank holds its own ghost copy

Topology file format (line 1):
  total_nodes  switch_count  tor_count  link_count  leaf_bw  spine_bw

Switch list (next switch_count values on one line):
  sid0 sid1 ... sidN
  (first tor_count entries are ToR IDs, the rest are Spine IDs)

Link entries (one per line):
  src  dst  data_rate  delay  error_rate

Output: node_partition.txt
  node_id,rank
  (one line per node; Spine nodes appear as rank=-1, meaning ghost / all ranks)
"""

import argparse
import sys
import os


def parse_topology(topo_file):
    """Parse Leaf-Spine topology file, return topology metadata and switch IDs."""
    with open(topo_file, 'r') as f:
        lines = f.read().split()

    idx = 0
    total_nodes  = int(lines[idx]); idx += 1
    switch_count = int(lines[idx]); idx += 1
    tor_count    = int(lines[idx]); idx += 1
    link_count   = int(lines[idx]); idx += 1
    leaf_bw      = int(lines[idx]); idx += 1
    spine_bw     = int(lines[idx]); idx += 1

    spine_count = switch_count - tor_count
    server_count = total_nodes - switch_count
    servers_per_tor = server_count // tor_count

    # Read switch IDs
    switch_ids = []
    for i in range(switch_count):
        switch_ids.append(int(lines[idx])); idx += 1

    tor_ids   = switch_ids[:tor_count]   # first tor_count are ToR IDs
    spine_ids = switch_ids[tor_count:]   # remainder are Spine IDs

    # Read links to determine server->ToR connectivity
    links = []
    for _ in range(link_count):
        src   = int(lines[idx]);   idx += 1
        dst   = int(lines[idx]);   idx += 1
        rate  = lines[idx];        idx += 1
        delay = lines[idx];        idx += 1
        err   = float(lines[idx]); idx += 1
        links.append((src, dst, rate, delay, err))

    return {
        'total_nodes':    total_nodes,
        'switch_count':   switch_count,
        'tor_count':      tor_count,
        'spine_count':    spine_count,
        'server_count':   server_count,
        'servers_per_tor': servers_per_tor,
        'tor_ids':        tor_ids,
        'spine_ids':      spine_ids,
        'links':          links,
    }


def partition_topology(topo_file, num_ranks, output_file):
    """
    Assign each network node to an MPI rank.

    Rules:
      - Build a server->ToR mapping by scanning links where one endpoint is a
        server (node_type==0) and the other is a ToR.
      - Group ToRs into num_ranks pods: rank = tor_index % num_ranks
      - Each server inherits the rank of its ToR.
      - Spine switches are assigned rank -1 (ghost: all ranks create them).
    """
    meta = parse_topology(topo_file)

    tor_id_set   = set(meta['tor_ids'])
    spine_id_set = set(meta['spine_ids'])
    switch_id_set = tor_id_set | spine_id_set

    # Build server -> ToR mapping from links
    server_to_tor = {}
    for src, dst, *_ in meta['links']:
        if src not in switch_id_set and dst in tor_id_set:
            # src is server, dst is ToR
            server_to_tor[src] = dst
        elif dst not in switch_id_set and src in tor_id_set:
            # dst is server, src is ToR
            server_to_tor[dst] = src

    # Map tor_id -> rank (round-robin across num_ranks)
    tor_to_rank = {}
    for idx, tor_id in enumerate(meta['tor_ids']):
        tor_to_rank[tor_id] = idx % num_ranks

    # Assign ranks
    node_rank = {}
    for node_id in range(meta['total_nodes']):
        if node_id in spine_id_set:
            node_rank[node_id] = -1          # ghost node
        elif node_id in tor_id_set:
            node_rank[node_id] = tor_to_rank[node_id]
        else:
            # server
            tor = server_to_tor.get(node_id)
            if tor is None:
                print(f"WARNING: server {node_id} has no ToR connection, assigned to rank 0")
                node_rank[node_id] = 0
            else:
                node_rank[node_id] = tor_to_rank[tor]

    # Write output
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    with open(output_file, 'w') as f:
        f.write("# node_id,rank  (rank=-1 means ghost/all-ranks Spine node)\n")
        for node_id in sorted(node_rank.keys()):
            f.write(f"{node_id},{node_rank[node_id]}\n")

    # Print summary
    rank_counts = {}
    ghost_count = 0
    for nid, r in node_rank.items():
        if r == -1:
            ghost_count += 1
        else:
            rank_counts[r] = rank_counts.get(r, 0) + 1

    print(f"Topology: {meta['total_nodes']} nodes, "
          f"{meta['tor_count']} ToRs, {meta['spine_count']} Spines, "
          f"{meta['server_count']} servers")
    print(f"Partition into {num_ranks} ranks:")
    for r in sorted(rank_counts.keys()):
        print(f"  Rank {r}: {rank_counts[r]} nodes")
    print(f"  Ghost (Spine, all ranks): {ghost_count} nodes")
    print(f"Output written to: {output_file}")

    return node_rank


def main():
    parser = argparse.ArgumentParser(
        description="Partition Leaf-Spine topology for MPI parallel simulation")
    parser.add_argument('--topo',    required=True,
                        help="Path to topology file (e.g. leaf-spine.txt)")
    parser.add_argument('--ranks',   type=int, required=True,
                        help="Number of MPI ranks (should equal number of Pods or divisor thereof)")
    parser.add_argument('--output',  required=True,
                        help="Output partition file path (e.g. task1/node_partition.txt)")
    args = parser.parse_args()

    if not os.path.exists(args.topo):
        print(f"ERROR: topology file not found: {args.topo}")
        sys.exit(1)

    partition_topology(args.topo, args.ranks, args.output)


if __name__ == '__main__':
    main()
