"""
Network Topology Builder
Constructs the multi-path topology specified in Figure 5 of the project doc:
S -> R1 (8), S -> R3 (14), R1 -> R2 (5), R1 -> R3 (6), R2 -> R4 (4), R3 -> R4 (3), R4 -> D (7).
"""
from typing import Dict, List, Tuple, Optional
import heapq
from .node import Node, RouterNode, EndpointNode
from .link import NetworkLink

class NetworkTopology:
    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.links: Dict[str, NetworkLink] = {}
        self.adj_list: Dict[str, List[str]] = {}
        self._build_default_topology()

    def _build_default_topology(self):
        """Constructs Figure 5 topology."""
        # 1. Add Endpoints & Routers
        self.nodes["S"] = EndpointNode("S", "10.0.0.1", "00:00:00:00:00:01")
        self.nodes["R1"] = RouterNode("R1", "10.0.1.1", "00:00:00:00:00:02", max_queue_size=50)
        self.nodes["R2"] = RouterNode("R2", "10.0.2.1", "00:00:00:00:00:03", max_queue_size=50)
        self.nodes["R3"] = RouterNode("R3", "10.0.3.1", "00:00:00:00:00:04", max_queue_size=50)
        self.nodes["R4"] = RouterNode("R4", "10.0.4.1", "00:00:00:00:00:05", max_queue_size=50)
        self.nodes["D"] = EndpointNode("D", "10.0.5.1", "00:00:00:00:00:06")

        for nid in self.nodes:
            self.adj_list[nid] = []

        # 2. Add Links according to Figure 5
        # S-R1 (8), S-R3 (14), R1-R2 (5), R1-R3 (6), R2-R4 (4), R3-R4 (3), R4-D (7)
        self.add_bidirectional_link("S", "R1", base_latency_ms=8.0, bandwidth_mbps=10.0)
        self.add_bidirectional_link("S", "R3", base_latency_ms=14.0, bandwidth_mbps=10.0)
        self.add_bidirectional_link("R1", "R2", base_latency_ms=5.0, bandwidth_mbps=10.0)
        self.add_bidirectional_link("R1", "R3", base_latency_ms=6.0, bandwidth_mbps=10.0)
        self.add_bidirectional_link("R2", "R4", base_latency_ms=4.0, bandwidth_mbps=10.0)
        self.add_bidirectional_link("R3", "R4", base_latency_ms=3.0, bandwidth_mbps=10.0)
        self.add_bidirectional_link("R4", "D", base_latency_ms=7.0, bandwidth_mbps=10.0)

    def add_bidirectional_link(self, u: str, v: str, base_latency_ms: float, bandwidth_mbps: float = 10.0):
        link_uv = NetworkLink(u=u, v=v, base_latency_ms=base_latency_ms, bandwidth_mbps=bandwidth_mbps)
        link_vu = NetworkLink(u=v, v=u, base_latency_ms=base_latency_ms, bandwidth_mbps=bandwidth_mbps)
        self.links[f"{u}->{v}"] = link_uv
        self.links[f"{v}->{u}"] = link_vu
        self.adj_list[u].append(v)
        self.adj_list[v].append(u)

    def get_link(self, u: str, v: str) -> Optional[NetworkLink]:
        return self.links.get(f"{u}->{v}")

    def get_candidate_paths(self, src: str = "S", dst: str = "D") -> List[List[str]]:
        """Finds all simple loop-free paths from src to dst."""
        paths = []
        def dfs(curr: str, target: str, current_path: List[str], visited: set):
            if curr == target:
                paths.append(list(current_path))
                return
            for neighbor in self.adj_list.get(curr, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    current_path.append(neighbor)
                    dfs(neighbor, target, current_path, visited)
                    current_path.pop()
                    visited.remove(neighbor)

        dfs(src, dst, [src], {src})
        # Sort candidate paths by base hop count or base delay
        return paths

    def calculate_path_cost(self, path: List[str], use_live_metrics: bool = False) -> float:
        total = 0.0
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            link = self.get_link(u, v)
            if not link:
                return float("inf")
            if use_live_metrics:
                # include link stress and router queue latency
                total += link.calculate_link_delay()
                # If v is a router, add queue delay
                node_v = self.nodes.get(v)
                if isinstance(node_v, RouterNode):
                    total += node_v.queue_length * 2.0  # 2ms per queued packet
            else:
                total += link.base_latency_ms
        return total

    def dijkstra_shortest_path(self, src: str = "S", dst: str = "D", use_live_metrics: bool = False) -> Tuple[List[str], float]:
        """Computes shortest path based on either static base latency or live network metrics."""
        pq = [(0.0, src, [src])]
        distances = {node: float("inf") for node in self.nodes}
        distances[src] = 0.0

        best_path = []
        best_cost = float("inf")

        while pq:
            curr_dist, u, path = heapq.heappop(pq)
            if curr_dist > distances[u]:
                continue
            if u == dst:
                return path, curr_dist

            for v in self.adj_list.get(u, []):
                link = self.get_link(u, v)
                if not link:
                    continue
                weight = link.calculate_link_delay() if use_live_metrics else link.base_latency_ms
                if use_live_metrics and isinstance(self.nodes.get(v), RouterNode):
                    weight += self.nodes[v].queue_length * 2.0

                new_dist = curr_dist + weight
                if new_dist < distances[v]:
                    distances[v] = new_dist
                    heapq.heappush(pq, (new_dist, v, path + [v]))

        return best_path, best_cost

    def stress_router_r2(self, stress_active: bool = True):
        """
        Simulates congestion on Router R2 as shown in Figure 5:
        Queue fills, delay rises on connected links (R1->R2, R2->R4).
        """
        r2 = self.nodes.get("R2")
        if isinstance(r2, RouterNode):
            if stress_active:
                # fill queue to 80% capacity
                r2.queue.clear()
                for i in range(40):
                    r2.queue.append(f"STRESS_PKT_{i}")
            else:
                r2.queue.clear()

        # Stress the links connected to R2
        l12 = self.get_link("R1", "R2")
        l21 = self.get_link("R2", "R1")
        l24 = self.get_link("R2", "R4")
        l42 = self.get_link("R4", "R2")

        for l in [l12, l21, l24, l42]:
            if l:
                l.set_stress(stress_active, stress_multiplier=6.0, additional_loss=0.20)

    def stress_router_r1(self, stress_active: bool = True):
        """
        Simulates congestion on Router R1:
        Queue fills, delay rises on connected links (S->R1, R1->R2, R1->R3).
        Forces traffic to consider the direct bypass route S->R3->R4->D.
        """
        r1 = self.nodes.get("R1")
        if isinstance(r1, RouterNode):
            if stress_active:
                r1.queue.clear()
                for i in range(40):
                    r1.queue.append(f"STRESS_PKT_{i}")
            else:
                r1.queue.clear()

        links = [
            self.get_link("S", "R1"), self.get_link("R1", "S"),
            self.get_link("R1", "R2"), self.get_link("R2", "R1"),
            self.get_link("R1", "R3"), self.get_link("R3", "R1")
        ]
        for l in links:
            if l:
                l.set_stress(stress_active, stress_multiplier=6.0, additional_loss=0.20)

    def inject_link_corruption(self, u: str, v: str, bit_error_rate: float):
        """Injects bit error rate corruption bidirectionally between u and v."""
        l_uv = self.get_link(u, v)
        l_vu = self.get_link(v, u)
        if l_uv:
            l_uv.inject_corruption(bit_error_rate)
        if l_vu:
            l_vu.inject_corruption(bit_error_rate)

