from datetime import datetime, date
import os
from typing import Any, Optional, Sequence, Union
from collections import OrderedDict, defaultdict
from rich.table import Table
from rich.console import Console


def get_user_input(text: str, choices: Optional[Sequence[Any]]) -> str:
    from InquirerPy import inquirer
    import termios
    import sys

    termios.tcflush(sys.stdin, termios.TCIOFLUSH)

    if choices:
        return inquirer.select(text, choices=choices).execute()
    else:
        return inquirer.text(text).execute()


class LruCache(OrderedDict):
    def __init__(self, max_size: int = 100) -> None:
        super().__init__()
        self.max_size = max_size

    def __setitem__(self, key: Any, value: Any) -> None:
        if len(self) >= self.max_size:
            self.popitem(last=False)
        super().__setitem__(key, value)


def output_table(
    rows: list[dict[str, Any]],
    markdown_file: Optional[str] = None,
    title: Optional[str] = None,
    column_names: Optional[list[str]] = None,
    print_table: bool = True,
):
    """Outputs a table to the terminal and writes it as markdown to a file if markdown_file is provided."""
    if len(rows) == 0:
        return

    if column_names is None:
        column_names = []
        for row in rows:
            for column_name in row.keys():
                if column_name not in column_names:
                    column_names.append(column_name)

    table = Table(title=title, show_lines=True, title_justify="left")
    md_data = f"| {' | '.join(column_names)} |\n"
    md_data += "| " + " | ".join(["---"] * len(column_names)) + " |\n"

    for column_name in column_names:
        table.add_column(column_name)

    for row in rows:
        column_values = [row.get(column_name, "") for column_name in column_names]
        column_value_strs = [
            f"{column_value: .2f}" if type(column_value) == float else str(column_value)
            for column_value in column_values
        ]
        table.add_row(*column_value_strs)
        md_data += f"| {' | '.join(column_value_strs)} |\n"

    if print_table:
        print()
        console = Console()
        console.print(table)

    if markdown_file is not None:
        dir = os.path.dirname(markdown_file)
        if not os.path.exists(dir):
            os.makedirs(dir)

        with open(markdown_file, "w+") as f:
            f.write(md_data)


def get_edges_n_hops(
    node: str, edges: dict[str, dict[str, float]], n_hops: int
) -> list[tuple[str, float]]:
    """Returns a list of edges that are exactly n hops of the given node and the weight of the n_hops -1 -> n_hops path to that edge."""
    if n_hops <= 0:
        return []

    edges_n_hops = []
    visited = set()

    def _recursive_edge_traversal(
        _node: str, _current_weight: float, _num_hops_taken: int
    ):
        if _num_hops_taken > n_hops:
            return

        if _num_hops_taken == n_hops:
            edges_n_hops.append((_node, _current_weight))
            return

        if _node in visited:
            return

        visited.add(_node)
        if _node in edges:
            for edge, weight in edges[_node].items():
                _recursive_edge_traversal(edge, weight, _num_hops_taken + 1)

    _recursive_edge_traversal(node, 0.0, 0)
    return edges_n_hops


def get_score_map(
    edges: dict[str, dict[str, float]],
) -> dict[str, float]:
    """Returns a dict of node to score for the given nodes and edges. Propagates scores to nodes that are 1 hop away,
    then 2 hops away, etc. so if a graph is
    A -> B -> C
    then C will get the score of B + A, and B will get the score of A.
    """
    score_map: dict[str, float] = defaultdict(float)
    nodes = set(edges.keys())
    for n_hops in range(1, len(nodes) + 1):
        for node in nodes:
            edges_n_hops = get_edges_n_hops(node, edges, n_hops)
            for edge, weight in edges_n_hops:
                score_map[edge] += max(weight, 0.0)
            if node not in score_map:
                score_map[node] = 0.0
    return score_map


def time_range_to_str(start_time: datetime, end_time: datetime) -> str:
    if start_time.date() == end_time.date():
        return f"{start_time.strftime('%b %-d %H:%M %p')} - {end_time.strftime('%H:%M %p')}"
    else:
        return f"{start_time.strftime('%b %-d')} - {end_time.strftime('%b %-d')}"


def time_to_str(time: Union[datetime, date]) -> str:
    hm_format = ", %H:%M %p" if type(time) == datetime else ""

    if time.year != datetime.now().year:
        return time.strftime(f"%b %-d %Y{hm_format}")
    else:
        return time.strftime(f"%b %-d{hm_format}")
