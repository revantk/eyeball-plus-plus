import pytest
from eyeball_pp.utils import get_score_map

def test_get_score_map_empty_input():
    """
    Test that get_score_map returns an empty dictionary when given empty input.
    """
    nodes = []
    edges = {}
    assert get_score_map(nodes, edges) == {}

def test_get_score_map_single_node():
    """
    Test that get_score_map returns a dictionary with a single node and score 0 when given a single node.
    """
    nodes = ['A']
    edges = {}
    assert get_score_map(nodes, edges) == {'A': 0.0}

def test_get_score_map_single_edge():
    """
    Test that get_score_map returns a dictionary with two nodes and scores 0 and 1 when given a single edge.
    """
    nodes = ['A', 'B']
    edges = {'A': {'B': 1.0}}
    assert get_score_map(nodes, edges) == {'A': 0.0, 'B': 1.0}

def test_get_score_map_multiple_edges():
    """
    Test that get_score_map returns a dictionary with three nodes and scores 0, 1, and 3 when given multiple edges.
    """
    nodes = ['A', 'B', 'C']
    edges = {'A': {'B': 1.0}, 'B': {'C': 1.0}}
    assert get_score_map(nodes, edges) == {'A': 0.0, 'B': 1.0, 'C': 3.0}

def test_get_score_map_multiple_hops():
    """
    Test that get_score_map returns a dictionary with four nodes and scores 0, 1, 3, and 6 when given multiple edges with multiple hops.
    """
    nodes = ['A', 'B', 'C', 'D']
    edges = {'A': {'B': 1.0}, 'B': {'C': 1.0}, 'C': {'D': 1.0}}
    assert get_score_map(nodes, edges) == {'A': 0.0, 'B': 1.0, 'C': 3.0, 'D': 6.0}