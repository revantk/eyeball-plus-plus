import pytest
from eyeball_pp.utils import get_edges_within_n_hops

def test_get_edges_within_n_hops_returns_empty_list_for_invalid_node():
    """
    Test that get_edges_within_n_hops returns an empty list when an invalid node is provided.
    """
    edges = {'A': {'B': 1.0, 'C': 2.0}, 'B': {'D': 3.0}, 'C': {'E': 4.0}}
    assert get_edges_within_n_hops('Z', edges, 2) == []

def test_get_edges_within_n_hops_returns_empty_list_for_zero_hops():
    """
    Test that get_edges_within_n_hops returns an empty list when zero hops are provided.
    """
    edges = {'A': {'B': 1.0, 'C': 2.0}, 'B': {'D': 3.0}, 'C': {'E': 4.0}}
    assert get_edges_within_n_hops('A', edges, 0) == []

def test_get_edges_within_n_hops_returns_all_edges_for_max_hops():
    """
    Test that get_edges_within_n_hops returns all edges when max hops are provided.
    """
    edges = {'A': {'B': 1.0, 'C': 2.0}, 'B': {'D': 3.0}, 'C': {'E': 4.0}}
    assert get_edges_within_n_hops('A', edges, 3) == [('B', 1.0), ('C', 2.0), ('D', 4.0), ('E', 6.0)]

def test_get_edges_within_n_hops_returns_correct_edges_for_valid_input():
    """
    Test that get_edges_within_n_hops returns the correct edges for valid input.
    """
    edges = {'A': {'B': 1.0, 'C': 2.0}, 'B': {'D': 3.0}, 'C': {'E': 4.0}}
    assert get_edges_within_n_hops('A', edges, 2) == [('B', 1.0), ('C', 2.0), ('D', 4.0), ('E', 6.0)]

def test_get_edges_within_n_hops_returns_correct_edges_for_single_edge():
    """
    Test that get_edges_within_n_hops returns the correct edge when there is only one edge.
    """
    edges = {'A': {'B': 1.0}}
    assert get_edges_within_n_hops('A', edges, 1) == [('B', 1.0)]

def test_get_edges_within_n_hops_returns_correct_edges_for_cyclic_graph():
    """
    Test that get_edges_within_n_hops returns the correct edges for a cyclic graph.
    """
    edges = {'A': {'B': 1.0}, 'B': {'C': 2.0}, 'C': {'A': 3.0}}
    assert get_edges_within_n_hops('A', edges, 2) == [('B', 1.0), ('C', 2.0), ('A', 3.0), ('B', 4.0)]

def test_get_edges_within_n_hops_returns_correct_edges_for_disconnected_graph():
    """
    Test that get_edges_within_n_hops returns the correct edges for a disconnected graph.
    """
    edges = {'A': {'B': 1.0}, 'C': {'D': 2.0}}
    assert get_edges_within_n_hops('A', edges, 2) == [('B', 1.0)]