"""
Module for graphing the results of lamdscape calculations and VQE experiments.
"""
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from landscape import TotalLandscapeResult, flatten_energies
from vqe import MyVQEResult

# Constants figure generation.
FIG_SIZE = (8,5)
DPI = 400


def display_energy_landscape(energy_landscape_results: TotalLandscapeResult, graph_title: str="Energy landscape",
                                show_legend=False):
    """Display the results of an energy landscape calculation.

    Args:
        energy_landscape_results (TotalLandscapeResult): results of an energy landscape calculation.
        graph_title (str, optional): title of the graph. Defaults to "Energy landscape".
        show_legend (bool, optional): whether to add a legend showing the different indexes. Defaults to False.
    """
    plt.figure(figsize=FIG_SIZE, dpi=DPI)
    idx_counter = 0
    basis_size = energy_landscape_results.basis_size
    mub_results_size = basis_size * energy_landscape_results.subset_num
    for i, mub_res in enumerate(energy_landscape_results.mub_results):
        for j, subset_res in enumerate(mub_res):
            energies_only = [result.value for result in subset_res]
            # For plots with full MUBs (no different subsets in each basis), have a distinct color for each basis.
            # For plots with partial MUBs (several subsets in each basis), keep distinct bases 
            state_color = f"C{i}" if len(mub_res) == 1 else f"C{j}"
            plt.plot(list(range(idx_counter, idx_counter+basis_size)), energies_only, 'o', color=state_color, lw=0.4, label=f"MUB {i}, subset {j}")
            idx_counter += basis_size
        # Show separation between different MUBs
        plt.axvspan(idx_counter - mub_results_size, idx_counter, alpha=0.1, color=f"C{i}")
    # Show exact result
    plt.axhline(y=energy_landscape_results.ground_energy, lw=0.6, color='red')
    # Show comp. basis specifically
    _, _, ymin, ymax = plt.axis()
    plt.text(x=basis_size*0.25, y=ymin + (ymax-ymin)*0.8, s='COMP', fontsize=10)
    
    plt.xlabel("MUB state index")
    plt.ylabel("Cost function result")
    if show_legend:
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title(graph_title)
    plt.show()


def display_energy_histogram(energy_landscape_results: TotalLandscapeResult, bins=100,
                                graph_title="Energy landscape histogram"):
    """Display a histogram of the energy landscape values of a Hamiltonian.

    Args:
        energy_landscape_results (TotalLandscapeResult): results of an energy landscape calculation.
        bins (int, optional): number of histogram bins. Defaults to 100.
        graph_title (str, optional): title of the graph. Defaults to "Energy landscape histogram".
    """
    fig = plt.figure(figsize=FIG_SIZE, dpi=DPI)
    plt.locator_params(axis='x', nbins=min(bins//2, 30), tight=True)
    plt.xticks(fontsize=10, rotation=60)
    plt.locator_params(axis='y', nbins=10)

    flat_results = flatten_energies(energy_landscape_results)
    plt.hist(flat_results, bins)
    # Show exact result
    plt.axvline(x=energy_landscape_results.ground_energy, lw=1, color='red')

    plt.xlabel("Cost function result")
    plt.ylabel("number of results")
    plt.title(graph_title)
    plt.show()


def draw_graph(G: nx.Graph):
    """
    Draw a networkx graph (the vertices and edges kind.)
    """
    colors = ["r" for _ in G.nodes()]
    pos = nx.spring_layout(G)
    default_axes = plt.axes(frameon=True)
    nx.draw_networkx(G, node_color=colors, node_size=600, alpha=0.8, ax=default_axes, pos=pos)
    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=edge_labels)
    plt.show()


def plot_VQE_evals(vqe_result_list: list[MyVQEResult], title: str = "", linewidth: float = 1) -> None:
    """
    Plot the progress of a VQE experiment.

    Args:
        vqe_result_list (list[MyVQEResult]): the list of VQE experuments to show on a single graph.
        title (str, optional): title of the graph. Defaults to "".
        linewidth (float, optional): the width of the plot line. Defaults to 1.
    """
    plt.figure(figsize=FIG_SIZE, dpi=DPI)
    plt.title(title)
    for vqe_result in vqe_result_list:
        assert vqe_result.costs_list_included
    vqe_result_list.sort(key=(lambda res: res.costs_list[0]))
    for vqe_result in vqe_result_list:
        plt.scatter([0], vqe_result.costs_list[:1], s=20)
        plt.plot(np.arange(len(vqe_result.costs_list)), np.array(vqe_result.costs_list),
                 label=vqe_result.desc, linewidth=linewidth)
    plt.legend()
    plt.show()
