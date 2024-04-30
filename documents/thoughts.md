# Current graphing method
The current method defines the x axis as an ordering on the (Basis, state) composite index.

There are two other elements involved: the color and the y axis.

The color is usually used to indicatre the basis. But what's funny is that, in a more reasonable ordering,
you'd find all states of the same basis bunched together. But that's not true in Dekel's graphs!
In his graphs, the colors always the same pattern, but are messy.
This indicates that he takes **the state** as the primary index, then the basis as the secondary one.
$$ graph_index = state_idx * num_of_states + basis_idx $$
And forgive me, but that's just silly.

If we did it the other way around, there would be a spatial sepaaration between the colors. Not only it will be *way* prettier, it will allow us to discern more easily which bases gave the best performance.

The Y axis is good. It always means the specific thing that we want it to mean (usually the energy expectation value).


## Idea 1: Transpose the index, keep colors and Y axis
### Advantages:
-   Cleaner visually.
-   Gives meaningful data about bases.
-   
### Disadvantages:
-  The colors will be a bit useless because they can be replaced by seperating vertical lines
-   understanding the "heat map" is still done by "summing with your eyes" over horizontal lines.
-   

## Idea 2: Historgam w. combined colors.
Change the Y axis to be the X axis (because it's a histogram), and the Y will be thee amount of states that reached that value.
We will,. of course, need to decice on a granularity. A basic idea would be to take the highest and lowest values, and paretition the range equally to, say, 100 intervals.
### Advantages:
-   Clearer representation of density.
-   Removing redundant data points that have no real meaning to the viewer.
-   
### Disadvantages:
-   Losing the indication on what came from which basis, which is very important to us.
-   

## Idea 2.1: Add mutliple colors to each column
Same as 2, but color each bar depending on the bases that contributed to it. Make the colors proportional to the amount.
### Advaantages:
-   Best of all worlds.
-   
## Disadvantages:
-   Not very easy to view.


## Idea 3: 1 and 2, side by side
### Advantages
-   Best of all worlds.
-   Easy to read.
-   
### Disadvantages
-   More space.
-   





Wait so I think I messed up.
Dekel did do the right indexing method in the 2-3 qubit case. There's just a THIRD indexing parameter that was not considered when you go to more than 3 qubits: the subset.
There's: 
1. Which subset of qubits toyou apply the MUB preparatio on
2. Which basis don you use
3. Which state do you take from said basis.

What Dekel forgot to show in his graph is the separation (perhaps by vertical lines?) between different subset choices.

## Idea 4: Vertical lines to separate values from different subset chocies.

## Idea 5: Idea 3, but with the vertical lines.
This is what I ended up wgoint with, with a slight modification:
Instead of lines separating the different bases, I colored the x-y plane to in different colors for different intervals, where each interval describhes a specific MUB.



# The required experiments
okay.
Let's use this to compile all of the experiments that I agreed with Tal that I'd do.

## 2 qubits
molecular and tr-is hams, landscaping and Dekel's expressive ansatz VQE.

## 3 qubits
maxcuts and tr-is hams (no option for moleculars), using shifted MUBs for landscaping and VQE.

## 4 qubits
maxcuts, molecular and tr-is hams, using shifted half-MUBs for landscaping and VQE.

## 5-8 qubits
maxvuts and tr-is hams, using shifted half-MUBs for landscaping and VQE.













Right, we're almost there.
What's mising, then?
- Decide on how to display both order and value.
    -   For order I think we can define a line, and for value the color scheme.
    -   making it into a line would be harder though.
-   Add the molecular 4-qubit example.
-   see what else Tal wanted.