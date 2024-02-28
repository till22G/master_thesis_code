import matplotlib.pyplot as plt
import numpy as np
import argparse
import os 

from matplotlib import colormaps

parser = argparse.ArgumentParser(prog="plot model size vs perfomance")
parser.add_argument("--output-path", default=os.path.join("..", "plots", "model_size_vs_performance", "model_size_vs_performance_rep.png"), type=str)
args = parser.parse_args()

# switch to script dir so paths work
script_dir = os.path.dirname(__file__)
os.chdir(script_dir)

# Sample data (replace with your actual data)
#model_sizes = [8000000, 22000000, 57000000, 82000000, 132000000, 218000000, 670000000]
model_sizes = [8000000, 22000000, 57000000, 82000000, 132000000, 218000000]
#hit1_scores = [[33.4, 46.9, 54.9, 56.7, 58.2, 58.8, 60.45], 
#               [15.8, 20.4, 22.7, 24.1, 24.6, 24.7, 24.8], 
#               [23.0, 27.8, 30.0, 31.2, 32.3, 31.3, 33.2], 
#               [23.9, 41.8, 52.6, 54.6, 60.4, 60.9, 63.4]]


hit1_scores_with_neighbors = [[34.3, 48.4, 54.7, 57.2, 58.6, 60.6], 
                             [ 15.4, 20.0, 22.3, 23.7, 24.1, 24.9],
                             [ 30.7, 43.7, 52.4, 55.7, 56.1, 58.9], 
                             [ 15.3, 20.0, 22.4, 23.5, 23.54, 25.2]]

hit1_scores = hit1_scores_with_neighbors


#hit1_scores_without_neighbors = [[30.7, 43.7 52.4, 55.4, 56.1] 
#                                [ 15.3, 20.0 22.4, 23.5, 22.6]]

# Set up the color map
viridis = colormaps['viridis'].resampled(4)

# Create a log-scale plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(model_sizes, hit1_scores[0], label=f'WN18RR neighbors', marker='o', color=viridis(0))
ax.plot(model_sizes, hit1_scores[1], label=f'FB15k237 neighbors', marker='o', color=viridis(1))
ax.plot(model_sizes, hit1_scores[2], label=f'WN18RR no neighbors', marker='o', color=viridis(2))
ax.plot(model_sizes, hit1_scores[3], label=f'FB15k237 no neighbors', marker='o', color=viridis(3))
#ax.plot(model_sizes, hit1_scores[2], label=f'Wikidata-Trans', marker='o', color=viridis(2))
#ax.plot(model_sizes, hit1_scores[3], label=f'Wikidata-Ind', marker='o', color=viridis(3))

# Create a twin Axes for the bottom ticks
ax_top = ax.twiny()
ax_top.set_xscale('log')
ax_top.set_xlim(left=7000000, right=10e8)
ax_top.minorticks_off()

# Top ticks
#top_ticks = [8000000, 22000000, 57000000, 82000000, 132000000, 218000000, 670000000]
#top_tick_labels = ['BERT-tiny', 'BERT-mini', 'BERT-small', 'BERT-medium', 'distilBERT', 'BERT-base', 'BERT-large']
top_ticks = [8000000, 22000000, 57000000, 82000000, 132000000, 218000000]
top_tick_labels = ['BERT-tiny', 'BERT-mini', 'BERT-small', 'BERT-medium', 'distilBERT', 'BERT-base']
ax_top.set_xticks(top_ticks)
ax_top.set_xticklabels(top_tick_labels, rotation=20)

# Set plot properties
ax.set_xscale('log')
ax.set_xlim(left=7000000, right=10e8)
ax.set_xlabel('Number of Parameters')
ax.set_ylabel('Hit@1 Score')
ax.set_title('Model Hit1 Scores vs. Number of Parameters')
ax.legend()
ax.grid(True)

# Save the plot to a specified location
save_path = args.output_path
dirname = os.path.dirname(save_path)
if not os.path.exists(dirname):
    os.makedirs(dirname)
plt.savefig(save_path, bbox_inches='tight')
print(f"Saved plot to {save_path}")
#plt.show()