import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import textwrap
from config import is_rag_context
import os
def create_comparison_plot(original_matrix, ai_matrix, questions, party_names):
    # Calculate difference matrix
    diff_matrix = np.abs(original_matrix - ai_matrix)
    
    # Create color matrix
    color_matrix = np.zeros_like(diff_matrix)
    color_matrix[diff_matrix == 0] = 2    # Full agreement (blue)
    color_matrix[diff_matrix == 1] = 1    # Partial agreement (light blue)
    color_matrix[diff_matrix == 2] = 0    # Disagreement (white)

    # Determine figure size based on the number of questions and parties
    num_questions = len(questions)
    num_parties = len(party_names)
    fig_height = max(5, num_questions * 0.5)  # Adjust the multiplier as needed for spacing
    fig_width = max(5, num_parties * 0.5)  # Adjust the multiplier as needed for spacing
    plt.figure(figsize=(fig_width, fig_height))
    
    colors = ['white', 'lightblue', 'blue']
    cmap = sns.color_palette(colors)
    
    # Format long questions with more width to reduce line breaks
    wrapped_questions = ['\n'.join(textwrap.wrap(q, width=70)) for q in questions]
    
    sns.heatmap(color_matrix, 
                xticklabels=party_names, 
                yticklabels=wrapped_questions,
                cmap=cmap,
                cbar=False)

    plt.title('Comparison between Original Party Answers\nand AI Predictions')
    plt.xlabel('Political Parties')
    plt.ylabel('Questions')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Adjust subplot parameters to give more space to y-axis labels
    plt.subplots_adjust(left=0.3)

    # Add legend
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor='blue', label='Full Agreement'),
                      plt.Rectangle((0,0),1,1, facecolor='lightblue', label='Partial Agreement'),
                      plt.Rectangle((0,0),1,1, facecolor='white', label='Disagreement')]
    plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    def save_plot_with_incremental_name(base_name):
        counter = 1
        file_name = f"{base_name}_{counter}.png"
        while os.path.exists(file_name):
            counter += 1
            file_name = f"{base_name}_{counter}.png"
        plt.savefig(file_name, bbox_inches='tight', dpi=300)

    if is_rag_context:
        save_plot_with_incremental_name('comparison_plot_rag')
    else:
        save_plot_with_incremental_name('comparison_plot_GPT')
    plt.show()