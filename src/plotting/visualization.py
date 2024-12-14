import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import textwrap

def create_comparison_plot(original_matrix, ai_matrix, questions, party_names):
    # Calculate difference matrix
    diff_matrix = np.abs(original_matrix - ai_matrix)
    
    # Create color matrix
    color_matrix = np.zeros_like(diff_matrix)
    color_matrix[diff_matrix == 0] = 2    # Full agreement (blue)
    color_matrix[diff_matrix == 1] = 1    # Partial agreement (light blue)
    color_matrix[diff_matrix == 2] = 0    # Disagreement (white)

    # Create plot with increased height to prevent overlap
    plt.figure(figsize=(15, 15))
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
    plt.savefig('comparison_plot.png', bbox_inches='tight', dpi=300)
    plt.show()