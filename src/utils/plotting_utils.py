import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

end_conditions_color_dict={'reach_stay_no_collision':'green','reach_no_stay_no_collision':'yellow',
                            'reach_stay_collision':'brown','reach_no_stay_collision':'red',
                            'no_reach_no_collision':'gray',
                            'no_reach_collision':'black'}

def plot_training(logs_path,plots_path):
    dataframe = pd.read_csv(os.path.join(logs_path, f"training.csv"))
    rewards_path = os.path.join(plots_path, f"rewards.png")
    robustness_path = os.path.join(plots_path, f"robustness.png")

    num_of_envs=dataframe['Environment'].max()+1
    dataframe['Total_Episode']=dataframe['Episode']*num_of_envs+dataframe['Environment']

    dataframe.drop(columns=['Episode'],inplace=True)

    means_rewards=dataframe.groupby('Total_Episode')["Reward"].mean().reset_index()
    means_robustness=dataframe.groupby('Total_Episode')["Robustness"].mean().reset_index()

    plt.plot(means_rewards['Total_Episode'],means_rewards["Reward"])
    plt.xlabel('Episode')
    plt.ylabel("Mean Reward")
    plt.title("Mean Reward per Episode")
    plt.savefig(rewards_path)
    plt.close()

    plt.plot(means_rewards['Total_Episode'],means_robustness["Robustness"])
    plt.xlabel('Episode')
    plt.ylabel("Mean Robustness")
    plt.title("Mean Robustness per Episode")
    plt.savefig(robustness_path)
    plt.close()


def plot_training_end(logs_path, plots_path):
    dataframe = pd.read_csv(os.path.join(logs_path, 'end_conditions.csv'))
    robustness_path = os.path.join(plots_path, 'end_robustness.png')

    num_of_envs = dataframe['Environment'].max() + 1
    dataframe['Total_Episode'] = dataframe['Episode'] * num_of_envs + dataframe['Environment']

    # Map the 'End_Condition' to the corresponding colors
    colors = dataframe['End_Condition'].map(end_conditions_color_dict)

    plt.scatter(dataframe['Total_Episode'], dataframe["Robustness"], color=colors)

    legend_elements=[Line2D([0], [0], marker='o', color='w', label=key, markerfacecolor=value, markersize=10) for key,value in end_conditions_color_dict.items()]

    plt.legend(handles=legend_elements, loc="lower right", title="End Condition")

    plt.xlabel("Episode")
    plt.ylabel('Reward')
    plt.title("Final Robustness per episode")
    plt.savefig(robustness_path)
    plt.close()

def plot_test_end_condition(logs_path, plots_path):
    path = os.path.join(plots_path, 'end_conditions.png')
    dataframe = pd.read_csv(os.path.join(logs_path, 'end_conditions.csv'))

    # Map the 'End_Condition' to the corresponding colors
    colors = dataframe['End_Condition'].map(end_conditions_color_dict)

    plt.scatter(dataframe['Run'], dataframe["Robustness"], color=colors)

    # Set up a custom legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='reach_stay_no_collision', markerfacecolor='green', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='not_satisfied', markerfacecolor='red', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='not_satisfied', markerfacecolor='red', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='not_satisfied', markerfacecolor='red', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='not_satisfied', markerfacecolor='red', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='not_satisfied', markerfacecolor='red', markersize=10)
    ]
    legend_elements=[Line2D([0], [0], marker='o', color='w', label=key, markerfacecolor=value, markersize=10) for key,value in end_conditions_color_dict.items()]
    
    plt.legend(handles=legend_elements, loc="lower right", title="End Condition")

    plt.xlabel("Run")
    plt.ylabel('Robustness')
    plt.title("Final Robustness per episode")
    plt.savefig(path)
    plt.close()