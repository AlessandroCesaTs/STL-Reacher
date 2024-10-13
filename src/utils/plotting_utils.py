import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

single_end_conditions_color_dict={'reach_stay_no_collision':'green','reach_no_stay_no_collision':'yellow',
                            'reach_stay_collision':'brown','reach_no_stay_collision':'red',
                            'no_reach_no_collision':'gray',
                            'no_reach_collision':'black'}

single_end_labels_dict={'reach_stay_no_collision':"Goal reached, stayed on goal, no collision",'reach_no_stay_no_collision':"Goal reahced, didn't stay on goal, no collision",
                            'reach_stay_collision':"Goal reached, stayed on goal, collision",'reach_no_stay_collision':"Goal reahced, didn't stay on goal, collision",
                            'no_reach_no_collision':"Didn't reach goal, no collision",
                            'no_reach_collision':"Didn't reach goal, collision"}

double_end_conditions_color_dict={'perfect':'green','first_part_completed_but_not_second':'red',
                            'no_part_completed':'black'}
double_end_conditions_labels_dict={'perfect':"Perfect Episode",'first_part_completed_but_not_second':"Completed first part but not second",
                            'no_part_completed':"Not completed first part"}

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


def plot_training_end(logs_path, plots_path,double=False):
    dataframe = pd.read_csv(os.path.join(logs_path, 'end_conditions.csv'))
    robustness_path = os.path.join(plots_path, 'end_robustness.png')

    num_of_envs = dataframe['Environment'].max() + 1
    dataframe['Total_Episode'] = dataframe['Episode'] * num_of_envs + dataframe['Environment']

    # Map the 'End_Condition' to the corresponding colors
    if double:
        colors = dataframe['End_Condition'].map(double_end_conditions_color_dict)
    else:
        colors = dataframe['End_Condition'].map(single_end_conditions_color_dict)

    plt.scatter(dataframe['Total_Episode'], dataframe["Robustness"], color=colors)

    legend_elements=[Line2D([0], [0], marker='o', color='w', label=key, markerfacecolor=value, markersize=10) for key,value in single_end_conditions_color_dict.items()]

    plt.legend(handles=legend_elements, loc="lower right", title="End Condition")

    plt.xlabel("Episode")
    plt.ylabel('Reward')
    plt.title("Final Robustness per episode")
    plt.savefig(robustness_path)
    plt.close()

def plot_test_end_condition(logs_path, plots_path,double=False):
    path = os.path.join(plots_path, 'end_conditions.png')
    dataframe = pd.read_csv(os.path.join(logs_path, 'end_conditions.csv'))

    # Map the 'End_Condition' to the corresponding colors
    if double:
        colors = dataframe['End_Condition'].map(double_end_conditions_color_dict)
    else:
        colors = dataframe['End_Condition'].map(single_end_conditions_color_dict)

    plt.scatter(dataframe['Run'], dataframe["Robustness"], color=colors)

    legend_elements=[Line2D([0], [0], marker='o', color='w', label=key, markerfacecolor=value, markersize=10) for key,value in single_end_conditions_color_dict.items()]
    
    plt.legend(handles=legend_elements, loc="lower right", title="End Condition")

    plt.xlabel("Run")
    plt.ylabel('Robustness')
    plt.title("Final Robustness per episode")
    plt.savefig(path)
    plt.close()
