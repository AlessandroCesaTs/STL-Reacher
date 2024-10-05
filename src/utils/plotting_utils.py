import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

end_conditions_color_dict={'satisfied':'green','not_satisfied':'red'}

def plot_train_means(dataframe,plots_path,column):

    num_of_envs=dataframe['Environment'].max()+1
    dataframe['Total_Episode']=dataframe['Episode']*num_of_envs+dataframe['Environment']

    dataframe.drop(columns=['Episode'],inplace=True)

    means=dataframe.groupby('Total_Episode')[column].mean().reset_index()

    plt.plot(means['Total_Episode'],means[column])
    plt.xlabel('Episode')
    plt.ylabel(f"Mean {column}")
    plt.title(f"Mean {column} per Episode")
    plt.savefig(plots_path)
    plt.close()


def plot_final_train_robustness(logs_path, plots_path):
    path = os.path.join(plots_path, f"final_state.png")
    dataframe = pd.read_csv(os.path.join(logs_path, f"final_state.csv"))

    num_of_envs = dataframe['Environment'].max() + 1
    dataframe['Total_Episode'] = dataframe['Episode'] * num_of_envs + dataframe['Environment']

    # Map the 'End_Condition' to the corresponding colors
    colors = dataframe['End_Condition'].map(end_conditions_color_dict)

    plt.scatter(dataframe['Total_Episode'], dataframe["Robustness"], color=colors)

    # Set up a custom legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='satisfied', markerfacecolor='green', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='not_satisfied', markerfacecolor='red', markersize=10),
    ]
    
    plt.legend(handles=legend_elements, loc="lower right", title="End Condition")

    plt.xlabel("Episode")
    plt.ylabel('Robustness')
    plt.title("Final Robustness per episode")
    plt.savefig(path)
    plt.close()


def plot_final_train_boolean(logs_path,plots_path):
    path=os.path.join(plots_path,"final_boolean.png")
    dataframe=pd.read_csv(os.path.join(logs_path,"final_boolean.csv"))

    num_of_envs=dataframe['Environment'].max()+1
    dataframe['Total_Episode']=dataframe['Episode']*num_of_envs+dataframe['Environment']
    values=dataframe['Boolean']
    
    plt.scatter(dataframe['Total_Episode'][values > 0], values[values > 0], color='green', label="Formula satisfied")
    plt.scatter(dataframe['Total_Episode'][values <= -0.02], values[values  <= -0.02], color='black', label="Dead")


    plt.xlabel("Episode")
    plt.ylabel('Boolean')
    plt.title("Final Boolean per episode")
    plt.savefig(path)
    plt.close()


def plot_final_test_robustness(logs_path,plots_path):
    path=os.path.join(plots_path,"final_robustness.png")
    dataframe=pd.read_csv(os.path.join(logs_path,"final_robustness.csv"))

    values=dataframe['Robustness']


    plt.scatter(dataframe['Run'][values > 0], values[values > 0], color='green', label="Formula satisfied")
    plt.scatter(dataframe['Run'][values <= 0], values[values <= 0], color='red', label="Formula not satisfied")


    plt.xlabel("Run")
    plt.ylabel('Robustness')
    plt.title(f"Final Robustness per run")

    plt.savefig(path)
    plt.close()

def plot_final_test_boolean(logs_path,plots_path):
    path=os.path.join(plots_path,"final_boolean.png")
    dataframe=pd.read_csv(os.path.join(logs_path,"final_boolean.csv"))

    values=dataframe['Boolean']

    plt.scatter(dataframe['Run'][values > 0], values[values > 0], color='green', label="Formula satisfied")
    plt.scatter(dataframe['Run'][values <= 0], values[values <= 0], color='red', label="Formula not satisfied")


    plt.xlabel("Run")
    plt.ylabel('Boolean')
    plt.title(f"Final Boolean per run")

    plt.savefig(path)
    plt.close()


def plot_test_values(dataframe,plots_path,column,run):

    plt.plot(dataframe[dataframe['Run']==run]['Step'],dataframe[dataframe['Run']==run][column])
    plt.xlabel('Step')
    plt.ylabel(column)
    plt.title(f"{column} per Step at run {run}")
    plt.savefig(plots_path)
    plt.close()
