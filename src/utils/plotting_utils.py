import os
import matplotlib.pyplot as plt
import pandas as pd

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


def plot_final_train_values(logs_path,plots_path,rob_or_bool):
    path=os.path.join(plots_path,f"final_{rob_or_bool}.png")
    column=rob_or_bool.capitalize()
    dataframe=pd.read_csv(os.path.join(logs_path,f"final_{rob_or_bool}.csv"))

    num_of_envs=dataframe['Environment'].max()+1
    dataframe['Total_Episode']=dataframe['Episode']*num_of_envs+dataframe['Environment']
    values=dataframe[column]

    """

    for i in range(len(values) - 1):
        x_values = dataframe['Total_Episode'].iloc[i:i+2]
        y_values = values.iloc[i:i+2]
        
        # Color red if both values are <= 0, otherwise red
        color = 'red' if all(y <=0 for y in y_values) else 'green'
        plt.plot(x_values, y_values, color=color)
    """

    #plt.plot(dataframe['Total_Episode'],values)
    plt.scatter(dataframe['Total_Episode'][values > 0], values[values > 0], color='green', label="Formula satisfied")
    plt.scatter(dataframe['Total_Episode'][values <= 0], values[values <= 0], color='red', label="Formula not satisfied")


    plt.xlabel("Episode")
    plt.ylabel(column)
    plt.title(f"Final {column} per episode")
    plt.legend()
    plt.savefig(path)
    plt.close()


def plot_final_test_values(logs_path,plots_path,rob_or_bool):
        path=os.path.join(plots_path,f"final_{rob_or_bool}.png")
        column=rob_or_bool.capitalize()
        dataframe=pd.read_csv(os.path.join(logs_path,f"final_{rob_or_bool}.csv"))

        values=dataframe[column]


        """
        for i in range(len(values)-1):
            x_values=dataframe['Run'].iloc[i:i+2]
            y_values = values.iloc[i:i+2]
        
            # Color red if both values are <= 0, otherwise red
            color = 'red' if all(y <=0 for y in y_values) else 'green'
            plt.plot(x_values, y_values, color=color)
        """

        plt.scatter(dataframe['Run'][values > 0], values[values > 0], color='green', label="Formula satisfied")
        plt.scatter(dataframe['Run'][values <= 0], values[values <= 0], color='red', label="Formula not satisfied")


        plt.xlabel("Run")
        plt.ylabel(column)
        plt.title(f"Final {column} per run")

        plt.savefig(path)
        plt.close()


def plot_test_values(dataframe,plots_path,column,run):

    plt.plot(dataframe[dataframe['Run']==run]['Step'],dataframe[dataframe['Run']==run][column])
    plt.xlabel('Step')
    plt.ylabel(column)
    plt.title(f"{column} per Step at run {run}")
    plt.savefig(plots_path)
    plt.close()