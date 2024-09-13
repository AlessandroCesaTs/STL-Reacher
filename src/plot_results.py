import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def plot_means(dataframe,plots_path,column):

    num_of_envs=dataframe['Environment'].max()+1
    dataframe['Total_Episode']=dataframe['Episode']*num_of_envs+dataframe['Environment']
    dataframe.drop(columns=['Environment'],inplace=True)

    mean_rewards=dataframe.groupby('Episode')[column].mean().reset_index()

    plt.plot(mean_rewards['Episode'],mean_rewards[column])
    plt.xlabel('Episode')
    plt.ylabel(f"Mean {column}")
    plt.title(f"Mean {column} per Episode")
    plt.savefig(plots_path)
    plt.close()

if __name__=="main":
    parser=argparse.ArgumentParser()
    parser.add_argment('--base_path',type=str)
    parser.add_argment('--num_of_robustnesses',type=str,default=1)

    args=parser.parse_args()
    base_path=args.base_path
    num_of_robustnesses=num_of_robustnesses

    training_logs_path=os.path.join(base_path,'training','logs')

    training_plots_path=os.path.join(base_path,'training','plots')
    os.makedirs(training_plots_path,exist_ok=True)

    rewards_plot_path=os.path.join(training_plots_path,'mean_rewards.png')
    rewards_dataframe=pd.read_csv(os.path.join(training_logs_path),'rewards.csv')

    plot_means(rewards_dataframe,'Reward')

    for robustness_index in range(num_of_robustnesses)

def plot_mean_robustnesses(logs_path,plots_path,robustness_index):
    path=os.path.join(plots_path,f"robustness_{robustness_index}.png")
    df=pd.read_csv(os.path.join(logs_path),f"robustnesses_{robustness_index}.png")



def plot_mean_rewards(logs_path,plots_path):
    path=os.path.join(plots_path,'mean_rewards.png')
    df=pd.read_csv(os.path.join(logs_path),'rewards.csv')

    num_of_envs=df['Environment'].max()+1
    df['Total_Episode']=df['Episode']*num_of_envs+df['Environment']
    df.drop(columns=['Environment'],inplace=True)

    mean_rewards=df.groupby('Episode')['Reward'].mean().reset_index()

    plt.plot(mean_rewards['Episode'],mean_rewards['Reward'])
    plt.xlabel('Episode')
    plt.ylabel('Mean Reward')
    plt.title('Mean Reward per Episode')
    plt.savefig(path)
    plt.close()




def plot_final_train_values(logs_path,plots_path,rob_or_bool):
    path=os.path.join(plots_path,f"final_{rob_or_bool}.png")
    column=rob_or_bool.capitalize()
    df=pd.read_csv(os.path.join(logs_path,f"final_{rob_or_bool}.csv"))

    num_of_envs=df['Environment'].max()+1
    df['Total_Episode']=df['Episode']*num_of_envs+df['Environment']

    values=df[column]

    positive_values=df[values>0]
    negative_values=df[values<=0]

    plt.plot(df['Total_Episode'],positive_values[column],color='green',label="Formula satisfied")
    plt.plot(df['Total_Episode'],negative_values[column],color='red',label="Formula not satisfied")
    plt.xlabel("Episode")
    plt.ylabel(column)
    plt.legend()
    plt.title(f"Final {column} per episode")
    
    plt.savefig(path)
    plt.close()



    

