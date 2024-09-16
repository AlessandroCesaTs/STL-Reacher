import os
import argparse
import pandas as pd

from utils.plotting_utils import plot_train_means,plot_test_values,plot_final_train_robustness, plot_final_train_boolean, plot_final_test_robustness, plot_final_test_boolean

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('--base_path',type=str,default=os.path.join(os.getcwd(),'output'))
    parser.add_argument('--plot_train',action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument('--plot_test',action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument('--num_of_robustnesses',type=int,default=1)
    parser.add_argument('--test_runs',type=int,default=1)

    args=parser.parse_args()
    base_path=args.base_path
    plot_train=args.plot_train
    plot_test=args.plot_test
    num_of_robustnesses=args.num_of_robustnesses

    if plot_train:

        train_logs_path=os.path.join(base_path,'train','logs')

        train_plots_path=os.path.join(base_path,'train','plots')
        os.makedirs(train_plots_path,exist_ok=True)

        test_rewards_dataframe=pd.read_csv(os.path.join(train_logs_path,'rewards.csv'))
        rewards_plot_path=os.path.join(train_plots_path,'rewards.png')

        plot_train_means(test_rewards_dataframe,rewards_plot_path,'Reward')

        for robustness_index in range(num_of_robustnesses):
            path=os.path.join(train_plots_path,f"robustness_{robustness_index}.png")
            dataframe=pd.read_csv(os.path.join(train_logs_path,f"robustness_{robustness_index}.csv"))
            plot_train_means(dataframe,path,'Robustness')
        
        plot_final_train_robustness(train_logs_path,train_plots_path)
        plot_final_train_boolean(train_logs_path,train_plots_path)

    if plot_test:
        test_runs=args.test_runs

        test_logs_path=os.path.join(base_path,'test','logs')

        test_plots_path=os.path.join(base_path,'test','plots')
        os.makedirs(test_plots_path,exist_ok=True)

        test_rewards_dataframe=pd.read_csv(os.path.join(test_logs_path,'rewards.csv'))
        test_robustnesses_dataframes=[pd.read_csv(os.path.join(test_logs_path,f"robustness_{robustness_index}.csv")) for robustness_index in range(num_of_robustnesses)]
        
        plot_final_test_robustness(test_logs_path,test_plots_path)
        plot_final_test_boolean(test_logs_path,test_plots_path)

        for run in range(test_runs):
            test_run_plots_path=os.path.join(test_plots_path,f"run_{run}")
            os.makedirs(test_run_plots_path,exist_ok=True)
            test_run_rewards_plot_path=os.path.join(test_run_plots_path,'rewards.png')
            plot_test_values(test_rewards_dataframe,test_run_rewards_plot_path,'Reward',run)

            for robustness_index in range(num_of_robustnesses):
                path=os.path.join(test_run_plots_path,f"robustness_{robustness_index}.png")
                dataframe=test_robustnesses_dataframes[robustness_index]
                plot_test_values(dataframe,path,'Robustness',run)



             



    



    











    

