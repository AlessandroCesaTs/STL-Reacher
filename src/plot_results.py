import os
import argparse
import pandas as pd

from utils.plotting_utils import plot_train_rewards,plot_test_rewards,plot_final_test_robustness, plot_test_final_state, plot_final_test_boolean

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('--output_path',type=str,default=os.path.join(os.getcwd(),'output'))
    parser.add_argument('--plot_train',action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument('--plot_test',action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument('--test_runs',type=int,default=1)

    args=parser.parse_args()
    output_path=args.output_path
    plot_train=args.plot_train
    plot_test=args.plot_test

    if plot_train:

        train_logs_path=os.path.join(output_path,'train','logs')

        train_plots_path=os.path.join(output_path,'train','plots')
        os.makedirs(train_plots_path,exist_ok=True)

        train_rewards_dataframe=pd.read_csv(os.path.join(train_logs_path,'rewards.csv'))
        train_safeties_dataframe=pd.read_csv(os.path.join(train_logs_path,'safeties.csv'))
        rewards_plot_path=os.path.join(train_plots_path,'rewards.png')

        plot_train_rewards(train_rewards_dataframe,rewards_plot_path,'Reward')
        
        plot_final_test_robustness(train_logs_path,train_plots_path)

    if plot_test:
        test_runs=args.test_runs

        test_logs_path=os.path.join(output_path,'test','logs')

        test_plots_path=os.path.join(output_path,'test','plots')
        os.makedirs(test_plots_path,exist_ok=True)

        train_rewards_dataframe=pd.read_csv(os.path.join(test_logs_path,'rewards.csv'))
        train_final_state_dataframe=pd.read_csv(os.path.join(test_logs_path,'final_state.csv'))
        
        plot_test_final_state(test_logs_path,test_plots_path)

        for run in range(test_runs):
            test_run_plots_path=os.path.join(test_plots_path,f"run_{run}")
            os.makedirs(test_run_plots_path,exist_ok=True)
            test_run_rewards_plot_path=os.path.join(test_run_plots_path,'rewards.png')
            plot_test_rewards(train_rewards_dataframe,test_run_rewards_plot_path,run)
             



    



    











    

