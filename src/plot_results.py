import os
import argparse
import pandas as pd

from utils.plotting_utils import plot_train_means,plot_test_values,plot_final_train_robustness, plot_final_train_boolean, plot_final_test_robustness, plot_final_test_boolean,plot_train_finals

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('--output_path',type=str,default=os.path.join(os.getcwd(),'output'))
    parser.add_argument('--plot_train',action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument('--plot_test',action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument('--num_of_robustnesses',type=int,default=1)
    parser.add_argument('--test_runs',type=int,default=1)

    args=parser.parse_args()
    output_path=args.output_path
    plot_train=args.plot_train
    plot_test=args.plot_test
    num_of_robustnesses=args.num_of_robustnesses

    if plot_train:

        train_logs_path=os.path.join(output_path,'train','logs')

        train_plots_path=os.path.join(output_path,'train','plots')
        os.makedirs(train_plots_path,exist_ok=True)

        train_rewards_dataframe=pd.read_csv(os.path.join(train_logs_path,'rewards.csv'))
        train_safeties_dataframe=pd.read_csv(os.path.join(train_logs_path,'safeties.csv'))
        rewards_plot_path=os.path.join(train_plots_path,'rewards.png')
        safeties_plot_path=os.path.join(train_plots_path,'safeties.png')

        plot_train_means(train_rewards_dataframe,rewards_plot_path,'Reward')
        plot_train_finals(train_safeties_dataframe,safeties_plot_path,'Robustness')

        for robustness_index in range(num_of_robustnesses):
            path=os.path.join(train_plots_path,f"robustness_{robustness_index}.png")
            dataframe=pd.read_csv(os.path.join(train_logs_path,f"robustness_{robustness_index}.csv"))
            plot_train_finals(dataframe,path,'Robustness')
        
        plot_final_train_robustness(train_logs_path,train_plots_path)
        plot_final_train_boolean(train_logs_path,train_plots_path)

    if plot_test:
        test_runs=args.test_runs

        test_logs_path=os.path.join(output_path,'test','logs')

        test_plots_path=os.path.join(output_path,'test','plots')
        os.makedirs(test_plots_path,exist_ok=True)

        train_rewards_dataframe=pd.read_csv(os.path.join(test_logs_path,'rewards.csv'))
        train_safeties_dataframe=pd.read_csv(os.path.join(test_logs_path,'safeties.csv'))
        test_robustnesses_dataframes=[pd.read_csv(os.path.join(test_logs_path,f"robustness_{robustness_index}.csv")) for robustness_index in range(num_of_robustnesses)]
        
        plot_final_test_robustness(test_logs_path,test_plots_path)
        plot_final_test_boolean(test_logs_path,test_plots_path)

        for run in range(test_runs):
            test_run_plots_path=os.path.join(test_plots_path,f"run_{run}")
            os.makedirs(test_run_plots_path,exist_ok=True)
            test_run_rewards_plot_path=os.path.join(test_run_plots_path,'rewards.png')
            plot_test_values(train_rewards_dataframe,test_run_rewards_plot_path,'Reward',run)

            test_run_safeties_plot_path=os.path.join(test_run_plots_path,'safeties.png')
            plot_test_values(train_safeties_dataframe,test_run_safeties_plot_path,'Robustness',run)


            for robustness_index in range(num_of_robustnesses):
                path=os.path.join(test_run_plots_path,f"robustness_{robustness_index}.png")
                dataframe=test_robustnesses_dataframes[robustness_index]
                plot_test_values(dataframe,path,'Robustness',run)



             



    



    











    

