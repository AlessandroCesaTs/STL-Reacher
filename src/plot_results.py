import os
import argparse
import pandas as pd

from utils.plotting_utils import plot_train_rewards,plot_train_end_condition,plot_test_end_condition

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('--output_path',type=str,default=os.path.join(os.getcwd(),'output'))
    parser.add_argument('--plot_train',action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument('--plot_test',action=argparse.BooleanOptionalAction,default=True)

    args=parser.parse_args()
    output_path=args.output_path
    plot_train=args.plot_train
    plot_test=args.plot_test

    if plot_train:

        train_logs_path=os.path.join(output_path,'train','logs')

        train_plots_path=os.path.join(output_path,'train','plots')
        os.makedirs(train_plots_path,exist_ok=True)


        plot_train_rewards(train_logs_path,train_plots_path)
        
        plot_train_end_condition(train_logs_path,train_plots_path)

    if plot_test:

        test_logs_path=os.path.join(output_path,'test','logs')

        test_plots_path=os.path.join(output_path,'test','plots')
        os.makedirs(test_plots_path,exist_ok=True)
        
        plot_test_end_condition(test_logs_path,test_plots_path)



    



    











    

