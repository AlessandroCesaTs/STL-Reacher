import os
import argparse
import pandas as pd

from utils.plotting_utils import plot_training,plot_training_end,plot_test_end_condition

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('--output_path',type=str,default=os.path.join(os.getcwd(),'output'))
    parser.add_argument('--plot_train',action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument('--plot_test',action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument('--double',action=argparse.BooleanOptionalAction,default=True)

    args=parser.parse_args()
    output_path=args.output_path
    plot_train=args.plot_train
    plot_test=args.plot_test
    double=args.double

    if plot_train:

        train_logs_path=os.path.join(output_path,'train','logs')

        train_plots_path=os.path.join(output_path,'train','plots')
        os.makedirs(train_plots_path,exist_ok=True)

        plot_training(train_logs_path,train_plots_path)
        
        plot_training_end(train_logs_path,train_plots_path,double)



    











    

