#! /usr/bin/python3
import os
os.environ["OMP_NUM_THREADS"] = "4" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "6" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6" # export NUMEXPR_NUM_THREADS=6
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import numpy as np
import time
from tqdm import tqdm
import random
import dimentioality_reduction
import subtest_feature_selection
import prediction




if __name__ == "__main__":
#     num_f = [256,64,16]
    num_f = [512,256,128,64,16]
    years_list = [6]
    for n_y in tqdm(years_list):
        # dimentionality reduction for both sub-test scores and the randomly selected segment of the transcript
        dimentioality_reduction.main(n_y,num_f)
        # results are saved in the following folder and they can be restored through the lines below
        res_path = f'val_folder_ad{n_y}/optimizing/df_perf_rs.pkl'
        perf1 = pd.read_pickle(res_path)
        res_path = f'val_folder_ad{n_y}/optimizing/df_perf_sts.pkl'
        perf2 = pd.read_pickle(res_path)
        
        # Finding the dimention giving the highest performance on the validation
        s1_rs = num_f[np.argmax(perf1["0vs2"])]
        s1_sts = num_f[np.argmax(perf2["0vs2"])]
        s2_rs = num_f[np.argmax(perf1["2vs4"])]
        s2_sts = num_f[np.argmax(perf2["2vs4"])]
#         print(f"for year {n_y}")
        # The tuned dimention
        print(s1_rs,s1_sts,s2_rs,s2_sts)
        
        # sub-test selection by performace error analysis 
        subtest_feature_selection.main(s1_rs,s1_sts,s2_rs,s2_sts,n_y)
        # downloading the the results of feature selection to use in the prediction process
        res_path = f'val_folder_ad{n_y}/feature_extraction/df_extracted_features.pkl'
        df_feature = pd.read_pickle(res_path)
        # solving different classification models using available features -- the results will be saved in an excel sheet
        prediction.main(s1_rs,s1_sts,s2_rs,s2_sts,n_y,df_feature)
        
        
