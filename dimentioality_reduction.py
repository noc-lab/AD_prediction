#! /usr/bin/python3
import os
# os.environ["OMP_NUM_THREADS"] = "4" # export OMP_NUM_THREADS=4
# os.environ["OPENBLAS_NUM_THREADS"] = "4" # export OPENBLAS_NUM_THREADS=4 
# os.environ["MKL_NUM_THREADS"] = "6" # export MKL_NUM_THREADS=6
# os.environ["VECLIB_MAXIMUM_THREADS"] = "4" # export VECLIB_MAXIMUM_THREADS=4
# os.environ["NUMEXPR_NUM_THREADS"] = "6" # export NUMEXPR_NUM_THREADS=6
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import numpy as np
import re
import random
from copy import deepcopy
from pandas import ExcelWriter

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.feature_selection import RFE
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm


class data_generator():
    def __init__(self, sections, data_rs, data_sts, stage, n_year, balanced = False):
        self.sections = sections
        assert stage in ['0vs4','0vs2','2vs4'], "This has to be one of ['0vs4','0vs2','2vs4']"
        self.stage = stage
        self.balanced = balanced
        ## feature names for each section
#         health_var = ['bg','bmi','calc_ldl','cpd','creat','hx_diab','dbp','hdl','hgt','has_hyper','liprx','sbp','trig','vent_rt']
        features = {}
        for section in sections:
            if section != "WAIS":
                features[section] = [col for col in data_sts.columns if section in col]
            else:
                features[section] = [col for col in data_sts.columns if (section in col) & ("WAIS-R" not in col) ]
            features[section][0:0] = ['id_date','diagnosis','age_at_event','closest_mmse_score','education','sex','apoe']
        self.features = features
        
        ## setting the diagnosis based on stage
        if self.stage == '0vs2':
            df0 = pd.read_pickle(f"20220218_20_23_37_0649/data/NtoN_ad{n_year}.pkl")
            df1 = pd.read_pickle(f"20220218_20_23_37_0649/data/NtoMCI_ad{n_year}.pkl")
        elif self.stage == '2vs4':
            df0 = pd.read_pickle(f"20220218_20_23_37_0649/data/MCItoMCI_ad{n_year}.pkl")
            df1 = pd.read_pickle(f"20220218_20_23_37_0649/data/MCItoD_ad{n_year}.pkl")
            
        self.n_id = df0.id_date.unique()
        self.d_id = df1.id_date.unique()
        self.id_date_all = list(self.n_id) + list(self.d_id)
        print(f"class0 = {len(self.n_id)} and class1 = {len(self.d_id)} and total {len(self.id_date_all)}.")
        
        ## intersecting two encoded data over RS and STS methods
        data_sts = data_sts[data_sts['id_date'].isin(data_rs.id_date.unique())]
        data_rs = data_rs[data_rs['id_date'].isin(data_sts.id_date.unique())]
        
        self.data_sts = data_sts[data_sts.id_date.isin(self.id_date_all)]
        self.data_rs = data_rs[data_rs.id_date.isin(self.id_date_all)]
        
        self.id_date_list = self.data_rs["id_date"].unique()
        
        ## Setting the diagnosis based on stage
        sts_diag = self.data_sts["id_date"].apply(lambda x:1 if x in self.d_id else 0)
        self.data_sts.loc[:,"diagnosis"] = sts_diag
        rs_diag = self.data_rs["id_date"].apply(lambda x:1 if x in self.d_id else 0) 
        self.data_rs.loc[:,"diagnosis"] = rs_diag
        
    def get_data(self):
        df = {}
        for section in self.sections:
            df[section] = self.data_sts[self.features[section]].copy()
        return self.data_rs,df
        
    def get_train_test_id(self, kfold = 10, test_ratio = 0.1):        
        
        n_idx = [idx for idx, val in enumerate(self.data_sts.diagnosis) if val==0] 
        d_idx = [idx for idx, val in enumerate(self.data_sts.diagnosis) if val==1] 
        n_id= self.data_sts.iloc[n_idx].id_date.unique()
        d_id= self.data_sts.iloc[d_idx].id_date.unique()
        
        print(f"Sahred ids: {bool(set(n_id) & set(d_id))}. length of normal {len(n_id)} and dementia {len(d_id)}")

        if self.balanced:
            if len(n_id)>=len(d_id):
                undersamples=len(d_id)
                id_list1 = [*n_id[0:undersamples],*d_id]
            else:
                undersamples=len(n_id)
                id_list1 = [*d_id[0:undersamples],*n_id]
        else:
            id_list1 = [*n_id,*d_id]

        random.seed(41)
        random.shuffle(id_list1)
        print(f"length of the IDs for classification is {len(id_list1)}")

        id_label1 = []
        for i in id_list1:
            id_label1.append(self.data_sts[self.data_sts["id_date"]==i].diagnosis.values[0])

        sss = StratifiedShuffleSplit(n_splits = kfold, test_size = test_ratio, random_state = 42)
        train_id1 = []
        test_id1 = []
        for train_index, test_index in sss.split(range(len(id_label1)), id_label1):
            train_id1.append(train_index)
            test_id1.append(test_index)

        return id_list1, train_id1, test_id1
    
def metrics_cal(y_pred,y_test):
    fpr, tpr, thresholds = metrics.roc_curve(np.array(y_test), np.array(y_pred),drop_intermediate = False)
    gmeans = np.sqrt(tpr * (1-fpr))
    ix = np.argmax(gmeans)
    threshold = thresholds[ix]*0.999

    auc = metrics.auc(fpr,tpr)
    acc = metrics.accuracy_score(y_test, (1*(np.array(y_pred)>threshold)).tolist())

    tn, fp, fn, tp = metrics.confusion_matrix(np.array(y_test), np.array(y_pred)>threshold).ravel()
    spe = tn / (tn+fp)
    sen = tp / (tp+fn)
    ppv = tp / (tp+fp)
    npv = tn / (tn+fn)
    
    return [auc,acc,sen,spe,ppv,npv]


def random_sampling_score(id_list_,data_,train_id_,test_id_,stage,folder_name,num_,n=10):
    
    df_list = list()
    id_list = np.array(id_list_)
#     health_var = ['bg','bmi','calc_ldl','cpd','creat','hx_diab','dbp','hdl','hgt','has_hyper','liprx','sbp','trig','vent_rt']
#     health_cat = ['hx_diab','has_hyper','liprx']
    col_to_drop = ['id_date','diagnosis','age_at_event','closest_mmse_score','education','sex','apoe'] 
    mat_avg = [{'results_rs':[]} for i in range(6)]
    
    for m in range(n):
        datacp = deepcopy(data_)
        test = datacp[datacp['id_date'].isin(id_list[test_id_[m]])]
        train = datacp[datacp['id_date'].isin(id_list[train_id_[m]])]
        y_train = train.diagnosis
        x_train = train.drop(columns=col_to_drop)
        y_test = test.diagnosis
        x_test = test.drop(columns=col_to_drop)


        if num_ != 512:
            X = x_train.values
            Y = y_train.values
            model = LogisticRegression(solver='lbfgs',max_iter=500)
            rfe = RFE(model,n_features_to_select = num_)
            fit = rfe.fit(X, Y)
            arr = x_train.keys().array[0:512]
            remove_col=np.ma.masked_array(arr, fit.support_)
            x_train = x_train.drop(columns=remove_col[~remove_col.mask].data)
            x_test = x_test.drop(columns=remove_col[~remove_col.mask].data)
            test = test.drop(columns=remove_col[~remove_col.mask].data)
            datacp = datacp.drop(columns=remove_col[~remove_col.mask].data)


        model_log = LogisticRegression(random_state=0)
        y_train = y_train.astype('int')
        model_log.fit(x_train, y_train)
        
        y_pred=[]
        y_test=[]
        for i in test['id_date'].unique():
            pred_arr = model_log.predict_proba(test[test.id_date==i].drop(columns=col_to_drop))
            label = test[test.id_date==i]['diagnosis'].tolist()[0]
            y_pred.append(np.mean(pred_arr[:,1]))
            y_test.append(label)

        mat = metrics_cal(y_pred,y_test)
        for idx,performance in enumerate(mat):
            mat_avg[idx]['results_rs'].append(performance)
            
        

        lis1 = col_to_drop + ["rs_score"]
        data_s = pd.DataFrame(columns=lis1) 
        arr = []
        for id_date in datacp['id_date']:
            pred_arr = model_log.predict_proba(datacp[datacp.id_date==id_date].drop(columns=col_to_drop))
            arr = []
            arr.extend(datacp.loc[datacp.id_date == id_date,col_to_drop].values[0].tolist())
            arr.append(np.mean(pred_arr[:,1]))
            data_s = data_s.merge(pd.Series(arr, index=lis1).to_frame().T,how='outer')
            
        newpath = f"{folder_name}/{stage}"
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        data_s.to_pickle(f"{newpath}/df_rs{m}")

    print(f"AUC is {np.mean(mat_avg[0]['results_rs'])}")
    
    df_res = pd.DataFrame(mat_avg,index=['AUC','Accuracy','Sensitivity','Specificity','PPV','NPV'])
    
    return df_res


def Sections_score(id_list_,data_,train_id_,test_id_,sections_,stage,folder_name,num_,n=10): 
    
    stg = ['0vs4','0vs2','2vs4']
    assert stage in stg, f"This has to be one of {stg}"
    
    df_list = list()
    id_list = np.array(id_list_)
#     health_var = ['bg','bmi','calc_ldl','cpd','creat','hx_diab','dbp','hdl','hgt','has_hyper','liprx','sbp','trig','vent_rt']
    col_to_drop = ['id_date','diagnosis','age_at_event','closest_mmse_score','education','sex','apoe']
    
    perf = ["AUC",'Acc','Sen','Spe','PPV','NPV']
    mat_avg = {metrics:{sec:[] for sec in sections_} for metrics in perf}
    
    for m in range(n):
        for section in sections_:
            datacp = data_[section].copy()
            test = datacp[datacp['id_date'].isin(id_list[test_id_[m]])]
            train = datacp[datacp['id_date'].isin(id_list[train_id_[m]])]
            y_train = train.diagnosis
            x_train = train.drop(columns=col_to_drop)
            y_test = test.diagnosis
            x_test = test.drop(columns=col_to_drop)


            if num_ != 512:
                X = x_train.values
                Y = y_train.values
                model = LogisticRegression(solver='lbfgs',max_iter=500)
                rfe = RFE(model,n_features_to_select = num_)
                fit = rfe.fit(X, Y)
                arr = x_train.keys().array[0:512]
                remove_col=np.ma.masked_array(arr, fit.support_)
                x_train = x_train.drop(columns=remove_col[~remove_col.mask].data)
                x_test = x_test.drop(columns=remove_col[~remove_col.mask].data)
                datacp = datacp.drop(columns=remove_col[~remove_col.mask].data)


            model_log = LogisticRegression(random_state=0)
            y_train = y_train.astype('int')
            model_log.fit(x_train, y_train)
            
            y_pred = model_log.predict_proba(x_test)
            mat = metrics_cal(y_pred[:,1],y_test)
            for idx,performance in enumerate(mat):
                mat_avg[perf[idx]][section].append(performance)

            if section == sections_[0]:
                lis1 = col_to_drop + [section]
                data_s = pd.DataFrame(columns=lis1) 
            else: 
                lis1 = ['id_date',section]
                data_s = pd.DataFrame(columns=lis1) 
            arr = []
            for id_date in datacp['id_date']:
                pred_arr = model_log.predict_proba(datacp[datacp['id_date']==id_date].drop(columns=col_to_drop))
                if section == sections_[0]:
                    data_sections = pd.DataFrame(columns=['id_date'])
                    diag = datacp[datacp['id_date']==id_date].diagnosis.values[0]
                    arr = []
                    arr.extend(datacp.loc[datacp.id_date == id_date,col_to_drop].values[0].tolist())
                    arr.append(pred_arr[0][1])
                    data_s = data_s.merge(pd.Series(arr, index=lis1).to_frame().T,how='outer')
                else:
                    arr = [id_date, pred_arr[0][1]]
                    data_s = data_s.merge(pd.Series(arr, index=lis1).to_frame().T,how='outer')
                    
            data_sections = pd.merge(data_sections, data_s,how='outer', on = 'id_date')
            
        newpath = f"{folder_name}/{stage}"
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        data_sections.to_pickle(f"{newpath}/df{m}")
    result = [mat_avg[p] for p in perf]
    
    auc_max = sorted([(s,np.mean(result[0][s])) for s in sections_],key=lambda x:x[1])[-1]
    print(f"Highest AUC is {auc_max}")
    
    df_res = pd.DataFrame(result,index=['AUC','Accuracy','Sensitivity','Specificity','PPV','NPV'])
    
    return df_res

def section_classifier(id_list_,train_id_,test_id_,sections_,folder_name,selected_section,stage,n=10):

    stg = ['0vs4','0vs2','2vs4']
    assert stage in stg, f"This has to be one of {stg}"
    
    id_list = np.array(id_list_)
    
    mat_avg = [{'results_sts':[]} for i in range(6)]
#     health_var = ['bg','bmi','calc_ldl','cpd','creat','hx_diab','dbp','hdl','hgt','has_hyper','liprx','sbp','trig','vent_rt']
    col_to_drop = ['id_date','diagnosis','age_at_event','closest_mmse_score','education','sex','apoe'] 
    
    lis1 = [section for section in sections_]
    lis1[0:0] = col_to_drop

    for m in range(n):
        data_sec1 = pd.read_pickle(f"{folder_name}/{stage}/df{m}")
        
        data_combined = pd.DataFrame(columns=lis1)  
        arr = []
        for i in data_sec1['id_date'].unique():
            #label = data_sec1[data_sec1['id_date']==i]['diagnosis'].tolist()[0]
            arr = data_sec1.loc[data_sec1.id_date == i,col_to_drop].values[0].tolist()
            if len(data_sec1.loc[data_sec1.id_date == i,sections_]):
                arr.extend(data_sec1.loc[data_sec1.id_date == i,sections_].values[0].tolist())
            else:
                arr.extend(np.tile(np.nan,len(sections_)).tolist())
                
            data_combined = data_combined.merge(pd.Series(arr, index=lis1).to_frame().T,how='outer')

        data_combined.fillna((data_combined.mean()), inplace=True)

        for vari in lis1[2:]:
            try:
                data_combined[vari] = data_combined[vari].apply(lambda a: a/data_combined[vari].std() - data_combined[vari].mean()/data_combined[vari].std())
            except ZeroDivisionError:
                pass

        test_combined=data_combined[data_combined['id_date'].isin(id_list[test_id_[m]])]
        
        train_combined=data_combined[data_combined['id_date'].isin(id_list[train_id_[m]])]
        y_train=train_combined.diagnosis

        x_train=np.array(train_combined[selected_section]).reshape(-1, len(selected_section))
        y_test=test_combined.diagnosis
        x_test=np.array(test_combined[selected_section]).reshape(-1, len(selected_section))

        model_last = LogisticRegression(solver='liblinear', random_state=0,C=1)
        #model_last = RandomForestClassifier(random_state=0)
        y_train = y_train.astype('int')
        model_last.fit(x_train, y_train)
        y_test = y_test.astype('int')

        y_pred = model_last.predict_proba(np.asarray(x_test))

        mat = metrics_cal(y_pred[:,1],y_test)
        for idx,performance in enumerate(mat):
            mat_avg[idx]['results_sts'].append(performance)
    
    print(f"Highest AUC is {np.mean(mat_avg[0]['results_sts'])} for {selected_section}")
    
    df_res = pd.DataFrame(mat_avg,index=['AUC','Accuracy','Sensitivity','Specificity','PPV','NPV'])
    return df_res
    
def main(n_year,num_f):
    sections = ['DEMO','WMS','OTHER','FAS','BNT','CLOCK_DRAWING','WAIS','WAIS-R']
    data1 = pd.read_pickle('20220218_20_23_37_0649/data/encoded_rs_dem.pkl')
    data2 = pd.read_pickle('20220218_20_23_37_0649/data/encoded_sts_dem.pkl')
    
    stages = ['0vs2','2vs4']
    folder_name_main = f'20220218_20_23_37_0649/val_folder_ad{n_year}/optimizing'
    bal = False
#     num_f = [512,256,128,64,16] 
    
    for stage in tqdm(stages, desc='optmizing',position=0):
        data_class = data_generator(sections, data1, data2, stage, n_year, balanced = bal)
        data_rs, df_sts = data_class.get_data()
        id_list1, train_id1, test_id1 = data_class.get_train_test_id(kfold = 10, test_ratio = 0.1)
            
        train_id2 = []
        valid_id2 = []
        for f in range(len(train_id1)):
            random.seed(41)
            train_id2.append(np.array(random.sample(list(train_id1[f]), int(len(train_id1[f])*0.9))))
            valid_id2.append(np.array([ids for ids in train_id1[f] if ids not in train_id2[f]]))
        
        for num in num_f:
            print('num-features'+str(num))
            folder_name = f"{folder_name_main}/num_features{num}"
            
            res = random_sampling_score(id_list1,data_rs,train_id2,valid_id2,stage,folder_name,num,n=10)
            res_path = f'{folder_name}/{stage}/results_rs.pkl'
            res.to_pickle(res_path)

            res = Sections_score(id_list1,df_sts,train_id2,valid_id2,sections,stage,folder_name,num,n=10)
            res_path = f'{folder_name}/{stage}/results_subtests.pkl'
            res.to_pickle(res_path)

            sec_sel = [section for section in sections]
            res = section_classifier(id_list1,train_id2,valid_id2,sections,folder_name,sec_sel,stage,n=10)

            res_path = f'{folder_name}/{stage}/results_sts.pkl'
            res.to_pickle(res_path)
          
          
    perf1 = {}
    perf2 = {}
    for stage in stages:
        perf1[stage] = []
        perf2[stage] = []
        for num in num_f:
            folder_name = f"{folder_name_main}/num_features{num}"
            res_path = f'{folder_name}/{stage}/results_rs.pkl'
            df_rs = pd.read_pickle(res_path)
            res1 = df_rs.applymap(np.mean)
            perf1[stage].append(np.round(res1.iloc[0,0],3))
            res_path = f'{folder_name}/{stage}/results_sts.pkl'
            df_sts = pd.read_pickle(res_path)
            res3 = df_sts.applymap(np.mean)
            perf2[stage].append(np.round(res3.iloc[0,0],3))

    df = pd.DataFrame(perf1)
    res_path = f'{folder_name_main}/df_perf_rs.pkl'
    df.to_pickle(res_path)
    df = pd.DataFrame(perf2)
    res_path = f'{folder_name_main}/df_perf_sts.pkl'
    df.to_pickle(res_path)
    print(f'Done for optimization of folder_ad{n_year}!')


if __name__ == "__main__":
#     n_year = 5
#     num_f = [512,256,128,64,16] 
    main(n_year,num_f)
    

