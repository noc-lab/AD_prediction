#! /usr/bin/python3
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
import pandas as pd
import os
# os.environ["OMP_NUM_THREADS"] = "4" # export OMP_NUM_THREADS=4
# os.environ["OPENBLAS_NUM_THREADS"] = "4" # export OPENBLAS_NUM_THREADS=4 
# os.environ["MKL_NUM_THREADS"] = "6" # export MKL_NUM_THREADS=6
# os.environ["VECLIB_MAXIMUM_THREADS"] = "4" # export VECLIB_MAXIMUM_THREADS=4
# os.environ["NUMEXPR_NUM_THREADS"] = "6" # export NUMEXPR_NUM_THREADS=6
import numpy as np
import re
import random
from copy import deepcopy
from pandas import ExcelWriter
import time
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.feature_selection import RFE
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm
from sklearn.metrics import f1_score


class data_generator():
    def __init__(self, sections, data_rs, data_sts, stage, n_years, balanced = False):
        self.sections = sections
        assert stage in ['0vs4','0vs2','2vs4'], "This has to be one of ['0vs4','0vs2','2vs4']"
        self.stage = stage
        self.balanced = balanced
        ## feature names for each section
        ext_var = ['bg','bmi','calc_ldl','cpd','creat','hx_diab','dbp','hdl','hgt','has_hyper','liprx','sbp','trig','vent_rt']
        features = {}
        for section in sections:
            if section != "WAIS":
                features[section] = [col for col in data_sts.columns if section in col]
            else:
                features[section] = [col for col in data_sts.columns if (section in col) & ("WAIS-R" not in col) ]
            features[section][0:0] = ['id_date','diagnosis','age_at_event','closest_mmse_score','education','sex','apoe'] + ext_var
        self.features = features
        
        ## setting the diagnosis based on stage
        if self.stage == '0vs2':
            df0 = pd.read_pickle(f"data/NtoN_ad{n_years}.pkl")
            df1 = pd.read_pickle(f"data/NtoMCI_ad{n_years}.pkl")
        elif self.stage == '2vs4':
            df0 = pd.read_pickle(f"data/MCItoMCI_ad{n_years}.pkl")
            df1 = pd.read_pickle(f"data/MCItoD_ad{n_years}.pkl")
#         if self.stage == '0vs2':
#             df0 = pd.read_pickle(f"data/NtoN_7ad.pkl")
#             df1 = pd.read_pickle(f"data/NtoMCI_7ad.pkl")
#         elif self.stage == '2vs4':
#             df0 = pd.read_pickle(f"data/MCItoMCI_5ad.pkl")
#             df1 = pd.read_pickle(f"data/MCItoD_5ad.pkl")
        else:
            df0 = pd.read_pickle("data/NtoN12.pkl")
            df1 = pd.read_pickle("data/NtoD.pkl")
            
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
        
        print(f"Shared ids: {bool(set(n_id) & set(d_id))}. Length of normal {len(n_id)} and dementia {len(d_id)}")

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
#     threshold = 0.5
    
#     threshold = thresholds[np.argmax([f1_score(y_test, (1*(np.array(y_pred)>thr)).tolist()) for thr in thresholds])]
    f1 = f1_score(y_test, (1*(np.array(y_pred)>threshold)).tolist())

    auc = metrics.auc(fpr,tpr)
    acc = metrics.accuracy_score(y_test, (1*(np.array(y_pred)>threshold)).tolist())

    tn, fp, fn, tp = metrics.confusion_matrix(np.array(y_test), np.array(y_pred)>threshold).ravel()
    spe = tn / (tn+fp)
    sen = tp / (tp+fn)
    ppv = tp / (tp+fp)
    npv = tn / (tn+fn)
    
    return [auc,acc,sen,spe,ppv,npv,f1]


def random_sampling_scroe(id_list_,data_,train_id_,test_id_,num_,folder_name,stage,n=10):
    
    df_list = list()
    id_list = np.array(id_list_)
    ext_var = ['bg','bmi','calc_ldl','cpd','creat','hx_diab','dbp','hdl','hgt','has_hyper','liprx','sbp','trig','vent_rt']
    col_to_drop = ['id_date','diagnosis','age_at_event','closest_mmse_score','education','sex','apoe'] + ext_var
    
    mat_avg = [{'results_rs':[]} for i in range(7)]
    
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
        # if not os.path.exists(newpath):
        #     os.makedirs(newpath)
        data_s.to_pickle(f"{newpath}/df_rs{m}")

    print(f"AUC is {np.mean(mat_avg[0]['results_rs'])}")
    
    df_res = pd.DataFrame(mat_avg,index=['AUC','Accuracy','Sensitivity','Specificity','PPV','NPV','F1'])
    
    return df_res
          
def Sections_score(id_list_,data_,train_id_,test_id_,sections_,stage,folder_name,num_,n=10): 
    
    stg = ['0vs4','0vs2','2vs4']
    assert stage in stg, f"This has to be one of {stg}"
    
    df_list = list()
    id_list = np.array(id_list_)
    ext_var = ['bg','bmi','calc_ldl','cpd','creat','hx_diab','dbp','hdl','hgt','has_hyper','liprx','sbp','trig','vent_rt']
    col_to_drop = ['id_date','diagnosis','age_at_event','closest_mmse_score','education','sex','apoe'] + ext_var
    
    perf = ["AUC",'Acc','Sen','Spe','PPV','NPV','F1']
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
        # if not os.path.exists(newpath):
        #     os.makedirs(newpath)
        data_sections.to_pickle(f"{newpath}/df{m}")
    result = [mat_avg[p] for p in perf]
    
    auc_max = sorted([(s,np.mean(result[0][s])) for s in sections_],key=lambda x:x[1])[-1]
    print(f"Highest AUC is {auc_max}")
    
    df_res = pd.DataFrame(result,index=['AUC','Accuracy','Sensitivity','Specificity','PPV','NPV','F1'])
    
    return df_res

def selected_section_classifier(id_list_,train_id_,test_id_,sections_,folder_name,selected_section,stage,flag,n=10):

    stg = ['0vs4','0vs2','2vs4']
    assert stage in stg, f"This has to be one of {stg}"
    
    id_list = np.array(id_list_)
    
    mat_avg = [{'results_ensemble':[]} for i in range(7)]
    ext_var = ['bg','bmi','calc_ldl','cpd','creat','hx_diab','dbp','hdl','hgt','has_hyper','liprx','sbp','trig','vent_rt']
    col_to_drop = ['id_date','diagnosis','age_at_event','closest_mmse_score','education','sex','apoe'] + ext_var
    
    lis1 = [section for section in sections_]
    lis1[0:0] = ["rs_score"]
    lis1[0:0] = col_to_drop
    
    cat_var = ['education','sex','apoe','hx_diab','has_hyper','liprx']
#     cat_var = ['education','sex','apoe']
    dum_cols = list(set(cat_var) & set(selected_section))
    
    x_test_id = []
    predictions = []
    coeff = []

    #for m in tqdm(range(n), desc='1st loop-fcn2',position=0):
    for m in range(n):
        data_sec1 = pd.read_pickle(f"{folder_name}/{stage}/df{m}")
        data_rs = pd.read_pickle(f"{folder_name}/{stage}/df_rs{m}")
        data_sec1["education"] = data_sec1["education"].apply(lambda a: int(a>=1))
        data_rs["education"] = data_rs["education"].apply(lambda a: int(a>=1))
        
        data_combined = pd.DataFrame(columns=lis1)  
        
        arr = []
        for i in data_sec1['id_date'].unique():
            arr = data_sec1.loc[data_sec1.id_date == i,col_to_drop].values[0].tolist()
            if len(data_rs.loc[data_rs.id_date == i,"rs_score"]):
                arr.append(data_rs.loc[data_rs.id_date == i,"rs_score"].values[0])
            else:
                arr.append(np.nan)
            if len(data_sec1.loc[data_sec1.id_date == i,sections_]):
                arr.extend(data_sec1.loc[data_sec1.id_date == i,sections_].values[0].tolist())
            else:
                arr.extend(np.tile(np.nan,len(sections_)).tolist())
            data_combined = data_combined.merge(pd.Series(arr, index=lis1).to_frame().T,how='outer')
            
        data_combined['education'].replace(np.nan,0,inplace=True)
        data_combined['sex'].replace(np.nan,2,inplace=True)
        data_combined['apoe'].replace(np.nan,33,inplace=True)
        data_combined['hx_diab'].replace(np.nan,0.0,inplace=True)
        data_combined['has_hyper'].replace(np.nan,1,inplace=True)
        data_combined['liprx'].replace(np.nan,0,inplace=True)
        data_combined.fillna((data_combined.mean()), inplace=True)
        for vari in set(lis1)-set(['id_date','diagnosis']+cat_var):
            try:
                data_combined[vari] = data_combined[vari].apply(lambda a: a/data_combined[vari].std() - data_combined[vari].mean()/data_combined[vari].std())
            except ZeroDivisionError:
                pass
        print(f"education labels in data combined {data_combined.education.unique()}")
        if len(dum_cols)!=0:
            data_combined = pd.get_dummies(data_combined, columns=dum_cols)
            all_col = data_combined.columns.tolist()
#             print(f"all_col {all_col}")
            selcted_col_cat = [col for col in all_col for dum_col in dum_cols if col.startswith(dum_col)]
#             print(f"selcted_col_cat {selcted_col_cat}")
            selected_section2 = list(set(selected_section) - set(dum_cols)) + selcted_col_cat
#             print(f"selected_section2 {selected_section2}")
        else:
            selected_section2 = selected_section

        data_combined.fillna((data_combined.mean()), inplace=True)
        
        test_combined=data_combined[data_combined['id_date'].isin(id_list[test_id_[m]])]
        train_combined=data_combined[data_combined['id_date'].isin(id_list[train_id_[m]])]
        y_train=train_combined.diagnosis

        x_train = np.array(train_combined[selected_section2]).reshape(-1, len(selected_section2))
        y_test = test_combined.diagnosis
        x_test = np.array(test_combined[selected_section2]).reshape(-1, len(selected_section2))
        x_test_id.extend(test_combined["id_date"].tolist())

        model_last = LogisticRegression(solver='liblinear', random_state=0,C=1)
        #model_last = RandomForestClassifier(random_state=0)
        #model_last = LGBMClassifier(random_state=0)
        y_train = y_train.astype('int')
        model_last.fit(x_train, y_train)
        y_test = y_test.astype('int')

        y_pred = model_last.predict_proba(np.asarray(x_test))
        predictions.extend(y_pred[:,1])
        coeff.append(model_last.coef_[0])

        mat = metrics_cal(y_pred[:,1],y_test)
        for idx,performance in enumerate(mat):
            mat_avg[idx]['results_ensemble'].append(performance)
            
        lis3 = ['id_date','diagnosis',"probelty"]
        data_s = pd.DataFrame(columns=lis3) 
        arr = []
        for id_date in data_combined['id_date']:
            train_test = np.array(data_combined[data_combined.id_date==id_date][selected_section2]).reshape(-1, len(selected_section2))
            pred_arr = model_last.predict_proba(train_test)
            arr = []
            arr.extend(data_combined.loc[data_combined.id_date == id_date,['id_date','diagnosis']].values[0].tolist())
            arr.append(np.mean(pred_arr[:,1]))
            data_s = data_s.merge(pd.Series(arr, index=lis3).to_frame().T,how='outer')

        newpath = f"{folder_name}/{stage}"
        # if not os.path.exists(newpath):
        #     os.makedirs(newpath)
        if flag == "voice":
          data_s.to_pickle(f"{newpath}/df_voice{m}")
        elif flag == "demo_apoe":
          data_s.to_pickle(f"{newpath}/df_demo_apoe{m}")
        elif flag == "demo":
          data_s.to_pickle(f"{newpath}/df_demo{m}")
        else:
          data_s.to_pickle(f"{newpath}/df_health{m}")
    
    print(f"Highest AUC is {np.mean(mat_avg[0]['results_ensemble'])} for {selected_section2}")
    predictions_id = pd.DataFrame({'id_date': x_test_id, 'score': predictions})
    df_res = pd.DataFrame(mat_avg,index=['AUC','Accuracy','Sensitivity','Specificity','PPV','NPV','F1'])
    return df_res,predictions_id,coeff,selected_section2
          
def selected_section_classifier2(id_list_,train_id_,test_id_,sections_,folder_name,selected_section,stage,apoe=0,n=10):

    stg = ['0vs4','0vs2','2vs4']
    assert stage in stg, f"This has to be one of {stg}"
    
    id_list = np.array(id_list_)
    
    mat_avg = [{'results_ensemble':[]} for i in range(7)]
    col_to_drop = ['id_date','diagnosis']
    
    #
    lis1 = []
    lis1[0:0] = ["voice","demo","health"]
    lis1[0:0] = col_to_drop
    
    x_test_id = []
    predictions = []
    coeff = []

    #for m in tqdm(range(n), desc='1st loop-fcn2',position=0):
    for m in range(n):
        data_sec1 = pd.read_pickle(f"{folder_name}/{stage}/df_voice{m}")
        data_demo = pd.read_pickle(f"{folder_name}/{stage}/df_demo{m}")
        data_health = pd.read_pickle(f"{folder_name}/{stage}/df_health{m}")
        if apoe==1:
          data_demo = pd.read_pickle(f"{folder_name}/{stage}/df_demo_apoe{m}")

        
        data_combined = pd.DataFrame(columns=lis1)  
        arr = []
        for i in data_sec1['id_date'].unique():
            arr = data_sec1.loc[data_sec1.id_date == i,col_to_drop].values[0].tolist()
            if len(data_sec1.loc[data_sec1.id_date == i,"probelty"]):
                arr.append(data_sec1.loc[data_sec1.id_date == i,"probelty"].values[0])
            else:
                arr.append(np.nan)
                print(f"voice score is nan!!!!!!")
            if len(data_demo.loc[data_demo.id_date == i,"probelty"]):
                arr.append(data_demo.loc[data_demo.id_date == i,"probelty"].values[0])
            else:
                arr.append(np.nan)
                print(f"demo score is nan!!!!!!")
            if len(data_health.loc[data_health.id_date == i,"probelty"]):
                arr.append(data_health.loc[data_health.id_date == i,"probelty"].values[0])
            else:
                arr.append(np.nan)
                print(f"demo score is nan!!!!!!")
                
            data_combined = data_combined.merge(pd.Series(arr, index=lis1).to_frame().T,how='outer')
            
        data_combined.fillna((data_combined.mean()), inplace=True)
        for vari in set(lis1)-set(['id_date','diagnosis']):
            try:
                data_combined[vari] = data_combined[vari].apply(lambda a: a/data_combined[vari].std() - data_combined[vari].mean()/data_combined[vari].std())
            except ZeroDivisionError:
                pass

        data_combined.fillna((data_combined.mean()), inplace=True)
        
        test_combined=data_combined[data_combined['id_date'].isin(id_list[test_id_[m]])]
        train_combined=data_combined[data_combined['id_date'].isin(id_list[train_id_[m]])]
        y_train=train_combined.diagnosis

        x_train = np.array(train_combined[selected_section]).reshape(-1, len(selected_section))
        y_test = test_combined.diagnosis
        x_test = np.array(test_combined[selected_section]).reshape(-1, len(selected_section))
        x_test_id.extend(test_combined["id_date"].tolist())

        model_last = LogisticRegression(solver='liblinear', random_state=0,C=1)
        #model_last = RandomForestClassifier(random_state=0)
        #model_last = LGBMClassifier(random_state=0)
        y_train = y_train.astype('int')
        model_last.fit(x_train, y_train)
        y_test = y_test.astype('int')

        y_pred = model_last.predict_proba(np.asarray(x_test))
        predictions.extend(y_pred[:,1])

        mat = metrics_cal(y_pred[:,1],y_test)
        for idx,performance in enumerate(mat):
            mat_avg[idx]['results_ensemble'].append(performance)
    
    print(f"Highest AUC is {np.mean(mat_avg[0]['results_ensemble'])} for {selected_section}")
    predictions_id = pd.DataFrame({'id_date': x_test_id, 'score': predictions})
    df_res = pd.DataFrame(mat_avg,index=['AUC','Accuracy','Sensitivity','Specificity','PPV','NPV','F1'])
    return df_res,predictions_id,coeff,selected_section
          
          
def selected_demo_classifier(id_list_,train_id_,test_id_,data_sec,folder_name,selected_section,stage,n=10):

    stg = ['0vs2','2vs4']
    assert stage in stg, f"This has to be one of {stg}"
    
    id_list = np.array(id_list_)
    
    mat_avg = [{'results_ensemble':[]} for i in range(7)]
    ext_var = ['bg','bmi','calc_ldl','cpd','creat','hx_diab','dbp','hdl','hgt','has_hyper','liprx','sbp','trig','vent_rt']
    col_to_drop = ['id_date','diagnosis','age_at_event','closest_mmse_score','education','sex','apoe'] + ext_var
    
    lis1 = col_to_drop
    
    cat_var = ['education','sex','apoe','hx_diab','has_hyper','liprx']
#     cat_var = ['education','sex','apoe']
    dum_cols = list(set(cat_var) & set(selected_section))
    
    x_test_id = []
    predictions = []

    #for m in tqdm(range(n), desc='1st loop-fcn2',position=0):
    data_sec["education"] = data_sec["education"].apply(lambda a: int(a>1))
    for m in range(n):
        data_combined = pd.DataFrame(columns=lis1)  
        
        arr = []
        for i in data_sec['id_date'].unique():
            arr = data_sec.loc[data_sec.id_date == i,col_to_drop].values[0].tolist()
            data_combined = data_combined.merge(pd.Series(arr, index=lis1).to_frame().T,how='outer')
            
        data_combined['education'].replace(np.nan,0,inplace=True)
        data_combined['sex'].replace(np.nan,2,inplace=True)
        data_combined['apoe'].replace(np.nan,33,inplace=True)
        data_combined['hx_diab'].replace(np.nan,0.0,inplace=True)
        data_combined['has_hyper'].replace(np.nan,1,inplace=True)
        data_combined['liprx'].replace(np.nan,0,inplace=True)
        data_combined.fillna((data_combined.mean()), inplace=True)
        for vari in set(lis1)-set(['id_date','diagnosis']+cat_var):
            try:
                data_combined[vari] = data_combined[vari].apply(lambda a: a/data_combined[vari].std() - data_combined[vari].mean()/data_combined[vari].std())
            except ZeroDivisionError:
                pass
        
        if len(dum_cols)!=0:
            data_combined = pd.get_dummies(data_combined, columns=dum_cols)
            all_col = data_combined.columns.tolist()
            selcted_col_cat = [col for col in all_col for dum_col in dum_cols if col.startswith(dum_col)]
            selected_section2 = list(set(selected_section) - set(dum_cols)) + selcted_col_cat
        else:
            selected_section2 = selected_section

        data_combined.fillna((data_combined.mean()), inplace=True)
        
        test_combined=data_combined[data_combined['id_date'].isin(id_list[test_id_[m]])]
        train_combined=data_combined[data_combined['id_date'].isin(id_list[train_id_[m]])]
        y_train=train_combined.diagnosis

        x_train = np.array(train_combined[selected_section2]).reshape(-1, len(selected_section2))
        y_test = test_combined.diagnosis
        x_test = np.array(test_combined[selected_section2]).reshape(-1, len(selected_section2))
        x_test_id.extend(test_combined["id_date"].tolist())

        model_last = LogisticRegression(solver='liblinear', random_state=0,C=1)
        #model_last = RandomForestClassifier(random_state=0)
        #model_last = LGBMClassifier(random_state=0)
        y_train = y_train.astype('int')
        model_last.fit(x_train, y_train)
        y_test = y_test.astype('int')

        y_pred = model_last.predict_proba(np.asarray(x_test))
        predictions.extend(y_pred[:,1])

        mat = metrics_cal(y_pred[:,1],y_test)
        for idx,performance in enumerate(mat):
            mat_avg[idx]['results_ensemble'].append(performance)
    
    print(f"Highest AUC is {np.mean(mat_avg[0]['results_ensemble'])} for {selected_section2}")
    predictions_id = pd.DataFrame({'id_date': x_test_id, 'score': predictions})
    df_res = pd.DataFrame(mat_avg,index=['AUC','Accuracy','Sensitivity','Specificity','PPV','NPV','F1'])
    return df_res,predictions_id
def save_xls(list_dfs_mean,list_dfs_std, xls_path,stg):
    with ExcelWriter(xls_path) as writer:
        for n, df in enumerate(list_dfs_mean):
            df.to_excel(writer,f"{stg[n]}",startrow=2 , startcol=2,index_label=df.name)
            df2 = list_dfs_std[n]
            df2.to_excel(writer,f"{stg[n]}",startrow=df.shape[0]+5 , startcol=2,index_label=df2.name)  


def main(s1_rs,s1_sts,s2_rs,s2_sts,n_y,selected_feature):
    sections = ['DEMO','WMS','OTHER','FAS','BNT','CLOCK_DRAWING','WAIS','WAIS-R']
    data1 = pd.read_pickle('data/encoded_rs_dem_health.pkl')
    data2 = pd.read_pickle('data/encoded_sts_dem_health.pkl')
    folder_name = f'folder_ad{n_y}_paramgiven/VoiceDemo_ad'
    
    stages = ['2vs4']
    bal = False

#     num_f = {"0vs2":[16,512],"2vs4":[512,128]} # last one
    sec_tuned = {"0vs2":selected_feature["0vs2"][0],"2vs4":selected_feature["2vs4"][0]} #[OTHER, WAIS, CLOCK_DRAWING, BNT, DEMO]
    num_f = {"0vs2":[s1_rs,s1_sts],"2vs4":[s2_rs,s2_sts]} 
    mmse = ['closest_mmse_score']
    demo = ['age_at_event','education','sex']
    demo_apoe = ['age_at_event','education','sex','apoe']
    health_var = ['bg','bmi','calc_ldl','cpd','creat','hx_diab','dbp','hdl','hgt','has_hyper','liprx','sbp','trig','vent_rt']
          
    health_tuned = {"0vs2":['creat', 'hdl', 'vent_rt', 'hgt', 'bmi', 'hx_diab'],"2vs4":['bmi', 'calc_ldl', 'bg', 'hx_diab']}
    health_tuned_demo = {"0vs2":['age_at_event','education','sex','creat', 'hdl', 'vent_rt', 'hgt', 'bmi', 'hx_diab'],"2vs4":['age_at_event','education','sex','bmi', 'calc_ldl', 'bg', 'hx_diab']}
          
#     years_list = [1,2,3,4,5,6,7]
#     for n_y in years_list:
    num_fold = 10
    for stage in tqdm(stages, desc='final classifier',position=0):
        ## Data preparation
        data_class = data_generator(sections, data1, data2, stage, n_y, balanced = bal)
        data_rs, df_sts = data_class.get_data()

        id_list1, train_id1, test_id1 = data_class.get_train_test_id(kfold = 10, test_ratio = 0.1)
        if not os.path.exists(f'{folder_name}/{stage}'):
            os.makedirs(f'{folder_name}/{stage}')

        res = random_sampling_scroe(id_list1,data_rs,train_id1,test_id1,num_f[stage][0],folder_name,stage,n=num_fold)
        res_path = f'{folder_name}/{stage}/results_rs.pkl'
        res.to_pickle(res_path)

        res = Sections_score(id_list1,df_sts,train_id1,test_id1,sections,stage,folder_name,num_f[stage][1],n=num_fold)
        res_path = f'{folder_name}/{stage}/results_subtests.pkl' 
        res.to_pickle(res_path)

        flag="voice"
        res,pred_id = selected_section_classifier(id_list1,train_id1,test_id1,sections,folder_name,sec_tuned[stage]+["rs_score"],stage,flag,n=num_fold)
        res_path = f'{folder_name}/{stage}/results_sts_rs.pkl'
        res.to_pickle(res_path)
        pred_id.to_pickle(f'{folder_name}/{stage}/pred_voice.pkl')


        flag="demo"  
        res,pred_id = selected_section_classifier(id_list1,train_id1,test_id1,sections,folder_name,demo,stage,flag,n=num_fold)
        res_path = f'{folder_name}/{stage}/results_demo.pkl'
        res.to_pickle(res_path)
#         pred_id.to_pickle(f'{folder_name}/{stage}/pred_demo.pkl')
          
        flag="health"  
        res,pred_id = selected_section_classifier(id_list1,train_id1,test_id1,sections,folder_name,health_tuned[stage],stage,flag,n=num_fold)
        res_path = f'{folder_name}/{stage}/results_health.pkl'
        res.to_pickle(res_path)

        res, pred_id = selected_section_classifier2(id_list1,train_id1,test_id1,sections,folder_name,["voice","demo"],stage,apoe=0,n=num_fold)
        res_path = f'{folder_name}/{stage}/results_sts_rs_demo.pkl'
        res.to_pickle(res_path)
        pred_id.to_pickle(f'{folder_name}/{stage}/pred_voice_demo.pkl')
       

        flag="demo_apoe"  
        res,pred_id = selected_section_classifier(id_list1,train_id1,test_id1,sections,folder_name,demo_apoe,stage,flag,n=num_fold)
        res_path = f'{folder_name}/{stage}/results_demo_apoe.pkl'
        res.to_pickle(res_path)

        res,pred_id = selected_section_classifier2(id_list1,train_id1,test_id1,sections,folder_name,["voice","demo"],stage,apoe=1,n=num_fold)
        res_path = f'{folder_name}/{stage}/results_sts_rs_demo_apoe.pkl'
        res.to_pickle(res_path)

        res,pred_id = selected_demo_classifier(id_list1,train_id1,test_id1,df_sts["DEMO"],folder_name,mmse,stage,n=num_fold)
        res_path = f'{folder_name}/{stage}/results_mmse.pkl'
        res.to_pickle(res_path)

        res,pred_id = selected_demo_classifier(id_list1,train_id1,test_id1,df_sts["DEMO"],folder_name,health_var,stage,n=num_fold)
        res_path = f'{folder_name}/{stage}/results_full_health.pkl'
        res.to_pickle(res_path)
          
          
        res,pred_id = selected_section_classifier2(id_list1,train_id1,test_id1,sections,folder_name,["voice","demo","health"],stage,apoe=0,n=num_fold)
        res_path = f'{folder_name}/{stage}/results_sts_rs_demo_health.pkl'
        res.to_pickle(res_path)
          
        res,pred_id = selected_section_classifier2(id_list1,train_id1,test_id1,sections,folder_name,["demo","health"],stage,apoe=0,n=num_fold)
        res_path = f'{folder_name}/{stage}/results_demo_health.pkl'
        res.to_pickle(res_path)
          
        res,pred_id = selected_section_classifier2(id_list1,train_id1,test_id1,sections,folder_name,["voice","demo","health"],stage,apoe=1,n=num_fold)
        res_path = f'{folder_name}/{stage}/results_sts_rs_demo_health_apoe.pkl'
        res.to_pickle(res_path)
          
        res,pred_id = selected_section_classifier2(id_list1,train_id1,test_id1,sections,folder_name,["demo","health"],stage,apoe=1,n=num_fold)
        res_path = f'{folder_name}/{stage}/results_demo_health_apoe.pkl'
        res.to_pickle(res_path)
          


    list_dfs_mean = []
    list_dfs_std = []
    for s in stages:
        res1 = pd.read_pickle(f'{folder_name}/{s}/results_demo.pkl')
        res1.set_axis(['demos'], axis=1,inplace=True)
        res2 = pd.read_pickle(f'{folder_name}/{s}/results_sts_rs.pkl')
        res2.set_axis(['STS_RS'], axis=1,inplace=True)
        res3 = pd.read_pickle(f'{folder_name}/{s}/results_sts_rs_demo.pkl')
        res3.set_axis(['STS_RS_demo'], axis=1,inplace=True)
        res4 = pd.read_pickle(f'{folder_name}/{s}/results_mmse.pkl')
        res4.set_axis(['MMSE'], axis=1,inplace=True)
        res5 = pd.read_pickle(f'{folder_name}/{s}/results_full_health.pkl')
        res5.set_axis(['Full_health'], axis=1,inplace=True)
        res6 = pd.read_pickle(f'{folder_name}/{s}/results_health.pkl')
        res6.set_axis(['Tuned_health'], axis=1,inplace=True)
        res7 = pd.read_pickle(f'{folder_name}/{s}/results_demo_apoe.pkl')
        res7.set_axis(['demo_apoe'], axis=1,inplace=True)
        res8 = pd.read_pickle(f'{folder_name}/{s}/results_sts_rs_demo_apoe.pkl')
        res8.set_axis(['voice_demo_apoe'], axis=1,inplace=True)
        res9 = pd.read_pickle(f'{folder_name}/{s}/results_sts_rs_demo_health.pkl')
        res9.set_axis(['voice_demo_health'], axis=1,inplace=True)
        res10 = pd.read_pickle(f'{folder_name}/{s}/results_demo_health.pkl')
        res10.set_axis(['demo_health'], axis=1,inplace=True)
        res11 = pd.read_pickle(f'{folder_name}/{s}/results_sts_rs_demo_health_apoe.pkl')
        res11.set_axis(['voice_demo_health_apoe'], axis=1,inplace=True)
        res12 = pd.read_pickle(f'{folder_name}/{s}/results_demo_health_apoe.pkl')
        res12.set_axis(['demo_health_apoe'], axis=1,inplace=True)

        df_results = [res1,res2,res3,res4,res5,res6,res7,res8,res9,res10,res11,res12] 
        df_mean = pd.concat([result.applymap(np.mean) for result in df_results],axis=1)
        df_mean = df_mean.mul(100).round(1)
        df_mean.name = 'mean'
        df_std = pd.concat([result.applymap(np.std) for result in df_results],axis=1)
        df_std = df_std.mul(100).round(1)
        df_std.name = 'std'
        list_dfs_mean.append(df_mean)
        list_dfs_std.append(df_std)

    save_xls(list_dfs_mean,list_dfs_std,f'{folder_name}/{n_y}years_results_ad_education_fixed.xlsx',["NtoMCI","MCItoD"])

    print(f'Done with results of year{n_y}!')
          
if __name__ == "__main__": 
#     s1_rs = 16
#     s1_sts = 512
#     s2_rs = 512
#     s2_sts =128
#     n_y = 5
#     selected_feature = 0
    main(s1_rs,s1_sts,s2_rs,s2_sts,n_y,selected_feature)
    

