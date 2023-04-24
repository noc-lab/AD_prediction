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
import time
import sys

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.feature_selection import RFE
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm


class data_generator():
    def __init__(self, sections, data_rs, data_sts, stage, n_years, balanced = False):
        self.sections = sections
        assert stage in ['0vs4','0vs2','2vs4'], "This has to be one of ['0vs4','0vs2','2vs4']"
        self.stage = stage
        self.balanced = balanced
        ## feature names for each section
        #health_var = ['bg','bmi','calc_ldl','cpd','creat','hx_diab','dbp','hdl','hgt','has_hyper','liprx','sbp','trig','vent_rt']
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
            df0 = pd.read_pickle(f"20220218_20_23_37_0649/data/NtoN_ad{n_years}.pkl")
            df1 = pd.read_pickle(f"20220218_20_23_37_0649/data/NtoMCI_ad{n_years}.pkl")
        elif self.stage == '2vs4':
            df0 = pd.read_pickle(f"20220218_20_23_37_0649/data/MCItoMCI_ad{n_years}.pkl")
            df1 = pd.read_pickle(f"20220218_20_23_37_0649/data/MCItoD_ad{n_years}.pkl")

            
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

    auc = metrics.auc(fpr,tpr)
    acc = metrics.accuracy_score(y_test, (1*(np.array(y_pred)>threshold)).tolist())

    tn, fp, fn, tp = metrics.confusion_matrix(np.array(y_test), np.array(y_pred)>threshold).ravel()
    spe = tn / (tn+fp)
    sen = tp / (tp+fn)
    ppv = tp / (tp+fp)
    npv = tn / (tn+fn)
    
    return [auc,acc,sen,spe,ppv,npv]


def Sections_score(id_list_,data_,train_id_,test_id_,sections_,stage,folder_name,num_,n=10): 
    
    stg = ['0vs4','0vs2','2vs4']
    assert stage in stg, f"This has to be one of {stg}"
    
    df_list = list()
    id_list = np.array(id_list_)
    #health_var = ['bg','bmi','calc_ldl','cpd','creat','hx_diab','dbp','hdl','hgt','has_hyper','liprx','sbp','trig','vent_rt']
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

            if num_ != 768:
                X = x_train.values
                Y = y_train.values
                model = LogisticRegression(solver='lbfgs',max_iter=500)
                rfe = RFE(model,n_features_to_select = num_)
                fit = rfe.fit(X, Y)
                arr = x_train.keys().array[0:768]
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
    
    df_res = pd.DataFrame(result,index=['AUC','Accuracy','Sensitivity','Specificity','PPV','NPV'])
    
    return df_res
def selected_section_classifier(id_list_,train_id_,test_id_,sections_,folder_name,selected_section,stage,n=10):

    stg = ['0vs4','0vs2','2vs4']
    assert stage in stg, f"This has to be one of {stg}"
    
    id_list = np.array(id_list_)
    
    mat_avg = [{'results_ensemble':[]} for i in range(6)]
    #health_var = ['bg','bmi','calc_ldl','cpd','creat','hx_diab','dbp','hdl','hgt','has_hyper','liprx','sbp','trig','vent_rt']
    col_to_drop = ['id_date','diagnosis','age_at_event','closest_mmse_score','education','sex','apoe'] 
    
    lis1 = [section for section in sections_]
    # lis1[0:0] = ["rs_score"]
    lis1[0:0] = col_to_drop
    
    for m in range(n):
        data_sec1 = pd.read_pickle(f"{folder_name}/{stage}/df{m}")

        
        data_combined = pd.DataFrame(columns=lis1)  
        arr = []
        for i in data_sec1['id_date'].unique():
            arr = data_sec1.loc[data_sec1.id_date == i,col_to_drop].values[0].tolist()
            # if len(data_rs.loc[data_rs.id_date == i,"rs_score"]):
            #     arr.append(data_rs.loc[data_rs.id_date == i,"rs_score"].values[0])
            # else:
            #     arr.append(np.nan)
            if len(data_sec1.loc[data_sec1.id_date == i,sections_]):
                arr.extend(data_sec1.loc[data_sec1.id_date == i,sections_].values[0].tolist())
            else:
                arr.extend(np.tile(np.nan,len(sections_)).tolist())
                
            data_combined = data_combined.merge(pd.Series(arr, index=lis1).to_frame().T,how='outer')
            
        data_combined['education'].replace(np.nan,0,inplace=True)
        data_combined['sex'].replace(np.nan,2,inplace=True)
        data_combined['apoe'].replace(np.nan,33,inplace=True)
        data_combined.fillna((data_combined.mean()), inplace=True)
        for vari in set(lis1)-set(['id_date','diagnosis','education','sex','apoe']):
            try:
                data_combined[vari] = data_combined[vari].apply(lambda a: a/data_combined[vari].std() - data_combined[vari].mean()/data_combined[vari].std())
            except ZeroDivisionError:
                pass
     

        data_combined.fillna((data_combined.mean()), inplace=True)
        
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
            mat_avg[idx]['results_ensemble'].append(performance)
    
    print(f"Highest AUC is {np.mean(mat_avg[0]['results_ensemble'])} for {selected_section}")
    
    df_res = pd.DataFrame(mat_avg,index=['AUC','Accuracy','Sensitivity','Specificity','PPV','NPV'])
    return df_res

def main(s1_rs,s1_sts,s2_rs,s2_sts,n_years):
    sections = ['DEMO','WMS','OTHER','FAS','BNT','CLOCK_DRAWING','WAIS','WAIS-R']
    data1 = pd.read_pickle('20220218_20_23_37_0649/data/encoded_rs_dem.pkl')
    data2 = pd.read_pickle('20220218_20_23_37_0649/data/encoded_sts_dem.pkl')
    
    stages = ['0vs2','2vs4']
    folder_name = f'20220218_20_23_37_0649/val_folder_ad{n_years}/feature_extraction'
    bal = False
    num_f = {"0vs2":[s1_rs,s1_sts],"2vs4":[s2_rs,s2_sts]} 
    # num_f = {"0vs4":[64,512],"0vs2":[64,512],"2vs4":[64,32]}
#     num_f = {"0vs2":[16,512],"2vs4":[512,128]} ##final one for google speech 
    # num_f = {"0vs2":[512,512],"2vs4":[512,512]} final one for whisper
#     num_f = {"0vs2":[64,64],"2vs4":[768,64]} ## google speech by longformer
#     num_f = {"0vs2":[64,64],"2vs4":[64,64]}
    RSLT_subset = {s:[] for s in stages}
    extracted_features = {subtest:[] for subtest in stages}
    for stage in tqdm(stages, desc='feature extraction',position=0):
        ## Data preparation
        data_class = data_generator(sections, data1, data2, stage, n_years, balanced = bal)
        data_rs, df_sts = data_class.get_data()
        id_list1, train_id1, test_id1 = data_class.get_train_test_id(kfold = 10, test_ratio = 0.1)
        if not os.path.exists(f'{folder_name}/{stage}'):
            os.makedirs(f'{folder_name}/{stage}')

        train_id2 = []
        valid_id2 = []
        for f in range(len(train_id1)):
            random.seed(41)
            train_id2.append(np.array(random.sample(list(train_id1[f]), int(len(train_id1[f])*0.9))))
            valid_id2.append(np.array([ids for ids in train_id1[f] if ids not in train_id2[f]]))
            
        res = Sections_score(id_list1,df_sts,train_id2,valid_id2,sections,stage,folder_name,num_f[stage][1],n=10)
        res_path = f'{folder_name}/{stage}/results_subtests.pkl'
        res.to_pickle(res_path)

        sec_sel = ['DEMO','WMS','OTHER','FAS','BNT','CLOCK_DRAWING','WAIS','WAIS-R']
        for i in range(len(sec_sel)):
            lists = []
            lists[0:0] = sec_sel
            removed = lists.pop(i)
            res = selected_section_classifier(id_list1,train_id2,valid_id2,sections,folder_name,lists,stage,n=10)
            #res2 = res.applymap(np.mean)
            res2 = res
            RSLT_subset[stage].append(res2.iloc[0,0])
            print(f"AUC for {stage} after removing {removed} is {np.mean(res2.iloc[0,0])}")

        a = deepcopy(RSLT_subset[stage])
        a = list(1-np.array(a))
        f1 = [np.mean(a[i])for i in range(len(sec_sel))]
        f1 = [np.mean(a[i])/np.min(f1) for i in range(len(sec_sel))]
        f2 = [np.std(a[i])/np.min(f1) for i in range(len(sec_sel))]
        dictdata = {'mean':pd.Series(f1, index=sec_sel),'std':pd.Series(f2, index=sec_sel)}
        b_plot = pd.concat(dictdata,axis=1)
        b_plot = b_plot.sort_values('mean')
        res_path = f'{folder_name}/b_plot_{stage}'
        b_plot.to_pickle(res_path)
                       
        lists = []
        lists[0:0] = list(b_plot.index)
        RSLT_selction = {subtest:[] for subtest in lists}
        res_max = 0
        max_ind = 0
        for i in range(len(sec_sel)):
            if i != 0:
                lists.pop(0)
            res = selected_section_classifier(id_list1,train_id2,valid_id2,sections,folder_name,lists,stage,n=10)
            #res2 = res.applymap(np.mean)
            res2 = res
            if res_max <= np.mean(res2.iloc[0,0]):
                res_max = np.mean(res2.iloc[0,0])
                max_ind = i
            RSLT_selction[lists[0]].append(res2.iloc[0,0])
            print(f"AUC for {i} features is {np.mean(res2.iloc[0,0])}")
          
        np.save(f'{folder_name}/RSLT_selction_{stage}.npy', RSLT_selction) 
        extracted_features[stage].append(list(b_plot.index)[max_ind:])
                       
                       
    df = pd.DataFrame(extracted_features)
    res_path = f'{folder_name}/df_extracted_features.pkl'
    df.to_pickle(res_path)
        
    print(f'Done for feature extraction of folder{n_years}!')



if __name__ == "__main__":
          
    main(s1_rs,s1_sts,s2_rs,s2_sts,n_years)