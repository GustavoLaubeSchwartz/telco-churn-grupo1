from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold

import argparse
import hashlib
import importlib.metadata

FILES=['Telco_customer_churn', 'Telco_customer_churn_demographics', 'Telco_customer_churn_location', 'Telco_customer_churn_population', 'Telco_customer_churn_services', 'Telco_customer_churn_status', 'CustomerChurn']

def run(data_dir, output_dir):
    ROOT=Path(data_dir)
    OUT=Path(output_dir)
    frames={name:pd.read_excel(ROOT/(name+'.xlsx'),keep_default_na=False) for name in FILES}
    report={}
    for n,d in frames.items():
        blank=d.map(lambda x:isinstance(x,str) and not x.strip())|d.isna()
        report[n]={'shape':list(d.shape),'duplicados':int(d.duplicated().sum()),'ausentes':blank.sum()[blank.any()].to_dict(),'constantes':d.columns[d.nunique(dropna=False)==1].tolist(),'tipos':d.dtypes.astype(str).to_dict()}
        key=next((c for c in ['CustomerID','Customer ID','Zip Code'] if c in d),None)
        report[n]['chave']=key
        report[n]['duplicados_chave']=int(d[key].duplicated().sum())
    d=frames['Telco_customer_churn'].copy()
    d['Total Charges']=pd.to_numeric(d['Total Charges'].replace(r'^\s*$',np.nan,regex=True),errors='coerce')
    report['alvo']=d['Churn Value'].value_counts().sort_index().to_dict()
    report['alvo_coerente']=bool((d['Churn Label'].map({'No':0,'Yes':1})==d['Churn Value']).all())
    report['ausentes_total_charges']=d.loc[d['Total Charges'].isna(),['Tenure Months','Monthly Charges','Churn Value']].to_dict('records')
    report['razao_ausente']=d.groupby('Churn Label')['Churn Reason'].apply(lambda s:int(s.str.strip().eq('').sum())).to_dict()
    report['resumo_numerico']=d[['Tenure Months','Monthly Charges','Total Charges']].describe().to_dict()
    report['churn_contrato']=d.groupby('Contract')['Churn Value'].agg(['count','sum','mean']).to_dict('index')
    report['duplicados_sem_id']=int(d.drop(columns='CustomerID').duplicated().sum())
    report['comparacoes']={}
    for name,key in [('CustomerChurn','Customer ID'),('Telco_customer_churn_demographics','Customer ID'),('Telco_customer_churn_location','Customer ID'),('Telco_customer_churn_services','Customer ID'),('Telco_customer_churn_status','Customer ID')]:
        other=frames[name].set_index(key)
        base=d.set_index('CustomerID')
        report['comparacoes'][name]={'mesmos_clientes':set(base.index)==set(other.index)}
        for a,b in [('Dependents','Dependents'),('Zip Code','Zip Code'),('Tenure Months','Tenure'),('Tenure Months','Tenure in Months'),('Total Charges','Total Charges'),('Churn Label','Churn'),('Churn Label','Churn Label')]:
            if a in base and b in other:
                x=base[a];y=other.loc[base.index,b]
                if a=='Total Charges':y=pd.to_numeric(y,errors='coerce');eq=(x==y)|(x.isna()&y.isna())
                else:eq=x==y
                report['comparacoes'][name][a]=int((~eq).sum())
    num=['Tenure Months','Monthly Charges','Total Charges']
    cat=['Gender','Senior Citizen','Partner','Dependents','Phone Service','Multiple Lines','Internet Service','Online Security','Online Backup','Device Protection','Tech Support','Streaming TV','Streaming Movies','Contract','Paperless Billing','Payment Method']
    X=d[num+cat].copy(); y=d['Churn Value'].astype(int)
    for c in cat:X[c]=X[c].str.strip().replace('',np.nan)
    report['dominios']={c:sorted(X[c].dropna().unique().tolist()) for c in cat}
    report['negativos']={c:int((X[c]<0).sum()) for c in num}
    report['coerencia_servicos']={
     'multiple_sem_telefone':int(((d['Phone Service']=='No')!=(d['Multiple Lines']=='No phone service')).sum()),
     **{c:int(((d['Internet Service']=='No')!=(d[c]=='No internet service')).sum()) for c in ['Online Security','Online Backup','Device Protection','Tech Support','Streaming TV','Streaming Movies']}}
    def make_preprocessor():
        return ColumnTransformer([
            ('numericos',Pipeline([('imputar',SimpleImputer(strategy='median')),('padronizar',StandardScaler())]),num),
            ('categoricos',Pipeline([('imputar',SimpleImputer(strategy='most_frequent')),('codificar',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),cat)
        ])
    processed=OUT/'data/processed'
    processed.mkdir(parents=True,exist_ok=True)
    (processed/'folds').mkdir(exist_ok=True)
    (OUT/'reports').mkdir(exist_ok=True)
    clean=X.copy()
    clean['Churn Value']=y
    clean.to_csv(processed/'telco_base_limpa.csv',index=False,encoding='utf-8')
    pd.DataFrame({'row_index':np.arange(len(d)), 'CustomerID':d['CustomerID']}).to_csv(processed/'rastreabilidade.csv',index=False)
    dictionary=[]
    for col in d.columns:
        if col=='Churn Value': role='alvo'
        elif col in num: role='preditor numerico'
        elif col in cat: role='preditor categorico'
        else: role='excluido da matriz preditiva'
        dictionary.append({'atributo':col,'tipo_apos_limpeza':str(d[col].dtype),'papel':role})
    (OUT/'reports/dicionario_atributos.json').write_text(json.dumps(dictionary,ensure_ascii=False,indent=2),encoding='utf-8')
    fold_assignment=np.zeros(len(d),dtype=int)
    folds=[]
    for k,(tr,va) in enumerate(StratifiedKFold(n_splits=5,shuffle=True,random_state=42).split(X,y),1):
        pp=make_preprocessor(); xt=pp.fit_transform(X.iloc[tr]);xv=pp.transform(X.iloc[va])
        assert np.isfinite(xt).all() and np.isfinite(xv).all()
        assert np.allclose(xt[:,:3].mean(axis=0),0,atol=1e-10)
        fold_assignment[va]=k
        feature_names=pp.get_feature_names_out()
        np.savez_compressed(processed/'folds'/f'fold_{k}.npz',X_train=xt,X_valid=xv,y_train=y.iloc[tr].to_numpy(),y_valid=y.iloc[va].to_numpy(),train_indices=tr,valid_indices=va,feature_names=feature_names.astype(str))
        numeric_pipe=pp.named_transformers_['numericos']
        categorical_pipe=pp.named_transformers_['categoricos']
        state={'numeric_columns':num,'categorical_columns':cat,'medians':numeric_pipe['imputar'].statistics_.tolist(),'means':numeric_pipe['padronizar'].mean_.tolist(),'scales':numeric_pipe['padronizar'].scale_.tolist(),'categorical_fill':categorical_pipe['imputar'].statistics_.tolist(),'categories':[a.tolist() for a in categorical_pipe['codificar'].categories_],'feature_names':feature_names.tolist()}
        (processed/'folds'/f'fold_{k}_parametros.json').write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8')
        folds.append({'fold':k,'treino':len(tr),'validacao':len(va),'features':xt.shape[1],'medianas_treino':pp.named_transformers_['numericos']['imputar'].statistics_.tolist()})
    report['preprocessamento']=folds

    pd.DataFrame({'row_index':np.arange(len(d)),'validation_fold':fold_assignment}).to_csv(processed/'particoes.csv',index=False)
    report['ambiente']={name:importlib.metadata.version(name) for name in ['numpy','pandas','scikit-learn','scipy','openpyxl']}
    report['arquivos_origem']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(ROOT.glob('*.xlsx')) if p.stem in FILES}
    (OUT/'reports/analise.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,default=lambda x:x.item() if hasattr(x,'item') else str(x)),encoding='utf-8')
    print('Base limpa:',len(clean),'linhas e',len(clean.columns),'colunas (19 preditores + alvo).')
    print('Cinco particoes exportadas com',folds[0]['features'],'atributos transformados cada.')
    return report

if __name__ == '__main__':
    project=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description='Analise exploratoria e pre-processamento Telco, Grupo 1.')
    parser.add_argument('--data-dir',type=Path,default=project/'data/raw')
    parser.add_argument('--output-dir',type=Path,default=project)
    args=parser.parse_args()
    run(args.data_dir,args.output_dir)
