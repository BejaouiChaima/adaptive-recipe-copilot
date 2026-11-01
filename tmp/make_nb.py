import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(text): cells.append(nbf.v4.new_markdown_cell(text))
def code(text): cells.append(nbf.v4.new_code_cell(text))

md("# Modèles de recommandation de recettes\n\nAlternatives au TF-IDF : CountVectorizer, BM25, LSA (SVD), NMF, KNN.")

code("""import re, numpy as np, pandas as pd, math
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path('.').resolve().parents[0]
DATA_PATH = PROJECT_ROOT / 'data' / 'processed' / 'recipes_clean.csv'
TOP_K = 10
N_EVAL = 200
RANDOM_STATE = 42
USE_COLS = ['RecipeId','Name','RecipeIngredientParts','RecipeCategory','Description']

def load_recipes():
    df = pd.read_csv(DATA_PATH, usecols=USE_COLS, low_memory=False)
    df['RecipeId'] = pd.to_numeric(df['RecipeId'], errors='coerce')
    df = df.dropna(subset=['RecipeId','Name']).copy()
    df['RecipeId'] = df['RecipeId'].astype('int64')
    df = df.drop_duplicates(subset='RecipeId', keep='first').reset_index(drop=True)
    for c in ['Name','RecipeIngredientParts','RecipeCategory','Description']:
        df[c] = df[c].fillna('').astype(str)
    df['CategoryForEval'] = df['RecipeCategory'].str.lower().str.strip()
    df.loc[df['CategoryForEval'].isin(['','nan','none']),'CategoryForEval'] = ''
    return df

def clean_text(v):
    if not isinstance(v,str): return ''
    v = v.lower()
    v = re.sub(r'\\bc\\s*\\(', ' ', v)
    v = re.sub(r'[\\[\\]{}()\"\\x27`]', ' ', v)
    v = re.sub(r'[^a-z0-9\\s]', ' ', v)
    return re.sub(r'\\s+',' ',v).strip()

def build_text(df, include_category=True):
    names = df['Name'].map(clean_text)
    ing = df['RecipeIngredientParts'].map(clean_text)
    desc = df['Description'].map(clean_text)
    if include_category:
        cat = df['RecipeCategory'].map(clean_text)
        text = names+' '+ing+' '+ing+' '+cat+' '+cat+' '+desc
    else:
        text = names+' '+ing+' '+ing+' '+desc
    return text.replace('','recipe')

df = load_recipes()
print(df.shape)
""")

md("## Métriques communes")
code("""def precision_at_k(rel,k):
    rel=np.asarray(rel[:k],dtype=int); return rel.sum()/k
def recall_at_k(rel,tot,k):
    if tot==0: return np.nan
    return np.sum(rel[:k])/tot
def ndcg_at_k(rel,tot,k):
    rel=np.asarray(rel[:k],dtype=float)
    if tot==0: return np.nan
    disc=np.log2(np.arange(2,len(rel)+2))
    dcg=np.sum(rel/disc)
    il=min(tot,k); idcg=np.sum(1/np.log2(np.arange(2,il+2)))
    return dcg/idcg if idcg>0 else 0.0
def map_at_k(rel,tot,k):
    rel=np.asarray(rel[:k],dtype=int)
    if tot==0: return np.nan
    prec=np.cumsum(rel)/np.arange(1,len(rel)+1)
    return np.sum(prec*rel)/min(tot,k)

def evaluate_scores(train_text, test_text, train_df, test_df, score_fn):
    Xtr, Xte = vectorize(train_text, test_text)
    sims = score_fn(Xtr, Xte)
    train_cats = train_df['CategoryForEval'].to_numpy()
    metrics = {'P@10':[],'R@10':[],'MAP@10':[],'NDCG@10':[]}
    valid=0
    for i in range(len(test_df)):
        qc = test_df.loc[i,'CategoryForEval']
        if not qc: continue
        mask = (train_cats==qc)
        tot = int(mask.sum())
        if tot==0: continue
        scores = sims[i]
        idx = np.argpartition(scores,-min(TOP_K,len(scores)))[-min(TOP_K,len(scores)):]
        idx = idx[np.argsort(scores[idx])[::-1]]
        rel = mask[idx].astype(int)
        metrics['P@10'].append(precision_at_k(rel,TOP_K))
        metrics['R@10'].append(recall_at_k(rel,tot,TOP_K))
        metrics['MAP@10'].append(map_at_k(rel,tot,TOP_K))
        metrics['NDCG@10'].append(ndcg_at_k(rel,tot,TOP_K))
        valid+=1
    print('requetes eval:',valid)
    for m,v in metrics.items():
        print(f'{m}: {np.mean(v):.4f}')
""")

md("## Modèle 1 : CountVectorizer (Bag of Words)")
code("""def vectorize(tr,te):
    v=CountVectorizer(stop_words='english',max_features=50000,min_df=2,max_df=0.95)
    Xtr=v.fit_transform(tr); Xte=v.transform(te)
    return Xtr,Xte

train_df,test_df=train_test_split(df,test_size=0.2,random_state=RANDOM_STATE)
if len(test_df)>N_EVAL: test_df=test_df.sample(n=N_EVAL,random_state=RANDOM_STATE)
train_df=train_df.reset_index(drop=True); test_df=test_df.reset_index(drop=True)
tr=build_text(train_df,include_category=False); te=build_text(test_df,include_category=False)
def cb(Xtr,Xte):
    return cosine_similarity(Xte,Xtr)
print('--- CountVectorizer ---')
evaluate_scores(tr,te,train_df,test_df,cb)
""")

md("## Modèle 2 : BM25 (implémentation manuelle)")
code("""class BM25:
    def __init__(self,k1=1.5,b=0.75):
        self.k1=k1; self.b=b
    def fit(self,corpus):
        self.corpus=corpus
        self.N=len(corpus)
        self.doc_len=np.array([len(d.split()) for d in corpus])
        self.avgdl=self.doc_len.mean()
        self.df={}
        for d in corpus:
            for w in set(d.split()):
                self.df[w]=self.df.get(w,0)+1
        return self
    def transform(self,query):
        scores=np.zeros(self.N)
        qwords=query.split()
        for w in qwords:
            if w not in self.df: continue
            idf=math.log((self.N-self.df[w]+0.5)/(self.df[w]+0.5)+1)
            for i,d in enumerate(self.corpus):
                tf=d.split().count(w)
                denom=tf+self.k1*(1-self.b+self.b*self.doc_len[i]/self.avgdl)
                scores[i]+=idf*tf*(self.k1+1)/denom
        return scores

bm=BM25().fit(tr.tolist())
sims=np.array([bm.transform(q) for q in te])
train_cats=train_df['CategoryForEval'].to_numpy()
metrics={'P@10':[],'R@10':[],'MAP@10':[],'NDCG@10':[]}; valid=0
for i in range(len(test_df)):
    qc=test_df.loc[i,'CategoryForEval']
    if not qc: continue
    mask=(train_cats==qc); tot=int(mask.sum())
    if tot==0: continue
    scores=sims[i]
    idx=np.argpartition(scores,-min(TOP_K,len(scores)))[-min(TOP_K,len(scores)):]
    idx=idx[np.argsort(scores[idx])[::-1]]
    rel=mask[idx].astype(int)
    metrics['P@10'].append(precision_at_k(rel,TOP_K))
    metrics['R@10'].append(recall_at_k(rel,tot,TOP_K))
    metrics['MAP@10'].append(map_at_k(rel,tot,TOP_K))
    metrics['NDCG@10'].append(ndcg_at_k(rel,tot,TOP_K))
    valid+=1
print('--- BM25 ---'); print('requetes eval:',valid)
for m,v in metrics.items(): print(f'{m}: {np.mean(v):.4f}')
""")

md("## Modèle 3 : LSA (TruncatedSVD sur TF-IDF)")
code("""def vectorize(tr,te):
    v=TfidfVectorizer(stop_words='english',max_features=50000,min_df=2,max_df=0.95,sublinear_tf=True)
    Xtr=v.fit_transform(tr); Xte=v.transform(te)
    svd=TruncatedSVD(n_components=100,random_state=RANDOM_STATE)
    Xtr=svd.fit_transform(Xtr); Xte=svd.transform(Xte)
    return Xtr,Xte
def cb(Xtr,Xte):
    return cosine_similarity(Xte,Xtr)
print('--- LSA/SVD ---')
evaluate_scores(tr,te,train_df,test_df,cb)
""")

md("## Modèle 4 : NMF (topic modeling)")
code("""def vectorize(tr,te):
    v=TfidfVectorizer(stop_words='english',max_features=20000,min_df=2,max_df=0.95)
    Xtr=v.fit_transform(tr); Xte=v.transform(te)
    nmf=NMF(n_components=50,random_state=RANDOM_STATE,max_iter=200)
    Xtr=nmf.fit_transform(Xtr); Xte=nmf.transform(Xte)
    return Xtr,Xte
def cb(Xtr,Xte):
    return cosine_similarity(Xte,Xtr)
print('--- NMF ---')
evaluate_scores(tr,te,train_df,test_df,cb)
""")

md("## Modèle 5 : KNN (NearestNeighbors sur TF-IDF)")
code("""v=TfidfVectorizer(stop_words='english',max_features=50000,min_df=2,max_df=0.95,sublinear_tf=True)
Xtr=v.fit_transform(tr)
knn=NearestNeighbors(n_neighbors=TOP_K+1,metric='cosine',algorithm='brute')
knn.fit(Xtr)
Xte=v.transform(te)
dists,inds=knn.kneighbors(Xte)
train_cats=train_df['CategoryForEval'].to_numpy()
metrics={'P@10':[],'R@10':[],'MAP@10':[],'NDCG@10':[]}; valid=0
for i in range(len(test_df)):
    qc=test_df.loc[i,'CategoryForEval']
    if not qc: continue
    mask=(train_cats==qc); tot=int(mask.sum())
    if tot==0: continue
    idx=inds[i][1:][:TOP_K]
    rel=mask[idx].astype(int)
    metrics['P@10'].append(precision_at_k(rel,TOP_K))
    metrics['R@10'].append(recall_at_k(rel,tot,TOP_K))
    metrics['MAP@10'].append(map_at_k(rel,tot,TOP_K))
    metrics['NDCG@10'].append(ndcg_at_k(rel,tot,TOP_K))
    valid+=1
print('--- KNN ---'); print('requetes eval:',valid)
for m,vv in metrics.items(): print(f'{m}: {np.mean(vv):.4f}')
""")

md("## Recommandation finale (exemple)")
code("""texts=build_text(df,include_category=True)
v=TfidfVectorizer(stop_words='english',max_features=50000,min_df=2,max_df=0.95,sublinear_tf=True)
X=v.fit_transform(texts)
svd=TruncatedSVD(n_components=100,random_state=RANDOM_STATE)
Xr=svd.fit_transform(X)
def recommend(name,top_n=10):
    matches=df[df['Name'].str.contains(name,case=False,regex=False,na=False)]
    if matches.empty:
        print('aucune recette'); return
    i=matches.index[0]
    sims=cosine_similarity(Xr[i:i+1],Xr).ravel()
    sims[i]=-1
    idx=np.argpartition(sims,-top_n)[-top_n:]
    idx=idx[np.argsort(sims[idx])[::-1]]
    res=df.iloc[idx][['RecipeId','Name','RecipeCategory']].copy()
    res['score']=sims[idx].round(4)
    print(res.to_string(index=False))
recommend('Oatmeal Cookies')
""")

nb['cells']=cells
with open('notebooks/03_recommendation_models.ipynb','w',encoding='utf-8') as f:
    nbf.write(nb,f)
print('done')
