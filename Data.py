import pandas as pd
from sklearn.model_selection import train_test_split
#from imblearn.over_sampling import SMOTE
from catboost import CatBoostClassifier, Pool
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, classification_report, confusion_matrix,\
average_precision_score, precision_score, recall_score
import numpy as np
import json

df = pd.read_csv('creditcard.csv')


# cleaning duplicates and null data

print('\n\tNull data (amount)')
print(df.isnull().sum())
print('\n\tDuplicate data (amount)')
print(df.duplicated().sum())
print('\n\tNA data(amount)')
print(df.isna().sum())

df.drop_duplicates(inplace=True, ignore_index=True)
print('\n\tDuplicate data (amount)')
print(df.duplicated().sum())



# drop negative transaction
errores_monto = df[df['Amount'] < 0]
print('Negative transactions:',len(errores_monto))
q999 = df['Amount'].quantile(0.999)
print(f'99.9% amounts are less than: {q999}')
print(df[df['Amount'] < q999].head())
# if there any string in Amount, convert or drop
print(df.dtypes)


# drop class to train model
x = df.drop('Class', axis=1)
y = df['Class']

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, stratify=y, random_state=42)

print(y.value_counts())
print(y.value_counts()/len(df)*100)

#sm = SMOTE(sampling_strategy=0.10, random_state=42)
#x_res, y_res = sm.fit_resample(x_train, y_train)

#print('New classes:',pd.Series(y_res).value_counts())

model = CatBoostClassifier(
    task_type='GPU',
    iterations=600,
    learning_rate=0.01,
    depth=5,
    eval_metric='AUC',
    #class_weights=[1,15],
    auto_class_weights='SqrtBalanced',
    random_seed=42,
    verbose=100,
    loss_function='Logloss',
    custom_metric=['AUC','Recall'],
    l2_leaf_reg=15,
    #rsm=0.85,
    bagging_temperature=1.0
)


eval_set = (x_test, y_test)
#Early stopping to get best iteration
model.fit(
    x_train, y_train,
    eval_set=eval_set,
    use_best_model=True,      
    plot=False
)

results = model.get_evals_result()
print("learn:", list(results.get('learn', {}).keys()))
print("val:",   list(results.get('validation', {}).keys()))

train_pool = Pool(x_train, y_train)
val_pool = Pool(x_test, y_test)

metrics_to_track = ['AUC','Recall']

hist_train = model.eval_metrics(train_pool, metrics=metrics_to_track,
                                ntree_start=0, ntree_end=model.tree_count_)
hist_val = model.eval_metrics(val_pool, metrics=metrics_to_track, 
                              ntree_start=0, ntree_end=model.tree_count_)

train_auc = hist_train['AUC']
val_auc = hist_val['AUC']
train_recall = hist_train['Recall']
val_recall = hist_val['Recall']
train_loss = results['learn']['Logloss']
val_loss = results['validation']['Logloss']

# GRAPHIC TRAINING LOSS 
plt.plot(train_loss, label='Train Loss')
plt.plot(val_loss, label='Validation Loss')
plt.xlabel('Iterations')
plt.ylabel('Log Loss')
plt.title('Overfitting in CatBost (train vs validation)')
plt.legend()
plt.show()

# GRAPHIC AUC 
plt.figure(figsize=(8,4))
plt.plot(train_auc, label='Train AUC')
plt.plot(val_auc, label='Validation AUC')
plt.xlabel('Iterations')
plt.ylabel('AUC')
plt.title('AUC in CatBoost (Train vs Validation)')
plt.legend()
plt.show()

# GRAPHIC RECALL
plt.figure(figsize=(8,4))
plt.plot(train_recall, label='Train Recall')
plt.plot(val_recall, label='Validation Recall')
plt.xlabel('Iterations')
plt.ylabel('Recall')
plt.title('Recall in CatBoost (Train vs Validation)')
plt.legend()
plt.show()


# threshold block 
probs = model.predict_proba(x_test)[:, 1]
prec, rec, thr = precision_recall_curve(y_test, probs)
print('PR-AUC:', average_precision_score(y_test, probs))

# Business criterion

P0 = 0.90 # target precision
maskP = (prec[:-1] >= P0)
if maskP.any():
    thr_prec = thr[maskP].min()
    idxP = np.where(thr == thr_prec)[0][0]
    print(f'Threshold by precision≥{P0:.2f}: {thr_prec:.4f} | P={prec[idxP]:.4f} | R={rec[idxP]:.4f}')
else:
    idxP = np.argmax(prec[:-1]); thr_prec = thr[idxP]
    print(f'Maximum threshold achieved={thr_prec:.4f} | P={prec[idxP]:.4f} | R={rec[idxP]:.4f}')

R0 = 0.80
maskR = (rec[:-1] >= R0)
if maskR.any():
    thr_rec = thr[maskR].min()
    idxR = np.where(thr == thr_rec)[0][0]
    print(f'Threshold by Recall≥{R0:.2f}: {thr_rec:.4f} | P={prec[idxR]:.4f} | R={rec[idxR]:.4f}')
else:
    idxR = np.argmax(rec[:-1]); thr_rec = thr[idxR]
    print(f'Maximum Recall in thr={thr_rec:.4f} | P={prec[idxR]:.4f} | R={rec[idxR]:.4f}')

final_threshold = float(thr_prec)   # thr_rec if recall is the main target

# save  threshold to inference
with open('fraude_threshold.json', 'w') as f:
    json.dump({'threshold': final_threshold}, f)

def predict_final(model, X): # set threshold saved
    with open('fraude_threshold.json', 'r') as f:
        thr = json.load(f)['threshold']
    p = model.predict_proba(X)[:, 1]
    yhat = (p >= thr).astype(int)
    return yhat, p, thr

y_pred_final, y_proba_final, thr_used = predict_final(model, x_test)

print(f'\n[Final assessment] Threshold used: {thr_used:.4f}')
print(confusion_matrix(y_test, y_pred_final))
print(classification_report(y_test, y_pred_final, digits=4))

#  Recall/Precision for each iteration at final threshold 
steps = range(1, model.tree_count_ + 1)
val_rec_at_thr, val_prec_at_thr = [], []
for k in steps:
    proba_k = model.predict(val_pool, prediction_type='Probability', ntree_end=k)  # [n,2]
    ypk = (proba_k[:, 1] >= thr_used).astype(int)
    val_rec_at_thr.append(recall_score(y_test, ypk))
    val_prec_at_thr.append(precision_score(y_test, ypk, zero_division=0))

plt.figure(figsize=(8,4))
plt.plot(list(steps), val_recall,       label='Recall (thr=0.5, CatBoost)')
plt.plot(list(steps), val_rec_at_thr,   label=f'Recall @ thr={thr_used:.3f}')
plt.xlabel('Iterations'); plt.ylabel('Recall'); plt.title('Recall on Validation vs Iterations')
plt.legend()
plt.show()

plt.figure(figsize=(8,4))
plt.plot(list(steps), val_prec_at_thr,  label=f'Precision @ thr={thr_used:.3f}')
plt.xlabel('Iterations'); plt.ylabel('Precision'); plt.title('Precision on Validation vs Iterations')
plt.legend()
plt.show()

# saving model
model.save_model("fraude_model.cbm")
with open("fraude_threshold.json","w") as f:
    json.dump({"threshold": float(final_threshold)}, f)