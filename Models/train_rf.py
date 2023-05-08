import numpy as np
import pickle
from model import RF
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay,confusion_matrix,classification_report



directory = 'Models/post_processed/'
X_raw_50 = np.load(file=directory+'50/X_raw.npy')
y_raw_50 = np.load(file=directory+'50/Y_raw.npy')
X_filtered_50 = np.load(file=directory+'50/X_filtered.npy')
y_filtered_50 = np.load(file=directory+'50/Y_filtered.npy')
with open(directory+'50/labels','rb') as fp:
    columns_50 = pickle.load(fp)
    

X_raw_100 = np.load(file=directory+'100/X_raw.npy')
y_raw_100 = np.load(file=directory+'100/Y_raw.npy')
X_filtered_100 = np.load(file=directory+'100/X_filtered.npy')
y_filtered_100 = np.load(file=directory+'100/Y_filtered.npy')
with open(directory+'100/labels','rb') as fp:
    columns_100 = pickle.load(fp)
   

X_raw_150 = np.load(file=directory+'150/X_raw.npy')
y_raw_150 = np.load(file=directory+'150/Y_raw.npy')
X_filtered_150 = np.load(file=directory+'150/X_filtered.npy')
y_filtered_150 = np.load(file=directory+'150/Y_filtered.npy')
with open(directory+'150/labels','rb') as fp:
    columns_150 = pickle.load(fp)
    

X_raw_200 = np.load(file=directory+'200/X_raw.npy')
y_raw_200 = np.load(file=directory+'200/Y_raw.npy')
X_filtered_200 = np.load(file=directory+'200/X_filtered.npy')
y_filtered_200 = np.load(file=directory+'200/Y_filtered.npy')
with open(directory+'200/labels','rb') as fp:
    columns_200 = pickle.load(fp)
    
X_raw_250 = np.load(file=directory+'250/X_raw.npy')
y_raw_250 = np.load(file=directory+'250/Y_raw.npy')
X_filtered_250 = np.load(file=directory+'250/X_filtered.npy')
y_filtered_250 = np.load(file=directory+'250/Y_filtered.npy')
with open(directory+'250/labels','rb') as fp:
    columns_250 = pickle.load(fp)
    
X_raw_300 = np.load(file=directory+'300/X_raw.npy')
y_raw_300 = np.load(file=directory+'300/Y_raw.npy')
X_filtered_300 = np.load(file=directory+'300/X_filtered.npy')
y_filtered_300 = np.load(file=directory+'300/Y_filtered.npy')
with open(directory+'300/labels','rb') as fp:
    columns_300 = pickle.load(fp)
    
X_raw_350 = np.load(file=directory+'350/X_raw.npy')
y_raw_350 = np.load(file=directory+'350/Y_raw.npy')
X_filtered_350 = np.load(file=directory+'350/X_filtered.npy')
y_filtered_350 = np.load(file=directory+'350/Y_filtered.npy')
with open(directory+'350/labels','rb') as fp:
    columns_350 = pickle.load(fp)
    
X_raw_400 = np.load(file=directory+'400/X_raw.npy')
y_raw_400 = np.load(file=directory+'400/Y_raw.npy')
X_filtered_400 = np.load(file=directory+'400/X_filtered.npy')
y_filtered_400 = np.load(file=directory+'400/Y_filtered.npy')
with open(directory+'400/labels','rb') as fp:
    columns_400 = pickle.load(fp)

rf_50_raw = RF(X_raw_50,y_raw_50,'RF_50_Raw')
rf_50_filtered = RF(X_filtered_50,y_filtered_50,'RF_50_Filtered')
rf_100_raw = RF(X_raw_100,y_filtered_100,'RF_100_Raw')
rf_100_filtered = RF(X_filtered_100,y_raw_100,'RF_100_Filtered')
rf_150_raw = RF(X_raw_150,y_raw_150,'RF_150_Raw')
rf_150_filtered = RF(X_filtered_150,y_filtered_150,'RF_150_Filtered')
rf_200_raw = RF(X_raw_200,y_raw_200,'RF_200_Raw')
rf_200_filtered = RF(X_filtered_200,y_filtered_200,'RF_200_Filtered')
rf_250_raw = RF(X_raw_250,y_raw_250,'RF_250_Raw')
rf_250_filtered = RF(X_filtered_250,y_filtered_250,'RF_250_Filtered')
rf_300_raw = RF(X_raw_300,y_raw_300,'RF_300_Raw')
rf_300_filtered = RF(X_filtered_300,y_filtered_300,'RF_300_Filtered')
rf_350_raw = RF(X_raw_350,y_raw_350,'RF_350_raw')
rf_350_filtered = RF(X_filtered_350,y_filtered_350,'RF_400_Filtered')
rf_400_raw = RF(X_raw_400,y_raw_400,'RF_400_raw')
rf_400_filtered = RF(X_filtered_400,y_filtered_400,'RF_400_Filtered')

print(f"Now Training for window length:50 Raw & Filtered")
print('Training Raw')
rf_50_raw.train()
print('Training Filtered')
rf_50_filtered.train()
rf_50_raw_test = rf_50_raw.test()
rf_50_filtered_test = rf_50_filtered.test()



print(f"Now Training for window length:100 Raw & Filtered")
print('Training Raw')
rf_100_raw.train()
print('Training Filtered')
rf_100_filtered.train()
rf_100_raw_test = rf_100_raw.test()
rf_100_filtered_test = rf_100_filtered.test()

print(f"Now Training for window length:150 Raw & Filtered")
print('Training Raw')
rf_150_raw.train()
print('Training Filtered')
rf_150_filtered.train()
rf_150_raw_test = rf_150_raw.test()
rf_150_filtered_test = rf_150_filtered.test()

print(f"Now Training for window length:200 Raw & Filtered")
print('Training Raw')
rf_200_raw.train()
print('Training Filtered')
rf_200_filtered.train()
rf_200_raw_test = rf_200_raw.test()
rf_200_filtered_test = rf_200_filtered.test()


print(f"Now Training for window length:250 Raw & Filtered")
print('Training Raw')
rf_250_raw.train()
print('Training Filtered')
rf_250_filtered.train()
rf_250_raw_test = rf_250_raw.test()
rf_250_filtered_test = rf_250_filtered.test()


print(f"Now Training for window length:300 Raw & Filtered")
print('Training Raw')
rf_300_raw.train()
print('Training Filtered')
rf_300_filtered.train()
rf_300_raw_test = rf_300_raw.test()
rf_300_filtered_test = rf_300_filtered.test()


print(f"Now Training for window length:350 Raw & Filtered")
print('Training Raw')
rf_350_raw.train()
print('Training Filtered')
rf_350_filtered.train()
rf_350_raw_test = rf_350_raw.test()
rf_350_filtered_test = rf_350_filtered.test()


print(f"Now Training for window length:400 Raw & Filtered")
print('Training Raw')
rf_400_raw.train()
print('Training Filtered')
rf_400_filtered.train()
rf_400_raw_test = rf_400_raw.test()
rf_400_filtered_test = rf_400_filtered.test()

save_directory = 'Models/saved_models/'

accuracy_dataframe = pd.DataFrame(data = {'Segment Length(ms)':[50,50,100,100,150,150,200,200,250,250,300,300,350,350,400,400]
                                ,'Accuracy':[
                                            rf_50_raw_test,
                                            rf_50_filtered_test,
                                            rf_100_raw_test,
                                            rf_100_filtered_test,
                                            rf_150_raw_test,
                                            rf_150_filtered_test,
                                            rf_200_raw_test,
                                            rf_200_filtered_test,
                                            rf_250_raw_test,
                                            rf_250_filtered_test,
                                            rf_300_raw_test,
                                            rf_300_filtered_test,
                                            rf_350_raw_test,
                                            rf_350_filtered_test,
                                            rf_400_raw_test,
                                            rf_400_filtered_test,],
                                        'Type':['Raw','Filtered',
                                                'Raw','Filtered',
                                                'Raw','Filtered',
                                                'Raw','Filtered',
                                                'Raw','Filtered',
                                                'Raw','Filtered',
                                                'Raw','Filtered',
                                                'Raw','Filtered',
          ]})

print(accuracy_dataframe)
accuracy_dataframe.to_csv('Resources/model_results/rf_accuracy.csv',index=False)

accuracy_plot=sns.barplot(x=accuracy_dataframe['Segment Length(ms)'],y=accuracy_dataframe['Accuracy'],hue=accuracy_dataframe['Type'],)
sns.move_legend(accuracy_plot, "upper left", bbox_to_anchor=(1, 1))
accuracy_plot.figure.savefig('Resources/model_results/rf_accuracy_plot.png',bbox_inches='tight')



feature_names = pickle.load(open('Models/post_processed/feature_labels.sav','rb'))
importances_50_raw = rf_50_raw.model.feature_importances_
importances_50_filtered = rf_50_filtered.model.feature_importances_
importances_100_raw = rf_100_raw.model.feature_importances_
importances_100_filtered = rf_100_filtered.model.feature_importances_
importances_150_raw = rf_150_raw.model.feature_importances_
importances_150_filtered = rf_150_filtered.model.feature_importances_
importances_200_raw = rf_200_raw.model.feature_importances_
importances_200_filtered = rf_200_filtered.model.feature_importances_
importances_250_raw = rf_250_raw.model.feature_importances_
importances_250_filtered = rf_250_filtered.model.feature_importances_
importances_300_raw = rf_300_raw.model.feature_importances_
importances_300_filtered = rf_300_filtered.model.feature_importances_
importances_350_raw = rf_350_raw.model.feature_importances_
importances_350_filtered = rf_350_filtered.model.feature_importances_
importances_400_raw = rf_400_raw.model.feature_importances_
importances_400_filtered = rf_400_filtered.model.feature_importances_
avg_impurity_decrease = (importances_50_raw + importances_50_filtered + 
                         importances_100_raw + importances_100_filtered +
                         importances_150_raw + importances_150_filtered +
                         importances_200_raw + importances_200_filtered +
                         importances_250_raw + importances_250_filtered +
                         importances_300_raw + importances_300_filtered +
                         importances_350_raw + importances_350_filtered +
                         importances_400_raw + importances_400_filtered )/16
series = pd.Series(avg_impurity_decrease,feature_names)
series = series.sort_values(ascending=False)[:10]
plt.figure()
plt.barh(series.index,series.values)
plt.xlabel('Mean Decrease in Gini Entropy')
plt.ylabel('Feature Name')
plt.gcf().savefig('Resources/model_results/rf_mean_impurity.png',bbox_inches='tight')

freq_features={}
for key in feature_names:
    freq_features[key] = 0

def top_5_features(importances,feature_names):
    series=pd.Series(importances,feature_names)
    series=series.sort_values(ascending=False)[:5]
    top_5= series.index
    for feature in top_5:
        freq_features[feature]+=1
top_5_features(importances_50_raw,feature_names)
top_5_features(importances_50_filtered,feature_names)
top_5_features(importances_100_raw,feature_names)
top_5_features(importances_100_filtered,feature_names)
top_5_features(importances_150_raw,feature_names)
top_5_features(importances_150_filtered,feature_names)
top_5_features(importances_200_raw,feature_names)
top_5_features(importances_200_filtered,feature_names)
top_5_features(importances_250_raw,feature_names)
top_5_features(importances_250_filtered,feature_names)
top_5_features(importances_300_raw,feature_names)
top_5_features(importances_300_filtered,feature_names)
top_5_features(importances_350_raw,feature_names)
top_5_features(importances_350_filtered,feature_names)
top_5_features(importances_400_raw,feature_names)
top_5_features(importances_400_filtered,feature_names)
freq_series = pd.Series(freq_features)
freq_series = freq_series.sort_values(ascending=False)[:10]
plt.figure()
plt.barh(freq_series.index,freq_series.values)
plt.xlabel('Frequency in top 5')
plt.ylabel('Feature Name')
plt.gcf().savefig('Resources/model_results/rf_top5.png',bbox_inches='tight')

#save models
pickle.dump(rf_50_raw.model,open(save_directory+rf_50_raw.name+'.sav','wb'))
pickle.dump(rf_50_filtered.model,open(save_directory+rf_50_raw.name+'.sav','wb'))
pickle.dump(rf_100_raw.model,open(save_directory+rf_100_raw.name+'.sav','wb'))
pickle.dump(rf_100_filtered.model,open(save_directory+rf_100_filtered.name+'.sav','wb'))
pickle.dump(rf_150_raw.model,open(save_directory+rf_150_raw.name+'.sav','wb'))
pickle.dump(rf_150_filtered.model,open(save_directory+rf_150_filtered.name+'.sav','wb'))
pickle.dump(rf_200_raw.model,open(save_directory+rf_200_raw.name+'.sav','wb'))
pickle.dump(rf_200_filtered.model,open(save_directory+rf_200_filtered.name+'.sav','wb'))
pickle.dump(rf_250_raw.model,open(save_directory+rf_250_raw.name+'.sav','wb'))
pickle.dump(rf_250_filtered.model,open(save_directory+rf_250_filtered.name+'.sav','wb'))
pickle.dump(rf_300_raw.model,open(save_directory+rf_300_raw.name+'.sav','wb'))
pickle.dump(rf_300_filtered.model,open(save_directory+rf_300_filtered.name+'.sav','wb'))
pickle.dump(rf_350_raw.model,open(save_directory+rf_350_raw.name+'.sav','wb'))
pickle.dump(rf_350_filtered.model,open(save_directory+rf_350_filtered.name+'.sav','wb'))
pickle.dump(rf_400_raw.model,open(save_directory+rf_400_raw.name+'.sav','wb'))
pickle.dump(rf_400_filtered.model,open(save_directory+rf_400_filtered.name+'.sav','wb'))


