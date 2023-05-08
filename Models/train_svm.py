import numpy as np
import pickle
from model import RF,SVM
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

SVM_50_raw = SVM(X_raw_50,y_raw_50,'SVM_50_Raw')
SVM_50_filtered = SVM(X_filtered_50,y_filtered_50,'SVM_50_Filtered')
SVM_100_raw = SVM(X_raw_100,y_filtered_100,'SVM_100_Raw')
SVM_100_filtered = SVM(X_filtered_100,y_raw_100,'SVM_100_Filtered')
SVM_150_raw = SVM(X_raw_150,y_raw_150,'SVM_150_Raw')
SVM_150_filtered = SVM(X_filtered_150,y_filtered_150,'SVM_150_Filtered')
SVM_200_raw = SVM(X_raw_200,y_raw_200,'SVM_200_Raw')
SVM_200_filtered = SVM(X_filtered_200,y_filtered_200,'SVM_200_Filtered')
SVM_250_raw = SVM(X_raw_250,y_raw_250,'SVM_250_Raw')
SVM_250_filtered = SVM(X_filtered_250,y_filtered_250,'SVM_250_Filtered')
SVM_300_raw = SVM(X_raw_300,y_raw_300,'SVM_300_Raw')
SVM_300_filtered = SVM(X_filtered_300,y_filtered_300,'SVM_300_Filtered')
SVM_350_raw = SVM(X_raw_350,y_raw_350,'SVM_350_raw')
SVM_350_filtered = SVM(X_filtered_350,y_filtered_350,'SVM_400_Filtered')
SVM_400_raw = SVM(X_raw_400,y_raw_400,'SVM_400_raw')
SVM_400_filtered = SVM(X_filtered_400,y_filtered_400,'SVM_400_Filtered')

print(f"Now Training for window length:50 Raw & Filtered")
print('Training Raw')
SVM_50_raw.train()
print('Training Filtered')
SVM_50_filtered.train()
SVM_50_raw_test = SVM_50_raw.test()
SVM_50_filtered_test = SVM_50_filtered.test()



print(f"Now Training for window length:100 Raw & Filtered")
print('Training Raw')
SVM_100_raw.train()
print('Training Filtered')
SVM_100_filtered.train()
SVM_100_raw_test = SVM_100_raw.test()
SVM_100_filtered_test = SVM_100_filtered.test()

print(f"Now Training for window length:150 Raw & Filtered")
print('Training Raw')
SVM_150_raw.train()
print('Training Filtered')
SVM_150_filtered.train()
SVM_150_raw_test = SVM_150_raw.test()
SVM_150_filtered_test = SVM_150_filtered.test()

print(f"Now Training for window length:200 Raw & Filtered")
print('Training Raw')
SVM_200_raw.train()
print('Training Filtered')
SVM_200_filtered.train()
SVM_200_raw_test = SVM_200_raw.test()
SVM_200_filtered_test = SVM_200_filtered.test()


print(f"Now Training for window length:250 Raw & Filtered")
print('Training Raw')
SVM_250_raw.train()
print('Training Filtered')
SVM_250_filtered.train()
SVM_250_raw_test = SVM_250_raw.test()
SVM_250_filtered_test = SVM_250_filtered.test()


print(f"Now Training for window length:300 Raw & Filtered")
print('Training Raw')
SVM_300_raw.train()
print('Training Filtered')
SVM_300_filtered.train()
SVM_300_raw_test = SVM_300_raw.test()
SVM_300_filtered_test = SVM_300_filtered.test()


print(f"Now Training for window length:350 Raw & Filtered")
print('Training Raw')
SVM_350_raw.train()
print('Training Filtered')
SVM_350_filtered.train()
SVM_350_raw_test = SVM_350_raw.test()
SVM_350_filtered_test = SVM_350_filtered.test()


print(f"Now Training for window length:400 Raw & Filtered")
print('Training Raw')
SVM_400_raw.train()
print('Training Filtered')
SVM_400_filtered.train()
SVM_400_raw_test = SVM_400_raw.test()
SVM_400_filtered_test = SVM_400_filtered.test()

save_directory = 'Models/saved_models/'

accuracy_dataframe = pd.DataFrame(data = {'Segment Length(ms)':[50,50,100,100,150,150,200,200,250,250,300,300,350,350,400,400]
                                ,'Accuracy':[
                                            SVM_50_raw_test,
                                            SVM_50_filtered_test,
                                            SVM_100_raw_test,
                                            SVM_100_filtered_test,
                                            SVM_150_raw_test,
                                            SVM_150_filtered_test,
                                            SVM_200_raw_test,
                                            SVM_200_filtered_test,
                                            SVM_250_raw_test,
                                            SVM_250_filtered_test,
                                            SVM_300_raw_test,
                                            SVM_300_filtered_test,
                                            SVM_350_raw_test,
                                            SVM_350_filtered_test,
                                            SVM_400_raw_test,
                                            SVM_400_filtered_test,],
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
accuracy_dataframe.to_csv('Resources/model_results/svm_accuracy.csv',index=False)

accuracy_plot=sns.barplot(x=accuracy_dataframe['Segment Length(ms)'],y=accuracy_dataframe['Accuracy'],hue=accuracy_dataframe['Type'],)
sns.move_legend(accuracy_plot, "upper left", bbox_to_anchor=(1, 1))
accuracy_plot.figure.savefig('Resources/model_results/svm_accuracy_plot.png',bbox_inches='tight')

def get_classification_report(X,y_true,model):
    predictions = model.model.predict(X)
    cr = classification_report(y_true,predictions,target_names=[int(i) for i in model.categories],output_dict=True)
    return cr
cr= get_classification_report(SVM_400_raw.X_test,SVM_400_raw.y_test,SVM_400_raw)
cr_dataframe = pd.DataFrame(cr).transpose()
print(cr_dataframe.to_latex())
cr_dataframe.to_csv('Resources/model_results/svm_cr.csv',index=False)
with open('Resources/model_results/svm_cr_latex.txt','w') as tf:
    tf.write(cr_dataframe.to_latex())



#save models
pickle.dump(SVM_50_raw.model,open(save_directory+SVM_50_raw.name+'.sav','wb'))
pickle.dump(SVM_50_filtered.model,open(save_directory+SVM_50_raw.name+'.sav','wb'))
pickle.dump(SVM_100_raw.model,open(save_directory+SVM_100_raw.name+'.sav','wb'))
pickle.dump(SVM_100_filtered.model,open(save_directory+SVM_100_filtered.name+'.sav','wb'))
pickle.dump(SVM_150_raw.model,open(save_directory+SVM_150_raw.name+'.sav','wb'))
pickle.dump(SVM_150_filtered.model,open(save_directory+SVM_150_filtered.name+'.sav','wb'))
pickle.dump(SVM_200_raw.model,open(save_directory+SVM_200_raw.name+'.sav','wb'))
pickle.dump(SVM_200_filtered.model,open(save_directory+SVM_200_filtered.name+'.sav','wb'))
pickle.dump(SVM_250_raw.model,open(save_directory+SVM_250_raw.name+'.sav','wb'))
pickle.dump(SVM_250_filtered.model,open(save_directory+SVM_250_filtered.name+'.sav','wb'))
pickle.dump(SVM_300_raw.model,open(save_directory+SVM_300_raw.name+'.sav','wb'))
pickle.dump(SVM_300_filtered.model,open(save_directory+SVM_300_filtered.name+'.sav','wb'))
pickle.dump(SVM_350_raw.model,open(save_directory+SVM_350_raw.name+'.sav','wb'))
pickle.dump(SVM_350_filtered.model,open(save_directory+SVM_350_filtered.name+'.sav','wb'))
pickle.dump(SVM_400_raw.model,open(save_directory+SVM_400_raw.name+'.sav','wb'))
pickle.dump(SVM_400_filtered.model,open(save_directory+SVM_400_filtered.name+'.sav','wb'))