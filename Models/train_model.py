import numpy as np
import pickle
from model import RF


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



print(f"Now Training for window length:100 Raw & Filtered")
print('Training Raw')
rf_100_raw.train()
print('Training Filtered')
rf_100_filtered.train()

print(f"Now Training for window length:150 Raw & Filtered")
print('Training Raw')
rf_150_raw.train()
print('Training Filtered')
rf_150_filtered.train()

print(f"Now Training for window length:200 Raw & Filtered")
print('Training Raw')
rf_200_raw.train()
print('Training Filtered')
rf_200_filtered.train()


print(f"Now Training for window length:250 Raw & Filtered")
print('Training Raw')
rf_250_raw.train()
print('Training Filtered')
rf_250_filtered.train()


print(f"Now Training for window length:300 Raw & Filtered")
print('Training Raw')
rf_300_raw.train()
print('Training Filtered')
rf_300_filtered.train()


print(f"Now Training for window length:350 Raw & Filtered")
print('Training Raw')
rf_350_raw.train()
print('Training Filtered')
rf_350_filtered.train()


print(f"Now Training for window length:400 Raw & Filtered")
print('Training Raw')
rf_400_raw.train()
print('Training Filtered')
rf_400_filtered.train()

save_directory = 'Models/saved_models/'

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


