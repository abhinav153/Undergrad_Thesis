import pyxdf

filepath = 'Experimental_Setup/recordings/xdf_recordings/trial33.xdf'

data,header = pyxdf.load_xdf(filepath)
print(data[1]['time_series'])
