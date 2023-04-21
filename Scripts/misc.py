'''
Functions for miscnallenous tasks
'''

def validate_frames(facereader_df,imotions_df,openface_df):
    '''
    Ensures that datatframes have the same total no of frames,if they don't append rows to make same no of rows
    **Generally the case that Imotions has less frames for some reason for the same video
    '''

    if (len(facereader_df) > len(imotions_df)) and (len(openface_df) > len(imotions_df)):

        initial_length = len(imotions_df)
        for i in range(len(facereader_df) - len(imotions_df)):
            last_row = imotions_df.iloc[-1]
            imotions_df = imotions_df.append(last_row,ignore_index=True)
            imotions_df.loc[initial_length + i ]['frame'] = initial_length + i 

    return imotions_df
        

        
