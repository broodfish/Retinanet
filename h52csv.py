
# coding: utf-8

# In[16]:


import h5py
import construct_datasets as cd
import numpy as np
import pandas as pd
import cv2
import os


# In[2]:


train_dir = 'train/train/digitStruct.mat'


# In[3]:


f = h5py.File(train_dir,'r')


# In[4]:


row_dict = cd.get_bbox(0, f)
row_dict


# In[6]:


img_name = cd.get_name(0, f)
img_name


# In[7]:


row_dict['img_name'] = img_name
row_dict


# In[43]:


bbox_df = pd.DataFrame([], columns = ['img_name', 'left', 'top', 'width', 'height', 'label'])
for j in range(f['/digitStruct/bbox'].shape[0]):
    img_name = cd.get_name(j, f)
    row_dict = cd.get_bbox(j, f)
    row_dict['img_name'] = img_name
    all_rows = pd.DataFrame.from_dict(row_dict, orient = 'columns')
    all_rows['width'] = all_rows['left'] + all_rows['width']
    all_rows['height'] = all_rows['top'] + all_rows['height']
    all_rows = all_rows[['img_name', 'left', 'top', 'width', 'height', 'label']]
    bbox_df = pd.concat([bbox_df, all_rows])


# In[44]:


bbox_df = bbox_df.rename(columns={'label':'class_name'})
bbox_df = bbox_df.rename(columns={'left':'x1'})
bbox_df = bbox_df.rename(columns={'top':'y1'})
bbox_df = bbox_df.rename(columns={'width':'x2'})
bbox_df = bbox_df.rename(columns={'height':'y2'})


# In[45]:


bbox_df['x1'] = bbox_df['x1'].astype('int')
bbox_df['y2'] = bbox_df['y2'].astype('int')
bbox_df['y1'] = bbox_df['y1'].astype('int')
bbox_df['x2'] = bbox_df['x2'].astype('int')
bbox_df['class_name'] = bbox_df['class_name'].astype('int')


# In[46]:


bbox_df.head(5)


# In[47]:


bbox_df[bbox_df['img_name'] == '252.png']


# In[48]:


for i in range(len(bbox_df)):
    img = cv2.imread(os.path.join('train/train/',bbox_df.iloc[i, 0]))
    height = img.shape[0]
    width = img.shape[1]
    if bbox_df.iloc[i, 1] < 0:
        bbox_df.iloc[i, 1] = 0
    if bbox_df.iloc[i, 2] < 0:
        bbox_df.iloc[i, 2] = 0
    if bbox_df.iloc[i, 3] > width:
        bbox_df.iloc[i, 3] = width
    if bbox_df.iloc[i, 4] > height:
        bbox_df.iloc[i, 4] = height
bbox_df.iloc[544, 1]


# In[49]:


bbox_df[bbox_df['img_name'] == '724.png']


# In[50]:


bbox_df.head(5)


# In[51]:


bbox_df.to_csv('train_data.csv', index = False, header = False)


# In[136]:


class_name = [10, 1, 2, 3, 4, 5, 6, 7, 8, 9]
id = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
class_dict = {
                "class_name": class_name,
                "id": id
}
class_df = pd.DataFrame(class_dict)


# In[162]:


class_df.to_csv('train/train/class.csv', index = False, header = False)


# In[153]:


class_df.head(3)

