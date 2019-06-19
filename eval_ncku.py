#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sound.SoundReader as reader
from sklearn.metrics import confusion_matrix, accuracy_score
import os
import numpy as np
import requests
import librosa

DATA_ROOT = '/data1/pipeline_dataset/audio'


# In[2]:


# Load your model  
# e.g. model = load_model()...

# Global Setting
segment_sec = 6         # length per segment, simulated cubo's audio callback
sample_rate = 16000     # Cubo device's defult sample rate
cry_limit_sec = 3

# detail Setting
win_size_sec = 0.3      # detect window length
sensitivity = 20        # Will detect segment with avg_dB
hit_rate = 0.55  # Cry ratio in segment
segment_cry_threshold = 2  # Threshold for segment cry windows numbers

# Global statistic calculation
cry_confidences = []
cry_total_detections = 0
cry_total_hit_in_cry = 0
other_total_detections = 0
cry_total_hit_in_other = 0


# In[3]:


# other: return 0, cry return 1
def testSegment(segment):

    # if detected cry
    return 1

    # if no cry
    return 0
    


# In[4]:


def feature_normalize(dataset):
    mu = np.mean(dataset,axis = 0)
    sigma = np.var(dataset,axis = 0)
    return (dataset - mu)/sigma

# return: 0 -> other, 1 - >cry
def getSegmentLabel(segment, index, meta):
    abs_seg = np.abs(segment)
    abs_seg += 1
    dBs = 20 * np.log10(abs_seg)
    dB = np.mean(abs_seg)
    if dB < sensitivity:
        return 0
        
    seg_size = len(segment)
    seg_start = index * sample_rate * segment_sec
    seg_end = seg_start + seg_size
    
    # get overlapped labeled segment
    overlaps = []
    for segment in meta['segments']:
        start_point = segment['start_ms'] * 16
        end_point = segment['end_ms'] * 16
        
        if seg_end < start_point or seg_start > end_point:
            continue
        else:
            overlaps.append(segment)
        
    seg_label_overlaps = []
    # get overlap ratio
    for label in overlaps:
        label_start = label['start_ms'] * 16
        label_end = label['end_ms'] * 16
        
        over_start = label_start
        if label_start < seg_start:
            over_start = seg_start
            
        over_end = label_end
        if label_end > seg_end:
            over_end = seg_end
            
        seg_label_overlaps.append((over_end - over_start, label['class']))
        
    is_cry = 0
    for duration, audio_class in seg_label_overlaps:
        if audio_class == 'cry' and duration > cry_limit_sec * sample_rate:
            return 1
    return 0
    
def test(audio_path, meta, segment_sec):
    print('\n --------- test {}, labeller: {} ---------'.format(meta['name'], meta['labeller']))
    _, sig, sound = reader.readWave(audio_path)
        
    segments = []
    segment_size = sample_rate * segment_sec
    
    # split audio to segment_sec chunk, no overlap
    for i in range(0, len(sig)-segment_size, segment_size):
        segments.append(sig[i:i+segment_size])
    
    # detect per chunk
    label_segments = []
    resut_segments = []
    
    for i in range(len(segments)): 
        segment = segments[i]
        
        label_is_cry = getSegmentLabel(segment, i, meta)
        predict_is_cry = testSegment(segment)
        print("seg {} label: {} -> predict: {}".format(i, label_is_cry, predict_is_cry))
        
        label_segments.append(label_is_cry)
        resut_segments.append(predict_is_cry)
    return label_segments, resut_segments


# In[ ]:


# get validation split to test
url = 'http://dev.yunyun.cloud:6001/metas/audio?split=val'
resp = requests.get(url)
metas = resp.json()['metas']
print('total val count: {}'.format(len(metas)))

correct_count = 0
fail_meta = []
all_label_is_cry = []
all_detect_is_cry = []
for meta in metas:
    if 'segments' not in meta:
        continue
        
    audio_folder = os.path.join(DATA_ROOT, 'batch_{}'.format(meta['batch_date']))
    if not os.path.isdir(audio_folder):
        os.makedirs(audio_folder)
        
    audio_path = os.path.join(audio_folder, meta['name'])
    if not os.path.isfile(audio_path):
        # download file
        url = os.path.join('http://dev.yunyun.cloud:6001/audio', meta['batch_date'], meta['name'])
        print('downloading .... {}'.format(url))
        r = requests.get(url) 
        with open(audio_path, 'wb') as f:
            f.write(r.content)
        
    labels, results = test(audio_path, meta, segment_sec)
            
    all_label_is_cry += labels
    all_detect_is_cry += results

acc = accuracy_score(all_label_is_cry, all_detect_is_cry)
print('\n -------- Report ----------')
print("Correct rate: {}".format(acc))
print(confusion_matrix(all_label_is_cry, all_detect_is_cry))
# print fail case

print(' -------- Error Detail -------')
for meta in metas:
    classes = []
    if 'segments' not in meta:
        print('{} no label segments'.format(meta['name']))


# In[ ]:




