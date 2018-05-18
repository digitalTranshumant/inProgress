
# Analysis of frequency distribution of the answers by languages

* answer_rel_freq.p is a parquet file coming from 'wikimotifs2/notebooks/analysis/similarity_heat_map.ipynb' 
* answer_rel_freq.p show the relative freq distribution of answers grouped by languages 

## Compute distances between pairs of languages


```python
%matplotlib inline
import pandas as pd
from sklearn.metrics import pairwise_distances

responses = pd.read_pickle('answer_rel_freq.p')
responses.index.name =''
```


```python
# Answers grouped by question (motiviation, familiarty, information depth)
questions = {'info_depth':["fact", "in-depth", "overview"],'motivation':['motivation_intrinsic_learning', 'motivation_media', 'motivation_bored/random',
       'motivation_conversation', 'motivation_current_event', 'motivation_personal_decision', 'motivation_work/school', 'motivation_other'],'familiarty':['familiar', 'unfamiliar']}
```


```python
#all the distance metrics for sparse vectors implemented in sklearn
#metrics = ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan']
metrics = ['cosine', 'euclidean', 'l1'] #selected 3 just for visualization purposes
```


```python
# compute pairwise distances
results = {}
for question,answers in questions.items():
    results[question] = {}
    for metric in metrics:
        results[question][metric] = pd.DataFrame(pairwise_distances(responses[answers],metric=metric).round(2),responses.index,responses.index)
```

## Visualization

### Visualizing distributions 



```python
for question,answers in questions.items():
    display(responses[answers].style.background_gradient( low=0, high=1,axis=1))
```


<style  type="text/css" >
    #T_8122ec92_5ab1_11e8_abac_1866dafa0bc4row0_col0 {
            background-color:  #fff7fb;
        }    #T_8122ec92_5ab1_11e8_abac_1866dafa0bc4row0_col1 {
            background-color:  #73a9cf;
        }    #T_8122ec92_5ab1_11e8_abac_1866dafa0bc4row0_col2 {
            background-color:  #99b8d8;
        }    #T_8122ec92_5ab1_11e8_abac_1866dafa0bc4row1_col0 {
            background-color:  #fff7fb;
        }    #T_8122ec92_5ab1_11e8_abac_1866dafa0bc4row1_col1 {
            background-color:  #83afd3;
        }    #T_8122ec92_5ab1_11e8_abac_1866dafa0bc4row1_col2 {
            background-color:  #73a9cf;
        }    #T_8122ec92_5ab1_11e8_abac_1866dafa0bc4row2_col0 {
            background-color:  #73a9cf;
        }    #T_8122ec92_5ab1_11e8_abac_1866dafa0bc4row2_col1 {
            background-color:  #fff7fb;
        }    #T_8122ec92_5ab1_11e8_abac_1866dafa0bc4row2_col2 {
            background-color:  #b1c2de;
        }    #T_8122ec92_5ab1_11e8_abac_1866dafa0bc4row3_col0 {
            background-color:  #73a9cf;
        }    #T_8122ec92_5ab1_11e8_abac_1866dafa0bc4row3_col1 {
            background-color:  #fff7fb;
        }    #T_8122ec92_5ab1_11e8_abac_1866dafa0bc4row3_col2 {
            background-color:  #8cb3d5;
        }    #T_8122ec92_5ab1_11e8_abac_1866dafa0bc4row4_col0 {
            background-color:  #73a9cf;
        }    #T_8122ec92_5ab1_11e8_abac_1866dafa0bc4row4_col1 {
            background-color:  #fff7fb;
        }    #T_8122ec92_5ab1_11e8_abac_1866dafa0bc4row4_col2 {
            background-color:  #81aed2;
        }    #T_8122ec92_5ab1_11e8_abac_1866dafa0bc4row5_col0 {
            background-color:  #fbf4f9;
        }    #T_8122ec92_5ab1_11e8_abac_1866dafa0bc4row5_col1 {
            background-color:  #fff7fb;
        }    #T_8122ec92_5ab1_11e8_abac_1866dafa0bc4row5_col2 {
            background-color:  #75a9cf;
        }    #T_8122ec92_5ab1_11e8_abac_1866dafa0bc4row6_col0 {
            background-color:  #f6eff7;
        }    #T_8122ec92_5ab1_11e8_abac_1866dafa0bc4row6_col1 {
            background-color:  #73a9cf;
        }    #T_8122ec92_5ab1_11e8_abac_1866dafa0bc4row6_col2 {
            background-color:  #fff7fb;
        }    #T_8122ec92_5ab1_11e8_abac_1866dafa0bc4row7_col0 {
            background-color:  #75a9cf;
        }    #T_8122ec92_5ab1_11e8_abac_1866dafa0bc4row7_col1 {
            background-color:  #fff7fb;
        }    #T_8122ec92_5ab1_11e8_abac_1866dafa0bc4row7_col2 {
            background-color:  #cccfe5;
        }    #T_8122ec92_5ab1_11e8_abac_1866dafa0bc4row8_col0 {
            background-color:  #fff7fb;
        }    #T_8122ec92_5ab1_11e8_abac_1866dafa0bc4row8_col1 {
            background-color:  #8bb2d4;
        }    #T_8122ec92_5ab1_11e8_abac_1866dafa0bc4row8_col2 {
            background-color:  #73a9cf;
        }    #T_8122ec92_5ab1_11e8_abac_1866dafa0bc4row9_col0 {
            background-color:  #73a9cf;
        }    #T_8122ec92_5ab1_11e8_abac_1866dafa0bc4row9_col1 {
            background-color:  #fff7fb;
        }    #T_8122ec92_5ab1_11e8_abac_1866dafa0bc4row9_col2 {
            background-color:  #dbdaeb;
        }    #T_8122ec92_5ab1_11e8_abac_1866dafa0bc4row10_col0 {
            background-color:  #73a9cf;
        }    #T_8122ec92_5ab1_11e8_abac_1866dafa0bc4row10_col1 {
            background-color:  #fff7fb;
        }    #T_8122ec92_5ab1_11e8_abac_1866dafa0bc4row10_col2 {
            background-color:  #e6e2ef;
        }    #T_8122ec92_5ab1_11e8_abac_1866dafa0bc4row11_col0 {
            background-color:  #73a9cf;
        }    #T_8122ec92_5ab1_11e8_abac_1866dafa0bc4row11_col1 {
            background-color:  #fff7fb;
        }    #T_8122ec92_5ab1_11e8_abac_1866dafa0bc4row11_col2 {
            background-color:  #fef6fa;
        }    #T_8122ec92_5ab1_11e8_abac_1866dafa0bc4row12_col0 {
            background-color:  #73a9cf;
        }    #T_8122ec92_5ab1_11e8_abac_1866dafa0bc4row12_col1 {
            background-color:  #fff7fb;
        }    #T_8122ec92_5ab1_11e8_abac_1866dafa0bc4row12_col2 {
            background-color:  #ede8f3;
        }    #T_8122ec92_5ab1_11e8_abac_1866dafa0bc4row13_col0 {
            background-color:  #73a9cf;
        }    #T_8122ec92_5ab1_11e8_abac_1866dafa0bc4row13_col1 {
            background-color:  #9ab8d8;
        }    #T_8122ec92_5ab1_11e8_abac_1866dafa0bc4row13_col2 {
            background-color:  #fff7fb;
        }</style>  
<table id="T_8122ec92_5ab1_11e8_abac_1866dafa0bc4" > 
<thead>    <tr> 
        <th class="blank level0" ></th> 
        <th class="col_heading level0 col0" >fact</th> 
        <th class="col_heading level0 col1" >in-depth</th> 
        <th class="col_heading level0 col2" >overview</th> 
    </tr>    <tr> 
        <th class="index_name level0" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
    </tr></thead> 
<tbody>    <tr> 
        <th id="T_8122ec92_5ab1_11e8_abac_1866dafa0bc4level0_row0" class="row_heading level0 row0" >ar</th> 
        <td id="T_8122ec92_5ab1_11e8_abac_1866dafa0bc4row0_col0" class="data row0 col0" >0.320624</td> 
        <td id="T_8122ec92_5ab1_11e8_abac_1866dafa0bc4row0_col1" class="data row0 col1" >0.341601</td> 
        <td id="T_8122ec92_5ab1_11e8_abac_1866dafa0bc4row0_col2" class="data row0 col2" >0.337774</td> 
    </tr>    <tr> 
        <th id="T_8122ec92_5ab1_11e8_abac_1866dafa0bc4level0_row1" class="row_heading level0 row1" >bn</th> 
        <td id="T_8122ec92_5ab1_11e8_abac_1866dafa0bc4row1_col0" class="data row1 col0" >0.266495</td> 
        <td id="T_8122ec92_5ab1_11e8_abac_1866dafa0bc4row1_col1" class="data row1 col1" >0.360919</td> 
        <td id="T_8122ec92_5ab1_11e8_abac_1866dafa0bc4row1_col2" class="data row1 col2" >0.368491</td> 
    </tr>    <tr> 
        <th id="T_8122ec92_5ab1_11e8_abac_1866dafa0bc4level0_row2" class="row_heading level0 row2" >de</th> 
        <td id="T_8122ec92_5ab1_11e8_abac_1866dafa0bc4row2_col0" class="data row2 col0" >0.430681</td> 
        <td id="T_8122ec92_5ab1_11e8_abac_1866dafa0bc4row2_col1" class="data row2 col1" >0.208277</td> 
        <td id="T_8122ec92_5ab1_11e8_abac_1866dafa0bc4row2_col2" class="data row2 col2" >0.361042</td> 
    </tr>    <tr> 
        <th id="T_8122ec92_5ab1_11e8_abac_1866dafa0bc4level0_row3" class="row_heading level0 row3" >en</th> 
        <td id="T_8122ec92_5ab1_11e8_abac_1866dafa0bc4row3_col0" class="data row3 col0" >0.377275</td> 
        <td id="T_8122ec92_5ab1_11e8_abac_1866dafa0bc4row3_col1" class="data row3 col1" >0.259612</td> 
        <td id="T_8122ec92_5ab1_11e8_abac_1866dafa0bc4row3_col2" class="data row3 col2" >0.363113</td> 
    </tr>    <tr> 
        <th id="T_8122ec92_5ab1_11e8_abac_1866dafa0bc4level0_row4" class="row_heading level0 row4" >es</th> 
        <td id="T_8122ec92_5ab1_11e8_abac_1866dafa0bc4row4_col0" class="data row4 col0" >0.347871</td> 
        <td id="T_8122ec92_5ab1_11e8_abac_1866dafa0bc4row4_col1" class="data row4 col1" >0.307009</td> 
        <td id="T_8122ec92_5ab1_11e8_abac_1866dafa0bc4row4_col2" class="data row4 col2" >0.34512</td> 
    </tr>    <tr> 
        <th id="T_8122ec92_5ab1_11e8_abac_1866dafa0bc4level0_row5" class="row_heading level0 row5" >he</th> 
        <td id="T_8122ec92_5ab1_11e8_abac_1866dafa0bc4row5_col0" class="data row5 col0" >0.267004</td> 
        <td id="T_8122ec92_5ab1_11e8_abac_1866dafa0bc4row5_col1" class="data row5 col1" >0.25665</td> 
        <td id="T_8122ec92_5ab1_11e8_abac_1866dafa0bc4row5_col2" class="data row5 col2" >0.476345</td> 
    </tr>    <tr> 
        <th id="T_8122ec92_5ab1_11e8_abac_1866dafa0bc4level0_row6" class="row_heading level0 row6" >hi</th> 
        <td id="T_8122ec92_5ab1_11e8_abac_1866dafa0bc4row6_col0" class="data row6 col0" >0.194948</td> 
        <td id="T_8122ec92_5ab1_11e8_abac_1866dafa0bc4row6_col1" class="data row6 col1" >0.677854</td> 
        <td id="T_8122ec92_5ab1_11e8_abac_1866dafa0bc4row6_col2" class="data row6 col2" >0.127198</td> 
    </tr>    <tr> 
        <th id="T_8122ec92_5ab1_11e8_abac_1866dafa0bc4level0_row7" class="row_heading level0 row7" >hu</th> 
        <td id="T_8122ec92_5ab1_11e8_abac_1866dafa0bc4row7_col0" class="data row7 col0" >0.424649</td> 
        <td id="T_8122ec92_5ab1_11e8_abac_1866dafa0bc4row7_col1" class="data row7 col1" >0.238138</td> 
        <td id="T_8122ec92_5ab1_11e8_abac_1866dafa0bc4row7_col2" class="data row7 col2" >0.337213</td> 
    </tr>    <tr> 
        <th id="T_8122ec92_5ab1_11e8_abac_1866dafa0bc4level0_row8" class="row_heading level0 row8" >ja</th> 
        <td id="T_8122ec92_5ab1_11e8_abac_1866dafa0bc4row8_col0" class="data row8 col0" >0.294525</td> 
        <td id="T_8122ec92_5ab1_11e8_abac_1866dafa0bc4row8_col1" class="data row8 col1" >0.349336</td> 
        <td id="T_8122ec92_5ab1_11e8_abac_1866dafa0bc4row8_col2" class="data row8 col2" >0.356139</td> 
    </tr>    <tr> 
        <th id="T_8122ec92_5ab1_11e8_abac_1866dafa0bc4level0_row9" class="row_heading level0 row9" >nl</th> 
        <td id="T_8122ec92_5ab1_11e8_abac_1866dafa0bc4row9_col0" class="data row9 col0" >0.470806</td> 
        <td id="T_8122ec92_5ab1_11e8_abac_1866dafa0bc4row9_col1" class="data row9 col1" >0.212334</td> 
        <td id="T_8122ec92_5ab1_11e8_abac_1866dafa0bc4row9_col2" class="data row9 col2" >0.316859</td> 
    </tr>    <tr> 
        <th id="T_8122ec92_5ab1_11e8_abac_1866dafa0bc4level0_row10" class="row_heading level0 row10" >ro</th> 
        <td id="T_8122ec92_5ab1_11e8_abac_1866dafa0bc4row10_col0" class="data row10 col0" >0.335253</td> 
        <td id="T_8122ec92_5ab1_11e8_abac_1866dafa0bc4row10_col1" class="data row10 col1" >0.331844</td> 
        <td id="T_8122ec92_5ab1_11e8_abac_1866dafa0bc4row10_col2" class="data row10 col2" >0.332903</td> 
    </tr>    <tr> 
        <th id="T_8122ec92_5ab1_11e8_abac_1866dafa0bc4level0_row11" class="row_heading level0 row11" >ru</th> 
        <td id="T_8122ec92_5ab1_11e8_abac_1866dafa0bc4row11_col0" class="data row11 col0" >0.351721</td> 
        <td id="T_8122ec92_5ab1_11e8_abac_1866dafa0bc4row11_col1" class="data row11 col1" >0.323908</td> 
        <td id="T_8122ec92_5ab1_11e8_abac_1866dafa0bc4row11_col2" class="data row11 col2" >0.324371</td> 
    </tr>    <tr> 
        <th id="T_8122ec92_5ab1_11e8_abac_1866dafa0bc4level0_row12" class="row_heading level0 row12" >uk</th> 
        <td id="T_8122ec92_5ab1_11e8_abac_1866dafa0bc4row12_col0" class="data row12 col0" >0.374382</td> 
        <td id="T_8122ec92_5ab1_11e8_abac_1866dafa0bc4row12_col1" class="data row12 col1" >0.304396</td> 
        <td id="T_8122ec92_5ab1_11e8_abac_1866dafa0bc4row12_col2" class="data row12 col2" >0.321222</td> 
    </tr>    <tr> 
        <th id="T_8122ec92_5ab1_11e8_abac_1866dafa0bc4level0_row13" class="row_heading level0 row13" >zh</th> 
        <td id="T_8122ec92_5ab1_11e8_abac_1866dafa0bc4row13_col0" class="data row13 col0" >0.380335</td> 
        <td id="T_8122ec92_5ab1_11e8_abac_1866dafa0bc4row13_col1" class="data row13 col1" >0.357488</td> 
        <td id="T_8122ec92_5ab1_11e8_abac_1866dafa0bc4row13_col2" class="data row13 col2" >0.262177</td> 
    </tr></tbody> 
</table> 



<style  type="text/css" >
    #T_8128db52_5ab1_11e8_abac_1866dafa0bc4row0_col0 {
            background-color:  #fff7fb;
        }    #T_8128db52_5ab1_11e8_abac_1866dafa0bc4row0_col1 {
            background-color:  #75a9cf;
        }    #T_8128db52_5ab1_11e8_abac_1866dafa0bc4row1_col0 {
            background-color:  #fff7fb;
        }    #T_8128db52_5ab1_11e8_abac_1866dafa0bc4row1_col1 {
            background-color:  #75a9cf;
        }    #T_8128db52_5ab1_11e8_abac_1866dafa0bc4row2_col0 {
            background-color:  #73a9cf;
        }    #T_8128db52_5ab1_11e8_abac_1866dafa0bc4row2_col1 {
            background-color:  #fff7fb;
        }    #T_8128db52_5ab1_11e8_abac_1866dafa0bc4row3_col0 {
            background-color:  #75a9cf;
        }    #T_8128db52_5ab1_11e8_abac_1866dafa0bc4row3_col1 {
            background-color:  #fff7fb;
        }    #T_8128db52_5ab1_11e8_abac_1866dafa0bc4row4_col0 {
            background-color:  #73a9cf;
        }    #T_8128db52_5ab1_11e8_abac_1866dafa0bc4row4_col1 {
            background-color:  #fff7fb;
        }    #T_8128db52_5ab1_11e8_abac_1866dafa0bc4row5_col0 {
            background-color:  #73a9cf;
        }    #T_8128db52_5ab1_11e8_abac_1866dafa0bc4row5_col1 {
            background-color:  #fff7fb;
        }    #T_8128db52_5ab1_11e8_abac_1866dafa0bc4row6_col0 {
            background-color:  #fff7fb;
        }    #T_8128db52_5ab1_11e8_abac_1866dafa0bc4row6_col1 {
            background-color:  #73a9cf;
        }    #T_8128db52_5ab1_11e8_abac_1866dafa0bc4row7_col0 {
            background-color:  #73a9cf;
        }    #T_8128db52_5ab1_11e8_abac_1866dafa0bc4row7_col1 {
            background-color:  #fff7fb;
        }    #T_8128db52_5ab1_11e8_abac_1866dafa0bc4row8_col0 {
            background-color:  #73a9cf;
        }    #T_8128db52_5ab1_11e8_abac_1866dafa0bc4row8_col1 {
            background-color:  #fff7fb;
        }    #T_8128db52_5ab1_11e8_abac_1866dafa0bc4row9_col0 {
            background-color:  #75a9cf;
        }    #T_8128db52_5ab1_11e8_abac_1866dafa0bc4row9_col1 {
            background-color:  #fff7fb;
        }    #T_8128db52_5ab1_11e8_abac_1866dafa0bc4row10_col0 {
            background-color:  #73a9cf;
        }    #T_8128db52_5ab1_11e8_abac_1866dafa0bc4row10_col1 {
            background-color:  #fff7fb;
        }    #T_8128db52_5ab1_11e8_abac_1866dafa0bc4row11_col0 {
            background-color:  #75a9cf;
        }    #T_8128db52_5ab1_11e8_abac_1866dafa0bc4row11_col1 {
            background-color:  #fff7fb;
        }    #T_8128db52_5ab1_11e8_abac_1866dafa0bc4row12_col0 {
            background-color:  #75a9cf;
        }    #T_8128db52_5ab1_11e8_abac_1866dafa0bc4row12_col1 {
            background-color:  #fff7fb;
        }    #T_8128db52_5ab1_11e8_abac_1866dafa0bc4row13_col0 {
            background-color:  #fff7fb;
        }    #T_8128db52_5ab1_11e8_abac_1866dafa0bc4row13_col1 {
            background-color:  #73a9cf;
        }</style>  
<table id="T_8128db52_5ab1_11e8_abac_1866dafa0bc4" > 
<thead>    <tr> 
        <th class="blank level0" ></th> 
        <th class="col_heading level0 col0" >familiar</th> 
        <th class="col_heading level0 col1" >unfamiliar</th> 
    </tr>    <tr> 
        <th class="index_name level0" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
    </tr></thead> 
<tbody>    <tr> 
        <th id="T_8128db52_5ab1_11e8_abac_1866dafa0bc4level0_row0" class="row_heading level0 row0" >ar</th> 
        <td id="T_8128db52_5ab1_11e8_abac_1866dafa0bc4row0_col0" class="data row0 col0" >0.46967</td> 
        <td id="T_8128db52_5ab1_11e8_abac_1866dafa0bc4row0_col1" class="data row0 col1" >0.53033</td> 
    </tr>    <tr> 
        <th id="T_8128db52_5ab1_11e8_abac_1866dafa0bc4level0_row1" class="row_heading level0 row1" >bn</th> 
        <td id="T_8128db52_5ab1_11e8_abac_1866dafa0bc4row1_col0" class="data row1 col0" >0.375873</td> 
        <td id="T_8128db52_5ab1_11e8_abac_1866dafa0bc4row1_col1" class="data row1 col1" >0.614365</td> 
    </tr>    <tr> 
        <th id="T_8128db52_5ab1_11e8_abac_1866dafa0bc4level0_row2" class="row_heading level0 row2" >de</th> 
        <td id="T_8128db52_5ab1_11e8_abac_1866dafa0bc4row2_col0" class="data row2 col0" >0.520131</td> 
        <td id="T_8128db52_5ab1_11e8_abac_1866dafa0bc4row2_col1" class="data row2 col1" >0.479869</td> 
    </tr>    <tr> 
        <th id="T_8128db52_5ab1_11e8_abac_1866dafa0bc4level0_row3" class="row_heading level0 row3" >en</th> 
        <td id="T_8128db52_5ab1_11e8_abac_1866dafa0bc4row3_col0" class="data row3 col0" >0.522804</td> 
        <td id="T_8128db52_5ab1_11e8_abac_1866dafa0bc4row3_col1" class="data row3 col1" >0.477196</td> 
    </tr>    <tr> 
        <th id="T_8128db52_5ab1_11e8_abac_1866dafa0bc4level0_row4" class="row_heading level0 row4" >es</th> 
        <td id="T_8128db52_5ab1_11e8_abac_1866dafa0bc4row4_col0" class="data row4 col0" >0.51331</td> 
        <td id="T_8128db52_5ab1_11e8_abac_1866dafa0bc4row4_col1" class="data row4 col1" >0.48669</td> 
    </tr>    <tr> 
        <th id="T_8128db52_5ab1_11e8_abac_1866dafa0bc4level0_row5" class="row_heading level0 row5" >he</th> 
        <td id="T_8128db52_5ab1_11e8_abac_1866dafa0bc4row5_col0" class="data row5 col0" >0.59211</td> 
        <td id="T_8128db52_5ab1_11e8_abac_1866dafa0bc4row5_col1" class="data row5 col1" >0.40789</td> 
    </tr>    <tr> 
        <th id="T_8128db52_5ab1_11e8_abac_1866dafa0bc4level0_row6" class="row_heading level0 row6" >hi</th> 
        <td id="T_8128db52_5ab1_11e8_abac_1866dafa0bc4row6_col0" class="data row6 col0" >0.44885</td> 
        <td id="T_8128db52_5ab1_11e8_abac_1866dafa0bc4row6_col1" class="data row6 col1" >0.55115</td> 
    </tr>    <tr> 
        <th id="T_8128db52_5ab1_11e8_abac_1866dafa0bc4level0_row7" class="row_heading level0 row7" >hu</th> 
        <td id="T_8128db52_5ab1_11e8_abac_1866dafa0bc4row7_col0" class="data row7 col0" >0.725021</td> 
        <td id="T_8128db52_5ab1_11e8_abac_1866dafa0bc4row7_col1" class="data row7 col1" >0.274979</td> 
    </tr>    <tr> 
        <th id="T_8128db52_5ab1_11e8_abac_1866dafa0bc4level0_row8" class="row_heading level0 row8" >ja</th> 
        <td id="T_8128db52_5ab1_11e8_abac_1866dafa0bc4row8_col0" class="data row8 col0" >0.534733</td> 
        <td id="T_8128db52_5ab1_11e8_abac_1866dafa0bc4row8_col1" class="data row8 col1" >0.465267</td> 
    </tr>    <tr> 
        <th id="T_8128db52_5ab1_11e8_abac_1866dafa0bc4level0_row9" class="row_heading level0 row9" >nl</th> 
        <td id="T_8128db52_5ab1_11e8_abac_1866dafa0bc4row9_col0" class="data row9 col0" >0.664811</td> 
        <td id="T_8128db52_5ab1_11e8_abac_1866dafa0bc4row9_col1" class="data row9 col1" >0.335189</td> 
    </tr>    <tr> 
        <th id="T_8128db52_5ab1_11e8_abac_1866dafa0bc4level0_row10" class="row_heading level0 row10" >ro</th> 
        <td id="T_8128db52_5ab1_11e8_abac_1866dafa0bc4row10_col0" class="data row10 col0" >0.594366</td> 
        <td id="T_8128db52_5ab1_11e8_abac_1866dafa0bc4row10_col1" class="data row10 col1" >0.405634</td> 
    </tr>    <tr> 
        <th id="T_8128db52_5ab1_11e8_abac_1866dafa0bc4level0_row11" class="row_heading level0 row11" >ru</th> 
        <td id="T_8128db52_5ab1_11e8_abac_1866dafa0bc4row11_col0" class="data row11 col0" >0.537973</td> 
        <td id="T_8128db52_5ab1_11e8_abac_1866dafa0bc4row11_col1" class="data row11 col1" >0.462027</td> 
    </tr>    <tr> 
        <th id="T_8128db52_5ab1_11e8_abac_1866dafa0bc4level0_row12" class="row_heading level0 row12" >uk</th> 
        <td id="T_8128db52_5ab1_11e8_abac_1866dafa0bc4row12_col0" class="data row12 col0" >0.727423</td> 
        <td id="T_8128db52_5ab1_11e8_abac_1866dafa0bc4row12_col1" class="data row12 col1" >0.272577</td> 
    </tr>    <tr> 
        <th id="T_8128db52_5ab1_11e8_abac_1866dafa0bc4level0_row13" class="row_heading level0 row13" >zh</th> 
        <td id="T_8128db52_5ab1_11e8_abac_1866dafa0bc4row13_col0" class="data row13 col0" >0.403172</td> 
        <td id="T_8128db52_5ab1_11e8_abac_1866dafa0bc4row13_col1" class="data row13 col1" >0.596828</td> 
    </tr></tbody> 
</table> 



<style  type="text/css" >
    #T_8130832a_5ab1_11e8_abac_1866dafa0bc4row0_col0 {
            background-color:  #75a9cf;
        }    #T_8130832a_5ab1_11e8_abac_1866dafa0bc4row0_col1 {
            background-color:  #c1cae2;
        }    #T_8130832a_5ab1_11e8_abac_1866dafa0bc4row0_col2 {
            background-color:  #ced0e6;
        }    #T_8130832a_5ab1_11e8_abac_1866dafa0bc4row0_col3 {
            background-color:  #e0deed;
        }    #T_8130832a_5ab1_11e8_abac_1866dafa0bc4row0_col4 {
            background-color:  #d2d2e7;
        }    #T_8130832a_5ab1_11e8_abac_1866dafa0bc4row0_col5 {
            background-color:  #ede7f2;
        }    #T_8130832a_5ab1_11e8_abac_1866dafa0bc4row0_col6 {
            background-color:  #ede8f3;
        }    #T_8130832a_5ab1_11e8_abac_1866dafa0bc4row0_col7 {
            background-color:  #fff7fb;
        }    #T_8130832a_5ab1_11e8_abac_1866dafa0bc4row1_col0 {
            background-color:  #73a9cf;
        }    #T_8130832a_5ab1_11e8_abac_1866dafa0bc4row1_col1 {
            background-color:  #e7e3f0;
        }    #T_8130832a_5ab1_11e8_abac_1866dafa0bc4row1_col2 {
            background-color:  #f4eef6;
        }    #T_8130832a_5ab1_11e8_abac_1866dafa0bc4row1_col3 {
            background-color:  #cdd0e5;
        }    #T_8130832a_5ab1_11e8_abac_1866dafa0bc4row1_col4 {
            background-color:  #cdd0e5;
        }    #T_8130832a_5ab1_11e8_abac_1866dafa0bc4row1_col5 {
            background-color:  #dfddec;
        }    #T_8130832a_5ab1_11e8_abac_1866dafa0bc4row1_col6 {
            background-color:  #dcdaeb;
        }    #T_8130832a_5ab1_11e8_abac_1866dafa0bc4row1_col7 {
            background-color:  #fff7fb;
        }    #T_8130832a_5ab1_11e8_abac_1866dafa0bc4row2_col0 {
            background-color:  #73a9cf;
        }    #T_8130832a_5ab1_11e8_abac_1866dafa0bc4row2_col1 {
            background-color:  #b3c3de;
        }    #T_8130832a_5ab1_11e8_abac_1866dafa0bc4row2_col2 {
            background-color:  #eae6f1;
        }    #T_8130832a_5ab1_11e8_abac_1866dafa0bc4row2_col3 {
            background-color:  #c2cbe2;
        }    #T_8130832a_5ab1_11e8_abac_1866dafa0bc4row2_col4 {
            background-color:  #f5eef6;
        }    #T_8130832a_5ab1_11e8_abac_1866dafa0bc4row2_col5 {
            background-color:  #faf2f8;
        }    #T_8130832a_5ab1_11e8_abac_1866dafa0bc4row2_col6 {
            background-color:  #c4cbe3;
        }    #T_8130832a_5ab1_11e8_abac_1866dafa0bc4row2_col7 {
            background-color:  #fff7fb;
        }    #T_8130832a_5ab1_11e8_abac_1866dafa0bc4row3_col0 {
            background-color:  #a8bedc;
        }    #T_8130832a_5ab1_11e8_abac_1866dafa0bc4row3_col1 {
            background-color:  #73a9cf;
        }    #T_8130832a_5ab1_11e8_abac_1866dafa0bc4row3_col2 {
            background-color:  #d4d4e8;
        }    #T_8130832a_5ab1_11e8_abac_1866dafa0bc4row3_col3 {
            background-color:  #d2d3e7;
        }    #T_8130832a_5ab1_11e8_abac_1866dafa0bc4row3_col4 {
            background-color:  #f3edf5;
        }    #T_8130832a_5ab1_11e8_abac_1866dafa0bc4row3_col5 {
            background-color:  #fcf4fa;
        }    #T_8130832a_5ab1_11e8_abac_1866dafa0bc4row3_col6 {
            background-color:  #fbf3f9;
        }    #T_8130832a_5ab1_11e8_abac_1866dafa0bc4row3_col7 {
            background-color:  #fff7fb;
        }    #T_8130832a_5ab1_11e8_abac_1866dafa0bc4row4_col0 {
            background-color:  #73a9cf;
        }    #T_8130832a_5ab1_11e8_abac_1866dafa0bc4row4_col1 {
            background-color:  #d0d1e6;
        }    #T_8130832a_5ab1_11e8_abac_1866dafa0bc4row4_col2 {
            background-color:  #e6e2ef;
        }    #T_8130832a_5ab1_11e8_abac_1866dafa0bc4row4_col3 {
            background-color:  #c9cee4;
        }    #T_8130832a_5ab1_11e8_abac_1866dafa0bc4row4_col4 {
            background-color:  #eae6f1;
        }    #T_8130832a_5ab1_11e8_abac_1866dafa0bc4row4_col5 {
            background-color:  #fcf4fa;
        }    #T_8130832a_5ab1_11e8_abac_1866dafa0bc4row4_col6 {
            background-color:  #93b5d6;
        }    #T_8130832a_5ab1_11e8_abac_1866dafa0bc4row4_col7 {
            background-color:  #fff7fb;
        }    #T_8130832a_5ab1_11e8_abac_1866dafa0bc4row5_col0 {
            background-color:  #73a9cf;
        }    #T_8130832a_5ab1_11e8_abac_1866dafa0bc4row5_col1 {
            background-color:  #99b8d8;
        }    #T_8130832a_5ab1_11e8_abac_1866dafa0bc4row5_col2 {
            background-color:  #d9d8ea;
        }    #T_8130832a_5ab1_11e8_abac_1866dafa0bc4row5_col3 {
            background-color:  #83afd3;
        }    #T_8130832a_5ab1_11e8_abac_1866dafa0bc4row5_col4 {
            background-color:  #f8f1f8;
        }    #T_8130832a_5ab1_11e8_abac_1866dafa0bc4row5_col5 {
            background-color:  #fdf5fa;
        }    #T_8130832a_5ab1_11e8_abac_1866dafa0bc4row5_col6 {
            background-color:  #e3e0ee;
        }    #T_8130832a_5ab1_11e8_abac_1866dafa0bc4row5_col7 {
            background-color:  #fff7fb;
        }    #T_8130832a_5ab1_11e8_abac_1866dafa0bc4row6_col0 {
            background-color:  #73a9cf;
        }    #T_8130832a_5ab1_11e8_abac_1866dafa0bc4row6_col1 {
            background-color:  #f7f0f7;
        }    #T_8130832a_5ab1_11e8_abac_1866dafa0bc4row6_col2 {
            background-color:  #f7f0f7;
        }    #T_8130832a_5ab1_11e8_abac_1866dafa0bc4row6_col3 {
            background-color:  #d4d4e8;
        }    #T_8130832a_5ab1_11e8_abac_1866dafa0bc4row6_col4 {
            background-color:  #e5e1ef;
        }    #T_8130832a_5ab1_11e8_abac_1866dafa0bc4row6_col5 {
            background-color:  #f0eaf4;
        }    #T_8130832a_5ab1_11e8_abac_1866dafa0bc4row6_col6 {
            background-color:  #d6d6e9;
        }    #T_8130832a_5ab1_11e8_abac_1866dafa0bc4row6_col7 {
            background-color:  #fff7fb;
        }    #T_8130832a_5ab1_11e8_abac_1866dafa0bc4row7_col0 {
            background-color:  #73a9cf;
        }    #T_8130832a_5ab1_11e8_abac_1866dafa0bc4row7_col1 {
            background-color:  #abbfdc;
        }    #T_8130832a_5ab1_11e8_abac_1866dafa0bc4row7_col2 {
            background-color:  #dad9ea;
        }    #T_8130832a_5ab1_11e8_abac_1866dafa0bc4row7_col3 {
            background-color:  #b0c2de;
        }    #T_8130832a_5ab1_11e8_abac_1866dafa0bc4row7_col4 {
            background-color:  #ced0e6;
        }    #T_8130832a_5ab1_11e8_abac_1866dafa0bc4row7_col5 {
            background-color:  #b9c6e0;
        }    #T_8130832a_5ab1_11e8_abac_1866dafa0bc4row7_col6 {
            background-color:  #e3e0ee;
        }    #T_8130832a_5ab1_11e8_abac_1866dafa0bc4row7_col7 {
            background-color:  #fff7fb;
        }    #T_8130832a_5ab1_11e8_abac_1866dafa0bc4row8_col0 {
            background-color:  #8bb2d4;
        }    #T_8130832a_5ab1_11e8_abac_1866dafa0bc4row8_col1 {
            background-color:  #73a9cf;
        }    #T_8130832a_5ab1_11e8_abac_1866dafa0bc4row8_col2 {
            background-color:  #a4bcda;
        }    #T_8130832a_5ab1_11e8_abac_1866dafa0bc4row8_col3 {
            background-color:  #cacee5;
        }    #T_8130832a_5ab1_11e8_abac_1866dafa0bc4row8_col4 {
            background-color:  #d3d4e7;
        }    #T_8130832a_5ab1_11e8_abac_1866dafa0bc4row8_col5 {
            background-color:  #e7e3f0;
        }    #T_8130832a_5ab1_11e8_abac_1866dafa0bc4row8_col6 {
            background-color:  #ebe6f2;
        }    #T_8130832a_5ab1_11e8_abac_1866dafa0bc4row8_col7 {
            background-color:  #fff7fb;
        }    #T_8130832a_5ab1_11e8_abac_1866dafa0bc4row9_col0 {
            background-color:  #b5c4df;
        }    #T_8130832a_5ab1_11e8_abac_1866dafa0bc4row9_col1 {
            background-color:  #73a9cf;
        }    #T_8130832a_5ab1_11e8_abac_1866dafa0bc4row9_col2 {
            background-color:  #eee9f3;
        }    #T_8130832a_5ab1_11e8_abac_1866dafa0bc4row9_col3 {
            background-color:  #a9bfdc;
        }    #T_8130832a_5ab1_11e8_abac_1866dafa0bc4row9_col4 {
            background-color:  #efe9f3;
        }    #T_8130832a_5ab1_11e8_abac_1866dafa0bc4row9_col5 {
            background-color:  #fff7fb;
        }    #T_8130832a_5ab1_11e8_abac_1866dafa0bc4row9_col6 {
            background-color:  #d3d4e7;
        }    #T_8130832a_5ab1_11e8_abac_1866dafa0bc4row9_col7 {
            background-color:  #f1ebf5;
        }    #T_8130832a_5ab1_11e8_abac_1866dafa0bc4row10_col0 {
            background-color:  #75a9cf;
        }    #T_8130832a_5ab1_11e8_abac_1866dafa0bc4row10_col1 {
            background-color:  #d8d7e9;
        }    #T_8130832a_5ab1_11e8_abac_1866dafa0bc4row10_col2 {
            background-color:  #f6eff7;
        }    #T_8130832a_5ab1_11e8_abac_1866dafa0bc4row10_col3 {
            background-color:  #c8cde4;
        }    #T_8130832a_5ab1_11e8_abac_1866dafa0bc4row10_col4 {
            background-color:  #e7e3f0;
        }    #T_8130832a_5ab1_11e8_abac_1866dafa0bc4row10_col5 {
            background-color:  #f3edf5;
        }    #T_8130832a_5ab1_11e8_abac_1866dafa0bc4row10_col6 {
            background-color:  #d4d4e8;
        }    #T_8130832a_5ab1_11e8_abac_1866dafa0bc4row10_col7 {
            background-color:  #fff7fb;
        }    #T_8130832a_5ab1_11e8_abac_1866dafa0bc4row11_col0 {
            background-color:  #75a9cf;
        }    #T_8130832a_5ab1_11e8_abac_1866dafa0bc4row11_col1 {
            background-color:  #afc1dd;
        }    #T_8130832a_5ab1_11e8_abac_1866dafa0bc4row11_col2 {
            background-color:  #eae6f1;
        }    #T_8130832a_5ab1_11e8_abac_1866dafa0bc4row11_col3 {
            background-color:  #d4d4e8;
        }    #T_8130832a_5ab1_11e8_abac_1866dafa0bc4row11_col4 {
            background-color:  #e6e2ef;
        }    #T_8130832a_5ab1_11e8_abac_1866dafa0bc4row11_col5 {
            background-color:  #f4edf6;
        }    #T_8130832a_5ab1_11e8_abac_1866dafa0bc4row11_col6 {
            background-color:  #f0eaf4;
        }    #T_8130832a_5ab1_11e8_abac_1866dafa0bc4row11_col7 {
            background-color:  #fff7fb;
        }    #T_8130832a_5ab1_11e8_abac_1866dafa0bc4row12_col0 {
            background-color:  #73a9cf;
        }    #T_8130832a_5ab1_11e8_abac_1866dafa0bc4row12_col1 {
            background-color:  #cdd0e5;
        }    #T_8130832a_5ab1_11e8_abac_1866dafa0bc4row12_col2 {
            background-color:  #f2ecf5;
        }    #T_8130832a_5ab1_11e8_abac_1866dafa0bc4row12_col3 {
            background-color:  #c8cde4;
        }    #T_8130832a_5ab1_11e8_abac_1866dafa0bc4row12_col4 {
            background-color:  #c8cde4;
        }    #T_8130832a_5ab1_11e8_abac_1866dafa0bc4row12_col5 {
            background-color:  #e1dfed;
        }    #T_8130832a_5ab1_11e8_abac_1866dafa0bc4row12_col6 {
            background-color:  #d7d6e9;
        }    #T_8130832a_5ab1_11e8_abac_1866dafa0bc4row12_col7 {
            background-color:  #fff7fb;
        }    #T_8130832a_5ab1_11e8_abac_1866dafa0bc4row13_col0 {
            background-color:  #75a9cf;
        }    #T_8130832a_5ab1_11e8_abac_1866dafa0bc4row13_col1 {
            background-color:  #84b0d3;
        }    #T_8130832a_5ab1_11e8_abac_1866dafa0bc4row13_col2 {
            background-color:  #bcc7e1;
        }    #T_8130832a_5ab1_11e8_abac_1866dafa0bc4row13_col3 {
            background-color:  #b8c6e0;
        }    #T_8130832a_5ab1_11e8_abac_1866dafa0bc4row13_col4 {
            background-color:  #cdd0e5;
        }    #T_8130832a_5ab1_11e8_abac_1866dafa0bc4row13_col5 {
            background-color:  #ede7f2;
        }    #T_8130832a_5ab1_11e8_abac_1866dafa0bc4row13_col6 {
            background-color:  #d6d6e9;
        }    #T_8130832a_5ab1_11e8_abac_1866dafa0bc4row13_col7 {
            background-color:  #fff7fb;
        }</style>  
<table id="T_8130832a_5ab1_11e8_abac_1866dafa0bc4" > 
<thead>    <tr> 
        <th class="blank level0" ></th> 
        <th class="col_heading level0 col0" >motivation_intrinsic_learning</th> 
        <th class="col_heading level0 col1" >motivation_media</th> 
        <th class="col_heading level0 col2" >motivation_bored/random</th> 
        <th class="col_heading level0 col3" >motivation_conversation</th> 
        <th class="col_heading level0 col4" >motivation_current_event</th> 
        <th class="col_heading level0 col5" >motivation_personal_decision</th> 
        <th class="col_heading level0 col6" >motivation_work/school</th> 
        <th class="col_heading level0 col7" >motivation_other</th> 
    </tr>    <tr> 
        <th class="index_name level0" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
    </tr></thead> 
<tbody>    <tr> 
        <th id="T_8130832a_5ab1_11e8_abac_1866dafa0bc4level0_row0" class="row_heading level0 row0" >ar</th> 
        <td id="T_8130832a_5ab1_11e8_abac_1866dafa0bc4row0_col0" class="data row0 col0" >0.439278</td> 
        <td id="T_8130832a_5ab1_11e8_abac_1866dafa0bc4row0_col1" class="data row0 col1" >0.27851</td> 
        <td id="T_8130832a_5ab1_11e8_abac_1866dafa0bc4row0_col2" class="data row0 col2" >0.246633</td> 
        <td id="T_8130832a_5ab1_11e8_abac_1866dafa0bc4row0_col3" class="data row0 col3" >0.18329</td> 
        <td id="T_8130832a_5ab1_11e8_abac_1866dafa0bc4row0_col4" class="data row0 col4" >0.237035</td> 
        <td id="T_8130832a_5ab1_11e8_abac_1866dafa0bc4row0_col5" class="data row0 col5" >0.142975</td> 
        <td id="T_8130832a_5ab1_11e8_abac_1866dafa0bc4row0_col6" class="data row0 col6" >0.137219</td> 
        <td id="T_8130832a_5ab1_11e8_abac_1866dafa0bc4row0_col7" class="data row0 col7" >0.0443927</td> 
    </tr>    <tr> 
        <th id="T_8130832a_5ab1_11e8_abac_1866dafa0bc4level0_row1" class="row_heading level0 row1" >bn</th> 
        <td id="T_8130832a_5ab1_11e8_abac_1866dafa0bc4row1_col0" class="data row1 col0" >0.54833</td> 
        <td id="T_8130832a_5ab1_11e8_abac_1866dafa0bc4row1_col1" class="data row1 col1" >0.195927</td> 
        <td id="T_8130832a_5ab1_11e8_abac_1866dafa0bc4row1_col2" class="data row1 col2" >0.121325</td> 
        <td id="T_8130832a_5ab1_11e8_abac_1866dafa0bc4row1_col3" class="data row1 col3" >0.309081</td> 
        <td id="T_8130832a_5ab1_11e8_abac_1866dafa0bc4row1_col4" class="data row1 col4" >0.309311</td> 
        <td id="T_8130832a_5ab1_11e8_abac_1866dafa0bc4row1_col5" class="data row1 col5" >0.236975</td> 
        <td id="T_8130832a_5ab1_11e8_abac_1866dafa0bc4row1_col6" class="data row1 col6" >0.249234</td> 
        <td id="T_8130832a_5ab1_11e8_abac_1866dafa0bc4row1_col7" class="data row1 col7" >0.051178</td> 
    </tr>    <tr> 
        <th id="T_8130832a_5ab1_11e8_abac_1866dafa0bc4level0_row2" class="row_heading level0 row2" >de</th> 
        <td id="T_8130832a_5ab1_11e8_abac_1866dafa0bc4row2_col0" class="data row2 col0" >0.311934</td> 
        <td id="T_8130832a_5ab1_11e8_abac_1866dafa0bc4row2_col1" class="data row2 col1" >0.238805</td> 
        <td id="T_8130832a_5ab1_11e8_abac_1866dafa0bc4row2_col2" class="data row2 col2" >0.146312</td> 
        <td id="T_8130832a_5ab1_11e8_abac_1866dafa0bc4row2_col3" class="data row2 col3" >0.217664</td> 
        <td id="T_8130832a_5ab1_11e8_abac_1866dafa0bc4row2_col4" class="data row2 col4" >0.115091</td> 
        <td id="T_8130832a_5ab1_11e8_abac_1866dafa0bc4row2_col5" class="data row2 col5" >0.102162</td> 
        <td id="T_8130832a_5ab1_11e8_abac_1866dafa0bc4row2_col6" class="data row2 col6" >0.21596</td> 
        <td id="T_8130832a_5ab1_11e8_abac_1866dafa0bc4row2_col7" class="data row2 col7" >0.0849367</td> 
    </tr>    <tr> 
        <th id="T_8130832a_5ab1_11e8_abac_1866dafa0bc4level0_row3" class="row_heading level0 row3" >en</th> 
        <td id="T_8130832a_5ab1_11e8_abac_1866dafa0bc4row3_col0" class="data row3 col0" >0.268912</td> 
        <td id="T_8130832a_5ab1_11e8_abac_1866dafa0bc4row3_col1" class="data row3 col1" >0.333193</td> 
        <td id="T_8130832a_5ab1_11e8_abac_1866dafa0bc4row3_col2" class="data row3 col2" >0.200111</td> 
        <td id="T_8130832a_5ab1_11e8_abac_1866dafa0bc4row3_col3" class="data row3 col3" >0.205217</td> 
        <td id="T_8130832a_5ab1_11e8_abac_1866dafa0bc4row3_col4" class="data row3 col4" >0.126707</td> 
        <td id="T_8130832a_5ab1_11e8_abac_1866dafa0bc4row3_col5" class="data row3 col5" >0.0964685</td> 
        <td id="T_8130832a_5ab1_11e8_abac_1866dafa0bc4row3_col6" class="data row3 col6" >0.0997539</td> 
        <td id="T_8130832a_5ab1_11e8_abac_1866dafa0bc4row3_col7" class="data row3 col7" >0.086249</td> 
    </tr>    <tr> 
        <th id="T_8130832a_5ab1_11e8_abac_1866dafa0bc4level0_row4" class="row_heading level0 row4" >es</th> 
        <td id="T_8130832a_5ab1_11e8_abac_1866dafa0bc4row4_col0" class="data row4 col0" >0.346677</td> 
        <td id="T_8130832a_5ab1_11e8_abac_1866dafa0bc4row4_col1" class="data row4 col1" >0.211907</td> 
        <td id="T_8130832a_5ab1_11e8_abac_1866dafa0bc4row4_col2" class="data row4 col2" >0.156995</td> 
        <td id="T_8130832a_5ab1_11e8_abac_1866dafa0bc4row4_col3" class="data row4 col3" >0.221241</td> 
        <td id="T_8130832a_5ab1_11e8_abac_1866dafa0bc4row4_col4" class="data row4 col4" >0.147702</td> 
        <td id="T_8130832a_5ab1_11e8_abac_1866dafa0bc4row4_col5" class="data row4 col5" >0.0853688</td> 
        <td id="T_8130832a_5ab1_11e8_abac_1866dafa0bc4row4_col6" class="data row4 col6" >0.305932</td> 
        <td id="T_8130832a_5ab1_11e8_abac_1866dafa0bc4row4_col7" class="data row4 col7" >0.0731987</td> 
    </tr>    <tr> 
        <th id="T_8130832a_5ab1_11e8_abac_1866dafa0bc4level0_row5" class="row_heading level0 row5" >he</th> 
        <td id="T_8130832a_5ab1_11e8_abac_1866dafa0bc4row5_col0" class="data row5 col0" >0.295599</td> 
        <td id="T_8130832a_5ab1_11e8_abac_1866dafa0bc4row5_col1" class="data row5 col1" >0.25829</td> 
        <td id="T_8130832a_5ab1_11e8_abac_1866dafa0bc4row5_col2" class="data row5 col2" >0.17755</td> 
        <td id="T_8130832a_5ab1_11e8_abac_1866dafa0bc4row5_col3" class="data row5 col3" >0.280747</td> 
        <td id="T_8130832a_5ab1_11e8_abac_1866dafa0bc4row5_col4" class="data row5 col4" >0.113274</td> 
        <td id="T_8130832a_5ab1_11e8_abac_1866dafa0bc4row5_col5" class="data row5 col5" >0.100167</td> 
        <td id="T_8130832a_5ab1_11e8_abac_1866dafa0bc4row5_col6" class="data row5 col6" >0.160406</td> 
        <td id="T_8130832a_5ab1_11e8_abac_1866dafa0bc4row5_col7" class="data row5 col7" >0.0931851</td> 
    </tr>    <tr> 
        <th id="T_8130832a_5ab1_11e8_abac_1866dafa0bc4level0_row6" class="row_heading level0 row6" >hi</th> 
        <td id="T_8130832a_5ab1_11e8_abac_1866dafa0bc4row6_col0" class="data row6 col0" >0.478412</td> 
        <td id="T_8130832a_5ab1_11e8_abac_1866dafa0bc4row6_col1" class="data row6 col1" >0.0924416</td> 
        <td id="T_8130832a_5ab1_11e8_abac_1866dafa0bc4row6_col2" class="data row6 col2" >0.0877094</td> 
        <td id="T_8130832a_5ab1_11e8_abac_1866dafa0bc4row6_col3" class="data row6 col3" >0.243353</td> 
        <td id="T_8130832a_5ab1_11e8_abac_1866dafa0bc4row6_col4" class="data row6 col4" >0.179087</td> 
        <td id="T_8130832a_5ab1_11e8_abac_1866dafa0bc4row6_col5" class="data row6 col5" >0.129431</td> 
        <td id="T_8130832a_5ab1_11e8_abac_1866dafa0bc4row6_col6" class="data row6 col6" >0.239233</td> 
        <td id="T_8130832a_5ab1_11e8_abac_1866dafa0bc4row6_col7" class="data row6 col7" >0.0412398</td> 
    </tr>    <tr> 
        <th id="T_8130832a_5ab1_11e8_abac_1866dafa0bc4level0_row7" class="row_heading level0 row7" >hu</th> 
        <td id="T_8130832a_5ab1_11e8_abac_1866dafa0bc4row7_col0" class="data row7 col0" >0.381441</td> 
        <td id="T_8130832a_5ab1_11e8_abac_1866dafa0bc4row7_col1" class="data row7 col1" >0.286445</td> 
        <td id="T_8130832a_5ab1_11e8_abac_1866dafa0bc4row7_col2" class="data row7 col2" >0.182997</td> 
        <td id="T_8130832a_5ab1_11e8_abac_1866dafa0bc4row7_col3" class="data row7 col3" >0.277908</td> 
        <td id="T_8130832a_5ab1_11e8_abac_1866dafa0bc4row7_col4" class="data row7 col4" >0.217265</td> 
        <td id="T_8130832a_5ab1_11e8_abac_1866dafa0bc4row7_col5" class="data row7 col5" >0.257465</td> 
        <td id="T_8130832a_5ab1_11e8_abac_1866dafa0bc4row7_col6" class="data row7 col6" >0.155882</td> 
        <td id="T_8130832a_5ab1_11e8_abac_1866dafa0bc4row7_col7" class="data row7 col7" >0.0436678</td> 
    </tr>    <tr> 
        <th id="T_8130832a_5ab1_11e8_abac_1866dafa0bc4level0_row8" class="row_heading level0 row8" >ja</th> 
        <td id="T_8130832a_5ab1_11e8_abac_1866dafa0bc4row8_col0" class="data row8 col0" >0.247616</td> 
        <td id="T_8130832a_5ab1_11e8_abac_1866dafa0bc4row8_col1" class="data row8 col1" >0.271976</td> 
        <td id="T_8130832a_5ab1_11e8_abac_1866dafa0bc4row8_col2" class="data row8 col2" >0.220281</td> 
        <td id="T_8130832a_5ab1_11e8_abac_1866dafa0bc4row8_col3" class="data row8 col3" >0.169178</td> 
        <td id="T_8130832a_5ab1_11e8_abac_1866dafa0bc4row8_col4" class="data row8 col4" >0.156246</td> 
        <td id="T_8130832a_5ab1_11e8_abac_1866dafa0bc4row8_col5" class="data row8 col5" >0.115519</td> 
        <td id="T_8130832a_5ab1_11e8_abac_1866dafa0bc4row8_col6" class="data row8 col6" >0.109238</td> 
        <td id="T_8130832a_5ab1_11e8_abac_1866dafa0bc4row8_col7" class="data row8 col7" >0.0518171</td> 
    </tr>    <tr> 
        <th id="T_8130832a_5ab1_11e8_abac_1866dafa0bc4level0_row9" class="row_heading level0 row9" >nl</th> 
        <td id="T_8130832a_5ab1_11e8_abac_1866dafa0bc4row9_col0" class="data row9 col0" >0.215163</td> 
        <td id="T_8130832a_5ab1_11e8_abac_1866dafa0bc4row9_col1" class="data row9 col1" >0.292975</td> 
        <td id="T_8130832a_5ab1_11e8_abac_1866dafa0bc4row9_col2" class="data row9 col2" >0.117467</td> 
        <td id="T_8130832a_5ab1_11e8_abac_1866dafa0bc4row9_col3" class="data row9 col3" >0.23261</td> 
        <td id="T_8130832a_5ab1_11e8_abac_1866dafa0bc4row9_col4" class="data row9 col4" >0.115831</td> 
        <td id="T_8130832a_5ab1_11e8_abac_1866dafa0bc4row9_col5" class="data row9 col5" >0.0662981</td> 
        <td id="T_8130832a_5ab1_11e8_abac_1866dafa0bc4row9_col6" class="data row9 col6" >0.173641</td> 
        <td id="T_8130832a_5ab1_11e8_abac_1866dafa0bc4row9_col7" class="data row9 col7" >0.108319</td> 
    </tr>    <tr> 
        <th id="T_8130832a_5ab1_11e8_abac_1866dafa0bc4level0_row10" class="row_heading level0 row10" >ro</th> 
        <td id="T_8130832a_5ab1_11e8_abac_1866dafa0bc4row10_col0" class="data row10 col0" >0.416991</td> 
        <td id="T_8130832a_5ab1_11e8_abac_1866dafa0bc4row10_col1" class="data row10 col1" >0.208534</td> 
        <td id="T_8130832a_5ab1_11e8_abac_1866dafa0bc4row10_col2" class="data row10 col2" >0.0955572</td> 
        <td id="T_8130832a_5ab1_11e8_abac_1866dafa0bc4row10_col3" class="data row10 col3" >0.252034</td> 
        <td id="T_8130832a_5ab1_11e8_abac_1866dafa0bc4row10_col4" class="data row10 col4" >0.160215</td> 
        <td id="T_8130832a_5ab1_11e8_abac_1866dafa0bc4row10_col5" class="data row10 col5" >0.10724</td> 
        <td id="T_8130832a_5ab1_11e8_abac_1866dafa0bc4row10_col6" class="data row10 col6" >0.221786</td> 
        <td id="T_8130832a_5ab1_11e8_abac_1866dafa0bc4row10_col7" class="data row10 col7" >0.0496476</td> 
    </tr>    <tr> 
        <th id="T_8130832a_5ab1_11e8_abac_1866dafa0bc4level0_row11" class="row_heading level0 row11" >ru</th> 
        <td id="T_8130832a_5ab1_11e8_abac_1866dafa0bc4row11_col0" class="data row11 col0" >0.40899</td> 
        <td id="T_8130832a_5ab1_11e8_abac_1866dafa0bc4row11_col1" class="data row11 col1" >0.296745</td> 
        <td id="T_8130832a_5ab1_11e8_abac_1866dafa0bc4row11_col2" class="data row11 col2" >0.1386</td> 
        <td id="T_8130832a_5ab1_11e8_abac_1866dafa0bc4row11_col3" class="data row11 col3" >0.211078</td> 
        <td id="T_8130832a_5ab1_11e8_abac_1866dafa0bc4row11_col4" class="data row11 col4" >0.154007</td> 
        <td id="T_8130832a_5ab1_11e8_abac_1866dafa0bc4row11_col5" class="data row11 col5" >0.0953894</td> 
        <td id="T_8130832a_5ab1_11e8_abac_1866dafa0bc4row11_col6" class="data row11 col6" >0.115708</td> 
        <td id="T_8130832a_5ab1_11e8_abac_1866dafa0bc4row11_col7" class="data row11 col7" >0.0389807</td> 
    </tr>    <tr> 
        <th id="T_8130832a_5ab1_11e8_abac_1866dafa0bc4level0_row12" class="row_heading level0 row12" >uk</th> 
        <td id="T_8130832a_5ab1_11e8_abac_1866dafa0bc4row12_col0" class="data row12 col0" >0.412234</td> 
        <td id="T_8130832a_5ab1_11e8_abac_1866dafa0bc4row12_col1" class="data row12 col1" >0.227642</td> 
        <td id="T_8130832a_5ab1_11e8_abac_1866dafa0bc4row12_col2" class="data row12 col2" >0.096261</td> 
        <td id="T_8130832a_5ab1_11e8_abac_1866dafa0bc4row12_col3" class="data row12 col3" >0.241922</td> 
        <td id="T_8130832a_5ab1_11e8_abac_1866dafa0bc4row12_col4" class="data row12 col4" >0.241484</td> 
        <td id="T_8130832a_5ab1_11e8_abac_1866dafa0bc4row12_col5" class="data row12 col5" >0.16312</td> 
        <td id="T_8130832a_5ab1_11e8_abac_1866dafa0bc4row12_col6" class="data row12 col6" >0.197361</td> 
        <td id="T_8130832a_5ab1_11e8_abac_1866dafa0bc4row12_col7" class="data row12 col7" >0.0299612</td> 
    </tr>    <tr> 
        <th id="T_8130832a_5ab1_11e8_abac_1866dafa0bc4level0_row13" class="row_heading level0 row13" >zh</th> 
        <td id="T_8130832a_5ab1_11e8_abac_1866dafa0bc4row13_col0" class="data row13 col0" >0.364338</td> 
        <td id="T_8130832a_5ab1_11e8_abac_1866dafa0bc4row13_col1" class="data row13 col1" >0.337576</td> 
        <td id="T_8130832a_5ab1_11e8_abac_1866dafa0bc4row13_col2" class="data row13 col2" >0.240767</td> 
        <td id="T_8130832a_5ab1_11e8_abac_1866dafa0bc4row13_col3" class="data row13 col3" >0.24764</td> 
        <td id="T_8130832a_5ab1_11e8_abac_1866dafa0bc4row13_col4" class="data row13 col4" >0.208363</td> 
        <td id="T_8130832a_5ab1_11e8_abac_1866dafa0bc4row13_col5" class="data row13 col5" >0.118964</td> 
        <td id="T_8130832a_5ab1_11e8_abac_1866dafa0bc4row13_col6" class="data row13 col6" >0.184626</td> 
        <td id="T_8130832a_5ab1_11e8_abac_1866dafa0bc4row13_col7" class="data row13 col7" >0.0377849</td> 
    </tr></tbody> 
</table> 


### Viz Pairwise metrics


```python
# Code for visualization, print dataframes side by side
from IPython.display import display_html
def display_side_by_side(title,dfs):
    html_str=''
    for df in dfs:
        html_str+=df.style.set_properties(**{'font-size':'1pt'}).background_gradient(low=0,high=.8,axis=0).set_table_styles([{'selector': 'th', 'props': [('font-size', '7pt')]}]).render()
    titleHtml = "<h2>%s</h2>" % title
    display_html(titleHtml+html_str.replace('table','table style="display:inline"'),raw=True)
```


```python
for question,metric in results.items():
    title = question + ' ('+ ','.join([m for m in metric.keys()]) + ' )'
    display_side_by_side(title=title,dfs=metric.values())
```


<h2>info_depth (cosine,euclidean,l1 )</h2><style  type="text/css" >
    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4 th {
          font-size: 7pt;
    }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row0_col0 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row0_col1 {
            font-size:  1pt;
            background-color:  #faf3f9;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row0_col2 {
            font-size:  1pt;
            background-color:  #f6eff7;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row0_col3 {
            font-size:  1pt;
            background-color:  #faf2f8;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row0_col4 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row0_col5 {
            font-size:  1pt;
            background-color:  #f5eff6;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row0_col6 {
            font-size:  1pt;
            background-color:  #c4cbe3;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row0_col7 {
            font-size:  1pt;
            background-color:  #f7f0f7;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row0_col8 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row0_col9 {
            font-size:  1pt;
            background-color:  #f1ebf5;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row0_col10 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row0_col11 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row0_col12 {
            font-size:  1pt;
            background-color:  #fbf4f9;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row0_col13 {
            font-size:  1pt;
            background-color:  #faf3f9;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row1_col0 {
            font-size:  1pt;
            background-color:  #fbf3f9;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row1_col1 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row1_col2 {
            font-size:  1pt;
            background-color:  #f0eaf4;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row1_col3 {
            font-size:  1pt;
            background-color:  #f7f0f7;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row1_col4 {
            font-size:  1pt;
            background-color:  #fbf4f9;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row1_col5 {
            font-size:  1pt;
            background-color:  #f8f1f8;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row1_col6 {
            font-size:  1pt;
            background-color:  #ced0e6;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row1_col7 {
            font-size:  1pt;
            background-color:  #f0eaf4;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row1_col8 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row1_col9 {
            font-size:  1pt;
            background-color:  #e9e5f1;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row1_col10 {
            font-size:  1pt;
            background-color:  #fbf3f9;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row1_col11 {
            font-size:  1pt;
            background-color:  #f7f0f7;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row1_col12 {
            font-size:  1pt;
            background-color:  #f4eef6;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row1_col13 {
            font-size:  1pt;
            background-color:  #e9e5f1;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row2_col0 {
            font-size:  1pt;
            background-color:  #eee8f3;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row2_col1 {
            font-size:  1pt;
            background-color:  #d5d5e8;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row2_col2 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row2_col3 {
            font-size:  1pt;
            background-color:  #fdf5fa;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row2_col4 {
            font-size:  1pt;
            background-color:  #f8f1f8;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row2_col5 {
            font-size:  1pt;
            background-color:  #f0eaf4;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row2_col6 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row2_col7 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row2_col8 {
            font-size:  1pt;
            background-color:  #e6e2ef;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row2_col9 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row2_col10 {
            font-size:  1pt;
            background-color:  #eee9f3;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row2_col11 {
            font-size:  1pt;
            background-color:  #f2ecf5;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row2_col12 {
            font-size:  1pt;
            background-color:  #f8f1f8;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row2_col13 {
            font-size:  1pt;
            background-color:  #e1dfed;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row3_col0 {
            font-size:  1pt;
            background-color:  #f7f0f7;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row3_col1 {
            font-size:  1pt;
            background-color:  #f0eaf4;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row3_col2 {
            font-size:  1pt;
            background-color:  #fdf5fa;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row3_col3 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row3_col4 {
            font-size:  1pt;
            background-color:  #fbf4f9;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row3_col5 {
            font-size:  1pt;
            background-color:  #f8f1f8;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row3_col6 {
            font-size:  1pt;
            background-color:  #8eb3d5;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row3_col7 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row3_col8 {
            font-size:  1pt;
            background-color:  #f6eff7;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row3_col9 {
            font-size:  1pt;
            background-color:  #fbf3f9;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row3_col10 {
            font-size:  1pt;
            background-color:  #fbf3f9;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row3_col11 {
            font-size:  1pt;
            background-color:  #fbf3f9;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row3_col12 {
            font-size:  1pt;
            background-color:  #fbf4f9;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row3_col13 {
            font-size:  1pt;
            background-color:  #f0eaf4;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row4_col0 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row4_col1 {
            font-size:  1pt;
            background-color:  #faf3f9;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row4_col2 {
            font-size:  1pt;
            background-color:  #fbf3f9;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row4_col3 {
            font-size:  1pt;
            background-color:  #fdf5fa;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row4_col4 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row4_col5 {
            font-size:  1pt;
            background-color:  #f5eff6;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row4_col6 {
            font-size:  1pt;
            background-color:  #b0c2de;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row4_col7 {
            font-size:  1pt;
            background-color:  #faf3f9;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row4_col8 {
            font-size:  1pt;
            background-color:  #fbf3f9;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row4_col9 {
            font-size:  1pt;
            background-color:  #f8f1f8;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row4_col10 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row4_col11 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row4_col12 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row4_col13 {
            font-size:  1pt;
            background-color:  #f5eef6;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row5_col0 {
            font-size:  1pt;
            background-color:  #eee8f3;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row5_col1 {
            font-size:  1pt;
            background-color:  #f0eaf4;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row5_col2 {
            font-size:  1pt;
            background-color:  #f1ebf5;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row5_col3 {
            font-size:  1pt;
            background-color:  #f7f0f7;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row5_col4 {
            font-size:  1pt;
            background-color:  #f1ebf4;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row5_col5 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row5_col6 {
            font-size:  1pt;
            background-color:  #6fa7ce;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row5_col7 {
            font-size:  1pt;
            background-color:  #f0eaf4;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row5_col8 {
            font-size:  1pt;
            background-color:  #f1ebf5;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row5_col9 {
            font-size:  1pt;
            background-color:  #e9e5f1;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row5_col10 {
            font-size:  1pt;
            background-color:  #eee9f3;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row5_col11 {
            font-size:  1pt;
            background-color:  #e9e5f1;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row5_col12 {
            font-size:  1pt;
            background-color:  #ede8f3;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row5_col13 {
            font-size:  1pt;
            background-color:  #b0c2de;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row6_col0 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row6_col1 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row6_col2 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row6_col3 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row6_col4 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row6_col5 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row6_col6 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row6_col7 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row6_col8 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row6_col9 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row6_col10 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row6_col11 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row6_col12 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row6_col13 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row7_col0 {
            font-size:  1pt;
            background-color:  #f2ecf5;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row7_col1 {
            font-size:  1pt;
            background-color:  #dcdaeb;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row7_col2 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row7_col3 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row7_col4 {
            font-size:  1pt;
            background-color:  #f8f1f8;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row7_col5 {
            font-size:  1pt;
            background-color:  #f0eaf4;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row7_col6 {
            font-size:  1pt;
            background-color:  #7bacd1;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row7_col7 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row7_col8 {
            font-size:  1pt;
            background-color:  #ede7f2;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row7_col9 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row7_col10 {
            font-size:  1pt;
            background-color:  #f7f0f7;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row7_col11 {
            font-size:  1pt;
            background-color:  #f7f0f7;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row7_col12 {
            font-size:  1pt;
            background-color:  #fbf4f9;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row7_col13 {
            font-size:  1pt;
            background-color:  #f0eaf4;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row8_col0 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row8_col1 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row8_col2 {
            font-size:  1pt;
            background-color:  #f4edf6;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row8_col3 {
            font-size:  1pt;
            background-color:  #faf2f8;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row8_col4 {
            font-size:  1pt;
            background-color:  #fbf4f9;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row8_col5 {
            font-size:  1pt;
            background-color:  #f8f1f8;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row8_col6 {
            font-size:  1pt;
            background-color:  #c9cee4;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row8_col7 {
            font-size:  1pt;
            background-color:  #f5eef6;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row8_col8 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row8_col9 {
            font-size:  1pt;
            background-color:  #efe9f3;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row8_col10 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row8_col11 {
            font-size:  1pt;
            background-color:  #fbf3f9;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row8_col12 {
            font-size:  1pt;
            background-color:  #fbf4f9;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row8_col13 {
            font-size:  1pt;
            background-color:  #f5eef6;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row9_col0 {
            font-size:  1pt;
            background-color:  #e1dfed;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row9_col1 {
            font-size:  1pt;
            background-color:  #c1cae2;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row9_col2 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row9_col3 {
            font-size:  1pt;
            background-color:  #faf2f8;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row9_col4 {
            font-size:  1pt;
            background-color:  #f4eef6;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row9_col5 {
            font-size:  1pt;
            background-color:  #e7e3f0;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row9_col6 {
            font-size:  1pt;
            background-color:  #60a1ca;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row9_col7 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row9_col8 {
            font-size:  1pt;
            background-color:  #d8d7e9;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row9_col9 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row9_col10 {
            font-size:  1pt;
            background-color:  #e9e5f1;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row9_col11 {
            font-size:  1pt;
            background-color:  #eee9f3;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row9_col12 {
            font-size:  1pt;
            background-color:  #f8f1f8;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row9_col13 {
            font-size:  1pt;
            background-color:  #e9e5f1;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row10_col0 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row10_col1 {
            font-size:  1pt;
            background-color:  #faf3f9;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row10_col2 {
            font-size:  1pt;
            background-color:  #f6eff7;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row10_col3 {
            font-size:  1pt;
            background-color:  #fdf5fa;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row10_col4 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row10_col5 {
            font-size:  1pt;
            background-color:  #f5eff6;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row10_col6 {
            font-size:  1pt;
            background-color:  #c0c9e2;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row10_col7 {
            font-size:  1pt;
            background-color:  #faf3f9;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row10_col8 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row10_col9 {
            font-size:  1pt;
            background-color:  #f4edf6;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row10_col10 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row10_col11 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row10_col12 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row10_col13 {
            font-size:  1pt;
            background-color:  #faf3f9;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row11_col0 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row11_col1 {
            font-size:  1pt;
            background-color:  #f5eff6;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row11_col2 {
            font-size:  1pt;
            background-color:  #f8f1f8;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row11_col3 {
            font-size:  1pt;
            background-color:  #fdf5fa;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row11_col4 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row11_col5 {
            font-size:  1pt;
            background-color:  #f3edf5;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row11_col6 {
            font-size:  1pt;
            background-color:  #c0c9e2;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row11_col7 {
            font-size:  1pt;
            background-color:  #faf3f9;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row11_col8 {
            font-size:  1pt;
            background-color:  #fbf3f9;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row11_col9 {
            font-size:  1pt;
            background-color:  #f6eff7;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row11_col10 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row11_col11 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row11_col12 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row11_col13 {
            font-size:  1pt;
            background-color:  #faf3f9;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row12_col0 {
            font-size:  1pt;
            background-color:  #fbf3f9;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row12_col1 {
            font-size:  1pt;
            background-color:  #f0eaf4;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row12_col2 {
            font-size:  1pt;
            background-color:  #fbf3f9;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row12_col3 {
            font-size:  1pt;
            background-color:  #fdf5fa;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row12_col4 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row12_col5 {
            font-size:  1pt;
            background-color:  #f3edf5;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row12_col6 {
            font-size:  1pt;
            background-color:  #b0c2de;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row12_col7 {
            font-size:  1pt;
            background-color:  #fdf5fa;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row12_col8 {
            font-size:  1pt;
            background-color:  #fbf3f9;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row12_col9 {
            font-size:  1pt;
            background-color:  #fbf3f9;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row12_col10 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row12_col11 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row12_col12 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row12_col13 {
            font-size:  1pt;
            background-color:  #faf3f9;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row13_col0 {
            font-size:  1pt;
            background-color:  #fbf3f9;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row13_col1 {
            font-size:  1pt;
            background-color:  #ebe6f2;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row13_col2 {
            font-size:  1pt;
            background-color:  #f4edf6;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row13_col3 {
            font-size:  1pt;
            background-color:  #f7f0f7;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row13_col4 {
            font-size:  1pt;
            background-color:  #f8f1f8;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row13_col5 {
            font-size:  1pt;
            background-color:  #e4e1ef;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row13_col6 {
            font-size:  1pt;
            background-color:  #d2d3e7;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row13_col7 {
            font-size:  1pt;
            background-color:  #f7f0f7;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row13_col8 {
            font-size:  1pt;
            background-color:  #f6eff7;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row13_col9 {
            font-size:  1pt;
            background-color:  #f6eff7;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row13_col10 {
            font-size:  1pt;
            background-color:  #fbf3f9;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row13_col11 {
            font-size:  1pt;
            background-color:  #fbf3f9;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row13_col12 {
            font-size:  1pt;
            background-color:  #fbf4f9;
        }    #T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row13_col13 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }</style>  
<table style="display:inline" id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4" > 
<thead>    <tr> 
        <th class="index_name level0" ></th> 
        <th class="col_heading level0 col0" >ar</th> 
        <th class="col_heading level0 col1" >bn</th> 
        <th class="col_heading level0 col2" >de</th> 
        <th class="col_heading level0 col3" >en</th> 
        <th class="col_heading level0 col4" >es</th> 
        <th class="col_heading level0 col5" >he</th> 
        <th class="col_heading level0 col6" >hi</th> 
        <th class="col_heading level0 col7" >hu</th> 
        <th class="col_heading level0 col8" >ja</th> 
        <th class="col_heading level0 col9" >nl</th> 
        <th class="col_heading level0 col10" >ro</th> 
        <th class="col_heading level0 col11" >ru</th> 
        <th class="col_heading level0 col12" >uk</th> 
        <th class="col_heading level0 col13" >zh</th> 
    </tr>    <tr> 
        <th class="index_name level0" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
    </tr></thead> 
<tbody>    <tr> 
        <th id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4level0_row0" class="row_heading level0 row0" >ar</th> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row0_col0" class="data row0 col0" >0</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row0_col1" class="data row0 col1" >0.01</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row0_col2" class="data row0 col2" >0.04</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row0_col3" class="data row0 col3" >0.02</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row0_col4" class="data row0 col4" >0</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row0_col5" class="data row0 col5" >0.04</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row0_col6" class="data row0 col6" >0.19</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row0_col7" class="data row0 col7" >0.03</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row0_col8" class="data row0 col8" >0</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row0_col9" class="data row0 col9" >0.06</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row0_col10" class="data row0 col10" >0</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row0_col11" class="data row0 col11" >0</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row0_col12" class="data row0 col12" >0.01</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row0_col13" class="data row0 col13" >0.01</td> 
    </tr>    <tr> 
        <th id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4level0_row1" class="row_heading level0 row1" >bn</th> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row1_col0" class="data row1 col0" >0.01</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row1_col1" class="data row1 col1" >0</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row1_col2" class="data row1 col2" >0.07</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row1_col3" class="data row1 col3" >0.03</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row1_col4" class="data row1 col4" >0.01</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row1_col5" class="data row1 col5" >0.03</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row1_col6" class="data row1 col6" >0.17</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row1_col7" class="data row1 col7" >0.06</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row1_col8" class="data row1 col8" >0</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row1_col9" class="data row1 col9" >0.09</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row1_col10" class="data row1 col10" >0.01</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row1_col11" class="data row1 col11" >0.02</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row1_col12" class="data row1 col12" >0.03</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row1_col13" class="data row1 col13" >0.04</td> 
    </tr>    <tr> 
        <th id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4level0_row2" class="row_heading level0 row2" >de</th> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row2_col0" class="data row2 col0" >0.04</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row2_col1" class="data row2 col1" >0.07</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row2_col2" class="data row2 col2" >0</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row2_col3" class="data row2 col3" >0.01</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row2_col4" class="data row2 col4" >0.02</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row2_col5" class="data row2 col5" >0.06</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row2_col6" class="data row2 col6" >0.37</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row2_col7" class="data row2 col7" >0</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row2_col8" class="data row2 col8" >0.05</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row2_col9" class="data row2 col9" >0</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row2_col10" class="data row2 col10" >0.04</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row2_col11" class="data row2 col11" >0.03</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row2_col12" class="data row2 col12" >0.02</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row2_col13" class="data row2 col13" >0.05</td> 
    </tr>    <tr> 
        <th id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4level0_row3" class="row_heading level0 row3" >en</th> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row3_col0" class="data row3 col0" >0.02</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row3_col1" class="data row3 col1" >0.03</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row3_col2" class="data row3 col2" >0.01</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row3_col3" class="data row3 col3" >0</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row3_col4" class="data row3 col4" >0.01</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row3_col5" class="data row3 col5" >0.03</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row3_col6" class="data row3 col6" >0.29</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row3_col7" class="data row3 col7" >0</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row3_col8" class="data row3 col8" >0.02</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row3_col9" class="data row3 col9" >0.02</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row3_col10" class="data row3 col10" >0.01</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row3_col11" class="data row3 col11" >0.01</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row3_col12" class="data row3 col12" >0.01</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row3_col13" class="data row3 col13" >0.03</td> 
    </tr>    <tr> 
        <th id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4level0_row4" class="row_heading level0 row4" >es</th> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row4_col0" class="data row4 col0" >0</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row4_col1" class="data row4 col1" >0.01</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row4_col2" class="data row4 col2" >0.02</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row4_col3" class="data row4 col3" >0.01</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row4_col4" class="data row4 col4" >0</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row4_col5" class="data row4 col5" >0.04</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row4_col6" class="data row4 col6" >0.23</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row4_col7" class="data row4 col7" >0.02</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row4_col8" class="data row4 col8" >0.01</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row4_col9" class="data row4 col9" >0.03</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row4_col10" class="data row4 col10" >0</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row4_col11" class="data row4 col11" >0</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row4_col12" class="data row4 col12" >0</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row4_col13" class="data row4 col13" >0.02</td> 
    </tr>    <tr> 
        <th id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4level0_row5" class="row_heading level0 row5" >he</th> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row5_col0" class="data row5 col0" >0.04</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row5_col1" class="data row5 col1" >0.03</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row5_col2" class="data row5 col2" >0.06</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row5_col3" class="data row5 col3" >0.03</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row5_col4" class="data row5 col4" >0.04</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row5_col5" class="data row5 col5" >0</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row5_col6" class="data row5 col6" >0.34</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row5_col7" class="data row5 col7" >0.06</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row5_col8" class="data row5 col8" >0.03</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row5_col9" class="data row5 col9" >0.09</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row5_col10" class="data row5 col10" >0.04</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row5_col11" class="data row5 col11" >0.05</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row5_col12" class="data row5 col12" >0.05</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row5_col13" class="data row5 col13" >0.1</td> 
    </tr>    <tr> 
        <th id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4level0_row6" class="row_heading level0 row6" >hi</th> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row6_col0" class="data row6 col0" >0.19</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row6_col1" class="data row6 col1" >0.17</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row6_col2" class="data row6 col2" >0.37</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row6_col3" class="data row6 col3" >0.29</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row6_col4" class="data row6 col4" >0.23</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row6_col5" class="data row6 col5" >0.34</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row6_col6" class="data row6 col6" >0</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row6_col7" class="data row6 col7" >0.32</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row6_col8" class="data row6 col8" >0.18</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row6_col9" class="data row6 col9" >0.36</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row6_col10" class="data row6 col10" >0.2</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row6_col11" class="data row6 col11" >0.2</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row6_col12" class="data row6 col12" >0.23</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row6_col13" class="data row6 col13" >0.16</td> 
    </tr>    <tr> 
        <th id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4level0_row7" class="row_heading level0 row7" >hu</th> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row7_col0" class="data row7 col0" >0.03</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row7_col1" class="data row7 col1" >0.06</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row7_col2" class="data row7 col2" >0</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row7_col3" class="data row7 col3" >0</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row7_col4" class="data row7 col4" >0.02</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row7_col5" class="data row7 col5" >0.06</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row7_col6" class="data row7 col6" >0.32</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row7_col7" class="data row7 col7" >0</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row7_col8" class="data row7 col8" >0.04</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row7_col9" class="data row7 col9" >0</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row7_col10" class="data row7 col10" >0.02</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row7_col11" class="data row7 col11" >0.02</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row7_col12" class="data row7 col12" >0.01</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row7_col13" class="data row7 col13" >0.03</td> 
    </tr>    <tr> 
        <th id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4level0_row8" class="row_heading level0 row8" >ja</th> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row8_col0" class="data row8 col0" >0</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row8_col1" class="data row8 col1" >0</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row8_col2" class="data row8 col2" >0.05</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row8_col3" class="data row8 col3" >0.02</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row8_col4" class="data row8 col4" >0.01</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row8_col5" class="data row8 col5" >0.03</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row8_col6" class="data row8 col6" >0.18</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row8_col7" class="data row8 col7" >0.04</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row8_col8" class="data row8 col8" >0</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row8_col9" class="data row8 col9" >0.07</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row8_col10" class="data row8 col10" >0</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row8_col11" class="data row8 col11" >0.01</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row8_col12" class="data row8 col12" >0.01</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row8_col13" class="data row8 col13" >0.02</td> 
    </tr>    <tr> 
        <th id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4level0_row9" class="row_heading level0 row9" >nl</th> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row9_col0" class="data row9 col0" >0.06</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row9_col1" class="data row9 col1" >0.09</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row9_col2" class="data row9 col2" >0</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row9_col3" class="data row9 col3" >0.02</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row9_col4" class="data row9 col4" >0.03</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row9_col5" class="data row9 col5" >0.09</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row9_col6" class="data row9 col6" >0.36</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row9_col7" class="data row9 col7" >0</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row9_col8" class="data row9 col8" >0.07</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row9_col9" class="data row9 col9" >0</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row9_col10" class="data row9 col10" >0.05</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row9_col11" class="data row9 col11" >0.04</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row9_col12" class="data row9 col12" >0.02</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row9_col13" class="data row9 col13" >0.04</td> 
    </tr>    <tr> 
        <th id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4level0_row10" class="row_heading level0 row10" >ro</th> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row10_col0" class="data row10 col0" >0</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row10_col1" class="data row10 col1" >0.01</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row10_col2" class="data row10 col2" >0.04</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row10_col3" class="data row10 col3" >0.01</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row10_col4" class="data row10 col4" >0</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row10_col5" class="data row10 col5" >0.04</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row10_col6" class="data row10 col6" >0.2</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row10_col7" class="data row10 col7" >0.02</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row10_col8" class="data row10 col8" >0</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row10_col9" class="data row10 col9" >0.05</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row10_col10" class="data row10 col10" >0</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row10_col11" class="data row10 col11" >0</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row10_col12" class="data row10 col12" >0</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row10_col13" class="data row10 col13" >0.01</td> 
    </tr>    <tr> 
        <th id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4level0_row11" class="row_heading level0 row11" >ru</th> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row11_col0" class="data row11 col0" >0</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row11_col1" class="data row11 col1" >0.02</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row11_col2" class="data row11 col2" >0.03</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row11_col3" class="data row11 col3" >0.01</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row11_col4" class="data row11 col4" >0</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row11_col5" class="data row11 col5" >0.05</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row11_col6" class="data row11 col6" >0.2</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row11_col7" class="data row11 col7" >0.02</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row11_col8" class="data row11 col8" >0.01</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row11_col9" class="data row11 col9" >0.04</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row11_col10" class="data row11 col10" >0</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row11_col11" class="data row11 col11" >0</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row11_col12" class="data row11 col12" >0</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row11_col13" class="data row11 col13" >0.01</td> 
    </tr>    <tr> 
        <th id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4level0_row12" class="row_heading level0 row12" >uk</th> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row12_col0" class="data row12 col0" >0.01</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row12_col1" class="data row12 col1" >0.03</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row12_col2" class="data row12 col2" >0.02</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row12_col3" class="data row12 col3" >0.01</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row12_col4" class="data row12 col4" >0</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row12_col5" class="data row12 col5" >0.05</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row12_col6" class="data row12 col6" >0.23</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row12_col7" class="data row12 col7" >0.01</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row12_col8" class="data row12 col8" >0.01</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row12_col9" class="data row12 col9" >0.02</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row12_col10" class="data row12 col10" >0</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row12_col11" class="data row12 col11" >0</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row12_col12" class="data row12 col12" >0</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row12_col13" class="data row12 col13" >0.01</td> 
    </tr>    <tr> 
        <th id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4level0_row13" class="row_heading level0 row13" >zh</th> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row13_col0" class="data row13 col0" >0.01</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row13_col1" class="data row13 col1" >0.04</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row13_col2" class="data row13 col2" >0.05</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row13_col3" class="data row13 col3" >0.03</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row13_col4" class="data row13 col4" >0.02</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row13_col5" class="data row13 col5" >0.1</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row13_col6" class="data row13 col6" >0.16</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row13_col7" class="data row13 col7" >0.03</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row13_col8" class="data row13 col8" >0.02</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row13_col9" class="data row13 col9" >0.04</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row13_col10" class="data row13 col10" >0.01</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row13_col11" class="data row13 col11" >0.01</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row13_col12" class="data row13 col12" >0.01</td> 
        <td id="T_7aad09a6_5ab1_11e8_abac_1866dafa0bc4row13_col13" class="data row13 col13" >0</td> 
    </tr></tbody> 
</table style="display:inline"> <style  type="text/css" >
    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4 th {
          font-size: 7pt;
    }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row0_col0 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row0_col1 {
            font-size:  1pt;
            background-color:  #f1ebf4;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row0_col2 {
            font-size:  1pt;
            background-color:  #e4e1ef;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row0_col3 {
            font-size:  1pt;
            background-color:  #efe9f3;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row0_col4 {
            font-size:  1pt;
            background-color:  #f8f1f8;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row0_col5 {
            font-size:  1pt;
            background-color:  #e2dfee;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row0_col6 {
            font-size:  1pt;
            background-color:  #9cb9d9;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row0_col7 {
            font-size:  1pt;
            background-color:  #e6e2ef;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row0_col8 {
            font-size:  1pt;
            background-color:  #f9f2f8;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row0_col9 {
            font-size:  1pt;
            background-color:  #dddbec;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row0_col10 {
            font-size:  1pt;
            background-color:  #fbf4f9;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row0_col11 {
            font-size:  1pt;
            background-color:  #f7f0f7;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row0_col12 {
            font-size:  1pt;
            background-color:  #f2ecf5;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row0_col13 {
            font-size:  1pt;
            background-color:  #e8e4f0;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row1_col0 {
            font-size:  1pt;
            background-color:  #f1ebf5;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row1_col1 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row1_col2 {
            font-size:  1pt;
            background-color:  #d9d8ea;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row1_col3 {
            font-size:  1pt;
            background-color:  #e4e1ef;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row1_col4 {
            font-size:  1pt;
            background-color:  #ede8f3;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row1_col5 {
            font-size:  1pt;
            background-color:  #e7e3f0;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row1_col6 {
            font-size:  1pt;
            background-color:  #a2bcda;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row1_col7 {
            font-size:  1pt;
            background-color:  #dad9ea;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row1_col8 {
            font-size:  1pt;
            background-color:  #f9f2f8;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row1_col9 {
            font-size:  1pt;
            background-color:  #d0d1e6;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row1_col10 {
            font-size:  1pt;
            background-color:  #f0eaf4;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row1_col11 {
            font-size:  1pt;
            background-color:  #ebe6f2;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row1_col12 {
            font-size:  1pt;
            background-color:  #e5e1ef;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row1_col13 {
            font-size:  1pt;
            background-color:  #d5d5e8;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row2_col0 {
            font-size:  1pt;
            background-color:  #d6d6e9;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row2_col1 {
            font-size:  1pt;
            background-color:  #bdc8e1;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row2_col2 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row2_col3 {
            font-size:  1pt;
            background-color:  #f4edf6;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row2_col4 {
            font-size:  1pt;
            background-color:  #e5e1ef;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row2_col5 {
            font-size:  1pt;
            background-color:  #d9d8ea;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row2_col6 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row2_col7 {
            font-size:  1pt;
            background-color:  #f9f2f8;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row2_col8 {
            font-size:  1pt;
            background-color:  #c9cee4;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row2_col9 {
            font-size:  1pt;
            background-color:  #f7f0f7;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row2_col10 {
            font-size:  1pt;
            background-color:  #dad9ea;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row2_col11 {
            font-size:  1pt;
            background-color:  #e0dded;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row2_col12 {
            font-size:  1pt;
            background-color:  #e7e3f0;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row2_col13 {
            font-size:  1pt;
            background-color:  #c9cee4;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row3_col0 {
            font-size:  1pt;
            background-color:  #ebe6f2;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row3_col1 {
            font-size:  1pt;
            background-color:  #d9d8ea;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row3_col2 {
            font-size:  1pt;
            background-color:  #f5eef6;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row3_col3 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row3_col4 {
            font-size:  1pt;
            background-color:  #f4eef6;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row3_col5 {
            font-size:  1pt;
            background-color:  #e4e1ef;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row3_col6 {
            font-size:  1pt;
            background-color:  #78abd0;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row3_col7 {
            font-size:  1pt;
            background-color:  #f6eff7;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row3_col8 {
            font-size:  1pt;
            background-color:  #e4e1ef;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row3_col9 {
            font-size:  1pt;
            background-color:  #efe9f3;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row3_col10 {
            font-size:  1pt;
            background-color:  #eee8f3;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row3_col11 {
            font-size:  1pt;
            background-color:  #f0eaf4;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row3_col12 {
            font-size:  1pt;
            background-color:  #f4eef6;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row3_col13 {
            font-size:  1pt;
            background-color:  #dbdaeb;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row4_col0 {
            font-size:  1pt;
            background-color:  #f7f0f7;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row4_col1 {
            font-size:  1pt;
            background-color:  #e9e5f1;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row4_col2 {
            font-size:  1pt;
            background-color:  #ede7f2;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row4_col3 {
            font-size:  1pt;
            background-color:  #f5eff6;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row4_col4 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row4_col5 {
            font-size:  1pt;
            background-color:  #e4e1ef;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row4_col6 {
            font-size:  1pt;
            background-color:  #8cb3d5;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row4_col7 {
            font-size:  1pt;
            background-color:  #f0eaf4;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row4_col8 {
            font-size:  1pt;
            background-color:  #f1ebf4;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row4_col9 {
            font-size:  1pt;
            background-color:  #e6e2ef;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row4_col10 {
            font-size:  1pt;
            background-color:  #faf2f8;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row4_col11 {
            font-size:  1pt;
            background-color:  #faf2f8;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row4_col12 {
            font-size:  1pt;
            background-color:  #f8f1f8;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row4_col13 {
            font-size:  1pt;
            background-color:  #e8e4f0;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row5_col0 {
            font-size:  1pt;
            background-color:  #d6d6e9;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row5_col1 {
            font-size:  1pt;
            background-color:  #d9d8ea;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row5_col2 {
            font-size:  1pt;
            background-color:  #dbdaeb;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row5_col3 {
            font-size:  1pt;
            background-color:  #e1dfed;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row5_col4 {
            font-size:  1pt;
            background-color:  #dddbec;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row5_col5 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row5_col6 {
            font-size:  1pt;
            background-color:  #67a4cc;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row5_col7 {
            font-size:  1pt;
            background-color:  #d8d7e9;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row5_col8 {
            font-size:  1pt;
            background-color:  #dad9ea;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row5_col9 {
            font-size:  1pt;
            background-color:  #d0d1e6;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row5_col10 {
            font-size:  1pt;
            background-color:  #d4d4e8;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row5_col11 {
            font-size:  1pt;
            background-color:  #d2d2e7;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row5_col12 {
            font-size:  1pt;
            background-color:  #d5d5e8;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row5_col13 {
            font-size:  1pt;
            background-color:  #a8bedc;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row6_col0 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row6_col1 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row6_col2 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row6_col3 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row6_col4 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row6_col5 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row6_col6 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row6_col7 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row6_col8 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row6_col9 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row6_col10 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row6_col11 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row6_col12 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row6_col13 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row7_col0 {
            font-size:  1pt;
            background-color:  #dcdaeb;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row7_col1 {
            font-size:  1pt;
            background-color:  #c6cce3;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row7_col2 {
            font-size:  1pt;
            background-color:  #faf2f8;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row7_col3 {
            font-size:  1pt;
            background-color:  #f5eff6;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row7_col4 {
            font-size:  1pt;
            background-color:  #ede8f3;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row7_col5 {
            font-size:  1pt;
            background-color:  #d9d8ea;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row7_col6 {
            font-size:  1pt;
            background-color:  #6ba5cd;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row7_col7 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row7_col8 {
            font-size:  1pt;
            background-color:  #d5d5e8;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row7_col9 {
            font-size:  1pt;
            background-color:  #f7f0f7;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row7_col10 {
            font-size:  1pt;
            background-color:  #e3e0ee;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row7_col11 {
            font-size:  1pt;
            background-color:  #e8e4f0;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row7_col12 {
            font-size:  1pt;
            background-color:  #f1ebf4;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row7_col13 {
            font-size:  1pt;
            background-color:  #d9d8ea;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row8_col0 {
            font-size:  1pt;
            background-color:  #f9f2f8;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row8_col1 {
            font-size:  1pt;
            background-color:  #f9f2f8;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row8_col2 {
            font-size:  1pt;
            background-color:  #dddbec;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row8_col3 {
            font-size:  1pt;
            background-color:  #ebe6f2;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row8_col4 {
            font-size:  1pt;
            background-color:  #f2ecf5;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row8_col5 {
            font-size:  1pt;
            background-color:  #e7e3f0;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row8_col6 {
            font-size:  1pt;
            background-color:  #9fbad9;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row8_col7 {
            font-size:  1pt;
            background-color:  #e1dfed;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row8_col8 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row8_col9 {
            font-size:  1pt;
            background-color:  #d6d6e9;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row8_col10 {
            font-size:  1pt;
            background-color:  #f5eff6;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row8_col11 {
            font-size:  1pt;
            background-color:  #f1ebf5;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row8_col12 {
            font-size:  1pt;
            background-color:  #ede8f3;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row8_col13 {
            font-size:  1pt;
            background-color:  #dfddec;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row9_col0 {
            font-size:  1pt;
            background-color:  #cccfe5;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row9_col1 {
            font-size:  1pt;
            background-color:  #abbfdc;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row9_col2 {
            font-size:  1pt;
            background-color:  #f7f0f7;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row9_col3 {
            font-size:  1pt;
            background-color:  #ede8f3;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row9_col4 {
            font-size:  1pt;
            background-color:  #dddbec;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row9_col5 {
            font-size:  1pt;
            background-color:  #cccfe5;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row9_col6 {
            font-size:  1pt;
            background-color:  #5ea0ca;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row9_col7 {
            font-size:  1pt;
            background-color:  #f6eff7;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row9_col8 {
            font-size:  1pt;
            background-color:  #bcc7e1;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row9_col9 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row9_col10 {
            font-size:  1pt;
            background-color:  #d4d4e8;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row9_col11 {
            font-size:  1pt;
            background-color:  #dad9ea;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row9_col12 {
            font-size:  1pt;
            background-color:  #e5e1ef;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row9_col13 {
            font-size:  1pt;
            background-color:  #ced0e6;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row10_col0 {
            font-size:  1pt;
            background-color:  #fbf4f9;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row10_col1 {
            font-size:  1pt;
            background-color:  #eee9f3;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row10_col2 {
            font-size:  1pt;
            background-color:  #e6e2ef;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row10_col3 {
            font-size:  1pt;
            background-color:  #f0eaf4;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row10_col4 {
            font-size:  1pt;
            background-color:  #faf2f8;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row10_col5 {
            font-size:  1pt;
            background-color:  #e0dded;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row10_col6 {
            font-size:  1pt;
            background-color:  #97b7d7;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row10_col7 {
            font-size:  1pt;
            background-color:  #eae6f1;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row10_col8 {
            font-size:  1pt;
            background-color:  #f5eef6;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row10_col9 {
            font-size:  1pt;
            background-color:  #e1dfed;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row10_col10 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row10_col11 {
            font-size:  1pt;
            background-color:  #fbf4f9;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row10_col12 {
            font-size:  1pt;
            background-color:  #f6eff7;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row10_col13 {
            font-size:  1pt;
            background-color:  #ece7f2;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row11_col0 {
            font-size:  1pt;
            background-color:  #f7f0f7;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row11_col1 {
            font-size:  1pt;
            background-color:  #e9e5f1;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row11_col2 {
            font-size:  1pt;
            background-color:  #eae6f1;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row11_col3 {
            font-size:  1pt;
            background-color:  #f2ecf5;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row11_col4 {
            font-size:  1pt;
            background-color:  #faf2f8;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row11_col5 {
            font-size:  1pt;
            background-color:  #dddbec;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row11_col6 {
            font-size:  1pt;
            background-color:  #97b7d7;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row11_col7 {
            font-size:  1pt;
            background-color:  #eee9f3;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row11_col8 {
            font-size:  1pt;
            background-color:  #f1ebf4;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row11_col9 {
            font-size:  1pt;
            background-color:  #e6e2ef;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row11_col10 {
            font-size:  1pt;
            background-color:  #fbf4f9;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row11_col11 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row11_col12 {
            font-size:  1pt;
            background-color:  #faf2f8;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row11_col13 {
            font-size:  1pt;
            background-color:  #eee8f3;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row12_col0 {
            font-size:  1pt;
            background-color:  #f1ebf5;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row12_col1 {
            font-size:  1pt;
            background-color:  #e0dded;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row12_col2 {
            font-size:  1pt;
            background-color:  #eee8f3;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row12_col3 {
            font-size:  1pt;
            background-color:  #f5eff6;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row12_col4 {
            font-size:  1pt;
            background-color:  #f8f1f8;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row12_col5 {
            font-size:  1pt;
            background-color:  #dddbec;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row12_col6 {
            font-size:  1pt;
            background-color:  #8cb3d5;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row12_col7 {
            font-size:  1pt;
            background-color:  #f2ecf5;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row12_col8 {
            font-size:  1pt;
            background-color:  #eae6f1;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row12_col9 {
            font-size:  1pt;
            background-color:  #ece7f2;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row12_col10 {
            font-size:  1pt;
            background-color:  #f5eff6;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row12_col11 {
            font-size:  1pt;
            background-color:  #faf2f8;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row12_col12 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row12_col13 {
            font-size:  1pt;
            background-color:  #eee8f3;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row13_col0 {
            font-size:  1pt;
            background-color:  #ebe6f2;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row13_col1 {
            font-size:  1pt;
            background-color:  #d7d6e9;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row13_col2 {
            font-size:  1pt;
            background-color:  #e0dded;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row13_col3 {
            font-size:  1pt;
            background-color:  #e6e2ef;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row13_col4 {
            font-size:  1pt;
            background-color:  #ede8f3;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row13_col5 {
            font-size:  1pt;
            background-color:  #cccfe5;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row13_col6 {
            font-size:  1pt;
            background-color:  #a7bddb;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row13_col7 {
            font-size:  1pt;
            background-color:  #e6e2ef;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row13_col8 {
            font-size:  1pt;
            background-color:  #e0deed;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row13_col9 {
            font-size:  1pt;
            background-color:  #e1dfed;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row13_col10 {
            font-size:  1pt;
            background-color:  #eee8f3;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row13_col11 {
            font-size:  1pt;
            background-color:  #f0eaf4;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row13_col12 {
            font-size:  1pt;
            background-color:  #f1ebf4;
        }    #T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row13_col13 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }</style>  
<table style="display:inline" id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4" > 
<thead>    <tr> 
        <th class="index_name level0" ></th> 
        <th class="col_heading level0 col0" >ar</th> 
        <th class="col_heading level0 col1" >bn</th> 
        <th class="col_heading level0 col2" >de</th> 
        <th class="col_heading level0 col3" >en</th> 
        <th class="col_heading level0 col4" >es</th> 
        <th class="col_heading level0 col5" >he</th> 
        <th class="col_heading level0 col6" >hi</th> 
        <th class="col_heading level0 col7" >hu</th> 
        <th class="col_heading level0 col8" >ja</th> 
        <th class="col_heading level0 col9" >nl</th> 
        <th class="col_heading level0 col10" >ro</th> 
        <th class="col_heading level0 col11" >ru</th> 
        <th class="col_heading level0 col12" >uk</th> 
        <th class="col_heading level0 col13" >zh</th> 
    </tr>    <tr> 
        <th class="index_name level0" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
    </tr></thead> 
<tbody>    <tr> 
        <th id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4level0_row0" class="row_heading level0 row0" >ar</th> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row0_col0" class="data row0 col0" >0</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row0_col1" class="data row0 col1" >0.07</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row0_col2" class="data row0 col2" >0.17</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row0_col3" class="data row0 col3" >0.1</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row0_col4" class="data row0 col4" >0.04</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row0_col5" class="data row0 col5" >0.17</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row0_col6" class="data row0 col6" >0.42</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row0_col7" class="data row0 col7" >0.15</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row0_col8" class="data row0 col8" >0.03</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row0_col9" class="data row0 col9" >0.2</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row0_col10" class="data row0 col10" >0.02</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row0_col11" class="data row0 col11" >0.04</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row0_col12" class="data row0 col12" >0.07</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row0_col13" class="data row0 col13" >0.1</td> 
    </tr>    <tr> 
        <th id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4level0_row1" class="row_heading level0 row1" >bn</th> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row1_col0" class="data row1 col0" >0.07</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row1_col1" class="data row1 col1" >0</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row1_col2" class="data row1 col2" >0.22</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row1_col3" class="data row1 col3" >0.15</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row1_col4" class="data row1 col4" >0.1</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row1_col5" class="data row1 col5" >0.15</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row1_col6" class="data row1 col6" >0.4</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row1_col7" class="data row1 col7" >0.2</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row1_col8" class="data row1 col8" >0.03</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row1_col9" class="data row1 col9" >0.26</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row1_col10" class="data row1 col10" >0.08</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row1_col11" class="data row1 col11" >0.1</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row1_col12" class="data row1 col12" >0.13</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row1_col13" class="data row1 col13" >0.16</td> 
    </tr>    <tr> 
        <th id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4level0_row2" class="row_heading level0 row2" >de</th> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row2_col0" class="data row2 col0" >0.17</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row2_col1" class="data row2 col1" >0.22</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row2_col2" class="data row2 col2" >0</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row2_col3" class="data row2 col3" >0.07</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row2_col4" class="data row2 col4" >0.13</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row2_col5" class="data row2 col5" >0.21</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row2_col6" class="data row2 col6" >0.58</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row2_col7" class="data row2 col7" >0.04</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row2_col8" class="data row2 col8" >0.2</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row2_col9" class="data row2 col9" >0.06</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row2_col10" class="data row2 col10" >0.16</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row2_col11" class="data row2 col11" >0.14</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row2_col12" class="data row2 col12" >0.12</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row2_col13" class="data row2 col13" >0.19</td> 
    </tr>    <tr> 
        <th id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4level0_row3" class="row_heading level0 row3" >en</th> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row3_col0" class="data row3 col0" >0.1</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row3_col1" class="data row3 col1" >0.15</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row3_col2" class="data row3 col2" >0.07</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row3_col3" class="data row3 col3" >0</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row3_col4" class="data row3 col4" >0.06</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row3_col5" class="data row3 col5" >0.16</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row3_col6" class="data row3 col6" >0.51</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row3_col7" class="data row3 col7" >0.06</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row3_col8" class="data row3 col8" >0.12</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row3_col9" class="data row3 col9" >0.11</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row3_col10" class="data row3 col10" >0.09</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row3_col11" class="data row3 col11" >0.08</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row3_col12" class="data row3 col12" >0.06</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row3_col13" class="data row3 col13" >0.14</td> 
    </tr>    <tr> 
        <th id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4level0_row4" class="row_heading level0 row4" >es</th> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row4_col0" class="data row4 col0" >0.04</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row4_col1" class="data row4 col1" >0.1</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row4_col2" class="data row4 col2" >0.13</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row4_col3" class="data row4 col3" >0.06</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row4_col4" class="data row4 col4" >0</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row4_col5" class="data row4 col5" >0.16</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row4_col6" class="data row4 col6" >0.46</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row4_col7" class="data row4 col7" >0.1</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row4_col8" class="data row4 col8" >0.07</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row4_col9" class="data row4 col9" >0.16</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row4_col10" class="data row4 col10" >0.03</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row4_col11" class="data row4 col11" >0.03</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row4_col12" class="data row4 col12" >0.04</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row4_col13" class="data row4 col13" >0.1</td> 
    </tr>    <tr> 
        <th id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4level0_row5" class="row_heading level0 row5" >he</th> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row5_col0" class="data row5 col0" >0.17</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row5_col1" class="data row5 col1" >0.15</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row5_col2" class="data row5 col2" >0.21</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row5_col3" class="data row5 col3" >0.16</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row5_col4" class="data row5 col4" >0.16</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row5_col5" class="data row5 col5" >0</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row5_col6" class="data row5 col6" >0.55</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row5_col7" class="data row5 col7" >0.21</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row5_col8" class="data row5 col8" >0.15</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row5_col9" class="data row5 col9" >0.26</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row5_col10" class="data row5 col10" >0.18</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row5_col11" class="data row5 col11" >0.19</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row5_col12" class="data row5 col12" >0.19</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row5_col13" class="data row5 col13" >0.26</td> 
    </tr>    <tr> 
        <th id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4level0_row6" class="row_heading level0 row6" >hi</th> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row6_col0" class="data row6 col0" >0.42</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row6_col1" class="data row6 col1" >0.4</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row6_col2" class="data row6 col2" >0.58</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row6_col3" class="data row6 col3" >0.51</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row6_col4" class="data row6 col4" >0.46</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row6_col5" class="data row6 col5" >0.55</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row6_col6" class="data row6 col6" >0</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row6_col7" class="data row6 col7" >0.54</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row6_col8" class="data row6 col8" >0.41</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row6_col9" class="data row6 col9" >0.57</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row6_col10" class="data row6 col10" >0.43</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row6_col11" class="data row6 col11" >0.43</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row6_col12" class="data row6 col12" >0.46</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row6_col13" class="data row6 col13" >0.39</td> 
    </tr>    <tr> 
        <th id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4level0_row7" class="row_heading level0 row7" >hu</th> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row7_col0" class="data row7 col0" >0.15</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row7_col1" class="data row7 col1" >0.2</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row7_col2" class="data row7 col2" >0.04</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row7_col3" class="data row7 col3" >0.06</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row7_col4" class="data row7 col4" >0.1</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row7_col5" class="data row7 col5" >0.21</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row7_col6" class="data row7 col6" >0.54</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row7_col7" class="data row7 col7" >0</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row7_col8" class="data row7 col8" >0.17</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row7_col9" class="data row7 col9" >0.06</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row7_col10" class="data row7 col10" >0.13</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row7_col11" class="data row7 col11" >0.11</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row7_col12" class="data row7 col12" >0.08</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row7_col13" class="data row7 col13" >0.15</td> 
    </tr>    <tr> 
        <th id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4level0_row8" class="row_heading level0 row8" >ja</th> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row8_col0" class="data row8 col0" >0.03</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row8_col1" class="data row8 col1" >0.03</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row8_col2" class="data row8 col2" >0.2</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row8_col3" class="data row8 col3" >0.12</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row8_col4" class="data row8 col4" >0.07</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row8_col5" class="data row8 col5" >0.15</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row8_col6" class="data row8 col6" >0.41</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row8_col7" class="data row8 col7" >0.17</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row8_col8" class="data row8 col8" >0</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row8_col9" class="data row8 col9" >0.23</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row8_col10" class="data row8 col10" >0.05</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row8_col11" class="data row8 col11" >0.07</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row8_col12" class="data row8 col12" >0.1</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row8_col13" class="data row8 col13" >0.13</td> 
    </tr>    <tr> 
        <th id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4level0_row9" class="row_heading level0 row9" >nl</th> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row9_col0" class="data row9 col0" >0.2</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row9_col1" class="data row9 col1" >0.26</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row9_col2" class="data row9 col2" >0.06</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row9_col3" class="data row9 col3" >0.11</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row9_col4" class="data row9 col4" >0.16</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row9_col5" class="data row9 col5" >0.26</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row9_col6" class="data row9 col6" >0.57</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row9_col7" class="data row9 col7" >0.06</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row9_col8" class="data row9 col8" >0.23</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row9_col9" class="data row9 col9" >0</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row9_col10" class="data row9 col10" >0.18</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row9_col11" class="data row9 col11" >0.16</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row9_col12" class="data row9 col12" >0.13</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row9_col13" class="data row9 col13" >0.18</td> 
    </tr>    <tr> 
        <th id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4level0_row10" class="row_heading level0 row10" >ro</th> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row10_col0" class="data row10 col0" >0.02</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row10_col1" class="data row10 col1" >0.08</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row10_col2" class="data row10 col2" >0.16</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row10_col3" class="data row10 col3" >0.09</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row10_col4" class="data row10 col4" >0.03</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row10_col5" class="data row10 col5" >0.18</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row10_col6" class="data row10 col6" >0.43</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row10_col7" class="data row10 col7" >0.13</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row10_col8" class="data row10 col8" >0.05</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row10_col9" class="data row10 col9" >0.18</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row10_col10" class="data row10 col10" >0</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row10_col11" class="data row10 col11" >0.02</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row10_col12" class="data row10 col12" >0.05</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row10_col13" class="data row10 col13" >0.09</td> 
    </tr>    <tr> 
        <th id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4level0_row11" class="row_heading level0 row11" >ru</th> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row11_col0" class="data row11 col0" >0.04</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row11_col1" class="data row11 col1" >0.1</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row11_col2" class="data row11 col2" >0.14</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row11_col3" class="data row11 col3" >0.08</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row11_col4" class="data row11 col4" >0.03</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row11_col5" class="data row11 col5" >0.19</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row11_col6" class="data row11 col6" >0.43</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row11_col7" class="data row11 col7" >0.11</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row11_col8" class="data row11 col8" >0.07</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row11_col9" class="data row11 col9" >0.16</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row11_col10" class="data row11 col10" >0.02</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row11_col11" class="data row11 col11" >0</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row11_col12" class="data row11 col12" >0.03</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row11_col13" class="data row11 col13" >0.08</td> 
    </tr>    <tr> 
        <th id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4level0_row12" class="row_heading level0 row12" >uk</th> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row12_col0" class="data row12 col0" >0.07</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row12_col1" class="data row12 col1" >0.13</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row12_col2" class="data row12 col2" >0.12</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row12_col3" class="data row12 col3" >0.06</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row12_col4" class="data row12 col4" >0.04</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row12_col5" class="data row12 col5" >0.19</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row12_col6" class="data row12 col6" >0.46</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row12_col7" class="data row12 col7" >0.08</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row12_col8" class="data row12 col8" >0.1</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row12_col9" class="data row12 col9" >0.13</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row12_col10" class="data row12 col10" >0.05</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row12_col11" class="data row12 col11" >0.03</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row12_col12" class="data row12 col12" >0</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row12_col13" class="data row12 col13" >0.08</td> 
    </tr>    <tr> 
        <th id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4level0_row13" class="row_heading level0 row13" >zh</th> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row13_col0" class="data row13 col0" >0.1</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row13_col1" class="data row13 col1" >0.16</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row13_col2" class="data row13 col2" >0.19</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row13_col3" class="data row13 col3" >0.14</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row13_col4" class="data row13 col4" >0.1</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row13_col5" class="data row13 col5" >0.26</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row13_col6" class="data row13 col6" >0.39</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row13_col7" class="data row13 col7" >0.15</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row13_col8" class="data row13 col8" >0.13</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row13_col9" class="data row13 col9" >0.18</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row13_col10" class="data row13 col10" >0.09</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row13_col11" class="data row13 col11" >0.08</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row13_col12" class="data row13 col12" >0.08</td> 
        <td id="T_7abce0f6_5ab1_11e8_abac_1866dafa0bc4row13_col13" class="data row13 col13" >0</td> 
    </tr></tbody> 
</table style="display:inline"> <style  type="text/css" >
    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4 th {
          font-size: 7pt;
    }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row0_col0 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row0_col1 {
            font-size:  1pt;
            background-color:  #f2ecf5;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row0_col2 {
            font-size:  1pt;
            background-color:  #e5e1ef;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row0_col3 {
            font-size:  1pt;
            background-color:  #efe9f3;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row0_col4 {
            font-size:  1pt;
            background-color:  #f7f0f7;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row0_col5 {
            font-size:  1pt;
            background-color:  #dfddec;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row0_col6 {
            font-size:  1pt;
            background-color:  #9ebad9;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row0_col7 {
            font-size:  1pt;
            background-color:  #ebe6f2;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row0_col8 {
            font-size:  1pt;
            background-color:  #f9f2f8;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row0_col9 {
            font-size:  1pt;
            background-color:  #e0deed;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row0_col10 {
            font-size:  1pt;
            background-color:  #fbf4f9;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row0_col11 {
            font-size:  1pt;
            background-color:  #f8f1f8;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row0_col12 {
            font-size:  1pt;
            background-color:  #f3edf5;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row0_col13 {
            font-size:  1pt;
            background-color:  #ebe6f2;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row1_col0 {
            font-size:  1pt;
            background-color:  #f2ecf5;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row1_col1 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row1_col2 {
            font-size:  1pt;
            background-color:  #dedcec;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row1_col3 {
            font-size:  1pt;
            background-color:  #e7e3f0;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row1_col4 {
            font-size:  1pt;
            background-color:  #ede8f3;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row1_col5 {
            font-size:  1pt;
            background-color:  #e9e5f1;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row1_col6 {
            font-size:  1pt;
            background-color:  #a7bddb;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row1_col7 {
            font-size:  1pt;
            background-color:  #dcdaeb;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row1_col8 {
            font-size:  1pt;
            background-color:  #f9f2f8;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row1_col9 {
            font-size:  1pt;
            background-color:  #d2d3e7;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row1_col10 {
            font-size:  1pt;
            background-color:  #f0eaf4;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row1_col11 {
            font-size:  1pt;
            background-color:  #eae6f1;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row1_col12 {
            font-size:  1pt;
            background-color:  #e6e2ef;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row1_col13 {
            font-size:  1pt;
            background-color:  #dedcec;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row2_col0 {
            font-size:  1pt;
            background-color:  #d6d6e9;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row2_col1 {
            font-size:  1pt;
            background-color:  #c5cce3;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row2_col2 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row2_col3 {
            font-size:  1pt;
            background-color:  #f4eef6;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row2_col4 {
            font-size:  1pt;
            background-color:  #e7e3f0;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row2_col5 {
            font-size:  1pt;
            background-color:  #d8d7e9;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row2_col6 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row2_col7 {
            font-size:  1pt;
            background-color:  #faf2f8;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row2_col8 {
            font-size:  1pt;
            background-color:  #d3d4e7;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row2_col9 {
            font-size:  1pt;
            background-color:  #f7f0f7;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row2_col10 {
            font-size:  1pt;
            background-color:  #dbdaeb;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row2_col11 {
            font-size:  1pt;
            background-color:  #e0dded;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row2_col12 {
            font-size:  1pt;
            background-color:  #e8e4f0;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row2_col13 {
            font-size:  1pt;
            background-color:  #cdd0e5;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row3_col0 {
            font-size:  1pt;
            background-color:  #ebe6f2;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row3_col1 {
            font-size:  1pt;
            background-color:  #dddbec;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row3_col2 {
            font-size:  1pt;
            background-color:  #f5eff6;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row3_col3 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row3_col4 {
            font-size:  1pt;
            background-color:  #f5eef6;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row3_col5 {
            font-size:  1pt;
            background-color:  #e7e3f0;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row3_col6 {
            font-size:  1pt;
            background-color:  #75a9cf;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row3_col7 {
            font-size:  1pt;
            background-color:  #f7f0f7;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row3_col8 {
            font-size:  1pt;
            background-color:  #e7e3f0;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row3_col9 {
            font-size:  1pt;
            background-color:  #eee8f3;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row3_col10 {
            font-size:  1pt;
            background-color:  #eee9f3;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row3_col11 {
            font-size:  1pt;
            background-color:  #f0eaf4;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row3_col12 {
            font-size:  1pt;
            background-color:  #f5eef6;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row3_col13 {
            font-size:  1pt;
            background-color:  #e1dfed;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row4_col0 {
            font-size:  1pt;
            background-color:  #f7f0f7;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row4_col1 {
            font-size:  1pt;
            background-color:  #e8e4f0;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row4_col2 {
            font-size:  1pt;
            background-color:  #ede8f3;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row4_col3 {
            font-size:  1pt;
            background-color:  #f6eff7;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row4_col4 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row4_col5 {
            font-size:  1pt;
            background-color:  #e1dfed;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row4_col6 {
            font-size:  1pt;
            background-color:  #8eb3d5;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row4_col7 {
            font-size:  1pt;
            background-color:  #f1ebf4;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row4_col8 {
            font-size:  1pt;
            background-color:  #f1ebf5;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row4_col9 {
            font-size:  1pt;
            background-color:  #e7e3f0;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row4_col10 {
            font-size:  1pt;
            background-color:  #f9f2f8;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row4_col11 {
            font-size:  1pt;
            background-color:  #faf3f9;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row4_col12 {
            font-size:  1pt;
            background-color:  #faf2f8;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row4_col13 {
            font-size:  1pt;
            background-color:  #e7e3f0;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row5_col0 {
            font-size:  1pt;
            background-color:  #d4d4e8;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row5_col1 {
            font-size:  1pt;
            background-color:  #dfddec;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row5_col2 {
            font-size:  1pt;
            background-color:  #dddbec;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row5_col3 {
            font-size:  1pt;
            background-color:  #e7e3f0;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row5_col4 {
            font-size:  1pt;
            background-color:  #dddbec;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row5_col5 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row5_col6 {
            font-size:  1pt;
            background-color:  #75a9cf;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row5_col7 {
            font-size:  1pt;
            background-color:  #dbdaeb;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row5_col8 {
            font-size:  1pt;
            background-color:  #dbdaeb;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row5_col9 {
            font-size:  1pt;
            background-color:  #d2d2e7;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row5_col10 {
            font-size:  1pt;
            background-color:  #d4d4e8;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row5_col11 {
            font-size:  1pt;
            background-color:  #d3d4e7;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row5_col12 {
            font-size:  1pt;
            background-color:  #d5d5e8;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row5_col13 {
            font-size:  1pt;
            background-color:  #a7bddb;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row6_col0 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row6_col1 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row6_col2 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row6_col3 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row6_col4 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row6_col5 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row6_col6 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row6_col7 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row6_col8 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row6_col9 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row6_col10 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row6_col11 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row6_col12 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row6_col13 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row7_col0 {
            font-size:  1pt;
            background-color:  #e1dfed;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row7_col1 {
            font-size:  1pt;
            background-color:  #c9cee4;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row7_col2 {
            font-size:  1pt;
            background-color:  #faf2f8;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row7_col3 {
            font-size:  1pt;
            background-color:  #f6eff7;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row7_col4 {
            font-size:  1pt;
            background-color:  #eee9f3;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row7_col5 {
            font-size:  1pt;
            background-color:  #d9d8ea;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row7_col6 {
            font-size:  1pt;
            background-color:  #69a5cc;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row7_col7 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row7_col8 {
            font-size:  1pt;
            background-color:  #d7d6e9;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row7_col9 {
            font-size:  1pt;
            background-color:  #f7f0f7;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row7_col10 {
            font-size:  1pt;
            background-color:  #e6e2ef;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row7_col11 {
            font-size:  1pt;
            background-color:  #eae6f1;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row7_col12 {
            font-size:  1pt;
            background-color:  #f1ebf4;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row7_col13 {
            font-size:  1pt;
            background-color:  #d9d8ea;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row8_col0 {
            font-size:  1pt;
            background-color:  #f9f2f8;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row8_col1 {
            font-size:  1pt;
            background-color:  #f8f1f8;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row8_col2 {
            font-size:  1pt;
            background-color:  #e3e0ee;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row8_col3 {
            font-size:  1pt;
            background-color:  #ede8f3;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row8_col4 {
            font-size:  1pt;
            background-color:  #f2ecf5;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row8_col5 {
            font-size:  1pt;
            background-color:  #e5e1ef;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row8_col6 {
            font-size:  1pt;
            background-color:  #a1bbda;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row8_col7 {
            font-size:  1pt;
            background-color:  #e3e0ee;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row8_col8 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row8_col9 {
            font-size:  1pt;
            background-color:  #d9d8ea;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row8_col10 {
            font-size:  1pt;
            background-color:  #f5eff6;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row8_col11 {
            font-size:  1pt;
            background-color:  #f2ecf5;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row8_col12 {
            font-size:  1pt;
            background-color:  #ede8f3;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row8_col13 {
            font-size:  1pt;
            background-color:  #e3e0ee;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row9_col0 {
            font-size:  1pt;
            background-color:  #d1d2e6;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row9_col1 {
            font-size:  1pt;
            background-color:  #adc1dd;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row9_col2 {
            font-size:  1pt;
            background-color:  #f7f0f7;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row9_col3 {
            font-size:  1pt;
            background-color:  #ece7f2;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row9_col4 {
            font-size:  1pt;
            background-color:  #dedcec;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row9_col5 {
            font-size:  1pt;
            background-color:  #c9cee4;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row9_col6 {
            font-size:  1pt;
            background-color:  #5c9fc9;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row9_col7 {
            font-size:  1pt;
            background-color:  #f7f0f7;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row9_col8 {
            font-size:  1pt;
            background-color:  #c1cae2;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row9_col9 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row9_col10 {
            font-size:  1pt;
            background-color:  #d8d7e9;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row9_col11 {
            font-size:  1pt;
            background-color:  #dedcec;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row9_col12 {
            font-size:  1pt;
            background-color:  #e8e4f0;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row9_col13 {
            font-size:  1pt;
            background-color:  #d0d1e6;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row10_col0 {
            font-size:  1pt;
            background-color:  #fbf4f9;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row10_col1 {
            font-size:  1pt;
            background-color:  #eee8f3;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row10_col2 {
            font-size:  1pt;
            background-color:  #e7e3f0;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row10_col3 {
            font-size:  1pt;
            background-color:  #f1ebf5;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row10_col4 {
            font-size:  1pt;
            background-color:  #faf2f8;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row10_col5 {
            font-size:  1pt;
            background-color:  #dddbec;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row10_col6 {
            font-size:  1pt;
            background-color:  #99b8d8;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row10_col7 {
            font-size:  1pt;
            background-color:  #ede8f3;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row10_col8 {
            font-size:  1pt;
            background-color:  #f5eef6;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row10_col9 {
            font-size:  1pt;
            background-color:  #e4e1ef;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row10_col10 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row10_col11 {
            font-size:  1pt;
            background-color:  #fbf4f9;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row10_col12 {
            font-size:  1pt;
            background-color:  #f6eff7;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row10_col13 {
            font-size:  1pt;
            background-color:  #ede7f2;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row11_col0 {
            font-size:  1pt;
            background-color:  #f8f1f8;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row11_col1 {
            font-size:  1pt;
            background-color:  #e7e3f0;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row11_col2 {
            font-size:  1pt;
            background-color:  #eae6f1;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row11_col3 {
            font-size:  1pt;
            background-color:  #f2ecf5;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row11_col4 {
            font-size:  1pt;
            background-color:  #fbf3f9;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row11_col5 {
            font-size:  1pt;
            background-color:  #dcdaeb;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row11_col6 {
            font-size:  1pt;
            background-color:  #94b6d7;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row11_col7 {
            font-size:  1pt;
            background-color:  #efe9f3;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row11_col8 {
            font-size:  1pt;
            background-color:  #f1ebf5;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row11_col9 {
            font-size:  1pt;
            background-color:  #e8e4f0;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row11_col10 {
            font-size:  1pt;
            background-color:  #fbf4f9;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row11_col11 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row11_col12 {
            font-size:  1pt;
            background-color:  #faf2f8;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row11_col13 {
            font-size:  1pt;
            background-color:  #f0eaf4;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row12_col0 {
            font-size:  1pt;
            background-color:  #f1ebf5;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row12_col1 {
            font-size:  1pt;
            background-color:  #dfddec;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row12_col2 {
            font-size:  1pt;
            background-color:  #eee9f3;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row12_col3 {
            font-size:  1pt;
            background-color:  #f6eff7;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row12_col4 {
            font-size:  1pt;
            background-color:  #faf2f8;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row12_col5 {
            font-size:  1pt;
            background-color:  #dad9ea;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row12_col6 {
            font-size:  1pt;
            background-color:  #8bb2d4;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row12_col7 {
            font-size:  1pt;
            background-color:  #f2ecf5;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row12_col8 {
            font-size:  1pt;
            background-color:  #eae6f1;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row12_col9 {
            font-size:  1pt;
            background-color:  #eee8f3;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row12_col10 {
            font-size:  1pt;
            background-color:  #f5eff6;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row12_col11 {
            font-size:  1pt;
            background-color:  #f9f2f8;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row12_col12 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row12_col13 {
            font-size:  1pt;
            background-color:  #f0eaf4;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row13_col0 {
            font-size:  1pt;
            background-color:  #ede7f2;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row13_col1 {
            font-size:  1pt;
            background-color:  #dddbec;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row13_col2 {
            font-size:  1pt;
            background-color:  #e0deed;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row13_col3 {
            font-size:  1pt;
            background-color:  #ebe6f2;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row13_col4 {
            font-size:  1pt;
            background-color:  #ece7f2;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row13_col5 {
            font-size:  1pt;
            background-color:  #c5cce3;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row13_col6 {
            font-size:  1pt;
            background-color:  #a5bddb;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row13_col7 {
            font-size:  1pt;
            background-color:  #e7e3f0;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row13_col8 {
            font-size:  1pt;
            background-color:  #e5e1ef;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row13_col9 {
            font-size:  1pt;
            background-color:  #e1dfed;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row13_col10 {
            font-size:  1pt;
            background-color:  #eee9f3;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row13_col11 {
            font-size:  1pt;
            background-color:  #f1ebf4;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row13_col12 {
            font-size:  1pt;
            background-color:  #f2ecf5;
        }    #T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row13_col13 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }</style>  
<table style="display:inline" id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4" > 
<thead>    <tr> 
        <th class="index_name level0" ></th> 
        <th class="col_heading level0 col0" >ar</th> 
        <th class="col_heading level0 col1" >bn</th> 
        <th class="col_heading level0 col2" >de</th> 
        <th class="col_heading level0 col3" >en</th> 
        <th class="col_heading level0 col4" >es</th> 
        <th class="col_heading level0 col5" >he</th> 
        <th class="col_heading level0 col6" >hi</th> 
        <th class="col_heading level0 col7" >hu</th> 
        <th class="col_heading level0 col8" >ja</th> 
        <th class="col_heading level0 col9" >nl</th> 
        <th class="col_heading level0 col10" >ro</th> 
        <th class="col_heading level0 col11" >ru</th> 
        <th class="col_heading level0 col12" >uk</th> 
        <th class="col_heading level0 col13" >zh</th> 
    </tr>    <tr> 
        <th class="index_name level0" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
    </tr></thead> 
<tbody>    <tr> 
        <th id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4level0_row0" class="row_heading level0 row0" >ar</th> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row0_col0" class="data row0 col0" >0</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row0_col1" class="data row0 col1" >0.1</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row0_col2" class="data row0 col2" >0.27</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row0_col3" class="data row0 col3" >0.16</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row0_col4" class="data row0 col4" >0.07</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row0_col5" class="data row0 col5" >0.28</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row0_col6" class="data row0 col6" >0.67</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row0_col7" class="data row0 col7" >0.21</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row0_col8" class="data row0 col8" >0.05</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row0_col9" class="data row0 col9" >0.3</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row0_col10" class="data row0 col10" >0.03</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row0_col11" class="data row0 col11" >0.06</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row0_col12" class="data row0 col12" >0.11</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row0_col13" class="data row0 col13" >0.15</td> 
    </tr>    <tr> 
        <th id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4level0_row1" class="row_heading level0 row1" >bn</th> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row1_col0" class="data row1 col0" >0.1</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row1_col1" class="data row1 col1" >0</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row1_col2" class="data row1 col2" >0.32</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row1_col3" class="data row1 col3" >0.22</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row1_col4" class="data row1 col4" >0.16</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row1_col5" class="data row1 col5" >0.21</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row1_col6" class="data row1 col6" >0.63</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row1_col7" class="data row1 col7" >0.31</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row1_col8" class="data row1 col8" >0.05</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row1_col9" class="data row1 col9" >0.4</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row1_col10" class="data row1 col10" >0.13</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row1_col11" class="data row1 col11" >0.17</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row1_col12" class="data row1 col12" >0.21</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row1_col13" class="data row1 col13" >0.22</td> 
    </tr>    <tr> 
        <th id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4level0_row2" class="row_heading level0 row2" >de</th> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row2_col0" class="data row2 col0" >0.27</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row2_col1" class="data row2 col1" >0.32</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row2_col2" class="data row2 col2" >0</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row2_col3" class="data row2 col3" >0.11</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row2_col4" class="data row2 col4" >0.2</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row2_col5" class="data row2 col5" >0.33</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row2_col6" class="data row2 col6" >0.94</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row2_col7" class="data row2 col7" >0.06</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row2_col8" class="data row2 col8" >0.28</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row2_col9" class="data row2 col9" >0.09</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row2_col10" class="data row2 col10" >0.25</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row2_col11" class="data row2 col11" >0.23</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row2_col12" class="data row2 col12" >0.19</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row2_col13" class="data row2 col13" >0.3</td> 
    </tr>    <tr> 
        <th id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4level0_row3" class="row_heading level0 row3" >en</th> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row3_col0" class="data row3 col0" >0.16</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row3_col1" class="data row3 col1" >0.22</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row3_col2" class="data row3 col2" >0.11</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row3_col3" class="data row3 col3" >0</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row3_col4" class="data row3 col4" >0.09</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row3_col5" class="data row3 col5" >0.23</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row3_col6" class="data row3 col6" >0.84</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row3_col7" class="data row3 col7" >0.09</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row3_col8" class="data row3 col8" >0.18</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row3_col9" class="data row3 col9" >0.19</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row3_col10" class="data row3 col10" >0.14</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row3_col11" class="data row3 col11" >0.13</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row3_col12" class="data row3 col12" >0.09</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row3_col13" class="data row3 col13" >0.2</td> 
    </tr>    <tr> 
        <th id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4level0_row4" class="row_heading level0 row4" >es</th> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row4_col0" class="data row4 col0" >0.07</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row4_col1" class="data row4 col1" >0.16</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row4_col2" class="data row4 col2" >0.2</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row4_col3" class="data row4 col3" >0.09</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row4_col4" class="data row4 col4" >0</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row4_col5" class="data row4 col5" >0.26</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row4_col6" class="data row4 col6" >0.74</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row4_col7" class="data row4 col7" >0.15</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row4_col8" class="data row4 col8" >0.11</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row4_col9" class="data row4 col9" >0.25</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row4_col10" class="data row4 col10" >0.05</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row4_col11" class="data row4 col11" >0.04</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row4_col12" class="data row4 col12" >0.05</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row4_col13" class="data row4 col13" >0.17</td> 
    </tr>    <tr> 
        <th id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4level0_row5" class="row_heading level0 row5" >he</th> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row5_col0" class="data row5 col0" >0.28</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row5_col1" class="data row5 col1" >0.21</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row5_col2" class="data row5 col2" >0.33</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row5_col3" class="data row5 col3" >0.23</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row5_col4" class="data row5 col4" >0.26</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row5_col5" class="data row5 col5" >0</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row5_col6" class="data row5 col6" >0.84</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row5_col7" class="data row5 col7" >0.32</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row5_col8" class="data row5 col8" >0.24</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row5_col9" class="data row5 col9" >0.41</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row5_col10" class="data row5 col10" >0.29</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row5_col11" class="data row5 col11" >0.3</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row5_col12" class="data row5 col12" >0.31</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row5_col13" class="data row5 col13" >0.43</td> 
    </tr>    <tr> 
        <th id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4level0_row6" class="row_heading level0 row6" >hi</th> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row6_col0" class="data row6 col0" >0.67</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row6_col1" class="data row6 col1" >0.63</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row6_col2" class="data row6 col2" >0.94</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row6_col3" class="data row6 col3" >0.84</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row6_col4" class="data row6 col4" >0.74</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row6_col5" class="data row6 col5" >0.84</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row6_col6" class="data row6 col6" >0</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row6_col7" class="data row6 col7" >0.88</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row6_col8" class="data row6 col8" >0.66</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row6_col9" class="data row6 col9" >0.93</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row6_col10" class="data row6 col10" >0.69</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row6_col11" class="data row6 col11" >0.71</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row6_col12" class="data row6 col12" >0.75</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row6_col13" class="data row6 col13" >0.64</td> 
    </tr>    <tr> 
        <th id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4level0_row7" class="row_heading level0 row7" >hu</th> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row7_col0" class="data row7 col0" >0.21</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row7_col1" class="data row7 col1" >0.31</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row7_col2" class="data row7 col2" >0.06</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row7_col3" class="data row7 col3" >0.09</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row7_col4" class="data row7 col4" >0.15</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row7_col5" class="data row7 col5" >0.32</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row7_col6" class="data row7 col6" >0.88</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row7_col7" class="data row7 col7" >0</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row7_col8" class="data row7 col8" >0.26</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row7_col9" class="data row7 col9" >0.09</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row7_col10" class="data row7 col10" >0.19</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row7_col11" class="data row7 col11" >0.17</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row7_col12" class="data row7 col12" >0.13</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row7_col13" class="data row7 col13" >0.24</td> 
    </tr>    <tr> 
        <th id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4level0_row8" class="row_heading level0 row8" >ja</th> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row8_col0" class="data row8 col0" >0.05</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row8_col1" class="data row8 col1" >0.05</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row8_col2" class="data row8 col2" >0.28</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row8_col3" class="data row8 col3" >0.18</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row8_col4" class="data row8 col4" >0.11</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row8_col5" class="data row8 col5" >0.24</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row8_col6" class="data row8 col6" >0.66</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row8_col7" class="data row8 col7" >0.26</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row8_col8" class="data row8 col8" >0</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row8_col9" class="data row8 col9" >0.35</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row8_col10" class="data row8 col10" >0.08</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row8_col11" class="data row8 col11" >0.11</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row8_col12" class="data row8 col12" >0.16</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row8_col13" class="data row8 col13" >0.19</td> 
    </tr>    <tr> 
        <th id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4level0_row9" class="row_heading level0 row9" >nl</th> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row9_col0" class="data row9 col0" >0.3</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row9_col1" class="data row9 col1" >0.4</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row9_col2" class="data row9 col2" >0.09</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row9_col3" class="data row9 col3" >0.19</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row9_col4" class="data row9 col4" >0.25</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row9_col5" class="data row9 col5" >0.41</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row9_col6" class="data row9 col6" >0.93</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row9_col7" class="data row9 col7" >0.09</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row9_col8" class="data row9 col8" >0.35</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row9_col9" class="data row9 col9" >0</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row9_col10" class="data row9 col10" >0.27</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row9_col11" class="data row9 col11" >0.24</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row9_col12" class="data row9 col12" >0.19</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row9_col13" class="data row9 col13" >0.29</td> 
    </tr>    <tr> 
        <th id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4level0_row10" class="row_heading level0 row10" >ro</th> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row10_col0" class="data row10 col0" >0.03</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row10_col1" class="data row10 col1" >0.13</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row10_col2" class="data row10 col2" >0.25</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row10_col3" class="data row10 col3" >0.14</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row10_col4" class="data row10 col4" >0.05</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row10_col5" class="data row10 col5" >0.29</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row10_col6" class="data row10 col6" >0.69</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row10_col7" class="data row10 col7" >0.19</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row10_col8" class="data row10 col8" >0.08</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row10_col9" class="data row10 col9" >0.27</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row10_col10" class="data row10 col10" >0</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row10_col11" class="data row10 col11" >0.03</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row10_col12" class="data row10 col12" >0.08</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row10_col13" class="data row10 col13" >0.14</td> 
    </tr>    <tr> 
        <th id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4level0_row11" class="row_heading level0 row11" >ru</th> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row11_col0" class="data row11 col0" >0.06</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row11_col1" class="data row11 col1" >0.17</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row11_col2" class="data row11 col2" >0.23</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row11_col3" class="data row11 col3" >0.13</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row11_col4" class="data row11 col4" >0.04</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row11_col5" class="data row11 col5" >0.3</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row11_col6" class="data row11 col6" >0.71</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row11_col7" class="data row11 col7" >0.17</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row11_col8" class="data row11 col8" >0.11</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row11_col9" class="data row11 col9" >0.24</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row11_col10" class="data row11 col10" >0.03</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row11_col11" class="data row11 col11" >0</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row11_col12" class="data row11 col12" >0.05</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row11_col13" class="data row11 col13" >0.12</td> 
    </tr>    <tr> 
        <th id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4level0_row12" class="row_heading level0 row12" >uk</th> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row12_col0" class="data row12 col0" >0.11</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row12_col1" class="data row12 col1" >0.21</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row12_col2" class="data row12 col2" >0.19</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row12_col3" class="data row12 col3" >0.09</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row12_col4" class="data row12 col4" >0.05</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row12_col5" class="data row12 col5" >0.31</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row12_col6" class="data row12 col6" >0.75</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row12_col7" class="data row12 col7" >0.13</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row12_col8" class="data row12 col8" >0.16</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row12_col9" class="data row12 col9" >0.19</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row12_col10" class="data row12 col10" >0.08</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row12_col11" class="data row12 col11" >0.05</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row12_col12" class="data row12 col12" >0</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row12_col13" class="data row12 col13" >0.12</td> 
    </tr>    <tr> 
        <th id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4level0_row13" class="row_heading level0 row13" >zh</th> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row13_col0" class="data row13 col0" >0.15</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row13_col1" class="data row13 col1" >0.22</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row13_col2" class="data row13 col2" >0.3</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row13_col3" class="data row13 col3" >0.2</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row13_col4" class="data row13 col4" >0.17</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row13_col5" class="data row13 col5" >0.43</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row13_col6" class="data row13 col6" >0.64</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row13_col7" class="data row13 col7" >0.24</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row13_col8" class="data row13 col8" >0.19</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row13_col9" class="data row13 col9" >0.29</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row13_col10" class="data row13 col10" >0.14</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row13_col11" class="data row13 col11" >0.12</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row13_col12" class="data row13 col12" >0.12</td> 
        <td id="T_7acc6562_5ab1_11e8_abac_1866dafa0bc4row13_col13" class="data row13 col13" >0</td> 
    </tr></tbody> 
</table style="display:inline"> 



<h2>motivation (cosine,euclidean,l1 )</h2><style  type="text/css" >
    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4 th {
          font-size: 7pt;
    }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row0_col0 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row0_col1 {
            font-size:  1pt;
            background-color:  #dcdaeb;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row0_col2 {
            font-size:  1pt;
            background-color:  #9ebad9;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row0_col3 {
            font-size:  1pt;
            background-color:  #e7e3f0;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row0_col4 {
            font-size:  1pt;
            background-color:  #8fb4d6;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row0_col5 {
            font-size:  1pt;
            background-color:  #bfc9e1;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row0_col6 {
            font-size:  1pt;
            background-color:  #cccfe5;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row0_col7 {
            font-size:  1pt;
            background-color:  #dfddec;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row0_col8 {
            font-size:  1pt;
            background-color:  #f0eaf4;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row0_col9 {
            font-size:  1pt;
            background-color:  #b7c5df;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row0_col10 {
            font-size:  1pt;
            background-color:  #bcc7e1;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row0_col11 {
            font-size:  1pt;
            background-color:  #e9e5f1;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row0_col12 {
            font-size:  1pt;
            background-color:  #d1d2e6;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row0_col13 {
            font-size:  1pt;
            background-color:  #f0eaf4;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row1_col0 {
            font-size:  1pt;
            background-color:  #c6cce3;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row1_col1 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row1_col2 {
            font-size:  1pt;
            background-color:  #7eadd1;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row1_col3 {
            font-size:  1pt;
            background-color:  #99b8d8;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row1_col4 {
            font-size:  1pt;
            background-color:  #a8bedc;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row1_col5 {
            font-size:  1pt;
            background-color:  #9ab8d8;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row1_col6 {
            font-size:  1pt;
            background-color:  #f7f0f7;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row1_col7 {
            font-size:  1pt;
            background-color:  #d1d2e6;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row1_col8 {
            font-size:  1pt;
            background-color:  #abbfdc;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row1_col9 {
            font-size:  1pt;
            background-color:  #93b5d6;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row1_col10 {
            font-size:  1pt;
            background-color:  #ede7f2;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row1_col11 {
            font-size:  1pt;
            background-color:  #96b6d7;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row1_col12 {
            font-size:  1pt;
            background-color:  #f6eff7;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row1_col13 {
            font-size:  1pt;
            background-color:  #adc1dd;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row2_col0 {
            font-size:  1pt;
            background-color:  #c6cce3;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row2_col1 {
            font-size:  1pt;
            background-color:  #d3d4e7;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row2_col2 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row2_col3 {
            font-size:  1pt;
            background-color:  #e7e3f0;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row2_col4 {
            font-size:  1pt;
            background-color:  #f6eff7;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row2_col5 {
            font-size:  1pt;
            background-color:  #f8f1f8;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row2_col6 {
            font-size:  1pt;
            background-color:  #dad9ea;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row2_col7 {
            font-size:  1pt;
            background-color:  #d1d2e6;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row2_col8 {
            font-size:  1pt;
            background-color:  #e4e1ef;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row2_col9 {
            font-size:  1pt;
            background-color:  #f0eaf4;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row2_col10 {
            font-size:  1pt;
            background-color:  #ede7f2;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row2_col11 {
            font-size:  1pt;
            background-color:  #c6cce3;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row2_col12 {
            font-size:  1pt;
            background-color:  #d1d2e6;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row2_col13 {
            font-size:  1pt;
            background-color:  #f0eaf4;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row3_col0 {
            font-size:  1pt;
            background-color:  #c6cce3;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row3_col1 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row3_col2 {
            font-size:  1pt;
            background-color:  #9ebad9;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row3_col3 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row3_col4 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row3_col5 {
            font-size:  1pt;
            background-color:  #e7e3f0;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row3_col6 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row3_col7 {
            font-size:  1pt;
            background-color:  #bcc7e1;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row3_col8 {
            font-size:  1pt;
            background-color:  #faf3f9;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row3_col9 {
            font-size:  1pt;
            background-color:  #f0eaf4;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row3_col10 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row3_col11 {
            font-size:  1pt;
            background-color:  #c6cce3;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row3_col12 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row3_col13 {
            font-size:  1pt;
            background-color:  #f0eaf4;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row4_col0 {
            font-size:  1pt;
            background-color:  #a1bbda;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row4_col1 {
            font-size:  1pt;
            background-color:  #d3d4e7;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row4_col2 {
            font-size:  1pt;
            background-color:  #f3edf5;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row4_col3 {
            font-size:  1pt;
            background-color:  #cccfe5;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row4_col4 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row4_col5 {
            font-size:  1pt;
            background-color:  #dbdaeb;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row4_col6 {
            font-size:  1pt;
            background-color:  #e1dfed;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row4_col7 {
            font-size:  1pt;
            background-color:  #8fb4d6;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row4_col8 {
            font-size:  1pt;
            background-color:  #cdd0e5;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row4_col9 {
            font-size:  1pt;
            background-color:  #dcdaeb;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row4_col10 {
            font-size:  1pt;
            background-color:  #ede7f2;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row4_col11 {
            font-size:  1pt;
            background-color:  #79abd0;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row4_col12 {
            font-size:  1pt;
            background-color:  #bcc7e1;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row4_col13 {
            font-size:  1pt;
            background-color:  #d0d1e6;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row5_col0 {
            font-size:  1pt;
            background-color:  #b4c4df;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row5_col1 {
            font-size:  1pt;
            background-color:  #b9c6e0;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row5_col2 {
            font-size:  1pt;
            background-color:  #f3edf5;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row5_col3 {
            font-size:  1pt;
            background-color:  #f2ecf5;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row5_col4 {
            font-size:  1pt;
            background-color:  #d1d2e6;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row5_col5 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row5_col6 {
            font-size:  1pt;
            background-color:  #b8c6e0;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row5_col7 {
            font-size:  1pt;
            background-color:  #d1d2e6;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row5_col8 {
            font-size:  1pt;
            background-color:  #f0eaf4;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row5_col9 {
            font-size:  1pt;
            background-color:  #f5eff6;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row5_col10 {
            font-size:  1pt;
            background-color:  #d1d2e6;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row5_col11 {
            font-size:  1pt;
            background-color:  #c6cce3;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row5_col12 {
            font-size:  1pt;
            background-color:  #a8bedc;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row5_col13 {
            font-size:  1pt;
            background-color:  #f0eaf4;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row6_col0 {
            font-size:  1pt;
            background-color:  #75a9cf;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row6_col1 {
            font-size:  1pt;
            background-color:  #f3edf5;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row6_col2 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row6_col3 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row6_col4 {
            font-size:  1pt;
            background-color:  #a8bedc;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row6_col5 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row6_col6 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row6_col7 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row6_col8 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row6_col9 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row6_col10 {
            font-size:  1pt;
            background-color:  #ede7f2;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row6_col11 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row6_col12 {
            font-size:  1pt;
            background-color:  #d1d2e6;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row6_col13 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row7_col0 {
            font-size:  1pt;
            background-color:  #e3e0ee;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row7_col1 {
            font-size:  1pt;
            background-color:  #e5e1ef;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row7_col2 {
            font-size:  1pt;
            background-color:  #b9c6e0;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row7_col3 {
            font-size:  1pt;
            background-color:  #e7e3f0;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row7_col4 {
            font-size:  1pt;
            background-color:  #8fb4d6;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row7_col5 {
            font-size:  1pt;
            background-color:  #dbdaeb;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row7_col6 {
            font-size:  1pt;
            background-color:  #cccfe5;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row7_col7 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row7_col8 {
            font-size:  1pt;
            background-color:  #ebe6f2;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row7_col9 {
            font-size:  1pt;
            background-color:  #d5d5e8;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row7_col10 {
            font-size:  1pt;
            background-color:  #bcc7e1;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row7_col11 {
            font-size:  1pt;
            background-color:  #c6cce3;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row7_col12 {
            font-size:  1pt;
            background-color:  #dfddec;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row7_col13 {
            font-size:  1pt;
            background-color:  #e7e3f0;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row8_col0 {
            font-size:  1pt;
            background-color:  #e3e0ee;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row8_col1 {
            font-size:  1pt;
            background-color:  #8eb3d5;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row8_col2 {
            font-size:  1pt;
            background-color:  #9ebad9;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row8_col3 {
            font-size:  1pt;
            background-color:  #fbf3f9;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row8_col4 {
            font-size:  1pt;
            background-color:  #76aad0;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row8_col5 {
            font-size:  1pt;
            background-color:  #e7e3f0;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row8_col6 {
            font-size:  1pt;
            background-color:  #75a9cf;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row8_col7 {
            font-size:  1pt;
            background-color:  #d1d2e6;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row8_col8 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row8_col9 {
            font-size:  1pt;
            background-color:  #e4e1ef;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row8_col10 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row8_col11 {
            font-size:  1pt;
            background-color:  #c6cce3;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row8_col12 {
            font-size:  1pt;
            background-color:  #76aad0;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row8_col13 {
            font-size:  1pt;
            background-color:  #f8f1f8;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row9_col0 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row9_col1 {
            font-size:  1pt;
            background-color:  #6ba5cd;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row9_col2 {
            font-size:  1pt;
            background-color:  #d3d4e7;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row9_col3 {
            font-size:  1pt;
            background-color:  #f2ecf5;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row9_col4 {
            font-size:  1pt;
            background-color:  #a8bedc;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row9_col5 {
            font-size:  1pt;
            background-color:  #f0eaf4;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row9_col6 {
            font-size:  1pt;
            background-color:  #75a9cf;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row9_col7 {
            font-size:  1pt;
            background-color:  #8fb4d6;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row9_col8 {
            font-size:  1pt;
            background-color:  #e4e1ef;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row9_col9 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row9_col10 {
            font-size:  1pt;
            background-color:  #8fb4d6;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row9_col11 {
            font-size:  1pt;
            background-color:  #96b6d7;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row9_col12 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row9_col13 {
            font-size:  1pt;
            background-color:  #dbdaeb;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row10_col0 {
            font-size:  1pt;
            background-color:  #c6cce3;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row10_col1 {
            font-size:  1pt;
            background-color:  #f3edf5;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row10_col2 {
            font-size:  1pt;
            background-color:  #e5e1ef;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row10_col3 {
            font-size:  1pt;
            background-color:  #cccfe5;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row10_col4 {
            font-size:  1pt;
            background-color:  #ede7f2;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row10_col5 {
            font-size:  1pt;
            background-color:  #dbdaeb;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row10_col6 {
            font-size:  1pt;
            background-color:  #f7f0f7;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row10_col7 {
            font-size:  1pt;
            background-color:  #bcc7e1;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row10_col8 {
            font-size:  1pt;
            background-color:  #c1cae2;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row10_col9 {
            font-size:  1pt;
            background-color:  #d5d5e8;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row10_col10 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row10_col11 {
            font-size:  1pt;
            background-color:  #d9d8ea;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row10_col12 {
            font-size:  1pt;
            background-color:  #f6eff7;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row10_col13 {
            font-size:  1pt;
            background-color:  #d0d1e6;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row11_col0 {
            font-size:  1pt;
            background-color:  #eee9f3;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row11_col1 {
            font-size:  1pt;
            background-color:  #d3d4e7;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row11_col2 {
            font-size:  1pt;
            background-color:  #b9c6e0;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row11_col3 {
            font-size:  1pt;
            background-color:  #eee8f3;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row11_col4 {
            font-size:  1pt;
            background-color:  #8fb4d6;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row11_col5 {
            font-size:  1pt;
            background-color:  #dbdaeb;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row11_col6 {
            font-size:  1pt;
            background-color:  #d4d4e8;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row11_col7 {
            font-size:  1pt;
            background-color:  #d1d2e6;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row11_col8 {
            font-size:  1pt;
            background-color:  #ebe6f2;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row11_col9 {
            font-size:  1pt;
            background-color:  #dcdaeb;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row11_col10 {
            font-size:  1pt;
            background-color:  #dfddec;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row11_col11 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row11_col12 {
            font-size:  1pt;
            background-color:  #dfddec;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row11_col13 {
            font-size:  1pt;
            background-color:  #f0eaf4;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row12_col0 {
            font-size:  1pt;
            background-color:  #d7d6e9;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row12_col1 {
            font-size:  1pt;
            background-color:  #f9f2f8;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row12_col2 {
            font-size:  1pt;
            background-color:  #b9c6e0;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row12_col3 {
            font-size:  1pt;
            background-color:  #cccfe5;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row12_col4 {
            font-size:  1pt;
            background-color:  #bcc7e1;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row12_col5 {
            font-size:  1pt;
            background-color:  #bfc9e1;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row12_col6 {
            font-size:  1pt;
            background-color:  #eee8f3;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row12_col7 {
            font-size:  1pt;
            background-color:  #dfddec;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row12_col8 {
            font-size:  1pt;
            background-color:  #cdd0e5;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row12_col9 {
            font-size:  1pt;
            background-color:  #c1cae2;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row12_col10 {
            font-size:  1pt;
            background-color:  #f6eff7;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row12_col11 {
            font-size:  1pt;
            background-color:  #d9d8ea;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row12_col12 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row12_col13 {
            font-size:  1pt;
            background-color:  #dbdaeb;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row13_col0 {
            font-size:  1pt;
            background-color:  #eee9f3;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row13_col1 {
            font-size:  1pt;
            background-color:  #c6cce3;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row13_col2 {
            font-size:  1pt;
            background-color:  #e5e1ef;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row13_col3 {
            font-size:  1pt;
            background-color:  #f7f0f7;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row13_col4 {
            font-size:  1pt;
            background-color:  #bcc7e1;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row13_col5 {
            font-size:  1pt;
            background-color:  #f0eaf4;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row13_col6 {
            font-size:  1pt;
            background-color:  #b8c6e0;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row13_col7 {
            font-size:  1pt;
            background-color:  #dfddec;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row13_col8 {
            font-size:  1pt;
            background-color:  #faf3f9;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row13_col9 {
            font-size:  1pt;
            background-color:  #ebe6f2;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row13_col10 {
            font-size:  1pt;
            background-color:  #bcc7e1;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row13_col11 {
            font-size:  1pt;
            background-color:  #e9e5f1;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row13_col12 {
            font-size:  1pt;
            background-color:  #d1d2e6;
        }    #T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row13_col13 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }</style>  
<table style="display:inline" id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4" > 
<thead>    <tr> 
        <th class="index_name level0" ></th> 
        <th class="col_heading level0 col0" >ar</th> 
        <th class="col_heading level0 col1" >bn</th> 
        <th class="col_heading level0 col2" >de</th> 
        <th class="col_heading level0 col3" >en</th> 
        <th class="col_heading level0 col4" >es</th> 
        <th class="col_heading level0 col5" >he</th> 
        <th class="col_heading level0 col6" >hi</th> 
        <th class="col_heading level0 col7" >hu</th> 
        <th class="col_heading level0 col8" >ja</th> 
        <th class="col_heading level0 col9" >nl</th> 
        <th class="col_heading level0 col10" >ro</th> 
        <th class="col_heading level0 col11" >ru</th> 
        <th class="col_heading level0 col12" >uk</th> 
        <th class="col_heading level0 col13" >zh</th> 
    </tr>    <tr> 
        <th class="index_name level0" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
    </tr></thead> 
<tbody>    <tr> 
        <th id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4level0_row0" class="row_heading level0 row0" >ar</th> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row0_col0" class="data row0 col0" >0</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row0_col1" class="data row0 col1" >0.05</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row0_col2" class="data row0 col2" >0.05</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row0_col3" class="data row0 col3" >0.05</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row0_col4" class="data row0 col4" >0.07</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row0_col5" class="data row0 col5" >0.06</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row0_col6" class="data row0 col6" >0.09</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row0_col7" class="data row0 col7" >0.03</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row0_col8" class="data row0 col8" >0.03</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row0_col9" class="data row0 col9" >0.1</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row0_col10" class="data row0 col10" >0.05</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row0_col11" class="data row0 col11" >0.02</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row0_col12" class="data row0 col12" >0.04</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row0_col13" class="data row0 col13" >0.02</td> 
    </tr>    <tr> 
        <th id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4level0_row1" class="row_heading level0 row1" >bn</th> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row1_col0" class="data row1 col0" >0.05</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row1_col1" class="data row1 col1" >0</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row1_col2" class="data row1 col2" >0.06</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row1_col3" class="data row1 col3" >0.14</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row1_col4" class="data row1 col4" >0.06</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row1_col5" class="data row1 col5" >0.08</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row1_col6" class="data row1 col6" >0.02</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row1_col7" class="data row1 col7" >0.04</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row1_col8" class="data row1 col8" >0.11</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row1_col9" class="data row1 col9" >0.13</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row1_col10" class="data row1 col10" >0.02</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row1_col11" class="data row1 col11" >0.06</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row1_col12" class="data row1 col12" >0.01</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row1_col13" class="data row1 col13" >0.07</td> 
    </tr>    <tr> 
        <th id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4level0_row2" class="row_heading level0 row2" >de</th> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row2_col0" class="data row2 col0" >0.05</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row2_col1" class="data row2 col1" >0.06</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row2_col2" class="data row2 col2" >0</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row2_col3" class="data row2 col3" >0.05</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row2_col4" class="data row2 col4" >0.01</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row2_col5" class="data row2 col5" >0.01</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row2_col6" class="data row2 col6" >0.07</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row2_col7" class="data row2 col7" >0.04</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row2_col8" class="data row2 col8" >0.05</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row2_col9" class="data row2 col9" >0.03</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row2_col10" class="data row2 col10" >0.02</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row2_col11" class="data row2 col11" >0.04</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row2_col12" class="data row2 col12" >0.04</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row2_col13" class="data row2 col13" >0.02</td> 
    </tr>    <tr> 
        <th id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4level0_row3" class="row_heading level0 row3" >en</th> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row3_col0" class="data row3 col0" >0.05</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row3_col1" class="data row3 col1" >0.14</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row3_col2" class="data row3 col2" >0.05</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row3_col3" class="data row3 col3" >0</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row3_col4" class="data row3 col4" >0.09</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row3_col5" class="data row3 col5" >0.03</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row3_col6" class="data row3 col6" >0.19</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row3_col7" class="data row3 col7" >0.05</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row3_col8" class="data row3 col8" >0.01</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row3_col9" class="data row3 col9" >0.03</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row3_col10" class="data row3 col10" >0.09</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row3_col11" class="data row3 col11" >0.04</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row3_col12" class="data row3 col12" >0.09</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row3_col13" class="data row3 col13" >0.02</td> 
    </tr>    <tr> 
        <th id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4level0_row4" class="row_heading level0 row4" >es</th> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row4_col0" class="data row4 col0" >0.07</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row4_col1" class="data row4 col1" >0.06</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row4_col2" class="data row4 col2" >0.01</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row4_col3" class="data row4 col3" >0.09</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row4_col4" class="data row4 col4" >0</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row4_col5" class="data row4 col5" >0.04</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row4_col6" class="data row4 col6" >0.06</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row4_col7" class="data row4 col7" >0.07</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row4_col8" class="data row4 col8" >0.08</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row4_col9" class="data row4 col9" >0.06</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row4_col10" class="data row4 col10" >0.02</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row4_col11" class="data row4 col11" >0.07</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row4_col12" class="data row4 col12" >0.05</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row4_col13" class="data row4 col13" >0.05</td> 
    </tr>    <tr> 
        <th id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4level0_row5" class="row_heading level0 row5" >he</th> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row5_col0" class="data row5 col0" >0.06</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row5_col1" class="data row5 col1" >0.08</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row5_col2" class="data row5 col2" >0.01</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row5_col3" class="data row5 col3" >0.03</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row5_col4" class="data row5 col4" >0.04</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row5_col5" class="data row5 col5" >0</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row5_col6" class="data row5 col6" >0.11</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row5_col7" class="data row5 col7" >0.04</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row5_col8" class="data row5 col8" >0.03</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row5_col9" class="data row5 col9" >0.02</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row5_col10" class="data row5 col10" >0.04</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row5_col11" class="data row5 col11" >0.04</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row5_col12" class="data row5 col12" >0.06</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row5_col13" class="data row5 col13" >0.02</td> 
    </tr>    <tr> 
        <th id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4level0_row6" class="row_heading level0 row6" >hi</th> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row6_col0" class="data row6 col0" >0.09</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row6_col1" class="data row6 col1" >0.02</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row6_col2" class="data row6 col2" >0.07</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row6_col3" class="data row6 col3" >0.19</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row6_col4" class="data row6 col4" >0.06</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row6_col5" class="data row6 col5" >0.11</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row6_col6" class="data row6 col6" >0</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row6_col7" class="data row6 col7" >0.09</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row6_col8" class="data row6 col8" >0.17</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row6_col9" class="data row6 col9" >0.17</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row6_col10" class="data row6 col10" >0.02</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row6_col11" class="data row6 col11" >0.08</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row6_col12" class="data row6 col12" >0.04</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row6_col13" class="data row6 col13" >0.11</td> 
    </tr>    <tr> 
        <th id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4level0_row7" class="row_heading level0 row7" >hu</th> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row7_col0" class="data row7 col0" >0.03</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row7_col1" class="data row7 col1" >0.04</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row7_col2" class="data row7 col2" >0.04</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row7_col3" class="data row7 col3" >0.05</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row7_col4" class="data row7 col4" >0.07</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row7_col5" class="data row7 col5" >0.04</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row7_col6" class="data row7 col6" >0.09</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row7_col7" class="data row7 col7" >0</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row7_col8" class="data row7 col8" >0.04</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row7_col9" class="data row7 col9" >0.07</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row7_col10" class="data row7 col10" >0.05</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row7_col11" class="data row7 col11" >0.04</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row7_col12" class="data row7 col12" >0.03</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row7_col13" class="data row7 col13" >0.03</td> 
    </tr>    <tr> 
        <th id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4level0_row8" class="row_heading level0 row8" >ja</th> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row8_col0" class="data row8 col0" >0.03</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row8_col1" class="data row8 col1" >0.11</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row8_col2" class="data row8 col2" >0.05</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row8_col3" class="data row8 col3" >0.01</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row8_col4" class="data row8 col4" >0.08</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row8_col5" class="data row8 col5" >0.03</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row8_col6" class="data row8 col6" >0.17</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row8_col7" class="data row8 col7" >0.04</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row8_col8" class="data row8 col8" >0</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row8_col9" class="data row8 col9" >0.05</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row8_col10" class="data row8 col10" >0.09</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row8_col11" class="data row8 col11" >0.04</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row8_col12" class="data row8 col12" >0.08</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row8_col13" class="data row8 col13" >0.01</td> 
    </tr>    <tr> 
        <th id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4level0_row9" class="row_heading level0 row9" >nl</th> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row9_col0" class="data row9 col0" >0.1</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row9_col1" class="data row9 col1" >0.13</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row9_col2" class="data row9 col2" >0.03</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row9_col3" class="data row9 col3" >0.03</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row9_col4" class="data row9 col4" >0.06</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row9_col5" class="data row9 col5" >0.02</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row9_col6" class="data row9 col6" >0.17</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row9_col7" class="data row9 col7" >0.07</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row9_col8" class="data row9 col8" >0.05</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row9_col9" class="data row9 col9" >0</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row9_col10" class="data row9 col10" >0.07</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row9_col11" class="data row9 col11" >0.06</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row9_col12" class="data row9 col12" >0.09</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row9_col13" class="data row9 col13" >0.04</td> 
    </tr>    <tr> 
        <th id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4level0_row10" class="row_heading level0 row10" >ro</th> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row10_col0" class="data row10 col0" >0.05</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row10_col1" class="data row10 col1" >0.02</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row10_col2" class="data row10 col2" >0.02</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row10_col3" class="data row10 col3" >0.09</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row10_col4" class="data row10 col4" >0.02</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row10_col5" class="data row10 col5" >0.04</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row10_col6" class="data row10 col6" >0.02</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row10_col7" class="data row10 col7" >0.05</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row10_col8" class="data row10 col8" >0.09</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row10_col9" class="data row10 col9" >0.07</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row10_col10" class="data row10 col10" >0</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row10_col11" class="data row10 col11" >0.03</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row10_col12" class="data row10 col12" >0.01</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row10_col13" class="data row10 col13" >0.05</td> 
    </tr>    <tr> 
        <th id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4level0_row11" class="row_heading level0 row11" >ru</th> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row11_col0" class="data row11 col0" >0.02</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row11_col1" class="data row11 col1" >0.06</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row11_col2" class="data row11 col2" >0.04</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row11_col3" class="data row11 col3" >0.04</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row11_col4" class="data row11 col4" >0.07</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row11_col5" class="data row11 col5" >0.04</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row11_col6" class="data row11 col6" >0.08</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row11_col7" class="data row11 col7" >0.04</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row11_col8" class="data row11 col8" >0.04</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row11_col9" class="data row11 col9" >0.06</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row11_col10" class="data row11 col10" >0.03</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row11_col11" class="data row11 col11" >0</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row11_col12" class="data row11 col12" >0.03</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row11_col13" class="data row11 col13" >0.02</td> 
    </tr>    <tr> 
        <th id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4level0_row12" class="row_heading level0 row12" >uk</th> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row12_col0" class="data row12 col0" >0.04</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row12_col1" class="data row12 col1" >0.01</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row12_col2" class="data row12 col2" >0.04</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row12_col3" class="data row12 col3" >0.09</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row12_col4" class="data row12 col4" >0.05</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row12_col5" class="data row12 col5" >0.06</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row12_col6" class="data row12 col6" >0.04</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row12_col7" class="data row12 col7" >0.03</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row12_col8" class="data row12 col8" >0.08</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row12_col9" class="data row12 col9" >0.09</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row12_col10" class="data row12 col10" >0.01</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row12_col11" class="data row12 col11" >0.03</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row12_col12" class="data row12 col12" >0</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row12_col13" class="data row12 col13" >0.04</td> 
    </tr>    <tr> 
        <th id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4level0_row13" class="row_heading level0 row13" >zh</th> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row13_col0" class="data row13 col0" >0.02</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row13_col1" class="data row13 col1" >0.07</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row13_col2" class="data row13 col2" >0.02</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row13_col3" class="data row13 col3" >0.02</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row13_col4" class="data row13 col4" >0.05</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row13_col5" class="data row13 col5" >0.02</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row13_col6" class="data row13 col6" >0.11</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row13_col7" class="data row13 col7" >0.03</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row13_col8" class="data row13 col8" >0.01</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row13_col9" class="data row13 col9" >0.04</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row13_col10" class="data row13 col10" >0.05</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row13_col11" class="data row13 col11" >0.02</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row13_col12" class="data row13 col12" >0.04</td> 
        <td id="T_7adca7a6_5ab1_11e8_abac_1866dafa0bc4row13_col13" class="data row13 col13" >0</td> 
    </tr></tbody> 
</table style="display:inline"> <style  type="text/css" >
    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4 th {
          font-size: 7pt;
    }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row0_col0 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row0_col1 {
            font-size:  1pt;
            background-color:  #b0c2de;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row0_col2 {
            font-size:  1pt;
            background-color:  #a9bfdc;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row0_col3 {
            font-size:  1pt;
            background-color:  #c2cbe2;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row0_col4 {
            font-size:  1pt;
            background-color:  #8eb3d5;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row0_col5 {
            font-size:  1pt;
            background-color:  #abbfdc;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row0_col6 {
            font-size:  1pt;
            background-color:  #99b8d8;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row0_col7 {
            font-size:  1pt;
            background-color:  #b7c5df;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row0_col8 {
            font-size:  1pt;
            background-color:  #c9cee4;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row0_col9 {
            font-size:  1pt;
            background-color:  #a4bcda;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row0_col10 {
            font-size:  1pt;
            background-color:  #80aed2;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row0_col11 {
            font-size:  1pt;
            background-color:  #cdd0e5;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row0_col12 {
            font-size:  1pt;
            background-color:  #a8bedc;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row0_col13 {
            font-size:  1pt;
            background-color:  #d6d6e9;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row1_col0 {
            font-size:  1pt;
            background-color:  #73a9cf;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row1_col1 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row1_col2 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row1_col3 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row1_col4 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row1_col5 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row1_col6 {
            font-size:  1pt;
            background-color:  #b8c6e0;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row1_col7 {
            font-size:  1pt;
            background-color:  #84b0d3;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row1_col8 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row1_col9 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row1_col10 {
            font-size:  1pt;
            background-color:  #63a2cb;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row1_col11 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row1_col12 {
            font-size:  1pt;
            background-color:  #9fbad9;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row1_col13 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row2_col0 {
            font-size:  1pt;
            background-color:  #97b7d7;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row2_col1 {
            font-size:  1pt;
            background-color:  #8fb4d6;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row2_col2 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row2_col3 {
            font-size:  1pt;
            background-color:  #d9d8ea;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row2_col4 {
            font-size:  1pt;
            background-color:  #dedcec;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row2_col5 {
            font-size:  1pt;
            background-color:  #eae6f1;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row2_col6 {
            font-size:  1pt;
            background-color:  #a9bfdc;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row2_col7 {
            font-size:  1pt;
            background-color:  #8cb3d5;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row2_col8 {
            font-size:  1pt;
            background-color:  #d7d6e9;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row2_col9 {
            font-size:  1pt;
            background-color:  #e4e1ef;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row2_col10 {
            font-size:  1pt;
            background-color:  #c0c9e2;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row2_col11 {
            font-size:  1pt;
            background-color:  #c6cce3;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row2_col12 {
            font-size:  1pt;
            background-color:  #9fbad9;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row2_col13 {
            font-size:  1pt;
            background-color:  #b5c4df;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row3_col0 {
            font-size:  1pt;
            background-color:  #97b7d7;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row3_col1 {
            font-size:  1pt;
            background-color:  #5ea0ca;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row3_col2 {
            font-size:  1pt;
            background-color:  #c9cee4;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row3_col3 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row3_col4 {
            font-size:  1pt;
            background-color:  #88b1d4;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row3_col5 {
            font-size:  1pt;
            background-color:  #dddbec;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row3_col6 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row3_col7 {
            font-size:  1pt;
            background-color:  #84b0d3;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row3_col8 {
            font-size:  1pt;
            background-color:  #eee8f3;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row3_col9 {
            font-size:  1pt;
            background-color:  #e1dfed;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row3_col10 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row3_col11 {
            font-size:  1pt;
            background-color:  #c1cae2;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row3_col12 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row3_col13 {
            font-size:  1pt;
            background-color:  #c1cae2;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row4_col0 {
            font-size:  1pt;
            background-color:  #89b1d4;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row4_col1 {
            font-size:  1pt;
            background-color:  #9ebad9;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row4_col2 {
            font-size:  1pt;
            background-color:  #e1dfed;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row4_col3 {
            font-size:  1pt;
            background-color:  #b5c4df;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row4_col4 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row4_col5 {
            font-size:  1pt;
            background-color:  #c9cee4;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row4_col6 {
            font-size:  1pt;
            background-color:  #bdc8e1;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row4_col7 {
            font-size:  1pt;
            background-color:  #75a9cf;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row4_col8 {
            font-size:  1pt;
            background-color:  #b8c6e0;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row4_col9 {
            font-size:  1pt;
            background-color:  #cdd0e5;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row4_col10 {
            font-size:  1pt;
            background-color:  #c6cce3;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row4_col11 {
            font-size:  1pt;
            background-color:  #a4bcda;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row4_col12 {
            font-size:  1pt;
            background-color:  #9fbad9;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row4_col13 {
            font-size:  1pt;
            background-color:  #a9bfdc;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row5_col0 {
            font-size:  1pt;
            background-color:  #8fb4d6;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row5_col1 {
            font-size:  1pt;
            background-color:  #86b0d3;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row5_col2 {
            font-size:  1pt;
            background-color:  #e8e4f0;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row5_col3 {
            font-size:  1pt;
            background-color:  #e3e0ee;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row5_col4 {
            font-size:  1pt;
            background-color:  #bcc7e1;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row5_col5 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row5_col6 {
            font-size:  1pt;
            background-color:  #93b5d6;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row5_col7 {
            font-size:  1pt;
            background-color:  #94b6d7;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row5_col8 {
            font-size:  1pt;
            background-color:  #dddbec;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row5_col9 {
            font-size:  1pt;
            background-color:  #e7e3f0;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row5_col10 {
            font-size:  1pt;
            background-color:  #a2bcda;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row5_col11 {
            font-size:  1pt;
            background-color:  #c1cae2;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row5_col12 {
            font-size:  1pt;
            background-color:  #88b1d4;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row5_col13 {
            font-size:  1pt;
            background-color:  #c1cae2;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row6_col0 {
            font-size:  1pt;
            background-color:  #73a9cf;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row6_col1 {
            font-size:  1pt;
            background-color:  #c9cee4;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row6_col2 {
            font-size:  1pt;
            background-color:  #9ebad9;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row6_col3 {
            font-size:  1pt;
            background-color:  #7dacd1;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row6_col4 {
            font-size:  1pt;
            background-color:  #a9bfdc;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row6_col5 {
            font-size:  1pt;
            background-color:  #8eb3d5;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row6_col6 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row6_col7 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row6_col8 {
            font-size:  1pt;
            background-color:  #81aed2;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row6_col9 {
            font-size:  1pt;
            background-color:  #8bb2d4;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row6_col10 {
            font-size:  1pt;
            background-color:  #c0c9e2;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row6_col11 {
            font-size:  1pt;
            background-color:  #88b1d4;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row6_col12 {
            font-size:  1pt;
            background-color:  #afc1dd;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row6_col13 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row7_col0 {
            font-size:  1pt;
            background-color:  #bfc9e1;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row7_col1 {
            font-size:  1pt;
            background-color:  #c1cae2;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row7_col2 {
            font-size:  1pt;
            background-color:  #a9bfdc;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row7_col3 {
            font-size:  1pt;
            background-color:  #bfc9e1;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row7_col4 {
            font-size:  1pt;
            background-color:  #88b1d4;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row7_col5 {
            font-size:  1pt;
            background-color:  #b5c4df;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row7_col6 {
            font-size:  1pt;
            background-color:  #93b5d6;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row7_col7 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row7_col8 {
            font-size:  1pt;
            background-color:  #bcc7e1;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row7_col9 {
            font-size:  1pt;
            background-color:  #acc0dd;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row7_col10 {
            font-size:  1pt;
            background-color:  #89b1d4;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row7_col11 {
            font-size:  1pt;
            background-color:  #b0c2de;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row7_col12 {
            font-size:  1pt;
            background-color:  #b5c4df;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row7_col13 {
            font-size:  1pt;
            background-color:  #c1cae2;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row8_col0 {
            font-size:  1pt;
            background-color:  #a5bddb;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row8_col1 {
            font-size:  1pt;
            background-color:  #65a3cb;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row8_col2 {
            font-size:  1pt;
            background-color:  #c9cee4;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row8_col3 {
            font-size:  1pt;
            background-color:  #eee8f3;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row8_col4 {
            font-size:  1pt;
            background-color:  #8eb3d5;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row8_col5 {
            font-size:  1pt;
            background-color:  #d6d6e9;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row8_col6 {
            font-size:  1pt;
            background-color:  #67a4cc;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row8_col7 {
            font-size:  1pt;
            background-color:  #84b0d3;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row8_col8 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row8_col9 {
            font-size:  1pt;
            background-color:  #d9d8ea;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row8_col10 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row8_col11 {
            font-size:  1pt;
            background-color:  #b5c4df;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row8_col12 {
            font-size:  1pt;
            background-color:  #63a2cb;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row8_col13 {
            font-size:  1pt;
            background-color:  #bcc7e1;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row9_col0 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row9_col1 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row9_col2 {
            font-size:  1pt;
            background-color:  #dad9ea;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row9_col3 {
            font-size:  1pt;
            background-color:  #e0deed;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row9_col4 {
            font-size:  1pt;
            background-color:  #a9bfdc;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row9_col5 {
            font-size:  1pt;
            background-color:  #e0dded;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row9_col6 {
            font-size:  1pt;
            background-color:  #67a4cc;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row9_col7 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row9_col8 {
            font-size:  1pt;
            background-color:  #d7d6e9;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row9_col9 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row9_col10 {
            font-size:  1pt;
            background-color:  #6da6cd;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row9_col11 {
            font-size:  1pt;
            background-color:  #a4bcda;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row9_col12 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row9_col13 {
            font-size:  1pt;
            background-color:  #96b6d7;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row10_col0 {
            font-size:  1pt;
            background-color:  #9fbad9;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row10_col1 {
            font-size:  1pt;
            background-color:  #bcc7e1;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row10_col2 {
            font-size:  1pt;
            background-color:  #d7d6e9;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row10_col3 {
            font-size:  1pt;
            background-color:  #b5c4df;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row10_col4 {
            font-size:  1pt;
            background-color:  #d6d6e9;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row10_col5 {
            font-size:  1pt;
            background-color:  #c9cee4;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row10_col6 {
            font-size:  1pt;
            background-color:  #dad9ea;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row10_col7 {
            font-size:  1pt;
            background-color:  #9cb9d9;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row10_col8 {
            font-size:  1pt;
            background-color:  #b4c4df;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row10_col9 {
            font-size:  1pt;
            background-color:  #c1cae2;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row10_col10 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row10_col11 {
            font-size:  1pt;
            background-color:  #cdd0e5;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row10_col12 {
            font-size:  1pt;
            background-color:  #d6d6e9;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row10_col13 {
            font-size:  1pt;
            background-color:  #a9bfdc;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row11_col0 {
            font-size:  1pt;
            background-color:  #cacee5;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row11_col1 {
            font-size:  1pt;
            background-color:  #9ebad9;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row11_col2 {
            font-size:  1pt;
            background-color:  #ced0e6;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row11_col3 {
            font-size:  1pt;
            background-color:  #d9d8ea;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row11_col4 {
            font-size:  1pt;
            background-color:  #a4bcda;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row11_col5 {
            font-size:  1pt;
            background-color:  #ced0e6;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row11_col6 {
            font-size:  1pt;
            background-color:  #a4bcda;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row11_col7 {
            font-size:  1pt;
            background-color:  #a2bcda;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row11_col8 {
            font-size:  1pt;
            background-color:  #d2d2e7;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row11_col9 {
            font-size:  1pt;
            background-color:  #c9cee4;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row11_col10 {
            font-size:  1pt;
            background-color:  #b8c6e0;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row11_col11 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row11_col12 {
            font-size:  1pt;
            background-color:  #b5c4df;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row11_col13 {
            font-size:  1pt;
            background-color:  #cdd0e5;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row12_col0 {
            font-size:  1pt;
            background-color:  #b8c6e0;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row12_col1 {
            font-size:  1pt;
            background-color:  #d3d4e7;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row12_col2 {
            font-size:  1pt;
            background-color:  #bfc9e1;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row12_col3 {
            font-size:  1pt;
            background-color:  #b1c2de;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row12_col4 {
            font-size:  1pt;
            background-color:  #b5c4df;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row12_col5 {
            font-size:  1pt;
            background-color:  #b5c4df;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row12_col6 {
            font-size:  1pt;
            background-color:  #d1d2e6;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row12_col7 {
            font-size:  1pt;
            background-color:  #bdc8e1;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row12_col8 {
            font-size:  1pt;
            background-color:  #b4c4df;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row12_col9 {
            font-size:  1pt;
            background-color:  #b4c4df;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row12_col10 {
            font-size:  1pt;
            background-color:  #d3d4e7;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row12_col11 {
            font-size:  1pt;
            background-color:  #c6cce3;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row12_col12 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row12_col13 {
            font-size:  1pt;
            background-color:  #b0c2de;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row13_col0 {
            font-size:  1pt;
            background-color:  #d4d4e8;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row13_col1 {
            font-size:  1pt;
            background-color:  #9ebad9;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row13_col2 {
            font-size:  1pt;
            background-color:  #bfc9e1;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row13_col3 {
            font-size:  1pt;
            background-color:  #d9d8ea;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row13_col4 {
            font-size:  1pt;
            background-color:  #a9bfdc;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row13_col5 {
            font-size:  1pt;
            background-color:  #ced0e6;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row13_col6 {
            font-size:  1pt;
            background-color:  #81aed2;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row13_col7 {
            font-size:  1pt;
            background-color:  #b7c5df;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row13_col8 {
            font-size:  1pt;
            background-color:  #d4d4e8;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row13_col9 {
            font-size:  1pt;
            background-color:  #c1cae2;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row13_col10 {
            font-size:  1pt;
            background-color:  #89b1d4;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row13_col11 {
            font-size:  1pt;
            background-color:  #cdd0e5;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row13_col12 {
            font-size:  1pt;
            background-color:  #97b7d7;
        }    #T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row13_col13 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }</style>  
<table style="display:inline" id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4" > 
<thead>    <tr> 
        <th class="index_name level0" ></th> 
        <th class="col_heading level0 col0" >ar</th> 
        <th class="col_heading level0 col1" >bn</th> 
        <th class="col_heading level0 col2" >de</th> 
        <th class="col_heading level0 col3" >en</th> 
        <th class="col_heading level0 col4" >es</th> 
        <th class="col_heading level0 col5" >he</th> 
        <th class="col_heading level0 col6" >hi</th> 
        <th class="col_heading level0 col7" >hu</th> 
        <th class="col_heading level0 col8" >ja</th> 
        <th class="col_heading level0 col9" >nl</th> 
        <th class="col_heading level0 col10" >ro</th> 
        <th class="col_heading level0 col11" >ru</th> 
        <th class="col_heading level0 col12" >uk</th> 
        <th class="col_heading level0 col13" >zh</th> 
    </tr>    <tr> 
        <th class="index_name level0" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
    </tr></thead> 
<tbody>    <tr> 
        <th id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4level0_row0" class="row_heading level0 row0" >ar</th> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row0_col0" class="data row0 col0" >0</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row0_col1" class="data row0 col1" >0.28</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row0_col2" class="data row0 col2" >0.23</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row0_col3" class="data row0 col3" >0.23</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row0_col4" class="data row0 col4" >0.25</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row0_col5" class="data row0 col5" >0.24</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row0_col6" class="data row0 col6" >0.28</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row0_col7" class="data row0 col7" >0.17</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row0_col8" class="data row0 col8" >0.21</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row0_col9" class="data row0 col9" >0.31</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row0_col10" class="data row0 col10" >0.22</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row0_col11" class="data row0 col11" >0.15</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row0_col12" class="data row0 col12" >0.18</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row0_col13" class="data row0 col13" >0.13</td> 
    </tr>    <tr> 
        <th id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4level0_row1" class="row_heading level0 row1" >bn</th> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row1_col0" class="data row1 col0" >0.28</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row1_col1" class="data row1 col1" >0</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row1_col2" class="data row1 col2" >0.35</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row1_col3" class="data row1 col3" >0.44</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row1_col4" class="data row1 col4" >0.32</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row1_col5" class="data row1 col5" >0.37</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row1_col6" class="data row1 col6" >0.22</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row1_col7" class="data row1 col7" >0.24</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row1_col8" class="data row1 col8" >0.43</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row1_col9" class="data row1 col9" >0.45</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row1_col10" class="data row1 col10" >0.25</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row1_col11" class="data row1 col11" >0.32</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row1_col12" class="data row1 col12" >0.19</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row1_col13" class="data row1 col13" >0.32</td> 
    </tr>    <tr> 
        <th id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4level0_row2" class="row_heading level0 row2" >de</th> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row2_col0" class="data row2 col0" >0.23</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row2_col1" class="data row2 col1" >0.35</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row2_col2" class="data row2 col2" >0</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row2_col3" class="data row2 col3" >0.17</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row2_col4" class="data row2 col4" >0.11</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row2_col5" class="data row2 col5" >0.09</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row2_col6" class="data row2 col6" >0.25</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row2_col7" class="data row2 col7" >0.23</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row2_col8" class="data row2 col8" >0.17</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row2_col9" class="data row2 col9" >0.13</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row2_col10" class="data row2 col10" >0.14</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row2_col11" class="data row2 col11" >0.16</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row2_col12" class="data row2 col12" >0.19</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row2_col13" class="data row2 col13" >0.19</td> 
    </tr>    <tr> 
        <th id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4level0_row3" class="row_heading level0 row3" >en</th> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row3_col0" class="data row3 col0" >0.23</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row3_col1" class="data row3 col1" >0.44</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row3_col2" class="data row3 col2" >0.17</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row3_col3" class="data row3 col3" >0</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row3_col4" class="data row3 col4" >0.26</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row3_col5" class="data row3 col5" >0.13</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row3_col6" class="data row3 col6" >0.38</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row3_col7" class="data row3 col7" >0.24</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row3_col8" class="data row3 col8" >0.09</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row3_col9" class="data row3 col9" >0.14</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row3_col10" class="data row3 col10" >0.26</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row3_col11" class="data row3 col11" >0.17</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row3_col12" class="data row3 col12" >0.27</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row3_col13" class="data row3 col13" >0.17</td> 
    </tr>    <tr> 
        <th id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4level0_row4" class="row_heading level0 row4" >es</th> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row4_col0" class="data row4 col0" >0.25</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row4_col1" class="data row4 col1" >0.32</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row4_col2" class="data row4 col2" >0.11</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row4_col3" class="data row4 col3" >0.26</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row4_col4" class="data row4 col4" >0</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row4_col5" class="data row4 col5" >0.18</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row4_col6" class="data row4 col6" >0.21</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row4_col7" class="data row4 col7" >0.26</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row4_col8" class="data row4 col8" >0.25</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row4_col9" class="data row4 col9" >0.21</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row4_col10" class="data row4 col10" >0.13</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row4_col11" class="data row4 col11" >0.22</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row4_col12" class="data row4 col12" >0.19</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row4_col13" class="data row4 col13" >0.21</td> 
    </tr>    <tr> 
        <th id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4level0_row5" class="row_heading level0 row5" >he</th> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row5_col0" class="data row5 col0" >0.24</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row5_col1" class="data row5 col1" >0.37</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row5_col2" class="data row5 col2" >0.09</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row5_col3" class="data row5 col3" >0.13</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row5_col4" class="data row5 col4" >0.18</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row5_col5" class="data row5 col5" >0</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row5_col6" class="data row5 col6" >0.29</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row5_col7" class="data row5 col7" >0.22</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row5_col8" class="data row5 col8" >0.15</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row5_col9" class="data row5 col9" >0.12</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row5_col10" class="data row5 col10" >0.18</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row5_col11" class="data row5 col11" >0.17</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row5_col12" class="data row5 col12" >0.22</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row5_col13" class="data row5 col13" >0.17</td> 
    </tr>    <tr> 
        <th id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4level0_row6" class="row_heading level0 row6" >hi</th> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row6_col0" class="data row6 col0" >0.28</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row6_col1" class="data row6 col1" >0.22</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row6_col2" class="data row6 col2" >0.25</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row6_col3" class="data row6 col3" >0.38</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row6_col4" class="data row6 col4" >0.21</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row6_col5" class="data row6 col5" >0.29</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row6_col6" class="data row6 col6" >0</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row6_col7" class="data row6 col7" >0.29</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row6_col8" class="data row6 col8" >0.36</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row6_col9" class="data row6 col9" >0.36</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row6_col10" class="data row6 col10" >0.14</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row6_col11" class="data row6 col11" >0.26</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row6_col12" class="data row6 col12" >0.17</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row6_col13" class="data row6 col13" >0.32</td> 
    </tr>    <tr> 
        <th id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4level0_row7" class="row_heading level0 row7" >hu</th> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row7_col0" class="data row7 col0" >0.17</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row7_col1" class="data row7 col1" >0.24</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row7_col2" class="data row7 col2" >0.23</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row7_col3" class="data row7 col3" >0.24</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row7_col4" class="data row7 col4" >0.26</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row7_col5" class="data row7 col5" >0.22</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row7_col6" class="data row7 col6" >0.29</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row7_col7" class="data row7 col7" >0</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row7_col8" class="data row7 col8" >0.24</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row7_col9" class="data row7 col9" >0.29</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row7_col10" class="data row7 col10" >0.21</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row7_col11" class="data row7 col11" >0.2</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row7_col12" class="data row7 col12" >0.16</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row7_col13" class="data row7 col13" >0.17</td> 
    </tr>    <tr> 
        <th id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4level0_row8" class="row_heading level0 row8" >ja</th> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row8_col0" class="data row8 col0" >0.21</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row8_col1" class="data row8 col1" >0.43</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row8_col2" class="data row8 col2" >0.17</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row8_col3" class="data row8 col3" >0.09</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row8_col4" class="data row8 col4" >0.25</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row8_col5" class="data row8 col5" >0.15</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row8_col6" class="data row8 col6" >0.36</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row8_col7" class="data row8 col7" >0.24</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row8_col8" class="data row8 col8" >0</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row8_col9" class="data row8 col9" >0.17</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row8_col10" class="data row8 col10" >0.26</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row8_col11" class="data row8 col11" >0.19</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row8_col12" class="data row8 col12" >0.26</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row8_col13" class="data row8 col13" >0.18</td> 
    </tr>    <tr> 
        <th id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4level0_row9" class="row_heading level0 row9" >nl</th> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row9_col0" class="data row9 col0" >0.31</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row9_col1" class="data row9 col1" >0.45</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row9_col2" class="data row9 col2" >0.13</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row9_col3" class="data row9 col3" >0.14</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row9_col4" class="data row9 col4" >0.21</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row9_col5" class="data row9 col5" >0.12</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row9_col6" class="data row9 col6" >0.36</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row9_col7" class="data row9 col7" >0.29</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row9_col8" class="data row9 col8" >0.17</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row9_col9" class="data row9 col9" >0</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row9_col10" class="data row9 col10" >0.24</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row9_col11" class="data row9 col11" >0.22</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row9_col12" class="data row9 col12" >0.27</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row9_col13" class="data row9 col13" >0.24</td> 
    </tr>    <tr> 
        <th id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4level0_row10" class="row_heading level0 row10" >ro</th> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row10_col0" class="data row10 col0" >0.22</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row10_col1" class="data row10 col1" >0.25</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row10_col2" class="data row10 col2" >0.14</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row10_col3" class="data row10 col3" >0.26</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row10_col4" class="data row10 col4" >0.13</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row10_col5" class="data row10 col5" >0.18</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row10_col6" class="data row10 col6" >0.14</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row10_col7" class="data row10 col7" >0.21</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row10_col8" class="data row10 col8" >0.26</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row10_col9" class="data row10 col9" >0.24</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row10_col10" class="data row10 col10" >0</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row10_col11" class="data row10 col11" >0.15</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row10_col12" class="data row10 col12" >0.11</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row10_col13" class="data row10 col13" >0.21</td> 
    </tr>    <tr> 
        <th id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4level0_row11" class="row_heading level0 row11" >ru</th> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row11_col0" class="data row11 col0" >0.15</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row11_col1" class="data row11 col1" >0.32</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row11_col2" class="data row11 col2" >0.16</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row11_col3" class="data row11 col3" >0.17</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row11_col4" class="data row11 col4" >0.22</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row11_col5" class="data row11 col5" >0.17</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row11_col6" class="data row11 col6" >0.26</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row11_col7" class="data row11 col7" >0.2</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row11_col8" class="data row11 col8" >0.19</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row11_col9" class="data row11 col9" >0.22</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row11_col10" class="data row11 col10" >0.15</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row11_col11" class="data row11 col11" >0</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row11_col12" class="data row11 col12" >0.16</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row11_col13" class="data row11 col13" >0.15</td> 
    </tr>    <tr> 
        <th id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4level0_row12" class="row_heading level0 row12" >uk</th> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row12_col0" class="data row12 col0" >0.18</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row12_col1" class="data row12 col1" >0.19</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row12_col2" class="data row12 col2" >0.19</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row12_col3" class="data row12 col3" >0.27</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row12_col4" class="data row12 col4" >0.19</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row12_col5" class="data row12 col5" >0.22</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row12_col6" class="data row12 col6" >0.17</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row12_col7" class="data row12 col7" >0.16</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row12_col8" class="data row12 col8" >0.26</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row12_col9" class="data row12 col9" >0.27</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row12_col10" class="data row12 col10" >0.11</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row12_col11" class="data row12 col11" >0.16</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row12_col12" class="data row12 col12" >0</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row12_col13" class="data row12 col13" >0.2</td> 
    </tr>    <tr> 
        <th id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4level0_row13" class="row_heading level0 row13" >zh</th> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row13_col0" class="data row13 col0" >0.13</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row13_col1" class="data row13 col1" >0.32</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row13_col2" class="data row13 col2" >0.19</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row13_col3" class="data row13 col3" >0.17</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row13_col4" class="data row13 col4" >0.21</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row13_col5" class="data row13 col5" >0.17</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row13_col6" class="data row13 col6" >0.32</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row13_col7" class="data row13 col7" >0.17</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row13_col8" class="data row13 col8" >0.18</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row13_col9" class="data row13 col9" >0.24</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row13_col10" class="data row13 col10" >0.21</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row13_col11" class="data row13 col11" >0.15</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row13_col12" class="data row13 col12" >0.2</td> 
        <td id="T_7aec21c2_5ab1_11e8_abac_1866dafa0bc4row13_col13" class="data row13 col13" >0</td> 
    </tr></tbody> 
</table style="display:inline"> <style  type="text/css" >
    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4 th {
          font-size: 7pt;
    }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row0_col0 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row0_col1 {
            font-size:  1pt;
            background-color:  #a9bfdc;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row0_col2 {
            font-size:  1pt;
            background-color:  #99b8d8;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row0_col3 {
            font-size:  1pt;
            background-color:  #cccfe5;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row0_col4 {
            font-size:  1pt;
            background-color:  #7dacd1;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row0_col5 {
            font-size:  1pt;
            background-color:  #a8bedc;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row0_col6 {
            font-size:  1pt;
            background-color:  #9ebad9;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row0_col7 {
            font-size:  1pt;
            background-color:  #bbc7e0;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row0_col8 {
            font-size:  1pt;
            background-color:  #dad9ea;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row0_col9 {
            font-size:  1pt;
            background-color:  #9ebad9;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row0_col10 {
            font-size:  1pt;
            background-color:  #88b1d4;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row0_col11 {
            font-size:  1pt;
            background-color:  #d3d4e7;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row0_col12 {
            font-size:  1pt;
            background-color:  #bfc9e1;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row0_col13 {
            font-size:  1pt;
            background-color:  #d8d7e9;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row1_col0 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row1_col1 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row1_col2 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row1_col3 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row1_col4 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row1_col5 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row1_col6 {
            font-size:  1pt;
            background-color:  #b3c3de;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row1_col7 {
            font-size:  1pt;
            background-color:  #83afd3;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row1_col8 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row1_col9 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row1_col10 {
            font-size:  1pt;
            background-color:  #81aed2;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row1_col11 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row1_col12 {
            font-size:  1pt;
            background-color:  #abbfdc;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row1_col13 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row2_col0 {
            font-size:  1pt;
            background-color:  #8cb3d5;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row2_col1 {
            font-size:  1pt;
            background-color:  #9ebad9;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row2_col2 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row2_col3 {
            font-size:  1pt;
            background-color:  #e2dfee;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row2_col4 {
            font-size:  1pt;
            background-color:  #e1dfed;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row2_col5 {
            font-size:  1pt;
            background-color:  #ebe6f2;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row2_col6 {
            font-size:  1pt;
            background-color:  #acc0dd;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row2_col7 {
            font-size:  1pt;
            background-color:  #80aed2;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row2_col8 {
            font-size:  1pt;
            background-color:  #d7d6e9;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row2_col9 {
            font-size:  1pt;
            background-color:  #e3e0ee;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row2_col10 {
            font-size:  1pt;
            background-color:  #c9cee4;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row2_col11 {
            font-size:  1pt;
            background-color:  #d1d2e6;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row2_col12 {
            font-size:  1pt;
            background-color:  #b0c2de;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row2_col13 {
            font-size:  1pt;
            background-color:  #b9c6e0;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row3_col0 {
            font-size:  1pt;
            background-color:  #9ab8d8;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row3_col1 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row3_col2 {
            font-size:  1pt;
            background-color:  #d2d3e7;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row3_col3 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row3_col4 {
            font-size:  1pt;
            background-color:  #a1bbda;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row3_col5 {
            font-size:  1pt;
            background-color:  #e0dded;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row3_col6 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row3_col7 {
            font-size:  1pt;
            background-color:  #75a9cf;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row3_col8 {
            font-size:  1pt;
            background-color:  #ede7f2;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row3_col9 {
            font-size:  1pt;
            background-color:  #dfddec;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row3_col10 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row3_col11 {
            font-size:  1pt;
            background-color:  #d3d4e7;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row3_col12 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row3_col13 {
            font-size:  1pt;
            background-color:  #c2cbe2;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row4_col0 {
            font-size:  1pt;
            background-color:  #7dacd1;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row4_col1 {
            font-size:  1pt;
            background-color:  #a9bfdc;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row4_col2 {
            font-size:  1pt;
            background-color:  #e4e1ef;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row4_col3 {
            font-size:  1pt;
            background-color:  #ced0e6;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row4_col4 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row4_col5 {
            font-size:  1pt;
            background-color:  #d0d1e6;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row4_col6 {
            font-size:  1pt;
            background-color:  #b4c4df;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row4_col7 {
            font-size:  1pt;
            background-color:  #71a8ce;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row4_col8 {
            font-size:  1pt;
            background-color:  #c4cbe3;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row4_col9 {
            font-size:  1pt;
            background-color:  #cccfe5;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row4_col10 {
            font-size:  1pt;
            background-color:  #c9cee4;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row4_col11 {
            font-size:  1pt;
            background-color:  #c2cbe2;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row4_col12 {
            font-size:  1pt;
            background-color:  #a5bddb;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row4_col13 {
            font-size:  1pt;
            background-color:  #b0c2de;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row5_col0 {
            font-size:  1pt;
            background-color:  #8eb3d5;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row5_col1 {
            font-size:  1pt;
            background-color:  #8fb4d6;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row5_col2 {
            font-size:  1pt;
            background-color:  #e8e4f0;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row5_col3 {
            font-size:  1pt;
            background-color:  #e9e5f1;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row5_col4 {
            font-size:  1pt;
            background-color:  #c1cae2;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row5_col5 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row5_col6 {
            font-size:  1pt;
            background-color:  #89b1d4;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row5_col7 {
            font-size:  1pt;
            background-color:  #a9bfdc;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row5_col8 {
            font-size:  1pt;
            background-color:  #dbdaeb;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row5_col9 {
            font-size:  1pt;
            background-color:  #e5e1ef;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row5_col10 {
            font-size:  1pt;
            background-color:  #a1bbda;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row5_col11 {
            font-size:  1pt;
            background-color:  #c6cce3;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row5_col12 {
            font-size:  1pt;
            background-color:  #8fb4d6;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row5_col13 {
            font-size:  1pt;
            background-color:  #bdc8e1;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row6_col0 {
            font-size:  1pt;
            background-color:  #80aed2;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row6_col1 {
            font-size:  1pt;
            background-color:  #cccfe5;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row6_col2 {
            font-size:  1pt;
            background-color:  #9fbad9;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row6_col3 {
            font-size:  1pt;
            background-color:  #8eb3d5;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row6_col4 {
            font-size:  1pt;
            background-color:  #9ebad9;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row6_col5 {
            font-size:  1pt;
            background-color:  #88b1d4;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row6_col6 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row6_col7 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row6_col8 {
            font-size:  1pt;
            background-color:  #91b5d6;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row6_col9 {
            font-size:  1pt;
            background-color:  #94b6d7;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row6_col10 {
            font-size:  1pt;
            background-color:  #d5d5e8;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row6_col11 {
            font-size:  1pt;
            background-color:  #a5bddb;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row6_col12 {
            font-size:  1pt;
            background-color:  #c6cce3;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row6_col13 {
            font-size:  1pt;
            background-color:  #93b5d6;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row7_col0 {
            font-size:  1pt;
            background-color:  #c2cbe2;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row7_col1 {
            font-size:  1pt;
            background-color:  #c6cce3;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row7_col2 {
            font-size:  1pt;
            background-color:  #9cb9d9;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row7_col3 {
            font-size:  1pt;
            background-color:  #c0c9e2;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row7_col4 {
            font-size:  1pt;
            background-color:  #83afd3;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row7_col5 {
            font-size:  1pt;
            background-color:  #c5cce3;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row7_col6 {
            font-size:  1pt;
            background-color:  #91b5d6;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row7_col7 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row7_col8 {
            font-size:  1pt;
            background-color:  #c1cae2;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row7_col9 {
            font-size:  1pt;
            background-color:  #abbfdc;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row7_col10 {
            font-size:  1pt;
            background-color:  #88b1d4;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row7_col11 {
            font-size:  1pt;
            background-color:  #c2cbe2;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row7_col12 {
            font-size:  1pt;
            background-color:  #bfc9e1;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row7_col13 {
            font-size:  1pt;
            background-color:  #d3d4e7;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row8_col0 {
            font-size:  1pt;
            background-color:  #c2cbe2;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row8_col1 {
            font-size:  1pt;
            background-color:  #6da6cd;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row8_col2 {
            font-size:  1pt;
            background-color:  #c4cbe3;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row8_col3 {
            font-size:  1pt;
            background-color:  #eee8f3;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row8_col4 {
            font-size:  1pt;
            background-color:  #9ab8d8;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row8_col5 {
            font-size:  1pt;
            background-color:  #d2d3e7;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row8_col6 {
            font-size:  1pt;
            background-color:  #71a8ce;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row8_col7 {
            font-size:  1pt;
            background-color:  #86b0d3;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row8_col8 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row8_col9 {
            font-size:  1pt;
            background-color:  #d3d4e7;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row8_col10 {
            font-size:  1pt;
            background-color:  #73a9cf;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row8_col11 {
            font-size:  1pt;
            background-color:  #d2d2e7;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row8_col12 {
            font-size:  1pt;
            background-color:  #73a9cf;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row8_col13 {
            font-size:  1pt;
            background-color:  #c0c9e2;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row9_col0 {
            font-size:  1pt;
            background-color:  #5c9fc9;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row9_col1 {
            font-size:  1pt;
            background-color:  #71a8ce;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row9_col2 {
            font-size:  1pt;
            background-color:  #d9d8ea;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row9_col3 {
            font-size:  1pt;
            background-color:  #e2dfee;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row9_col4 {
            font-size:  1pt;
            background-color:  #a9bfdc;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row9_col5 {
            font-size:  1pt;
            background-color:  #dfddec;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row9_col6 {
            font-size:  1pt;
            background-color:  #79abd0;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row9_col7 {
            font-size:  1pt;
            background-color:  #5c9fc9;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row9_col8 {
            font-size:  1pt;
            background-color:  #d4d4e8;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row9_col9 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row9_col10 {
            font-size:  1pt;
            background-color:  #84b0d3;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row9_col11 {
            font-size:  1pt;
            background-color:  #c0c9e2;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row9_col12 {
            font-size:  1pt;
            background-color:  #7dacd1;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row9_col13 {
            font-size:  1pt;
            background-color:  #a1bbda;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row10_col0 {
            font-size:  1pt;
            background-color:  #a1bbda;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row10_col1 {
            font-size:  1pt;
            background-color:  #cccfe5;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row10_col2 {
            font-size:  1pt;
            background-color:  #d8d7e9;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row10_col3 {
            font-size:  1pt;
            background-color:  #bbc7e0;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row10_col4 {
            font-size:  1pt;
            background-color:  #d3d4e7;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row10_col5 {
            font-size:  1pt;
            background-color:  #c5cce3;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row10_col6 {
            font-size:  1pt;
            background-color:  #e3e0ee;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row10_col7 {
            font-size:  1pt;
            background-color:  #93b5d6;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row10_col8 {
            font-size:  1pt;
            background-color:  #bdc8e1;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row10_col9 {
            font-size:  1pt;
            background-color:  #c4cbe3;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row10_col10 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row10_col11 {
            font-size:  1pt;
            background-color:  #d7d6e9;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row10_col12 {
            font-size:  1pt;
            background-color:  #e2dfee;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row10_col13 {
            font-size:  1pt;
            background-color:  #bdc8e1;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row11_col0 {
            font-size:  1pt;
            background-color:  #cdd0e5;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row11_col1 {
            font-size:  1pt;
            background-color:  #9cb9d9;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row11_col2 {
            font-size:  1pt;
            background-color:  #d0d1e6;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row11_col3 {
            font-size:  1pt;
            background-color:  #e2dfee;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row11_col4 {
            font-size:  1pt;
            background-color:  #b9c6e0;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row11_col5 {
            font-size:  1pt;
            background-color:  #cdd0e5;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row11_col6 {
            font-size:  1pt;
            background-color:  #b0c2de;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row11_col7 {
            font-size:  1pt;
            background-color:  #afc1dd;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row11_col8 {
            font-size:  1pt;
            background-color:  #dedcec;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row11_col9 {
            font-size:  1pt;
            background-color:  #d3d4e7;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row11_col10 {
            font-size:  1pt;
            background-color:  #c5cce3;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row11_col11 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row11_col12 {
            font-size:  1pt;
            background-color:  #bfc9e1;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row11_col13 {
            font-size:  1pt;
            background-color:  #ced0e6;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row12_col0 {
            font-size:  1pt;
            background-color:  #c1cae2;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row12_col1 {
            font-size:  1pt;
            background-color:  #d3d4e7;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row12_col2 {
            font-size:  1pt;
            background-color:  #b9c6e0;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row12_col3 {
            font-size:  1pt;
            background-color:  #abbfdc;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row12_col4 {
            font-size:  1pt;
            background-color:  #a7bddb;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row12_col5 {
            font-size:  1pt;
            background-color:  #abbfdc;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row12_col6 {
            font-size:  1pt;
            background-color:  #d5d5e8;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row12_col7 {
            font-size:  1pt;
            background-color:  #b8c6e0;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row12_col8 {
            font-size:  1pt;
            background-color:  #afc1dd;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row12_col9 {
            font-size:  1pt;
            background-color:  #b1c2de;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row12_col10 {
            font-size:  1pt;
            background-color:  #dddbec;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row12_col11 {
            font-size:  1pt;
            background-color:  #c9cee4;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row12_col12 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row12_col13 {
            font-size:  1pt;
            background-color:  #c5cce3;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row13_col0 {
            font-size:  1pt;
            background-color:  #d3d4e7;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row13_col1 {
            font-size:  1pt;
            background-color:  #9cb9d9;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row13_col2 {
            font-size:  1pt;
            background-color:  #b8c6e0;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row13_col3 {
            font-size:  1pt;
            background-color:  #d9d8ea;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row13_col4 {
            font-size:  1pt;
            background-color:  #a4bcda;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row13_col5 {
            font-size:  1pt;
            background-color:  #c5cce3;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row13_col6 {
            font-size:  1pt;
            background-color:  #a1bbda;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row13_col7 {
            font-size:  1pt;
            background-color:  #c5cce3;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row13_col8 {
            font-size:  1pt;
            background-color:  #d4d4e8;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row13_col9 {
            font-size:  1pt;
            background-color:  #bdc8e1;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row13_col10 {
            font-size:  1pt;
            background-color:  #a1bbda;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row13_col11 {
            font-size:  1pt;
            background-color:  #ced0e6;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row13_col12 {
            font-size:  1pt;
            background-color:  #bbc7e0;
        }    #T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row13_col13 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }</style>  
<table style="display:inline" id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4" > 
<thead>    <tr> 
        <th class="index_name level0" ></th> 
        <th class="col_heading level0 col0" >ar</th> 
        <th class="col_heading level0 col1" >bn</th> 
        <th class="col_heading level0 col2" >de</th> 
        <th class="col_heading level0 col3" >en</th> 
        <th class="col_heading level0 col4" >es</th> 
        <th class="col_heading level0 col5" >he</th> 
        <th class="col_heading level0 col6" >hi</th> 
        <th class="col_heading level0 col7" >hu</th> 
        <th class="col_heading level0 col8" >ja</th> 
        <th class="col_heading level0 col9" >nl</th> 
        <th class="col_heading level0 col10" >ro</th> 
        <th class="col_heading level0 col11" >ru</th> 
        <th class="col_heading level0 col12" >uk</th> 
        <th class="col_heading level0 col13" >zh</th> 
    </tr>    <tr> 
        <th class="index_name level0" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
    </tr></thead> 
<tbody>    <tr> 
        <th id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4level0_row0" class="row_heading level0 row0" >ar</th> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row0_col0" class="data row0 col0" >0</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row0_col1" class="data row0 col1" >0.73</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row0_col2" class="data row0 col2" >0.58</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row0_col3" class="data row0 col3" >0.53</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row0_col4" class="data row0 col4" >0.63</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row0_col5" class="data row0 col5" >0.57</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row0_col6" class="data row0 col6" >0.62</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row0_col7" class="data row0 col7" >0.38</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row0_col8" class="data row0 col8" >0.38</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row0_col9" class="data row0 col9" >0.72</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row0_col10" class="data row0 col10" >0.51</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row0_col11" class="data row0 col11" >0.34</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row0_col12" class="data row0 col12" >0.39</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row0_col13" class="data row0 col13" >0.31</td> 
    </tr>    <tr> 
        <th id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4level0_row1" class="row_heading level0 row1" >bn</th> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row1_col0" class="data row1 col0" >0.73</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row1_col1" class="data row1 col1" >0</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row1_col2" class="data row1 col2" >0.79</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row1_col3" class="data row1 col3" >1.11</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row1_col4" class="data row1 col4" >0.73</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row1_col5" class="data row1 col5" >0.86</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row1_col6" class="data row1 col6" >0.53</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row1_col7" class="data row1 col7" >0.56</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row1_col8" class="data row1 col8" >1.03</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row1_col9" class="data row1 col9" >1.01</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row1_col10" class="data row1 col10" >0.53</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row1_col11" class="data row1 col11" >0.8</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row1_col12" class="data row1 col12" >0.47</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row1_col13" class="data row1 col13" >0.8</td> 
    </tr>    <tr> 
        <th id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4level0_row2" class="row_heading level0 row2" >de</th> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row2_col0" class="data row2 col0" >0.58</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row2_col1" class="data row2 col1" >0.79</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row2_col2" class="data row2 col2" >0</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row2_col3" class="data row2 col3" >0.34</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row2_col4" class="data row2 col4" >0.23</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row2_col5" class="data row2 col5" >0.2</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row2_col6" class="data row2 col6" >0.56</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row2_col7" class="data row2 col7" >0.57</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row2_col8" class="data row2 col8" >0.41</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row2_col9" class="data row2 col9" >0.3</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row2_col10" class="data row2 col10" >0.31</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row2_col11" class="data row2 col11" >0.36</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row2_col12" class="data row2 col12" >0.45</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row2_col13" class="data row2 col13" >0.46</td> 
    </tr>    <tr> 
        <th id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4level0_row3" class="row_heading level0 row3" >en</th> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row3_col0" class="data row3 col0" >0.53</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row3_col1" class="data row3 col1" >1.11</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row3_col2" class="data row3 col2" >0.34</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row3_col3" class="data row3 col3" >0</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row3_col4" class="data row3 col4" >0.51</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row3_col5" class="data row3 col5" >0.28</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row3_col6" class="data row3 col6" >0.87</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row3_col7" class="data row3 col7" >0.6</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row3_col8" class="data row3 col8" >0.23</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row3_col9" class="data row3 col9" >0.34</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row3_col10" class="data row3 col10" >0.63</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row3_col11" class="data row3 col11" >0.34</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row3_col12" class="data row3 col12" >0.72</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row3_col13" class="data row3 col13" >0.42</td> 
    </tr>    <tr> 
        <th id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4level0_row4" class="row_heading level0 row4" >es</th> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row4_col0" class="data row4 col0" >0.63</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row4_col1" class="data row4 col1" >0.73</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row4_col2" class="data row4 col2" >0.23</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row4_col3" class="data row4 col3" >0.51</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row4_col4" class="data row4 col4" >0</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row4_col5" class="data row4 col5" >0.39</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row4_col6" class="data row4 col6" >0.52</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row4_col7" class="data row4 col7" >0.61</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row4_col8" class="data row4 col8" >0.53</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row4_col9" class="data row4 col9" >0.48</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row4_col10" class="data row4 col10" >0.31</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row4_col11" class="data row4 col11" >0.42</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row4_col12" class="data row4 col12" >0.49</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row4_col13" class="data row4 col13" >0.5</td> 
    </tr>    <tr> 
        <th id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4level0_row5" class="row_heading level0 row5" >he</th> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row5_col0" class="data row5 col0" >0.57</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row5_col1" class="data row5 col1" >0.86</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row5_col2" class="data row5 col2" >0.2</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row5_col3" class="data row5 col3" >0.28</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row5_col4" class="data row5 col4" >0.39</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row5_col5" class="data row5 col5" >0</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row5_col6" class="data row5 col6" >0.7</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row5_col7" class="data row5 col7" >0.44</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row5_col8" class="data row5 col8" >0.37</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row5_col9" class="data row5 col9" >0.29</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row5_col10" class="data row5 col10" >0.44</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row5_col11" class="data row5 col11" >0.4</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row5_col12" class="data row5 col12" >0.56</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row5_col13" class="data row5 col13" >0.44</td> 
    </tr>    <tr> 
        <th id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4level0_row6" class="row_heading level0 row6" >hi</th> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row6_col0" class="data row6 col0" >0.62</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row6_col1" class="data row6 col1" >0.53</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row6_col2" class="data row6 col2" >0.56</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row6_col3" class="data row6 col3" >0.87</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row6_col4" class="data row6 col4" >0.52</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row6_col5" class="data row6 col5" >0.7</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row6_col6" class="data row6 col6" >0</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row6_col7" class="data row6 col7" >0.67</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row6_col8" class="data row6 col8" >0.79</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row6_col9" class="data row6 col9" >0.76</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row6_col10" class="data row6 col10" >0.26</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row6_col11" class="data row6 col11" >0.54</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row6_col12" class="data row6 col12" >0.36</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row6_col13" class="data row6 col13" >0.61</td> 
    </tr>    <tr> 
        <th id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4level0_row7" class="row_heading level0 row7" >hu</th> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row7_col0" class="data row7 col0" >0.38</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row7_col1" class="data row7 col1" >0.56</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row7_col2" class="data row7 col2" >0.57</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row7_col3" class="data row7 col3" >0.6</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row7_col4" class="data row7 col4" >0.61</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row7_col5" class="data row7 col5" >0.44</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row7_col6" class="data row7 col6" >0.67</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row7_col7" class="data row7 col7" >0</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row7_col8" class="data row7 col8" >0.55</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row7_col9" class="data row7 col9" >0.66</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row7_col10" class="data row7 col10" >0.51</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row7_col11" class="data row7 col11" >0.42</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row7_col12" class="data row7 col12" >0.39</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row7_col13" class="data row7 col13" >0.34</td> 
    </tr>    <tr> 
        <th id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4level0_row8" class="row_heading level0 row8" >ja</th> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row8_col0" class="data row8 col0" >0.38</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row8_col1" class="data row8 col1" >1.03</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row8_col2" class="data row8 col2" >0.41</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row8_col3" class="data row8 col3" >0.23</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row8_col4" class="data row8 col4" >0.53</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row8_col5" class="data row8 col5" >0.37</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row8_col6" class="data row8 col6" >0.79</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row8_col7" class="data row8 col7" >0.55</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row8_col8" class="data row8 col8" >0</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row8_col9" class="data row8 col9" >0.43</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row8_col10" class="data row8 col10" >0.57</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row8_col11" class="data row8 col11" >0.35</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row8_col12" class="data row8 col12" >0.65</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row8_col13" class="data row8 col13" >0.43</td> 
    </tr>    <tr> 
        <th id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4level0_row9" class="row_heading level0 row9" >nl</th> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row9_col0" class="data row9 col0" >0.72</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row9_col1" class="data row9 col1" >1.01</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row9_col2" class="data row9 col2" >0.3</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row9_col3" class="data row9 col3" >0.34</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row9_col4" class="data row9 col4" >0.48</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row9_col5" class="data row9 col5" >0.29</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row9_col6" class="data row9 col6" >0.76</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row9_col7" class="data row9 col7" >0.66</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row9_col8" class="data row9 col8" >0.43</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row9_col9" class="data row9 col9" >0</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row9_col10" class="data row9 col10" >0.52</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row9_col11" class="data row9 col11" >0.43</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row9_col12" class="data row9 col12" >0.62</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row9_col13" class="data row9 col13" >0.56</td> 
    </tr>    <tr> 
        <th id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4level0_row10" class="row_heading level0 row10" >ro</th> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row10_col0" class="data row10 col0" >0.51</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row10_col1" class="data row10 col1" >0.53</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row10_col2" class="data row10 col2" >0.31</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row10_col3" class="data row10 col3" >0.63</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row10_col4" class="data row10 col4" >0.31</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row10_col5" class="data row10 col5" >0.44</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row10_col6" class="data row10 col6" >0.26</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row10_col7" class="data row10 col7" >0.51</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row10_col8" class="data row10 col8" >0.57</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row10_col9" class="data row10 col9" >0.52</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row10_col10" class="data row10 col10" >0</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row10_col11" class="data row10 col11" >0.32</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row10_col12" class="data row10 col12" >0.22</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row10_col13" class="data row10 col13" >0.44</td> 
    </tr>    <tr> 
        <th id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4level0_row11" class="row_heading level0 row11" >ru</th> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row11_col0" class="data row11 col0" >0.34</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row11_col1" class="data row11 col1" >0.8</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row11_col2" class="data row11 col2" >0.36</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row11_col3" class="data row11 col3" >0.34</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row11_col4" class="data row11 col4" >0.42</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row11_col5" class="data row11 col5" >0.4</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row11_col6" class="data row11 col6" >0.54</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row11_col7" class="data row11 col7" >0.42</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row11_col8" class="data row11 col8" >0.35</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row11_col9" class="data row11 col9" >0.43</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row11_col10" class="data row11 col10" >0.32</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row11_col11" class="data row11 col11" >0</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row11_col12" class="data row11 col12" >0.39</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row11_col13" class="data row11 col13" >0.37</td> 
    </tr>    <tr> 
        <th id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4level0_row12" class="row_heading level0 row12" >uk</th> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row12_col0" class="data row12 col0" >0.39</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row12_col1" class="data row12 col1" >0.47</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row12_col2" class="data row12 col2" >0.45</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row12_col3" class="data row12 col3" >0.72</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row12_col4" class="data row12 col4" >0.49</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row12_col5" class="data row12 col5" >0.56</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row12_col6" class="data row12 col6" >0.36</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row12_col7" class="data row12 col7" >0.39</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row12_col8" class="data row12 col8" >0.65</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row12_col9" class="data row12 col9" >0.62</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row12_col10" class="data row12 col10" >0.22</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row12_col11" class="data row12 col11" >0.39</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row12_col12" class="data row12 col12" >0</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row12_col13" class="data row12 col13" >0.41</td> 
    </tr>    <tr> 
        <th id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4level0_row13" class="row_heading level0 row13" >zh</th> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row13_col0" class="data row13 col0" >0.31</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row13_col1" class="data row13 col1" >0.8</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row13_col2" class="data row13 col2" >0.46</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row13_col3" class="data row13 col3" >0.42</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row13_col4" class="data row13 col4" >0.5</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row13_col5" class="data row13 col5" >0.44</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row13_col6" class="data row13 col6" >0.61</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row13_col7" class="data row13 col7" >0.34</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row13_col8" class="data row13 col8" >0.43</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row13_col9" class="data row13 col9" >0.56</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row13_col10" class="data row13 col10" >0.44</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row13_col11" class="data row13 col11" >0.37</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row13_col12" class="data row13 col12" >0.41</td> 
        <td id="T_7afba9e4_5ab1_11e8_abac_1866dafa0bc4row13_col13" class="data row13 col13" >0</td> 
    </tr></tbody> 
</table style="display:inline"> 



<h2>familiarty (cosine,euclidean,l1 )</h2><style  type="text/css" >
    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4 th {
          font-size: 7pt;
    }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row0_col0 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row0_col1 {
            font-size:  1pt;
            background-color:  #f7f0f7;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row0_col2 {
            font-size:  1pt;
            background-color:  #f3edf5;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row0_col3 {
            font-size:  1pt;
            background-color:  #f3edf5;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row0_col4 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row0_col5 {
            font-size:  1pt;
            background-color:  #dfddec;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row0_col6 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row0_col7 {
            font-size:  1pt;
            background-color:  #c2cbe2;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row0_col8 {
            font-size:  1pt;
            background-color:  #f1ebf5;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row0_col9 {
            font-size:  1pt;
            background-color:  #cdd0e5;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row0_col10 {
            font-size:  1pt;
            background-color:  #dfddec;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row0_col11 {
            font-size:  1pt;
            background-color:  #f1ebf5;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row0_col12 {
            font-size:  1pt;
            background-color:  #b9c6e0;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row0_col13 {
            font-size:  1pt;
            background-color:  #fbf3f9;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row1_col0 {
            font-size:  1pt;
            background-color:  #f1ebf5;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row1_col1 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row1_col2 {
            font-size:  1pt;
            background-color:  #b9c6e0;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row1_col3 {
            font-size:  1pt;
            background-color:  #b9c6e0;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row1_col4 {
            font-size:  1pt;
            background-color:  #d9d8ea;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row1_col5 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row1_col6 {
            font-size:  1pt;
            background-color:  #f9f2f8;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row1_col7 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row1_col8 {
            font-size:  1pt;
            background-color:  #83afd3;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row1_col9 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row1_col10 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row1_col11 {
            font-size:  1pt;
            background-color:  #83afd3;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row1_col12 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row1_col13 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row2_col0 {
            font-size:  1pt;
            background-color:  #f8f1f8;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row2_col1 {
            font-size:  1pt;
            background-color:  #efe9f3;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row2_col2 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row2_col3 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row2_col4 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row2_col5 {
            font-size:  1pt;
            background-color:  #f6eff7;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row2_col6 {
            font-size:  1pt;
            background-color:  #f9f2f8;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row2_col7 {
            font-size:  1pt;
            background-color:  #dfddec;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row2_col8 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row2_col9 {
            font-size:  1pt;
            background-color:  #e7e3f0;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row2_col10 {
            font-size:  1pt;
            background-color:  #f6eff7;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row2_col11 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row2_col12 {
            font-size:  1pt;
            background-color:  #dfddec;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row2_col13 {
            font-size:  1pt;
            background-color:  #f2ecf5;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row3_col0 {
            font-size:  1pt;
            background-color:  #f8f1f8;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row3_col1 {
            font-size:  1pt;
            background-color:  #efe9f3;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row3_col2 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row3_col3 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row3_col4 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row3_col5 {
            font-size:  1pt;
            background-color:  #f6eff7;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row3_col6 {
            font-size:  1pt;
            background-color:  #f9f2f8;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row3_col7 {
            font-size:  1pt;
            background-color:  #dfddec;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row3_col8 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row3_col9 {
            font-size:  1pt;
            background-color:  #e7e3f0;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row3_col10 {
            font-size:  1pt;
            background-color:  #f6eff7;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row3_col11 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row3_col12 {
            font-size:  1pt;
            background-color:  #dfddec;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row3_col13 {
            font-size:  1pt;
            background-color:  #f2ecf5;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row4_col0 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row4_col1 {
            font-size:  1pt;
            background-color:  #f3edf5;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row4_col2 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row4_col3 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row4_col4 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row4_col5 {
            font-size:  1pt;
            background-color:  #f6eff7;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row4_col6 {
            font-size:  1pt;
            background-color:  #f9f2f8;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row4_col7 {
            font-size:  1pt;
            background-color:  #d9d8ea;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row4_col8 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row4_col9 {
            font-size:  1pt;
            background-color:  #e7e3f0;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row4_col10 {
            font-size:  1pt;
            background-color:  #f6eff7;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row4_col11 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row4_col12 {
            font-size:  1pt;
            background-color:  #d9d8ea;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row4_col13 {
            font-size:  1pt;
            background-color:  #f7f0f7;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row5_col0 {
            font-size:  1pt;
            background-color:  #e9e5f1;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row5_col1 {
            font-size:  1pt;
            background-color:  #d3d4e7;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row5_col2 {
            font-size:  1pt;
            background-color:  #f3edf5;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row5_col3 {
            font-size:  1pt;
            background-color:  #f3edf5;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row5_col4 {
            font-size:  1pt;
            background-color:  #f5eef6;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row5_col5 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row5_col6 {
            font-size:  1pt;
            background-color:  #e5e1ef;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row5_col7 {
            font-size:  1pt;
            background-color:  #f3edf5;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row5_col8 {
            font-size:  1pt;
            background-color:  #f1ebf5;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row5_col9 {
            font-size:  1pt;
            background-color:  #faf2f8;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row5_col10 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row5_col11 {
            font-size:  1pt;
            background-color:  #f1ebf5;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row5_col12 {
            font-size:  1pt;
            background-color:  #f3edf5;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row5_col13 {
            font-size:  1pt;
            background-color:  #dad9ea;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row6_col0 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row6_col1 {
            font-size:  1pt;
            background-color:  #fbf4f9;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row6_col2 {
            font-size:  1pt;
            background-color:  #f3edf5;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row6_col3 {
            font-size:  1pt;
            background-color:  #f3edf5;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row6_col4 {
            font-size:  1pt;
            background-color:  #f5eef6;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row6_col5 {
            font-size:  1pt;
            background-color:  #d1d2e6;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row6_col6 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row6_col7 {
            font-size:  1pt;
            background-color:  #b0c2de;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row6_col8 {
            font-size:  1pt;
            background-color:  #f1ebf5;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row6_col9 {
            font-size:  1pt;
            background-color:  #b4c4df;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row6_col10 {
            font-size:  1pt;
            background-color:  #d1d2e6;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row6_col11 {
            font-size:  1pt;
            background-color:  #dfddec;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row6_col12 {
            font-size:  1pt;
            background-color:  #a8bedc;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row6_col13 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row7_col0 {
            font-size:  1pt;
            background-color:  #6fa7ce;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row7_col1 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row7_col2 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row7_col3 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row7_col4 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row7_col5 {
            font-size:  1pt;
            background-color:  #dfddec;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row7_col6 {
            font-size:  1pt;
            background-color:  #6ba5cd;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row7_col7 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row7_col8 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row7_col9 {
            font-size:  1pt;
            background-color:  #faf2f8;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row7_col10 {
            font-size:  1pt;
            background-color:  #dfddec;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row7_col11 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row7_col12 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row7_col13 {
            font-size:  1pt;
            background-color:  #67a4cc;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row8_col0 {
            font-size:  1pt;
            background-color:  #f8f1f8;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row8_col1 {
            font-size:  1pt;
            background-color:  #ebe6f2;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row8_col2 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row8_col3 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row8_col4 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row8_col5 {
            font-size:  1pt;
            background-color:  #f6eff7;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row8_col6 {
            font-size:  1pt;
            background-color:  #f9f2f8;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row8_col7 {
            font-size:  1pt;
            background-color:  #e5e1ef;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row8_col8 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row8_col9 {
            font-size:  1pt;
            background-color:  #eee9f3;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row8_col10 {
            font-size:  1pt;
            background-color:  #f6eff7;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row8_col11 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row8_col12 {
            font-size:  1pt;
            background-color:  #e5e1ef;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row8_col13 {
            font-size:  1pt;
            background-color:  #f2ecf5;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row9_col0 {
            font-size:  1pt;
            background-color:  #b8c6e0;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row9_col1 {
            font-size:  1pt;
            background-color:  #9ebad9;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row9_col2 {
            font-size:  1pt;
            background-color:  #b9c6e0;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row9_col3 {
            font-size:  1pt;
            background-color:  #b9c6e0;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row9_col4 {
            font-size:  1pt;
            background-color:  #c6cce3;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row9_col5 {
            font-size:  1pt;
            background-color:  #f6eff7;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row9_col6 {
            font-size:  1pt;
            background-color:  #acc0dd;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row9_col7 {
            font-size:  1pt;
            background-color:  #fbf4f9;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row9_col8 {
            font-size:  1pt;
            background-color:  #c6cce3;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row9_col9 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row9_col10 {
            font-size:  1pt;
            background-color:  #f6eff7;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row9_col11 {
            font-size:  1pt;
            background-color:  #c6cce3;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row9_col12 {
            font-size:  1pt;
            background-color:  #fbf4f9;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row9_col13 {
            font-size:  1pt;
            background-color:  #a4bcda;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row10_col0 {
            font-size:  1pt;
            background-color:  #e9e5f1;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row10_col1 {
            font-size:  1pt;
            background-color:  #d3d4e7;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row10_col2 {
            font-size:  1pt;
            background-color:  #f3edf5;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row10_col3 {
            font-size:  1pt;
            background-color:  #f3edf5;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row10_col4 {
            font-size:  1pt;
            background-color:  #f5eef6;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row10_col5 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row10_col6 {
            font-size:  1pt;
            background-color:  #e5e1ef;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row10_col7 {
            font-size:  1pt;
            background-color:  #f3edf5;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row10_col8 {
            font-size:  1pt;
            background-color:  #f1ebf5;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row10_col9 {
            font-size:  1pt;
            background-color:  #faf2f8;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row10_col10 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row10_col11 {
            font-size:  1pt;
            background-color:  #f1ebf5;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row10_col12 {
            font-size:  1pt;
            background-color:  #f3edf5;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row10_col13 {
            font-size:  1pt;
            background-color:  #dad9ea;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row11_col0 {
            font-size:  1pt;
            background-color:  #f8f1f8;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row11_col1 {
            font-size:  1pt;
            background-color:  #ebe6f2;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row11_col2 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row11_col3 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row11_col4 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row11_col5 {
            font-size:  1pt;
            background-color:  #f6eff7;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row11_col6 {
            font-size:  1pt;
            background-color:  #f3edf5;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row11_col7 {
            font-size:  1pt;
            background-color:  #e5e1ef;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row11_col8 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row11_col9 {
            font-size:  1pt;
            background-color:  #eee9f3;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row11_col10 {
            font-size:  1pt;
            background-color:  #f6eff7;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row11_col11 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row11_col12 {
            font-size:  1pt;
            background-color:  #e5e1ef;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row11_col13 {
            font-size:  1pt;
            background-color:  #eee8f3;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row12_col0 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row12_col1 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row12_col2 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row12_col3 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row12_col4 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row12_col5 {
            font-size:  1pt;
            background-color:  #dfddec;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row12_col6 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row12_col7 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row12_col8 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row12_col9 {
            font-size:  1pt;
            background-color:  #faf2f8;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row12_col10 {
            font-size:  1pt;
            background-color:  #dfddec;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row12_col11 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row12_col12 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row12_col13 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row13_col0 {
            font-size:  1pt;
            background-color:  #f8f1f8;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row13_col1 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row13_col2 {
            font-size:  1pt;
            background-color:  #d3d4e7;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row13_col3 {
            font-size:  1pt;
            background-color:  #d3d4e7;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row13_col4 {
            font-size:  1pt;
            background-color:  #e9e5f1;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row13_col5 {
            font-size:  1pt;
            background-color:  #8fb4d6;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row13_col6 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row13_col7 {
            font-size:  1pt;
            background-color:  #7eadd1;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row13_col8 {
            font-size:  1pt;
            background-color:  #c6cce3;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row13_col9 {
            font-size:  1pt;
            background-color:  #7bacd1;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row13_col10 {
            font-size:  1pt;
            background-color:  #8fb4d6;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row13_col11 {
            font-size:  1pt;
            background-color:  #a8bedc;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row13_col12 {
            font-size:  1pt;
            background-color:  #73a9cf;
        }    #T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row13_col13 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }</style>  
<table style="display:inline" id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4" > 
<thead>    <tr> 
        <th class="index_name level0" ></th> 
        <th class="col_heading level0 col0" >ar</th> 
        <th class="col_heading level0 col1" >bn</th> 
        <th class="col_heading level0 col2" >de</th> 
        <th class="col_heading level0 col3" >en</th> 
        <th class="col_heading level0 col4" >es</th> 
        <th class="col_heading level0 col5" >he</th> 
        <th class="col_heading level0 col6" >hi</th> 
        <th class="col_heading level0 col7" >hu</th> 
        <th class="col_heading level0 col8" >ja</th> 
        <th class="col_heading level0 col9" >nl</th> 
        <th class="col_heading level0 col10" >ro</th> 
        <th class="col_heading level0 col11" >ru</th> 
        <th class="col_heading level0 col12" >uk</th> 
        <th class="col_heading level0 col13" >zh</th> 
    </tr>    <tr> 
        <th class="index_name level0" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
    </tr></thead> 
<tbody>    <tr> 
        <th id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4level0_row0" class="row_heading level0 row0" >ar</th> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row0_col0" class="data row0 col0" >0</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row0_col1" class="data row0 col1" >0.02</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row0_col2" class="data row0 col2" >0.01</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row0_col3" class="data row0 col3" >0.01</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row0_col4" class="data row0 col4" >0</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row0_col5" class="data row0 col5" >0.03</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row0_col6" class="data row0 col6" >0</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row0_col7" class="data row0 col7" >0.11</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row0_col8" class="data row0 col8" >0.01</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row0_col9" class="data row0 col9" >0.07</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row0_col10" class="data row0 col10" >0.03</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row0_col11" class="data row0 col11" >0.01</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row0_col12" class="data row0 col12" >0.12</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row0_col13" class="data row0 col13" >0.01</td> 
    </tr>    <tr> 
        <th id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4level0_row1" class="row_heading level0 row1" >bn</th> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row1_col0" class="data row1 col0" >0.02</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row1_col1" class="data row1 col1" >0</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row1_col2" class="data row1 col2" >0.04</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row1_col3" class="data row1 col3" >0.04</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row1_col4" class="data row1 col4" >0.03</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row1_col5" class="data row1 col5" >0.09</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row1_col6" class="data row1 col6" >0.01</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row1_col7" class="data row1 col7" >0.21</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row1_col8" class="data row1 col8" >0.05</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row1_col9" class="data row1 col9" >0.15</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row1_col10" class="data row1 col10" >0.09</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row1_col11" class="data row1 col11" >0.05</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row1_col12" class="data row1 col12" >0.21</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row1_col13" class="data row1 col13" >0</td> 
    </tr>    <tr> 
        <th id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4level0_row2" class="row_heading level0 row2" >de</th> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row2_col0" class="data row2 col0" >0.01</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row2_col1" class="data row2 col1" >0.04</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row2_col2" class="data row2 col2" >0</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row2_col3" class="data row2 col3" >0</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row2_col4" class="data row2 col4" >0</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row2_col5" class="data row2 col5" >0.01</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row2_col6" class="data row2 col6" >0.01</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row2_col7" class="data row2 col7" >0.07</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row2_col8" class="data row2 col8" >0</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row2_col9" class="data row2 col9" >0.04</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row2_col10" class="data row2 col10" >0.01</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row2_col11" class="data row2 col11" >0</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row2_col12" class="data row2 col12" >0.07</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row2_col13" class="data row2 col13" >0.03</td> 
    </tr>    <tr> 
        <th id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4level0_row3" class="row_heading level0 row3" >en</th> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row3_col0" class="data row3 col0" >0.01</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row3_col1" class="data row3 col1" >0.04</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row3_col2" class="data row3 col2" >0</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row3_col3" class="data row3 col3" >0</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row3_col4" class="data row3 col4" >0</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row3_col5" class="data row3 col5" >0.01</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row3_col6" class="data row3 col6" >0.01</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row3_col7" class="data row3 col7" >0.07</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row3_col8" class="data row3 col8" >0</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row3_col9" class="data row3 col9" >0.04</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row3_col10" class="data row3 col10" >0.01</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row3_col11" class="data row3 col11" >0</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row3_col12" class="data row3 col12" >0.07</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row3_col13" class="data row3 col13" >0.03</td> 
    </tr>    <tr> 
        <th id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4level0_row4" class="row_heading level0 row4" >es</th> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row4_col0" class="data row4 col0" >0</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row4_col1" class="data row4 col1" >0.03</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row4_col2" class="data row4 col2" >0</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row4_col3" class="data row4 col3" >0</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row4_col4" class="data row4 col4" >0</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row4_col5" class="data row4 col5" >0.01</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row4_col6" class="data row4 col6" >0.01</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row4_col7" class="data row4 col7" >0.08</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row4_col8" class="data row4 col8" >0</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row4_col9" class="data row4 col9" >0.04</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row4_col10" class="data row4 col10" >0.01</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row4_col11" class="data row4 col11" >0</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row4_col12" class="data row4 col12" >0.08</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row4_col13" class="data row4 col13" >0.02</td> 
    </tr>    <tr> 
        <th id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4level0_row5" class="row_heading level0 row5" >he</th> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row5_col0" class="data row5 col0" >0.03</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row5_col1" class="data row5 col1" >0.09</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row5_col2" class="data row5 col2" >0.01</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row5_col3" class="data row5 col3" >0.01</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row5_col4" class="data row5 col4" >0.01</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row5_col5" class="data row5 col5" >0</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row5_col6" class="data row5 col6" >0.04</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row5_col7" class="data row5 col7" >0.03</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row5_col8" class="data row5 col8" >0.01</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row5_col9" class="data row5 col9" >0.01</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row5_col10" class="data row5 col10" >0</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row5_col11" class="data row5 col11" >0.01</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row5_col12" class="data row5 col12" >0.03</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row5_col13" class="data row5 col13" >0.07</td> 
    </tr>    <tr> 
        <th id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4level0_row6" class="row_heading level0 row6" >hi</th> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row6_col0" class="data row6 col0" >0</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row6_col1" class="data row6 col1" >0.01</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row6_col2" class="data row6 col2" >0.01</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row6_col3" class="data row6 col3" >0.01</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row6_col4" class="data row6 col4" >0.01</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row6_col5" class="data row6 col5" >0.04</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row6_col6" class="data row6 col6" >0</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row6_col7" class="data row6 col7" >0.13</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row6_col8" class="data row6 col8" >0.01</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row6_col9" class="data row6 col9" >0.09</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row6_col10" class="data row6 col10" >0.04</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row6_col11" class="data row6 col11" >0.02</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row6_col12" class="data row6 col12" >0.14</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row6_col13" class="data row6 col13" >0</td> 
    </tr>    <tr> 
        <th id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4level0_row7" class="row_heading level0 row7" >hu</th> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row7_col0" class="data row7 col0" >0.11</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row7_col1" class="data row7 col1" >0.21</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row7_col2" class="data row7 col2" >0.07</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row7_col3" class="data row7 col3" >0.07</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row7_col4" class="data row7 col4" >0.08</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row7_col5" class="data row7 col5" >0.03</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row7_col6" class="data row7 col6" >0.13</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row7_col7" class="data row7 col7" >0</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row7_col8" class="data row7 col8" >0.06</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row7_col9" class="data row7 col9" >0.01</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row7_col10" class="data row7 col10" >0.03</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row7_col11" class="data row7 col11" >0.06</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row7_col12" class="data row7 col12" >0</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row7_col13" class="data row7 col13" >0.18</td> 
    </tr>    <tr> 
        <th id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4level0_row8" class="row_heading level0 row8" >ja</th> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row8_col0" class="data row8 col0" >0.01</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row8_col1" class="data row8 col1" >0.05</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row8_col2" class="data row8 col2" >0</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row8_col3" class="data row8 col3" >0</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row8_col4" class="data row8 col4" >0</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row8_col5" class="data row8 col5" >0.01</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row8_col6" class="data row8 col6" >0.01</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row8_col7" class="data row8 col7" >0.06</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row8_col8" class="data row8 col8" >0</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row8_col9" class="data row8 col9" >0.03</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row8_col10" class="data row8 col10" >0.01</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row8_col11" class="data row8 col11" >0</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row8_col12" class="data row8 col12" >0.06</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row8_col13" class="data row8 col13" >0.03</td> 
    </tr>    <tr> 
        <th id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4level0_row9" class="row_heading level0 row9" >nl</th> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row9_col0" class="data row9 col0" >0.07</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row9_col1" class="data row9 col1" >0.15</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row9_col2" class="data row9 col2" >0.04</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row9_col3" class="data row9 col3" >0.04</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row9_col4" class="data row9 col4" >0.04</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row9_col5" class="data row9 col5" >0.01</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row9_col6" class="data row9 col6" >0.09</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row9_col7" class="data row9 col7" >0.01</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row9_col8" class="data row9 col8" >0.03</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row9_col9" class="data row9 col9" >0</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row9_col10" class="data row9 col10" >0.01</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row9_col11" class="data row9 col11" >0.03</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row9_col12" class="data row9 col12" >0.01</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row9_col13" class="data row9 col13" >0.13</td> 
    </tr>    <tr> 
        <th id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4level0_row10" class="row_heading level0 row10" >ro</th> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row10_col0" class="data row10 col0" >0.03</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row10_col1" class="data row10 col1" >0.09</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row10_col2" class="data row10 col2" >0.01</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row10_col3" class="data row10 col3" >0.01</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row10_col4" class="data row10 col4" >0.01</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row10_col5" class="data row10 col5" >0</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row10_col6" class="data row10 col6" >0.04</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row10_col7" class="data row10 col7" >0.03</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row10_col8" class="data row10 col8" >0.01</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row10_col9" class="data row10 col9" >0.01</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row10_col10" class="data row10 col10" >0</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row10_col11" class="data row10 col11" >0.01</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row10_col12" class="data row10 col12" >0.03</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row10_col13" class="data row10 col13" >0.07</td> 
    </tr>    <tr> 
        <th id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4level0_row11" class="row_heading level0 row11" >ru</th> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row11_col0" class="data row11 col0" >0.01</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row11_col1" class="data row11 col1" >0.05</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row11_col2" class="data row11 col2" >0</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row11_col3" class="data row11 col3" >0</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row11_col4" class="data row11 col4" >0</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row11_col5" class="data row11 col5" >0.01</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row11_col6" class="data row11 col6" >0.02</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row11_col7" class="data row11 col7" >0.06</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row11_col8" class="data row11 col8" >0</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row11_col9" class="data row11 col9" >0.03</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row11_col10" class="data row11 col10" >0.01</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row11_col11" class="data row11 col11" >0</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row11_col12" class="data row11 col12" >0.06</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row11_col13" class="data row11 col13" >0.04</td> 
    </tr>    <tr> 
        <th id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4level0_row12" class="row_heading level0 row12" >uk</th> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row12_col0" class="data row12 col0" >0.12</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row12_col1" class="data row12 col1" >0.21</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row12_col2" class="data row12 col2" >0.07</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row12_col3" class="data row12 col3" >0.07</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row12_col4" class="data row12 col4" >0.08</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row12_col5" class="data row12 col5" >0.03</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row12_col6" class="data row12 col6" >0.14</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row12_col7" class="data row12 col7" >0</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row12_col8" class="data row12 col8" >0.06</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row12_col9" class="data row12 col9" >0.01</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row12_col10" class="data row12 col10" >0.03</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row12_col11" class="data row12 col11" >0.06</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row12_col12" class="data row12 col12" >0</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row12_col13" class="data row12 col13" >0.19</td> 
    </tr>    <tr> 
        <th id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4level0_row13" class="row_heading level0 row13" >zh</th> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row13_col0" class="data row13 col0" >0.01</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row13_col1" class="data row13 col1" >0</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row13_col2" class="data row13 col2" >0.03</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row13_col3" class="data row13 col3" >0.03</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row13_col4" class="data row13 col4" >0.02</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row13_col5" class="data row13 col5" >0.07</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row13_col6" class="data row13 col6" >0</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row13_col7" class="data row13 col7" >0.18</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row13_col8" class="data row13 col8" >0.03</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row13_col9" class="data row13 col9" >0.13</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row13_col10" class="data row13 col10" >0.07</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row13_col11" class="data row13 col11" >0.04</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row13_col12" class="data row13 col12" >0.19</td> 
        <td id="T_7b0a35cc_5ab1_11e8_abac_1866dafa0bc4row13_col13" class="data row13 col13" >0</td> 
    </tr></tbody> 
</table style="display:inline"> <style  type="text/css" >
    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4 th {
          font-size: 7pt;
    }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row0_col0 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row0_col1 {
            font-size:  1pt;
            background-color:  #e7e3f0;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row0_col2 {
            font-size:  1pt;
            background-color:  #eae6f1;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row0_col3 {
            font-size:  1pt;
            background-color:  #e6e2ef;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row0_col4 {
            font-size:  1pt;
            background-color:  #eee9f3;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row0_col5 {
            font-size:  1pt;
            background-color:  #bbc7e0;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row0_col6 {
            font-size:  1pt;
            background-color:  #f9f2f8;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row0_col7 {
            font-size:  1pt;
            background-color:  #99b8d8;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row0_col8 {
            font-size:  1pt;
            background-color:  #dfddec;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row0_col9 {
            font-size:  1pt;
            background-color:  #a1bbda;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row0_col10 {
            font-size:  1pt;
            background-color:  #b4c4df;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row0_col11 {
            font-size:  1pt;
            background-color:  #dad9ea;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row0_col12 {
            font-size:  1pt;
            background-color:  #99b8d8;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row0_col13 {
            font-size:  1pt;
            background-color:  #efe9f3;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row1_col0 {
            font-size:  1pt;
            background-color:  #dbdaeb;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row1_col1 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row1_col2 {
            font-size:  1pt;
            background-color:  #a2bcda;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row1_col3 {
            font-size:  1pt;
            background-color:  #a2bcda;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row1_col4 {
            font-size:  1pt;
            background-color:  #adc1dd;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row1_col5 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row1_col6 {
            font-size:  1pt;
            background-color:  #e8e4f0;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row1_col7 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row1_col8 {
            font-size:  1pt;
            background-color:  #88b1d4;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row1_col9 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row1_col10 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row1_col11 {
            font-size:  1pt;
            background-color:  #88b1d4;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row1_col12 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row1_col13 {
            font-size:  1pt;
            background-color:  #faf2f8;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row2_col0 {
            font-size:  1pt;
            background-color:  #efe9f3;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row2_col1 {
            font-size:  1pt;
            background-color:  #d5d5e8;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row2_col2 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row2_col3 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row2_col4 {
            font-size:  1pt;
            background-color:  #fdf5fa;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row2_col5 {
            font-size:  1pt;
            background-color:  #dfddec;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row2_col6 {
            font-size:  1pt;
            background-color:  #e8e4f0;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row2_col7 {
            font-size:  1pt;
            background-color:  #b5c4df;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row2_col8 {
            font-size:  1pt;
            background-color:  #f9f2f8;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row2_col9 {
            font-size:  1pt;
            background-color:  #c6cce3;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row2_col10 {
            font-size:  1pt;
            background-color:  #dfddec;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row2_col11 {
            font-size:  1pt;
            background-color:  #f6eff7;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row2_col12 {
            font-size:  1pt;
            background-color:  #b5c4df;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row2_col13 {
            font-size:  1pt;
            background-color:  #dad9ea;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row3_col0 {
            font-size:  1pt;
            background-color:  #ede7f2;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row3_col1 {
            font-size:  1pt;
            background-color:  #d5d5e8;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row3_col2 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row3_col3 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row3_col4 {
            font-size:  1pt;
            background-color:  #fdf5fa;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row3_col5 {
            font-size:  1pt;
            background-color:  #dfddec;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row3_col6 {
            font-size:  1pt;
            background-color:  #e8e4f0;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row3_col7 {
            font-size:  1pt;
            background-color:  #b5c4df;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row3_col8 {
            font-size:  1pt;
            background-color:  #f9f2f8;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row3_col9 {
            font-size:  1pt;
            background-color:  #c6cce3;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row3_col10 {
            font-size:  1pt;
            background-color:  #dfddec;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row3_col11 {
            font-size:  1pt;
            background-color:  #f9f2f8;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row3_col12 {
            font-size:  1pt;
            background-color:  #b5c4df;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row3_col13 {
            font-size:  1pt;
            background-color:  #dad9ea;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row4_col0 {
            font-size:  1pt;
            background-color:  #f1ebf5;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row4_col1 {
            font-size:  1pt;
            background-color:  #d8d7e9;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row4_col2 {
            font-size:  1pt;
            background-color:  #fdf5fa;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row4_col3 {
            font-size:  1pt;
            background-color:  #fdf5fa;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row4_col4 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row4_col5 {
            font-size:  1pt;
            background-color:  #dad9ea;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row4_col6 {
            font-size:  1pt;
            background-color:  #ece7f2;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row4_col7 {
            font-size:  1pt;
            background-color:  #b1c2de;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row4_col8 {
            font-size:  1pt;
            background-color:  #f6eff7;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row4_col9 {
            font-size:  1pt;
            background-color:  #c2cbe2;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row4_col10 {
            font-size:  1pt;
            background-color:  #dad9ea;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row4_col11 {
            font-size:  1pt;
            background-color:  #f6eff7;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row4_col12 {
            font-size:  1pt;
            background-color:  #b1c2de;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row4_col13 {
            font-size:  1pt;
            background-color:  #dddbec;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row5_col0 {
            font-size:  1pt;
            background-color:  #cccfe5;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row5_col1 {
            font-size:  1pt;
            background-color:  #b1c2de;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row5_col2 {
            font-size:  1pt;
            background-color:  #dddbec;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row5_col3 {
            font-size:  1pt;
            background-color:  #dddbec;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row5_col4 {
            font-size:  1pt;
            background-color:  #dad9ea;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row5_col5 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row5_col6 {
            font-size:  1pt;
            background-color:  #c5cce3;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row5_col7 {
            font-size:  1pt;
            background-color:  #d8d7e9;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row5_col8 {
            font-size:  1pt;
            background-color:  #e3e0ee;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row5_col9 {
            font-size:  1pt;
            background-color:  #e9e5f1;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row5_col10 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row5_col11 {
            font-size:  1pt;
            background-color:  #e3e0ee;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row5_col12 {
            font-size:  1pt;
            background-color:  #d8d7e9;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row5_col13 {
            font-size:  1pt;
            background-color:  #b7c5df;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row6_col0 {
            font-size:  1pt;
            background-color:  #f8f1f8;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row6_col1 {
            font-size:  1pt;
            background-color:  #eee8f3;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row6_col2 {
            font-size:  1pt;
            background-color:  #dddbec;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row6_col3 {
            font-size:  1pt;
            background-color:  #dddbec;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row6_col4 {
            font-size:  1pt;
            background-color:  #e3e0ee;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row6_col5 {
            font-size:  1pt;
            background-color:  #a8bedc;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row6_col6 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row6_col7 {
            font-size:  1pt;
            background-color:  #8bb2d4;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row6_col8 {
            font-size:  1pt;
            background-color:  #d1d2e6;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row6_col9 {
            font-size:  1pt;
            background-color:  #8fb4d6;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row6_col10 {
            font-size:  1pt;
            background-color:  #a1bbda;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row6_col11 {
            font-size:  1pt;
            background-color:  #cacee5;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row6_col12 {
            font-size:  1pt;
            background-color:  #8bb2d4;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row6_col13 {
            font-size:  1pt;
            background-color:  #f4eef6;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row7_col0 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row7_col1 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row7_col2 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row7_col3 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row7_col4 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row7_col5 {
            font-size:  1pt;
            background-color:  #adc1dd;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row7_col6 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row7_col7 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row7_col8 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row7_col9 {
            font-size:  1pt;
            background-color:  #ede7f2;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row7_col10 {
            font-size:  1pt;
            background-color:  #b4c4df;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row7_col11 {
            font-size:  1pt;
            background-color:  #63a2cb;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row7_col12 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row7_col13 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row8_col0 {
            font-size:  1pt;
            background-color:  #e9e5f1;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row8_col1 {
            font-size:  1pt;
            background-color:  #d1d2e6;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row8_col2 {
            font-size:  1pt;
            background-color:  #faf2f8;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row8_col3 {
            font-size:  1pt;
            background-color:  #faf2f8;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row8_col4 {
            font-size:  1pt;
            background-color:  #f7f0f7;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row8_col5 {
            font-size:  1pt;
            background-color:  #e7e3f0;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row8_col6 {
            font-size:  1pt;
            background-color:  #e2dfee;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row8_col7 {
            font-size:  1pt;
            background-color:  #bdc8e1;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row8_col8 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row8_col9 {
            font-size:  1pt;
            background-color:  #d1d2e6;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row8_col10 {
            font-size:  1pt;
            background-color:  #e7e3f0;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row8_col11 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row8_col12 {
            font-size:  1pt;
            background-color:  #bdc8e1;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row8_col13 {
            font-size:  1pt;
            background-color:  #d5d5e8;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row9_col0 {
            font-size:  1pt;
            background-color:  #8fb4d6;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row9_col1 {
            font-size:  1pt;
            background-color:  #86b0d3;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row9_col2 {
            font-size:  1pt;
            background-color:  #a2bcda;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row9_col3 {
            font-size:  1pt;
            background-color:  #a2bcda;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row9_col4 {
            font-size:  1pt;
            background-color:  #a1bbda;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row9_col5 {
            font-size:  1pt;
            background-color:  #dfddec;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row9_col6 {
            font-size:  1pt;
            background-color:  #8bb2d4;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row9_col7 {
            font-size:  1pt;
            background-color:  #f0eaf4;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row9_col8 {
            font-size:  1pt;
            background-color:  #a8bedc;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row9_col9 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row9_col10 {
            font-size:  1pt;
            background-color:  #dfddec;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row9_col11 {
            font-size:  1pt;
            background-color:  #a8bedc;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row9_col12 {
            font-size:  1pt;
            background-color:  #f0eaf4;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row9_col13 {
            font-size:  1pt;
            background-color:  #89b1d4;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row10_col0 {
            font-size:  1pt;
            background-color:  #c6cce3;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row10_col1 {
            font-size:  1pt;
            background-color:  #b1c2de;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row10_col2 {
            font-size:  1pt;
            background-color:  #dddbec;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row10_col3 {
            font-size:  1pt;
            background-color:  #dddbec;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row10_col4 {
            font-size:  1pt;
            background-color:  #dad9ea;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row10_col5 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row10_col6 {
            font-size:  1pt;
            background-color:  #c0c9e2;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row10_col7 {
            font-size:  1pt;
            background-color:  #dad9ea;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row10_col8 {
            font-size:  1pt;
            background-color:  #e3e0ee;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row10_col9 {
            font-size:  1pt;
            background-color:  #e9e5f1;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row10_col10 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row10_col11 {
            font-size:  1pt;
            background-color:  #e3e0ee;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row10_col12 {
            font-size:  1pt;
            background-color:  #d8d7e9;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row10_col13 {
            font-size:  1pt;
            background-color:  #b7c5df;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row11_col0 {
            font-size:  1pt;
            background-color:  #e6e2ef;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row11_col1 {
            font-size:  1pt;
            background-color:  #d1d2e6;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row11_col2 {
            font-size:  1pt;
            background-color:  #f7f0f7;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row11_col3 {
            font-size:  1pt;
            background-color:  #faf2f8;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row11_col4 {
            font-size:  1pt;
            background-color:  #f7f0f7;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row11_col5 {
            font-size:  1pt;
            background-color:  #e7e3f0;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row11_col6 {
            font-size:  1pt;
            background-color:  #dfddec;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row11_col7 {
            font-size:  1pt;
            background-color:  #c1cae2;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row11_col8 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row11_col9 {
            font-size:  1pt;
            background-color:  #d1d2e6;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row11_col10 {
            font-size:  1pt;
            background-color:  #e7e3f0;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row11_col11 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row11_col12 {
            font-size:  1pt;
            background-color:  #bdc8e1;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row11_col13 {
            font-size:  1pt;
            background-color:  #d5d5e8;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row12_col0 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row12_col1 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row12_col2 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row12_col3 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row12_col4 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row12_col5 {
            font-size:  1pt;
            background-color:  #adc1dd;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row12_col6 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row12_col7 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row12_col8 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row12_col9 {
            font-size:  1pt;
            background-color:  #ede7f2;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row12_col10 {
            font-size:  1pt;
            background-color:  #adc1dd;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row12_col11 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row12_col12 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row12_col13 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row13_col0 {
            font-size:  1pt;
            background-color:  #e9e5f1;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row13_col1 {
            font-size:  1pt;
            background-color:  #faf3f9;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row13_col2 {
            font-size:  1pt;
            background-color:  #b7c5df;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row13_col3 {
            font-size:  1pt;
            background-color:  #b7c5df;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row13_col4 {
            font-size:  1pt;
            background-color:  #c1cae2;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row13_col5 {
            font-size:  1pt;
            background-color:  #73a9cf;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row13_col6 {
            font-size:  1pt;
            background-color:  #f2ecf5;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row13_col7 {
            font-size:  1pt;
            background-color:  #69a5cc;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row13_col8 {
            font-size:  1pt;
            background-color:  #9fbad9;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row13_col9 {
            font-size:  1pt;
            background-color:  #6da6cd;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row13_col10 {
            font-size:  1pt;
            background-color:  #73a9cf;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row13_col11 {
            font-size:  1pt;
            background-color:  #9fbad9;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row13_col12 {
            font-size:  1pt;
            background-color:  #69a5cc;
        }    #T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row13_col13 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }</style>  
<table style="display:inline" id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4" > 
<thead>    <tr> 
        <th class="index_name level0" ></th> 
        <th class="col_heading level0 col0" >ar</th> 
        <th class="col_heading level0 col1" >bn</th> 
        <th class="col_heading level0 col2" >de</th> 
        <th class="col_heading level0 col3" >en</th> 
        <th class="col_heading level0 col4" >es</th> 
        <th class="col_heading level0 col5" >he</th> 
        <th class="col_heading level0 col6" >hi</th> 
        <th class="col_heading level0 col7" >hu</th> 
        <th class="col_heading level0 col8" >ja</th> 
        <th class="col_heading level0 col9" >nl</th> 
        <th class="col_heading level0 col10" >ro</th> 
        <th class="col_heading level0 col11" >ru</th> 
        <th class="col_heading level0 col12" >uk</th> 
        <th class="col_heading level0 col13" >zh</th> 
    </tr>    <tr> 
        <th class="index_name level0" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
    </tr></thead> 
<tbody>    <tr> 
        <th id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4level0_row0" class="row_heading level0 row0" >ar</th> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row0_col0" class="data row0 col0" >0</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row0_col1" class="data row0 col1" >0.13</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row0_col2" class="data row0 col2" >0.07</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row0_col3" class="data row0 col3" >0.08</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row0_col4" class="data row0 col4" >0.06</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row0_col5" class="data row0 col5" >0.17</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row0_col6" class="data row0 col6" >0.03</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row0_col7" class="data row0 col7" >0.36</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row0_col8" class="data row0 col8" >0.09</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row0_col9" class="data row0 col9" >0.28</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row0_col10" class="data row0 col10" >0.18</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row0_col11" class="data row0 col11" >0.1</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row0_col12" class="data row0 col12" >0.36</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row0_col13" class="data row0 col13" >0.09</td> 
    </tr>    <tr> 
        <th id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4level0_row1" class="row_heading level0 row1" >bn</th> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row1_col0" class="data row1 col0" >0.13</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row1_col1" class="data row1 col1" >0</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row1_col2" class="data row1 col2" >0.2</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row1_col3" class="data row1 col3" >0.2</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row1_col4" class="data row1 col4" >0.19</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row1_col5" class="data row1 col5" >0.3</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row1_col6" class="data row1 col6" >0.1</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row1_col7" class="data row1 col7" >0.49</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row1_col8" class="data row1 col8" >0.22</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row1_col9" class="data row1 col9" >0.4</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row1_col10" class="data row1 col10" >0.3</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row1_col11" class="data row1 col11" >0.22</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row1_col12" class="data row1 col12" >0.49</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row1_col13" class="data row1 col13" >0.03</td> 
    </tr>    <tr> 
        <th id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4level0_row2" class="row_heading level0 row2" >de</th> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row2_col0" class="data row2 col0" >0.07</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row2_col1" class="data row2 col1" >0.2</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row2_col2" class="data row2 col2" >0</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row2_col3" class="data row2 col3" >0</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row2_col4" class="data row2 col4" >0.01</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row2_col5" class="data row2 col5" >0.1</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row2_col6" class="data row2 col6" >0.1</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row2_col7" class="data row2 col7" >0.29</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row2_col8" class="data row2 col8" >0.02</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row2_col9" class="data row2 col9" >0.2</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row2_col10" class="data row2 col10" >0.1</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row2_col11" class="data row2 col11" >0.03</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row2_col12" class="data row2 col12" >0.29</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row2_col13" class="data row2 col13" >0.17</td> 
    </tr>    <tr> 
        <th id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4level0_row3" class="row_heading level0 row3" >en</th> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row3_col0" class="data row3 col0" >0.08</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row3_col1" class="data row3 col1" >0.2</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row3_col2" class="data row3 col2" >0</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row3_col3" class="data row3 col3" >0</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row3_col4" class="data row3 col4" >0.01</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row3_col5" class="data row3 col5" >0.1</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row3_col6" class="data row3 col6" >0.1</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row3_col7" class="data row3 col7" >0.29</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row3_col8" class="data row3 col8" >0.02</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row3_col9" class="data row3 col9" >0.2</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row3_col10" class="data row3 col10" >0.1</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row3_col11" class="data row3 col11" >0.02</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row3_col12" class="data row3 col12" >0.29</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row3_col13" class="data row3 col13" >0.17</td> 
    </tr>    <tr> 
        <th id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4level0_row4" class="row_heading level0 row4" >es</th> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row4_col0" class="data row4 col0" >0.06</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row4_col1" class="data row4 col1" >0.19</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row4_col2" class="data row4 col2" >0.01</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row4_col3" class="data row4 col3" >0.01</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row4_col4" class="data row4 col4" >0</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row4_col5" class="data row4 col5" >0.11</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row4_col6" class="data row4 col6" >0.09</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row4_col7" class="data row4 col7" >0.3</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row4_col8" class="data row4 col8" >0.03</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row4_col9" class="data row4 col9" >0.21</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row4_col10" class="data row4 col10" >0.11</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row4_col11" class="data row4 col11" >0.03</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row4_col12" class="data row4 col12" >0.3</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row4_col13" class="data row4 col13" >0.16</td> 
    </tr>    <tr> 
        <th id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4level0_row5" class="row_heading level0 row5" >he</th> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row5_col0" class="data row5 col0" >0.17</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row5_col1" class="data row5 col1" >0.3</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row5_col2" class="data row5 col2" >0.1</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row5_col3" class="data row5 col3" >0.1</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row5_col4" class="data row5 col4" >0.11</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row5_col5" class="data row5 col5" >0</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row5_col6" class="data row5 col6" >0.2</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row5_col7" class="data row5 col7" >0.19</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row5_col8" class="data row5 col8" >0.08</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row5_col9" class="data row5 col9" >0.1</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row5_col10" class="data row5 col10" >0</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row5_col11" class="data row5 col11" >0.08</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row5_col12" class="data row5 col12" >0.19</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row5_col13" class="data row5 col13" >0.27</td> 
    </tr>    <tr> 
        <th id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4level0_row6" class="row_heading level0 row6" >hi</th> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row6_col0" class="data row6 col0" >0.03</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row6_col1" class="data row6 col1" >0.1</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row6_col2" class="data row6 col2" >0.1</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row6_col3" class="data row6 col3" >0.1</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row6_col4" class="data row6 col4" >0.09</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row6_col5" class="data row6 col5" >0.2</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row6_col6" class="data row6 col6" >0</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row6_col7" class="data row6 col7" >0.39</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row6_col8" class="data row6 col8" >0.12</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row6_col9" class="data row6 col9" >0.31</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row6_col10" class="data row6 col10" >0.21</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row6_col11" class="data row6 col11" >0.13</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row6_col12" class="data row6 col12" >0.39</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row6_col13" class="data row6 col13" >0.06</td> 
    </tr>    <tr> 
        <th id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4level0_row7" class="row_heading level0 row7" >hu</th> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row7_col0" class="data row7 col0" >0.36</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row7_col1" class="data row7 col1" >0.49</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row7_col2" class="data row7 col2" >0.29</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row7_col3" class="data row7 col3" >0.29</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row7_col4" class="data row7 col4" >0.3</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row7_col5" class="data row7 col5" >0.19</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row7_col6" class="data row7 col6" >0.39</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row7_col7" class="data row7 col7" >0</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row7_col8" class="data row7 col8" >0.27</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row7_col9" class="data row7 col9" >0.09</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row7_col10" class="data row7 col10" >0.18</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row7_col11" class="data row7 col11" >0.26</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row7_col12" class="data row7 col12" >0</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row7_col13" class="data row7 col13" >0.46</td> 
    </tr>    <tr> 
        <th id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4level0_row8" class="row_heading level0 row8" >ja</th> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row8_col0" class="data row8 col0" >0.09</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row8_col1" class="data row8 col1" >0.22</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row8_col2" class="data row8 col2" >0.02</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row8_col3" class="data row8 col3" >0.02</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row8_col4" class="data row8 col4" >0.03</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row8_col5" class="data row8 col5" >0.08</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row8_col6" class="data row8 col6" >0.12</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row8_col7" class="data row8 col7" >0.27</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row8_col8" class="data row8 col8" >0</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row8_col9" class="data row8 col9" >0.18</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row8_col10" class="data row8 col10" >0.08</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row8_col11" class="data row8 col11" >0</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row8_col12" class="data row8 col12" >0.27</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row8_col13" class="data row8 col13" >0.19</td> 
    </tr>    <tr> 
        <th id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4level0_row9" class="row_heading level0 row9" >nl</th> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row9_col0" class="data row9 col0" >0.28</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row9_col1" class="data row9 col1" >0.4</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row9_col2" class="data row9 col2" >0.2</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row9_col3" class="data row9 col3" >0.2</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row9_col4" class="data row9 col4" >0.21</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row9_col5" class="data row9 col5" >0.1</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row9_col6" class="data row9 col6" >0.31</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row9_col7" class="data row9 col7" >0.09</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row9_col8" class="data row9 col8" >0.18</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row9_col9" class="data row9 col9" >0</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row9_col10" class="data row9 col10" >0.1</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row9_col11" class="data row9 col11" >0.18</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row9_col12" class="data row9 col12" >0.09</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row9_col13" class="data row9 col13" >0.37</td> 
    </tr>    <tr> 
        <th id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4level0_row10" class="row_heading level0 row10" >ro</th> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row10_col0" class="data row10 col0" >0.18</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row10_col1" class="data row10 col1" >0.3</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row10_col2" class="data row10 col2" >0.1</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row10_col3" class="data row10 col3" >0.1</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row10_col4" class="data row10 col4" >0.11</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row10_col5" class="data row10 col5" >0</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row10_col6" class="data row10 col6" >0.21</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row10_col7" class="data row10 col7" >0.18</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row10_col8" class="data row10 col8" >0.08</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row10_col9" class="data row10 col9" >0.1</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row10_col10" class="data row10 col10" >0</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row10_col11" class="data row10 col11" >0.08</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row10_col12" class="data row10 col12" >0.19</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row10_col13" class="data row10 col13" >0.27</td> 
    </tr>    <tr> 
        <th id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4level0_row11" class="row_heading level0 row11" >ru</th> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row11_col0" class="data row11 col0" >0.1</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row11_col1" class="data row11 col1" >0.22</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row11_col2" class="data row11 col2" >0.03</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row11_col3" class="data row11 col3" >0.02</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row11_col4" class="data row11 col4" >0.03</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row11_col5" class="data row11 col5" >0.08</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row11_col6" class="data row11 col6" >0.13</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row11_col7" class="data row11 col7" >0.26</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row11_col8" class="data row11 col8" >0</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row11_col9" class="data row11 col9" >0.18</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row11_col10" class="data row11 col10" >0.08</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row11_col11" class="data row11 col11" >0</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row11_col12" class="data row11 col12" >0.27</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row11_col13" class="data row11 col13" >0.19</td> 
    </tr>    <tr> 
        <th id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4level0_row12" class="row_heading level0 row12" >uk</th> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row12_col0" class="data row12 col0" >0.36</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row12_col1" class="data row12 col1" >0.49</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row12_col2" class="data row12 col2" >0.29</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row12_col3" class="data row12 col3" >0.29</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row12_col4" class="data row12 col4" >0.3</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row12_col5" class="data row12 col5" >0.19</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row12_col6" class="data row12 col6" >0.39</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row12_col7" class="data row12 col7" >0</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row12_col8" class="data row12 col8" >0.27</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row12_col9" class="data row12 col9" >0.09</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row12_col10" class="data row12 col10" >0.19</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row12_col11" class="data row12 col11" >0.27</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row12_col12" class="data row12 col12" >0</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row12_col13" class="data row12 col13" >0.46</td> 
    </tr>    <tr> 
        <th id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4level0_row13" class="row_heading level0 row13" >zh</th> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row13_col0" class="data row13 col0" >0.09</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row13_col1" class="data row13 col1" >0.03</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row13_col2" class="data row13 col2" >0.17</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row13_col3" class="data row13 col3" >0.17</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row13_col4" class="data row13 col4" >0.16</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row13_col5" class="data row13 col5" >0.27</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row13_col6" class="data row13 col6" >0.06</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row13_col7" class="data row13 col7" >0.46</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row13_col8" class="data row13 col8" >0.19</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row13_col9" class="data row13 col9" >0.37</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row13_col10" class="data row13 col10" >0.27</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row13_col11" class="data row13 col11" >0.19</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row13_col12" class="data row13 col12" >0.46</td> 
        <td id="T_7b160eba_5ab1_11e8_abac_1866dafa0bc4row13_col13" class="data row13 col13" >0</td> 
    </tr></tbody> 
</table style="display:inline"> <style  type="text/css" >
    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4 th {
          font-size: 7pt;
    }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row0_col0 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row0_col1 {
            font-size:  1pt;
            background-color:  #e7e3f0;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row0_col2 {
            font-size:  1pt;
            background-color:  #eae6f1;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row0_col3 {
            font-size:  1pt;
            background-color:  #e7e3f0;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row0_col4 {
            font-size:  1pt;
            background-color:  #eee8f3;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row0_col5 {
            font-size:  1pt;
            background-color:  #b9c6e0;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row0_col6 {
            font-size:  1pt;
            background-color:  #f9f2f8;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row0_col7 {
            font-size:  1pt;
            background-color:  #97b7d7;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row0_col8 {
            font-size:  1pt;
            background-color:  #dfddec;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row0_col9 {
            font-size:  1pt;
            background-color:  #a4bcda;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row0_col10 {
            font-size:  1pt;
            background-color:  #b8c6e0;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row0_col11 {
            font-size:  1pt;
            background-color:  #dad9ea;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row0_col12 {
            font-size:  1pt;
            background-color:  #94b6d7;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row0_col13 {
            font-size:  1pt;
            background-color:  #eee9f3;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row1_col0 {
            font-size:  1pt;
            background-color:  #dddbec;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row1_col1 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row1_col2 {
            font-size:  1pt;
            background-color:  #a4bcda;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row1_col3 {
            font-size:  1pt;
            background-color:  #a4bcda;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row1_col4 {
            font-size:  1pt;
            background-color:  #afc1dd;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row1_col5 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row1_col6 {
            font-size:  1pt;
            background-color:  #e9e5f1;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row1_col7 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row1_col8 {
            font-size:  1pt;
            background-color:  #8bb2d4;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row1_col9 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row1_col10 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row1_col11 {
            font-size:  1pt;
            background-color:  #86b0d3;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row1_col12 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row1_col13 {
            font-size:  1pt;
            background-color:  #faf3f9;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row2_col0 {
            font-size:  1pt;
            background-color:  #efe9f3;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row2_col1 {
            font-size:  1pt;
            background-color:  #d6d6e9;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row2_col2 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row2_col3 {
            font-size:  1pt;
            background-color:  #fdf5fa;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row2_col4 {
            font-size:  1pt;
            background-color:  #fdf5fa;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row2_col5 {
            font-size:  1pt;
            background-color:  #dfddec;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row2_col6 {
            font-size:  1pt;
            background-color:  #e9e5f1;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row2_col7 {
            font-size:  1pt;
            background-color:  #b5c4df;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row2_col8 {
            font-size:  1pt;
            background-color:  #f9f2f8;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row2_col9 {
            font-size:  1pt;
            background-color:  #c5cce3;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row2_col10 {
            font-size:  1pt;
            background-color:  #dddbec;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row2_col11 {
            font-size:  1pt;
            background-color:  #f7f0f7;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row2_col12 {
            font-size:  1pt;
            background-color:  #b5c4df;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row2_col13 {
            font-size:  1pt;
            background-color:  #dcdaeb;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row3_col0 {
            font-size:  1pt;
            background-color:  #ede8f3;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row3_col1 {
            font-size:  1pt;
            background-color:  #d6d6e9;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row3_col2 {
            font-size:  1pt;
            background-color:  #fdf5fa;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row3_col3 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row3_col4 {
            font-size:  1pt;
            background-color:  #fbf4f9;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row3_col5 {
            font-size:  1pt;
            background-color:  #dfddec;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row3_col6 {
            font-size:  1pt;
            background-color:  #e7e3f0;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row3_col7 {
            font-size:  1pt;
            background-color:  #b8c6e0;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row3_col8 {
            font-size:  1pt;
            background-color:  #fbf3f9;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row3_col9 {
            font-size:  1pt;
            background-color:  #c9cee4;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row3_col10 {
            font-size:  1pt;
            background-color:  #e0dded;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row3_col11 {
            font-size:  1pt;
            background-color:  #f8f1f8;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row3_col12 {
            font-size:  1pt;
            background-color:  #b5c4df;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row3_col13 {
            font-size:  1pt;
            background-color:  #dad9ea;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row4_col0 {
            font-size:  1pt;
            background-color:  #f1ebf4;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row4_col1 {
            font-size:  1pt;
            background-color:  #d8d7e9;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row4_col2 {
            font-size:  1pt;
            background-color:  #fdf5fa;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row4_col3 {
            font-size:  1pt;
            background-color:  #fbf4f9;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row4_col4 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row4_col5 {
            font-size:  1pt;
            background-color:  #d9d8ea;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row4_col6 {
            font-size:  1pt;
            background-color:  #ebe6f2;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row4_col7 {
            font-size:  1pt;
            background-color:  #b3c3de;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row4_col8 {
            font-size:  1pt;
            background-color:  #f7f0f7;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row4_col9 {
            font-size:  1pt;
            background-color:  #c2cbe2;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row4_col10 {
            font-size:  1pt;
            background-color:  #dad9ea;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row4_col11 {
            font-size:  1pt;
            background-color:  #f4eef6;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row4_col12 {
            font-size:  1pt;
            background-color:  #b0c2de;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row4_col13 {
            font-size:  1pt;
            background-color:  #dedcec;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row5_col0 {
            font-size:  1pt;
            background-color:  #ced0e6;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row5_col1 {
            font-size:  1pt;
            background-color:  #b3c3de;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row5_col2 {
            font-size:  1pt;
            background-color:  #dedcec;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row5_col3 {
            font-size:  1pt;
            background-color:  #dedcec;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row5_col4 {
            font-size:  1pt;
            background-color:  #dad9ea;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row5_col5 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row5_col6 {
            font-size:  1pt;
            background-color:  #c4cbe3;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row5_col7 {
            font-size:  1pt;
            background-color:  #d8d7e9;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row5_col8 {
            font-size:  1pt;
            background-color:  #e5e1ef;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row5_col9 {
            font-size:  1pt;
            background-color:  #e7e3f0;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row5_col10 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row5_col11 {
            font-size:  1pt;
            background-color:  #e4e1ef;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row5_col12 {
            font-size:  1pt;
            background-color:  #d8d7e9;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row5_col13 {
            font-size:  1pt;
            background-color:  #b7c5df;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row6_col0 {
            font-size:  1pt;
            background-color:  #f9f2f8;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row6_col1 {
            font-size:  1pt;
            background-color:  #eee9f3;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row6_col2 {
            font-size:  1pt;
            background-color:  #dedcec;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row6_col3 {
            font-size:  1pt;
            background-color:  #dad9ea;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row6_col4 {
            font-size:  1pt;
            background-color:  #e3e0ee;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row6_col5 {
            font-size:  1pt;
            background-color:  #a2bcda;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row6_col6 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row6_col7 {
            font-size:  1pt;
            background-color:  #8bb2d4;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row6_col8 {
            font-size:  1pt;
            background-color:  #d2d3e7;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row6_col9 {
            font-size:  1pt;
            background-color:  #94b6d7;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row6_col10 {
            font-size:  1pt;
            background-color:  #a7bddb;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row6_col11 {
            font-size:  1pt;
            background-color:  #cccfe5;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row6_col12 {
            font-size:  1pt;
            background-color:  #88b1d4;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row6_col13 {
            font-size:  1pt;
            background-color:  #f4edf6;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row7_col0 {
            font-size:  1pt;
            background-color:  #5ea0ca;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row7_col1 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row7_col2 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row7_col3 {
            font-size:  1pt;
            background-color:  #60a1ca;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row7_col4 {
            font-size:  1pt;
            background-color:  #60a1ca;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row7_col5 {
            font-size:  1pt;
            background-color:  #acc0dd;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row7_col6 {
            font-size:  1pt;
            background-color:  #5ea0ca;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row7_col7 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row7_col8 {
            font-size:  1pt;
            background-color:  #60a1ca;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row7_col9 {
            font-size:  1pt;
            background-color:  #eee8f3;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row7_col10 {
            font-size:  1pt;
            background-color:  #b4c4df;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row7_col11 {
            font-size:  1pt;
            background-color:  #60a1ca;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row7_col12 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row7_col13 {
            font-size:  1pt;
            background-color:  #5c9fc9;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row8_col0 {
            font-size:  1pt;
            background-color:  #e9e5f1;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row8_col1 {
            font-size:  1pt;
            background-color:  #d1d2e6;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row8_col2 {
            font-size:  1pt;
            background-color:  #f9f2f8;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row8_col3 {
            font-size:  1pt;
            background-color:  #fbf4f9;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row8_col4 {
            font-size:  1pt;
            background-color:  #f7f0f7;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row8_col5 {
            font-size:  1pt;
            background-color:  #e7e3f0;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row8_col6 {
            font-size:  1pt;
            background-color:  #e2dfee;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row8_col7 {
            font-size:  1pt;
            background-color:  #bdc8e1;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row8_col8 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row8_col9 {
            font-size:  1pt;
            background-color:  #d0d1e6;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row8_col10 {
            font-size:  1pt;
            background-color:  #e6e2ef;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row8_col11 {
            font-size:  1pt;
            background-color:  #fdf5fa;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row8_col12 {
            font-size:  1pt;
            background-color:  #bbc7e0;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row8_col13 {
            font-size:  1pt;
            background-color:  #d7d6e9;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row9_col0 {
            font-size:  1pt;
            background-color:  #96b6d7;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row9_col1 {
            font-size:  1pt;
            background-color:  #84b0d3;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row9_col2 {
            font-size:  1pt;
            background-color:  #9fbad9;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row9_col3 {
            font-size:  1pt;
            background-color:  #a4bcda;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row9_col4 {
            font-size:  1pt;
            background-color:  #a1bbda;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row9_col5 {
            font-size:  1pt;
            background-color:  #dcdaeb;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row9_col6 {
            font-size:  1pt;
            background-color:  #91b5d6;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row9_col7 {
            font-size:  1pt;
            background-color:  #f1ebf4;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row9_col8 {
            font-size:  1pt;
            background-color:  #a8bedc;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row9_col9 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row9_col10 {
            font-size:  1pt;
            background-color:  #e0dded;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row9_col11 {
            font-size:  1pt;
            background-color:  #a9bfdc;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row9_col12 {
            font-size:  1pt;
            background-color:  #f0eaf4;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row9_col13 {
            font-size:  1pt;
            background-color:  #8bb2d4;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row10_col0 {
            font-size:  1pt;
            background-color:  #cacee5;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row10_col1 {
            font-size:  1pt;
            background-color:  #b0c2de;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row10_col2 {
            font-size:  1pt;
            background-color:  #dad9ea;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row10_col3 {
            font-size:  1pt;
            background-color:  #dedcec;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row10_col4 {
            font-size:  1pt;
            background-color:  #dad9ea;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row10_col5 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row10_col6 {
            font-size:  1pt;
            background-color:  #c4cbe3;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row10_col7 {
            font-size:  1pt;
            background-color:  #d9d8ea;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row10_col8 {
            font-size:  1pt;
            background-color:  #e2dfee;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row10_col9 {
            font-size:  1pt;
            background-color:  #eae6f1;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row10_col10 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row10_col11 {
            font-size:  1pt;
            background-color:  #e4e1ef;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row10_col12 {
            font-size:  1pt;
            background-color:  #d8d7e9;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row10_col13 {
            font-size:  1pt;
            background-color:  #b7c5df;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row11_col0 {
            font-size:  1pt;
            background-color:  #e7e3f0;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row11_col1 {
            font-size:  1pt;
            background-color:  #d1d2e6;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row11_col2 {
            font-size:  1pt;
            background-color:  #f7f0f7;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row11_col3 {
            font-size:  1pt;
            background-color:  #f9f2f8;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row11_col4 {
            font-size:  1pt;
            background-color:  #f5eff6;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row11_col5 {
            font-size:  1pt;
            background-color:  #e7e3f0;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row11_col6 {
            font-size:  1pt;
            background-color:  #e0deed;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row11_col7 {
            font-size:  1pt;
            background-color:  #c0c9e2;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row11_col8 {
            font-size:  1pt;
            background-color:  #fdf5fa;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row11_col9 {
            font-size:  1pt;
            background-color:  #d2d2e7;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row11_col10 {
            font-size:  1pt;
            background-color:  #e8e4f0;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row11_col11 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row11_col12 {
            font-size:  1pt;
            background-color:  #bdc8e1;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row11_col13 {
            font-size:  1pt;
            background-color:  #d4d4e8;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row12_col0 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row12_col1 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row12_col2 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row12_col3 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row12_col4 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row12_col5 {
            font-size:  1pt;
            background-color:  #acc0dd;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row12_col6 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row12_col7 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row12_col8 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row12_col9 {
            font-size:  1pt;
            background-color:  #ece7f2;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row12_col10 {
            font-size:  1pt;
            background-color:  #afc1dd;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row12_col11 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row12_col12 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row12_col13 {
            font-size:  1pt;
            background-color:  #589ec8;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row13_col0 {
            font-size:  1pt;
            background-color:  #e9e5f1;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row13_col1 {
            font-size:  1pt;
            background-color:  #faf3f9;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row13_col2 {
            font-size:  1pt;
            background-color:  #bcc7e1;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row13_col3 {
            font-size:  1pt;
            background-color:  #b7c5df;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row13_col4 {
            font-size:  1pt;
            background-color:  #c5cce3;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row13_col5 {
            font-size:  1pt;
            background-color:  #73a9cf;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row13_col6 {
            font-size:  1pt;
            background-color:  #f2ecf5;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row13_col7 {
            font-size:  1pt;
            background-color:  #6da6cd;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row13_col8 {
            font-size:  1pt;
            background-color:  #a8bedc;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row13_col9 {
            font-size:  1pt;
            background-color:  #71a8ce;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row13_col10 {
            font-size:  1pt;
            background-color:  #78abd0;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row13_col11 {
            font-size:  1pt;
            background-color:  #9ebad9;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row13_col12 {
            font-size:  1pt;
            background-color:  #69a5cc;
        }    #T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row13_col13 {
            font-size:  1pt;
            background-color:  #fff7fb;
        }</style>  
<table style="display:inline" id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4" > 
<thead>    <tr> 
        <th class="index_name level0" ></th> 
        <th class="col_heading level0 col0" >ar</th> 
        <th class="col_heading level0 col1" >bn</th> 
        <th class="col_heading level0 col2" >de</th> 
        <th class="col_heading level0 col3" >en</th> 
        <th class="col_heading level0 col4" >es</th> 
        <th class="col_heading level0 col5" >he</th> 
        <th class="col_heading level0 col6" >hi</th> 
        <th class="col_heading level0 col7" >hu</th> 
        <th class="col_heading level0 col8" >ja</th> 
        <th class="col_heading level0 col9" >nl</th> 
        <th class="col_heading level0 col10" >ro</th> 
        <th class="col_heading level0 col11" >ru</th> 
        <th class="col_heading level0 col12" >uk</th> 
        <th class="col_heading level0 col13" >zh</th> 
    </tr>    <tr> 
        <th class="index_name level0" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
    </tr></thead> 
<tbody>    <tr> 
        <th id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4level0_row0" class="row_heading level0 row0" >ar</th> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row0_col0" class="data row0 col0" >0</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row0_col1" class="data row0 col1" >0.18</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row0_col2" class="data row0 col2" >0.1</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row0_col3" class="data row0 col3" >0.11</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row0_col4" class="data row0 col4" >0.09</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row0_col5" class="data row0 col5" >0.24</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row0_col6" class="data row0 col6" >0.04</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row0_col7" class="data row0 col7" >0.51</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row0_col8" class="data row0 col8" >0.13</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row0_col9" class="data row0 col9" >0.39</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row0_col10" class="data row0 col10" >0.25</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row0_col11" class="data row0 col11" >0.14</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row0_col12" class="data row0 col12" >0.52</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row0_col13" class="data row0 col13" >0.13</td> 
    </tr>    <tr> 
        <th id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4level0_row1" class="row_heading level0 row1" >bn</th> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row1_col0" class="data row1 col0" >0.18</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row1_col1" class="data row1 col1" >0</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row1_col2" class="data row1 col2" >0.28</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row1_col3" class="data row1 col3" >0.28</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row1_col4" class="data row1 col4" >0.27</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row1_col5" class="data row1 col5" >0.42</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row1_col6" class="data row1 col6" >0.14</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row1_col7" class="data row1 col7" >0.69</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row1_col8" class="data row1 col8" >0.31</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row1_col9" class="data row1 col9" >0.57</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row1_col10" class="data row1 col10" >0.43</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row1_col11" class="data row1 col11" >0.31</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row1_col12" class="data row1 col12" >0.69</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row1_col13" class="data row1 col13" >0.04</td> 
    </tr>    <tr> 
        <th id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4level0_row2" class="row_heading level0 row2" >de</th> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row2_col0" class="data row2 col0" >0.1</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row2_col1" class="data row2 col1" >0.28</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row2_col2" class="data row2 col2" >0</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row2_col3" class="data row2 col3" >0.01</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row2_col4" class="data row2 col4" >0.01</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row2_col5" class="data row2 col5" >0.14</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row2_col6" class="data row2 col6" >0.14</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row2_col7" class="data row2 col7" >0.41</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row2_col8" class="data row2 col8" >0.03</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row2_col9" class="data row2 col9" >0.29</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row2_col10" class="data row2 col10" >0.15</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row2_col11" class="data row2 col11" >0.04</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row2_col12" class="data row2 col12" >0.41</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row2_col13" class="data row2 col13" >0.23</td> 
    </tr>    <tr> 
        <th id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4level0_row3" class="row_heading level0 row3" >en</th> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row3_col0" class="data row3 col0" >0.11</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row3_col1" class="data row3 col1" >0.28</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row3_col2" class="data row3 col2" >0.01</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row3_col3" class="data row3 col3" >0</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row3_col4" class="data row3 col4" >0.02</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row3_col5" class="data row3 col5" >0.14</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row3_col6" class="data row3 col6" >0.15</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row3_col7" class="data row3 col7" >0.4</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row3_col8" class="data row3 col8" >0.02</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row3_col9" class="data row3 col9" >0.28</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row3_col10" class="data row3 col10" >0.14</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row3_col11" class="data row3 col11" >0.03</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row3_col12" class="data row3 col12" >0.41</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row3_col13" class="data row3 col13" >0.24</td> 
    </tr>    <tr> 
        <th id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4level0_row4" class="row_heading level0 row4" >es</th> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row4_col0" class="data row4 col0" >0.09</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row4_col1" class="data row4 col1" >0.27</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row4_col2" class="data row4 col2" >0.01</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row4_col3" class="data row4 col3" >0.02</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row4_col4" class="data row4 col4" >0</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row4_col5" class="data row4 col5" >0.16</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row4_col6" class="data row4 col6" >0.13</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row4_col7" class="data row4 col7" >0.42</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row4_col8" class="data row4 col8" >0.04</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row4_col9" class="data row4 col9" >0.3</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row4_col10" class="data row4 col10" >0.16</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row4_col11" class="data row4 col11" >0.05</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row4_col12" class="data row4 col12" >0.43</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row4_col13" class="data row4 col13" >0.22</td> 
    </tr>    <tr> 
        <th id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4level0_row5" class="row_heading level0 row5" >he</th> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row5_col0" class="data row5 col0" >0.24</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row5_col1" class="data row5 col1" >0.42</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row5_col2" class="data row5 col2" >0.14</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row5_col3" class="data row5 col3" >0.14</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row5_col4" class="data row5 col4" >0.16</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row5_col5" class="data row5 col5" >0</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row5_col6" class="data row5 col6" >0.29</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row5_col7" class="data row5 col7" >0.27</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row5_col8" class="data row5 col8" >0.11</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row5_col9" class="data row5 col9" >0.15</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row5_col10" class="data row5 col10" >0</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row5_col11" class="data row5 col11" >0.11</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row5_col12" class="data row5 col12" >0.27</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row5_col13" class="data row5 col13" >0.38</td> 
    </tr>    <tr> 
        <th id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4level0_row6" class="row_heading level0 row6" >hi</th> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row6_col0" class="data row6 col0" >0.04</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row6_col1" class="data row6 col1" >0.14</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row6_col2" class="data row6 col2" >0.14</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row6_col3" class="data row6 col3" >0.15</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row6_col4" class="data row6 col4" >0.13</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row6_col5" class="data row6 col5" >0.29</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row6_col6" class="data row6 col6" >0</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row6_col7" class="data row6 col7" >0.55</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row6_col8" class="data row6 col8" >0.17</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row6_col9" class="data row6 col9" >0.43</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row6_col10" class="data row6 col10" >0.29</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row6_col11" class="data row6 col11" >0.18</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row6_col12" class="data row6 col12" >0.56</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row6_col13" class="data row6 col13" >0.09</td> 
    </tr>    <tr> 
        <th id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4level0_row7" class="row_heading level0 row7" >hu</th> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row7_col0" class="data row7 col0" >0.51</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row7_col1" class="data row7 col1" >0.69</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row7_col2" class="data row7 col2" >0.41</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row7_col3" class="data row7 col3" >0.4</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row7_col4" class="data row7 col4" >0.42</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row7_col5" class="data row7 col5" >0.27</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row7_col6" class="data row7 col6" >0.55</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row7_col7" class="data row7 col7" >0</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row7_col8" class="data row7 col8" >0.38</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row7_col9" class="data row7 col9" >0.12</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row7_col10" class="data row7 col10" >0.26</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row7_col11" class="data row7 col11" >0.37</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row7_col12" class="data row7 col12" >0</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row7_col13" class="data row7 col13" >0.64</td> 
    </tr>    <tr> 
        <th id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4level0_row8" class="row_heading level0 row8" >ja</th> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row8_col0" class="data row8 col0" >0.13</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row8_col1" class="data row8 col1" >0.31</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row8_col2" class="data row8 col2" >0.03</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row8_col3" class="data row8 col3" >0.02</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row8_col4" class="data row8 col4" >0.04</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row8_col5" class="data row8 col5" >0.11</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row8_col6" class="data row8 col6" >0.17</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row8_col7" class="data row8 col7" >0.38</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row8_col8" class="data row8 col8" >0</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row8_col9" class="data row8 col9" >0.26</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row8_col10" class="data row8 col10" >0.12</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row8_col11" class="data row8 col11" >0.01</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row8_col12" class="data row8 col12" >0.39</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row8_col13" class="data row8 col13" >0.26</td> 
    </tr>    <tr> 
        <th id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4level0_row9" class="row_heading level0 row9" >nl</th> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row9_col0" class="data row9 col0" >0.39</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row9_col1" class="data row9 col1" >0.57</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row9_col2" class="data row9 col2" >0.29</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row9_col3" class="data row9 col3" >0.28</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row9_col4" class="data row9 col4" >0.3</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row9_col5" class="data row9 col5" >0.15</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row9_col6" class="data row9 col6" >0.43</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row9_col7" class="data row9 col7" >0.12</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row9_col8" class="data row9 col8" >0.26</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row9_col9" class="data row9 col9" >0</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row9_col10" class="data row9 col10" >0.14</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row9_col11" class="data row9 col11" >0.25</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row9_col12" class="data row9 col12" >0.13</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row9_col13" class="data row9 col13" >0.52</td> 
    </tr>    <tr> 
        <th id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4level0_row10" class="row_heading level0 row10" >ro</th> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row10_col0" class="data row10 col0" >0.25</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row10_col1" class="data row10 col1" >0.43</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row10_col2" class="data row10 col2" >0.15</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row10_col3" class="data row10 col3" >0.14</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row10_col4" class="data row10 col4" >0.16</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row10_col5" class="data row10 col5" >0</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row10_col6" class="data row10 col6" >0.29</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row10_col7" class="data row10 col7" >0.26</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row10_col8" class="data row10 col8" >0.12</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row10_col9" class="data row10 col9" >0.14</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row10_col10" class="data row10 col10" >0</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row10_col11" class="data row10 col11" >0.11</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row10_col12" class="data row10 col12" >0.27</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row10_col13" class="data row10 col13" >0.38</td> 
    </tr>    <tr> 
        <th id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4level0_row11" class="row_heading level0 row11" >ru</th> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row11_col0" class="data row11 col0" >0.14</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row11_col1" class="data row11 col1" >0.31</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row11_col2" class="data row11 col2" >0.04</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row11_col3" class="data row11 col3" >0.03</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row11_col4" class="data row11 col4" >0.05</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row11_col5" class="data row11 col5" >0.11</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row11_col6" class="data row11 col6" >0.18</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row11_col7" class="data row11 col7" >0.37</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row11_col8" class="data row11 col8" >0.01</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row11_col9" class="data row11 col9" >0.25</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row11_col10" class="data row11 col10" >0.11</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row11_col11" class="data row11 col11" >0</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row11_col12" class="data row11 col12" >0.38</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row11_col13" class="data row11 col13" >0.27</td> 
    </tr>    <tr> 
        <th id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4level0_row12" class="row_heading level0 row12" >uk</th> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row12_col0" class="data row12 col0" >0.52</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row12_col1" class="data row12 col1" >0.69</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row12_col2" class="data row12 col2" >0.41</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row12_col3" class="data row12 col3" >0.41</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row12_col4" class="data row12 col4" >0.43</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row12_col5" class="data row12 col5" >0.27</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row12_col6" class="data row12 col6" >0.56</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row12_col7" class="data row12 col7" >0</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row12_col8" class="data row12 col8" >0.39</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row12_col9" class="data row12 col9" >0.13</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row12_col10" class="data row12 col10" >0.27</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row12_col11" class="data row12 col11" >0.38</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row12_col12" class="data row12 col12" >0</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row12_col13" class="data row12 col13" >0.65</td> 
    </tr>    <tr> 
        <th id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4level0_row13" class="row_heading level0 row13" >zh</th> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row13_col0" class="data row13 col0" >0.13</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row13_col1" class="data row13 col1" >0.04</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row13_col2" class="data row13 col2" >0.23</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row13_col3" class="data row13 col3" >0.24</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row13_col4" class="data row13 col4" >0.22</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row13_col5" class="data row13 col5" >0.38</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row13_col6" class="data row13 col6" >0.09</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row13_col7" class="data row13 col7" >0.64</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row13_col8" class="data row13 col8" >0.26</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row13_col9" class="data row13 col9" >0.52</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row13_col10" class="data row13 col10" >0.38</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row13_col11" class="data row13 col11" >0.27</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row13_col12" class="data row13 col12" >0.65</td> 
        <td id="T_7b21e0dc_5ab1_11e8_abac_1866dafa0bc4row13_col13" class="data row13 col13" >0</td> 
    </tr></tbody> 
</table style="display:inline"> 


### Correlations


```python
responsesShortName = responses.rename(columns=dict([(col,col.replace('motivation','m')) for col in responses.columns]))
```


```python
responsesShortName.corr().style.background_gradient(low=-0.5,high=.5,axis=0)
```




<style  type="text/css" >
    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row0_col0 {
            background-color:  #73a9cf;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row0_col1 {
            background-color:  #fff7fb;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row0_col2 {
            background-color:  #fff7fb;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row0_col3 {
            background-color:  #efe9f3;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row0_col4 {
            background-color:  #a7bddb;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row0_col5 {
            background-color:  #d3d4e7;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row0_col6 {
            background-color:  #e6e2ef;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row0_col7 {
            background-color:  #fff7fb;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row0_col8 {
            background-color:  #fff7fb;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row0_col9 {
            background-color:  #fff7fb;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row0_col10 {
            background-color:  #d0d1e6;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row0_col11 {
            background-color:  #fff7fb;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row0_col12 {
            background-color:  #dfddec;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row1_col0 {
            background-color:  #fff7fb;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row1_col1 {
            background-color:  #73a9cf;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row1_col2 {
            background-color:  #c2cbe2;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row1_col3 {
            background-color:  #fff7fb;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row1_col4 {
            background-color:  #fff7fb;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row1_col5 {
            background-color:  #fff7fb;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row1_col6 {
            background-color:  #fff7fb;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row1_col7 {
            background-color:  #fbf3f9;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row1_col8 {
            background-color:  #dfddec;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row1_col9 {
            background-color:  #c2cbe2;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row1_col10 {
            background-color:  #fff7fb;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row1_col11 {
            background-color:  #f2ecf5;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row1_col12 {
            background-color:  #fff7fb;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row2_col0 {
            background-color:  #fff7fb;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row2_col1 {
            background-color:  #bcc7e1;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row2_col2 {
            background-color:  #73a9cf;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row2_col3 {
            background-color:  #fff7fb;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row2_col4 {
            background-color:  #fff7fb;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row2_col5 {
            background-color:  #fff7fb;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row2_col6 {
            background-color:  #fff7fb;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row2_col7 {
            background-color:  #fff7fb;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row2_col8 {
            background-color:  #f1ebf4;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row2_col9 {
            background-color:  #fff7fb;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row2_col10 {
            background-color:  #fff7fb;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row2_col11 {
            background-color:  #fff7fb;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row2_col12 {
            background-color:  #e9e5f1;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row3_col0 {
            background-color:  #e2dfee;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row3_col1 {
            background-color:  #fff7fb;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row3_col2 {
            background-color:  #fff7fb;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row3_col3 {
            background-color:  #75a9cf;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row3_col4 {
            background-color:  #eae6f1;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row3_col5 {
            background-color:  #dbdaeb;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row3_col6 {
            background-color:  #e2dfee;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row3_col7 {
            background-color:  #fff7fb;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row3_col8 {
            background-color:  #fff7fb;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row3_col9 {
            background-color:  #fff7fb;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row3_col10 {
            background-color:  #fff7fb;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row3_col11 {
            background-color:  #fbf4f9;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row3_col12 {
            background-color:  #fff7fb;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row4_col0 {
            background-color:  #a7bddb;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row4_col1 {
            background-color:  #fff7fb;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row4_col2 {
            background-color:  #fff7fb;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row4_col3 {
            background-color:  #f5eff6;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row4_col4 {
            background-color:  #73a9cf;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row4_col5 {
            background-color:  #a7bddb;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row4_col6 {
            background-color:  #fbf4f9;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row4_col7 {
            background-color:  #fff7fb;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row4_col8 {
            background-color:  #fff7fb;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row4_col9 {
            background-color:  #fff7fb;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row4_col10 {
            background-color:  #f1ebf4;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row4_col11 {
            background-color:  #fff7fb;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row4_col12 {
            background-color:  #eee8f3;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row5_col0 {
            background-color:  #cccfe5;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row5_col1 {
            background-color:  #fff7fb;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row5_col2 {
            background-color:  #fff7fb;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row5_col3 {
            background-color:  #e0dded;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row5_col4 {
            background-color:  #a2bcda;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row5_col5 {
            background-color:  #73a9cf;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row5_col6 {
            background-color:  #fff7fb;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row5_col7 {
            background-color:  #fff7fb;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row5_col8 {
            background-color:  #fff7fb;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row5_col9 {
            background-color:  #fff7fb;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row5_col10 {
            background-color:  #fff7fb;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row5_col11 {
            background-color:  #fbf3f9;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row5_col12 {
            background-color:  #fff7fb;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row6_col0 {
            background-color:  #e5e1ef;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row6_col1 {
            background-color:  #fff7fb;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row6_col2 {
            background-color:  #fff7fb;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row6_col3 {
            background-color:  #f0eaf4;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row6_col4 {
            background-color:  #fbf3f9;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row6_col5 {
            background-color:  #fff7fb;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row6_col6 {
            background-color:  #75a9cf;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row6_col7 {
            background-color:  #fff7fb;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row6_col8 {
            background-color:  #fff7fb;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row6_col9 {
            background-color:  #fff7fb;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row6_col10 {
            background-color:  #f1ebf5;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row6_col11 {
            background-color:  #fff7fb;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row6_col12 {
            background-color:  #f0eaf4;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row7_col0 {
            background-color:  #fff7fb;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row7_col1 {
            background-color:  #fbf3f9;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row7_col2 {
            background-color:  #fff7fb;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row7_col3 {
            background-color:  #fff7fb;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row7_col4 {
            background-color:  #fff7fb;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row7_col5 {
            background-color:  #fff7fb;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row7_col6 {
            background-color:  #fff7fb;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row7_col7 {
            background-color:  #73a9cf;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row7_col8 {
            background-color:  #dcdaeb;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row7_col9 {
            background-color:  #ede7f2;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row7_col10 {
            background-color:  #fff7fb;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row7_col11 {
            background-color:  #f5eef6;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row7_col12 {
            background-color:  #fff7fb;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row8_col0 {
            background-color:  #fff7fb;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row8_col1 {
            background-color:  #e2dfee;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row8_col2 {
            background-color:  #fbf3f9;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row8_col3 {
            background-color:  #fff7fb;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row8_col4 {
            background-color:  #fff7fb;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row8_col5 {
            background-color:  #fff7fb;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row8_col6 {
            background-color:  #fff7fb;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row8_col7 {
            background-color:  #e0dded;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row8_col8 {
            background-color:  #73a9cf;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row8_col9 {
            background-color:  #f8f1f8;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row8_col10 {
            background-color:  #fff7fb;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row8_col11 {
            background-color:  #ede8f3;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row8_col12 {
            background-color:  #fff7fb;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row9_col0 {
            background-color:  #fff7fb;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row9_col1 {
            background-color:  #c6cce3;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row9_col2 {
            background-color:  #fff7fb;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row9_col3 {
            background-color:  #fff7fb;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row9_col4 {
            background-color:  #fff7fb;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row9_col5 {
            background-color:  #fff7fb;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row9_col6 {
            background-color:  #fff7fb;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row9_col7 {
            background-color:  #f0eaf4;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row9_col8 {
            background-color:  #f7f0f7;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row9_col9 {
            background-color:  #73a9cf;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row9_col10 {
            background-color:  #fff7fb;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row9_col11 {
            background-color:  #ced0e6;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row9_col12 {
            background-color:  #fff7fb;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row10_col0 {
            background-color:  #d2d3e7;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row10_col1 {
            background-color:  #fff7fb;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row10_col2 {
            background-color:  #fff7fb;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row10_col3 {
            background-color:  #fff7fb;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row10_col4 {
            background-color:  #f4edf6;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row10_col5 {
            background-color:  #fff7fb;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row10_col6 {
            background-color:  #f5eef6;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row10_col7 {
            background-color:  #fff7fb;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row10_col8 {
            background-color:  #fff7fb;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row10_col9 {
            background-color:  #fff7fb;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row10_col10 {
            background-color:  #73a9cf;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row10_col11 {
            background-color:  #fff7fb;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row10_col12 {
            background-color:  #d2d3e7;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row11_col0 {
            background-color:  #fff7fb;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row11_col1 {
            background-color:  #fef6fb;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row11_col2 {
            background-color:  #fff7fb;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row11_col3 {
            background-color:  #fff7fb;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row11_col4 {
            background-color:  #fff7fb;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row11_col5 {
            background-color:  #fff7fb;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row11_col6 {
            background-color:  #fff7fb;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row11_col7 {
            background-color:  #fff7fb;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row11_col8 {
            background-color:  #f4eef6;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row11_col9 {
            background-color:  #d6d6e9;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row11_col10 {
            background-color:  #fff7fb;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row11_col11 {
            background-color:  #75a9cf;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row11_col12 {
            background-color:  #fff7fb;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row12_col0 {
            background-color:  #ebe6f2;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row12_col1 {
            background-color:  #fff7fb;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row12_col2 {
            background-color:  #fbf4f9;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row12_col3 {
            background-color:  #fff7fb;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row12_col4 {
            background-color:  #f8f1f8;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row12_col5 {
            background-color:  #fff7fb;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row12_col6 {
            background-color:  #fbf3f9;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row12_col7 {
            background-color:  #fff7fb;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row12_col8 {
            background-color:  #fff7fb;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row12_col9 {
            background-color:  #fff7fb;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row12_col10 {
            background-color:  #d9d8ea;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row12_col11 {
            background-color:  #fff7fb;
        }    #T_692e079c_5ab2_11e8_abac_1866dafa0bc4row12_col12 {
            background-color:  #75a9cf;
        }</style>  
<table id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4" > 
<thead>    <tr> 
        <th class="blank level0" ></th> 
        <th class="col_heading level0 col0" >m_intrinsic_learning</th> 
        <th class="col_heading level0 col1" >m_media</th> 
        <th class="col_heading level0 col2" >m_bored/random</th> 
        <th class="col_heading level0 col3" >m_conversation</th> 
        <th class="col_heading level0 col4" >m_current_event</th> 
        <th class="col_heading level0 col5" >m_personal_decision</th> 
        <th class="col_heading level0 col6" >m_work/school</th> 
        <th class="col_heading level0 col7" >m_other</th> 
        <th class="col_heading level0 col8" >overview</th> 
        <th class="col_heading level0 col9" >fact</th> 
        <th class="col_heading level0 col10" >in-depth</th> 
        <th class="col_heading level0 col11" >familiar</th> 
        <th class="col_heading level0 col12" >unfamiliar</th> 
    </tr></thead> 
<tbody>    <tr> 
        <th id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4level0_row0" class="row_heading level0 row0" >m_intrinsic_learning</th> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row0_col0" class="data row0 col0" >1</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row0_col1" class="data row0 col1" >-0.553167</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row0_col2" class="data row0 col2" >-0.320287</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row0_col3" class="data row0 col3" >0.443594</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row0_col4" class="data row0 col4" >0.786511</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row0_col5" class="data row0 col5" >0.602044</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row0_col6" class="data row0 col6" >0.418336</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row0_col7" class="data row0 col7" >-0.695585</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row0_col8" class="data row0 col8" >-0.356596</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row0_col9" class="data row0 col9" >-0.515512</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row0_col10" class="data row0 col10" >0.557053</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row0_col11" class="data row0 col11" >-0.380738</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row0_col12" class="data row0 col12" >0.37111</td> 
    </tr>    <tr> 
        <th id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4level0_row1" class="row_heading level0 row1" >m_media</th> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row1_col0" class="data row1 col0" >-0.553167</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row1_col1" class="data row1 col1" >1</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row1_col2" class="data row1 col2" >0.679473</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row1_col3" class="data row1 col3" >-0.285836</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row1_col4" class="data row1 col4" >-0.194191</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row1_col5" class="data row1 col5" >-0.155595</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row1_col6" class="data row1 col6" >-0.688785</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row1_col7" class="data row1 col7" >0.203643</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row1_col8" class="data row1 col8" >0.442203</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row1_col9" class="data row1 col9" >0.630176</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row1_col10" class="data row1 col10" >-0.689154</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row1_col11" class="data row1 col11" >0.165813</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row1_col12" class="data row1 col12" >-0.161382</td> 
    </tr>    <tr> 
        <th id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4level0_row2" class="row_heading level0 row2" >m_bored/random</th> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row2_col0" class="data row2 col0" >-0.320287</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row2_col1" class="data row2 col1" >0.679473</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row2_col2" class="data row2 col2" >1</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row2_col3" class="data row2 col3" >-0.410577</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row2_col4" class="data row2 col4" >0.000221248</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row2_col5" class="data row2 col5" >-0.00917467</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row2_col6" class="data row2 col6" >-0.52275</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row2_col7" class="data row2 col7" >-0.00533014</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row2_col8" class="data row2 col8" >0.280901</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row2_col9" class="data row2 col9" >0.105349</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row2_col10" class="data row2 col10" >-0.247862</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row2_col11" class="data row2 col11" >-0.270797</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row2_col12" class="data row2 col12" >0.278704</td> 
    </tr>    <tr> 
        <th id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4level0_row3" class="row_heading level0 row3" >m_conversation</th> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row3_col0" class="data row3 col0" >0.443594</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row3_col1" class="data row3 col1" >-0.285836</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row3_col2" class="data row3 col2" >-0.410577</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row3_col3" class="data row3 col3" >1</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row3_col4" class="data row3 col4" >0.384476</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row3_col5" class="data row3 col5" >0.54899</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row3_col6" class="data row3 col6" >0.442348</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row3_col7" class="data row3 col7" >-0.026159</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row3_col8" class="data row3 col8" >0.0975257</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row3_col9" class="data row3 col9" >-0.139362</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row3_col10" class="data row3 col10" >0.0210073</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row3_col11" class="data row3 col11" >0.05315</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row3_col12" class="data row3 col12" >-0.0673708</td> 
    </tr>    <tr> 
        <th id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4level0_row4" class="row_heading level0 row4" >m_current_event</th> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row4_col0" class="data row4 col0" >0.786511</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row4_col1" class="data row4 col1" >-0.194191</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row4_col2" class="data row4 col2" >0.000221248</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row4_col3" class="data row4 col3" >0.384476</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row4_col4" class="data row4 col4" >1</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row4_col5" class="data row4 col5" >0.807111</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row4_col6" class="data row4 col6" >0.200199</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row4_col7" class="data row4 col7" >-0.693558</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row4_col8" class="data row4 col8" >-0.182412</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row4_col9" class="data row4 col9" >-0.261961</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row4_col10" class="data row4 col10" >0.280353</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row4_col11" class="data row4 col11" >-0.242071</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row4_col12" class="data row4 col12" >0.228714</td> 
    </tr>    <tr> 
        <th id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4level0_row5" class="row_heading level0 row5" >m_personal_decision</th> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row5_col0" class="data row5 col0" >0.602044</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row5_col1" class="data row5 col1" >-0.155595</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row5_col2" class="data row5 col2" >-0.00917467</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row5_col3" class="data row5 col3" >0.54899</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row5_col4" class="data row5 col4" >0.807111</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row5_col5" class="data row5 col5" >1</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row5_col6" class="data row5 col6" >0.0850555</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row5_col7" class="data row5 col7" >-0.514881</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row5_col8" class="data row5 col8" >-0.0133491</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row5_col9" class="data row5 col9" >-0.121606</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row5_col10" class="data row5 col10" >0.081477</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row5_col11" class="data row5 col11" >0.0596435</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row5_col12" class="data row5 col12" >-0.0739398</td> 
    </tr>    <tr> 
        <th id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4level0_row6" class="row_heading level0 row6" >m_work/school</th> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row6_col0" class="data row6 col0" >0.418336</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row6_col1" class="data row6 col1" >-0.688785</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row6_col2" class="data row6 col2" >-0.52275</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row6_col3" class="data row6 col3" >0.442348</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row6_col4" class="data row6 col4" >0.200199</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row6_col5" class="data row6 col5" >0.0850555</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row6_col6" class="data row6 col6" >1</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row6_col7" class="data row6 col7" >-0.0144304</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row6_col8" class="data row6 col8" >-0.246281</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row6_col9" class="data row6 col9" >-0.179889</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row6_col10" class="data row6 col10" >0.272197</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row6_col11" class="data row6 col11" >-0.211474</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row6_col12" class="data row6 col12" >0.205931</td> 
    </tr>    <tr> 
        <th id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4level0_row7" class="row_heading level0 row7" >m_other</th> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row7_col0" class="data row7 col0" >-0.695585</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row7_col1" class="data row7 col1" >0.203643</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row7_col2" class="data row7 col2" >-0.00533014</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row7_col3" class="data row7 col3" >-0.026159</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row7_col4" class="data row7 col4" >-0.693558</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row7_col5" class="data row7 col5" >-0.514881</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row7_col6" class="data row7 col6" >-0.0144304</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row7_col7" class="data row7 col7" >1</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row7_col8" class="data row7 col8" >0.459532</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row7_col9" class="data row7 col9" >0.330918</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row7_col10" class="data row7 col10" >-0.509594</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row7_col11" class="data row7 col11" >0.13519</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row7_col12" class="data row7 col12" >-0.134239</td> 
    </tr>    <tr> 
        <th id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4level0_row8" class="row_heading level0 row8" >overview</th> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row8_col0" class="data row8 col0" >-0.356596</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row8_col1" class="data row8 col1" >0.442203</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row8_col2" class="data row8 col2" >0.280901</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row8_col3" class="data row8 col3" >0.0975257</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row8_col4" class="data row8 col4" >-0.182412</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row8_col5" class="data row8 col5" >-0.0133491</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row8_col6" class="data row8 col6" >-0.246281</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row8_col7" class="data row8 col7" >0.459532</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row8_col8" class="data row8 col8" >1</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row8_col9" class="data row8 col9" >0.202573</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row8_col10" class="data row8 col10" >-0.779849</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row8_col11" class="data row8 col11" >0.239785</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row8_col12" class="data row8 col12" >-0.245966</td> 
    </tr>    <tr> 
        <th id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4level0_row9" class="row_heading level0 row9" >fact</th> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row9_col0" class="data row9 col0" >-0.515512</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row9_col1" class="data row9 col1" >0.630176</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row9_col2" class="data row9 col2" >0.105349</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row9_col3" class="data row9 col3" >-0.139362</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row9_col4" class="data row9 col4" >-0.261961</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row9_col5" class="data row9 col5" >-0.121606</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row9_col6" class="data row9 col6" >-0.179889</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row9_col7" class="data row9 col7" >0.330918</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row9_col8" class="data row9 col8" >0.202573</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row9_col9" class="data row9 col9" >1</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row9_col10" class="data row9 col10" >-0.770905</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row9_col11" class="data row9 col11" >0.51045</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row9_col12" class="data row9 col12" >-0.508434</td> 
    </tr>    <tr> 
        <th id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4level0_row10" class="row_heading level0 row10" >in-depth</th> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row10_col0" class="data row10 col0" >0.557053</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row10_col1" class="data row10 col1" >-0.689154</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row10_col2" class="data row10 col2" >-0.247862</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row10_col3" class="data row10 col3" >0.0210073</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row10_col4" class="data row10 col4" >0.280353</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row10_col5" class="data row10 col5" >0.081477</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row10_col6" class="data row10 col6" >0.272197</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row10_col7" class="data row10 col7" >-0.509594</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row10_col8" class="data row10 col8" >-0.779849</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row10_col9" class="data row10 col9" >-0.770905</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row10_col10" class="data row10 col10" >1</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row10_col11" class="data row10 col11" >-0.479113</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row10_col12" class="data row10 col12" >0.482011</td> 
    </tr>    <tr> 
        <th id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4level0_row11" class="row_heading level0 row11" >familiar</th> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row11_col0" class="data row11 col0" >-0.380738</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row11_col1" class="data row11 col1" >0.165813</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row11_col2" class="data row11 col2" >-0.270797</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row11_col3" class="data row11 col3" >0.05315</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row11_col4" class="data row11 col4" >-0.242071</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row11_col5" class="data row11 col5" >0.0596435</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row11_col6" class="data row11 col6" >-0.211474</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row11_col7" class="data row11 col7" >0.13519</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row11_col8" class="data row11 col8" >0.239785</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row11_col9" class="data row11 col9" >0.51045</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row11_col10" class="data row11 col10" >-0.479113</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row11_col11" class="data row11 col11" >1</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row11_col12" class="data row11 col12" >-0.999761</td> 
    </tr>    <tr> 
        <th id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4level0_row12" class="row_heading level0 row12" >unfamiliar</th> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row12_col0" class="data row12 col0" >0.37111</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row12_col1" class="data row12 col1" >-0.161382</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row12_col2" class="data row12 col2" >0.278704</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row12_col3" class="data row12 col3" >-0.0673708</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row12_col4" class="data row12 col4" >0.228714</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row12_col5" class="data row12 col5" >-0.0739398</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row12_col6" class="data row12 col6" >0.205931</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row12_col7" class="data row12 col7" >-0.134239</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row12_col8" class="data row12 col8" >-0.245966</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row12_col9" class="data row12 col9" >-0.508434</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row12_col10" class="data row12 col10" >0.482011</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row12_col11" class="data row12 col11" >-0.999761</td> 
        <td id="T_692e079c_5ab2_11e8_abac_1866dafa0bc4row12_col12" class="data row12 col12" >1</td> 
    </tr></tbody> 
</table> 


