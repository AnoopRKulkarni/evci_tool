EVCI Siting Tool
================

<!-- WARNING: THIS FILE WAS AUTOGENERATED! DO NOT EDIT! -->

## Install

`pip install evci_tool`

## How to use

The model inputs are provided in the form of excel files (xlsx). The
`analyze_sites()` is the entry level function and completes the analysis
for specified corridor

``` python
from evci_tool.config import *
from evci_tool.model import *
from evci_tool.analysis import *

ui_inputs = { 
    "backoff_factor": 1,
    "M": ["3WS", "4WS", "4WF"],
    "years_of_analysis": [1,2,3],
    "capex_2W": 2500,
    "capex_3WS": 112000,
    "capex_4WS": 250000,
    "capex_4WF": 1500000,
    "hoarding cost": 900000,
    "kiosk_cost": 180000,
    "year1_conversion": 0.02,
    "year2_conversion": 0.05,
    "year3_conversion": 0.1,
    "holiday_percentage": 0.3,
    "fast_charging": 0.3,
    "slow_charging": 0.15,
    "cluster": False,
    "cluster_th": 0.2,
    "plot_dendrogram": False,
    "use_defaults": False 
}


u_df = analyze_sites ('mumbai_goa',ui_inputs)
u_df.head()
```


    Initial Analysis
    ________________

    Number of sites: 244/244
    Total capex charges = INR Cr 45.68
    Total opex charges = INR Cr 87.77
    Total Margin = INR Cr 33.53
    confirmed sites with utilization > 20%: 0

      0%|          | 0/244 [00:00<?, ?it/s]  0%|          | 1/244 [00:00<00:29,  8.22it/s]  1%|          | 2/244 [00:00<00:29,  8.12it/s]  1%|1         | 3/244 [00:00<00:29,  8.15it/s]  2%|1         | 4/244 [00:00<00:29,  8.13it/s]  2%|2         | 5/244 [00:00<00:29,  8.09it/s]  2%|2         | 6/244 [00:00<00:29,  8.09it/s]  3%|2         | 7/244 [00:00<00:29,  8.11it/s]  3%|3         | 8/244 [00:00<00:29,  8.14it/s]  4%|3         | 9/244 [00:01<00:28,  8.13it/s]  4%|4         | 10/244 [00:01<00:28,  8.17it/s]  5%|4         | 11/244 [00:01<00:28,  8.17it/s]  5%|4         | 12/244 [00:01<00:28,  8.14it/s]  5%|5         | 13/244 [00:01<00:28,  8.18it/s]  6%|5         | 14/244 [00:01<00:28,  8.19it/s]  6%|6         | 15/244 [00:01<00:27,  8.19it/s]  7%|6         | 16/244 [00:01<00:27,  8.19it/s]  7%|6         | 17/244 [00:02<00:27,  8.20it/s]  7%|7         | 18/244 [00:02<00:27,  8.26it/s]  8%|7         | 19/244 [00:02<00:27,  8.28it/s]  8%|8         | 20/244 [00:02<00:27,  8.29it/s]  9%|8         | 21/244 [00:02<00:28,  7.83it/s]  9%|9         | 22/244 [00:02<00:28,  7.93it/s]  9%|9         | 23/244 [00:02<00:27,  8.03it/s] 10%|9         | 24/244 [00:02<00:27,  8.07it/s] 10%|#         | 25/244 [00:03<00:26,  8.11it/s] 11%|#         | 26/244 [00:03<00:26,  8.12it/s] 11%|#1        | 27/244 [00:03<00:26,  8.16it/s] 11%|#1        | 28/244 [00:03<00:26,  8.17it/s] 12%|#1        | 29/244 [00:03<00:26,  8.15it/s] 12%|#2        | 30/244 [00:03<00:26,  8.16it/s] 13%|#2        | 31/244 [00:03<00:26,  8.16it/s] 13%|#3        | 32/244 [00:03<00:25,  8.18it/s] 14%|#3        | 33/244 [00:04<00:25,  8.21it/s] 14%|#3        | 34/244 [00:04<00:25,  8.21it/s] 14%|#4        | 35/244 [00:04<00:25,  8.21it/s] 15%|#4        | 36/244 [00:04<00:25,  8.21it/s] 15%|#5        | 37/244 [00:04<00:25,  8.22it/s] 16%|#5        | 38/244 [00:04<00:25,  8.23it/s] 16%|#5        | 39/244 [00:04<00:24,  8.21it/s] 16%|#6        | 40/244 [00:04<00:24,  8.24it/s] 17%|#6        | 41/244 [00:05<00:24,  8.22it/s] 17%|#7        | 42/244 [00:05<00:24,  8.24it/s] 18%|#7        | 43/244 [00:05<00:24,  8.26it/s] 18%|#8        | 44/244 [00:05<00:24,  8.25it/s] 18%|#8        | 45/244 [00:05<00:24,  8.25it/s] 19%|#8        | 46/244 [00:05<00:24,  8.22it/s] 19%|#9        | 47/244 [00:05<00:24,  8.20it/s] 20%|#9        | 48/244 [00:05<00:23,  8.24it/s] 20%|##        | 49/244 [00:05<00:23,  8.23it/s] 20%|##        | 50/244 [00:06<00:23,  8.26it/s] 21%|##        | 51/244 [00:06<00:23,  8.28it/s] 21%|##1       | 52/244 [00:06<00:23,  8.24it/s] 22%|##1       | 53/244 [00:06<00:23,  8.26it/s] 22%|##2       | 54/244 [00:06<00:22,  8.27it/s] 23%|##2       | 55/244 [00:06<00:22,  8.29it/s] 23%|##2       | 56/244 [00:06<00:22,  8.30it/s] 23%|##3       | 57/244 [00:06<00:22,  8.31it/s] 24%|##3       | 58/244 [00:07<00:22,  8.33it/s] 24%|##4       | 59/244 [00:07<00:22,  8.33it/s] 25%|##4       | 60/244 [00:07<00:22,  8.33it/s] 25%|##5       | 61/244 [00:07<00:22,  8.28it/s] 25%|##5       | 62/244 [00:07<00:21,  8.29it/s] 26%|##5       | 63/244 [00:07<00:21,  8.32it/s] 26%|##6       | 64/244 [00:07<00:21,  8.28it/s] 27%|##6       | 65/244 [00:07<00:21,  8.31it/s] 27%|##7       | 66/244 [00:08<00:21,  8.20it/s] 27%|##7       | 67/244 [00:08<00:21,  8.27it/s] 28%|##7       | 68/244 [00:08<00:21,  8.28it/s] 28%|##8       | 69/244 [00:08<00:21,  8.32it/s] 29%|##8       | 70/244 [00:08<00:20,  8.32it/s] 29%|##9       | 71/244 [00:08<00:20,  8.31it/s] 30%|##9       | 72/244 [00:08<00:20,  8.31it/s] 30%|##9       | 73/244 [00:08<00:20,  8.32it/s] 30%|###       | 74/244 [00:09<00:20,  8.33it/s] 31%|###       | 75/244 [00:09<00:20,  8.34it/s] 31%|###1      | 76/244 [00:09<00:20,  8.34it/s] 32%|###1      | 77/244 [00:09<00:20,  8.31it/s] 32%|###1      | 78/244 [00:09<00:19,  8.31it/s] 32%|###2      | 79/244 [00:09<00:19,  8.32it/s] 33%|###2      | 80/244 [00:09<00:19,  8.29it/s] 33%|###3      | 81/244 [00:09<00:19,  8.30it/s] 34%|###3      | 82/244 [00:09<00:19,  8.12it/s] 34%|###4      | 83/244 [00:10<00:19,  8.17it/s] 34%|###4      | 84/244 [00:10<00:19,  8.24it/s] 35%|###4      | 85/244 [00:10<00:19,  8.25it/s] 35%|###5      | 86/244 [00:10<00:19,  8.28it/s] 36%|###5      | 87/244 [00:10<00:19,  8.23it/s] 36%|###6      | 88/244 [00:10<00:18,  8.22it/s] 36%|###6      | 89/244 [00:10<00:18,  8.26it/s] 37%|###6      | 90/244 [00:10<00:18,  8.31it/s] 37%|###7      | 91/244 [00:11<00:18,  8.24it/s] 38%|###7      | 92/244 [00:11<00:18,  8.25it/s] 38%|###8      | 93/244 [00:11<00:18,  8.31it/s] 39%|###8      | 94/244 [00:11<00:17,  8.35it/s] 39%|###8      | 95/244 [00:11<00:17,  8.30it/s] 39%|###9      | 96/244 [00:11<00:17,  8.29it/s] 40%|###9      | 97/244 [00:11<00:17,  8.32it/s] 40%|####      | 98/244 [00:11<00:17,  8.34it/s] 41%|####      | 99/244 [00:12<00:17,  8.31it/s] 41%|####      | 100/244 [00:12<00:17,  8.33it/s] 41%|####1     | 101/244 [00:12<00:17,  8.30it/s] 42%|####1     | 102/244 [00:12<00:17,  8.33it/s] 42%|####2     | 103/244 [00:12<00:16,  8.34it/s] 43%|####2     | 104/244 [00:12<00:16,  8.27it/s] 43%|####3     | 105/244 [00:12<00:16,  8.31it/s] 43%|####3     | 106/244 [00:12<00:16,  8.26it/s] 44%|####3     | 107/244 [00:12<00:16,  8.30it/s] 44%|####4     | 108/244 [00:13<00:16,  8.32it/s] 45%|####4     | 109/244 [00:13<00:16,  8.33it/s] 45%|####5     | 110/244 [00:13<00:16,  8.36it/s] 45%|####5     | 111/244 [00:13<00:15,  8.36it/s] 46%|####5     | 112/244 [00:13<00:15,  8.37it/s] 46%|####6     | 113/244 [00:13<00:15,  8.35it/s] 47%|####6     | 114/244 [00:13<00:15,  8.33it/s] 47%|####7     | 115/244 [00:13<00:15,  8.34it/s] 48%|####7     | 116/244 [00:14<00:15,  8.35it/s] 48%|####7     | 117/244 [00:14<00:15,  8.34it/s] 48%|####8     | 118/244 [00:14<00:15,  8.29it/s] 49%|####8     | 119/244 [00:14<00:15,  8.31it/s] 49%|####9     | 120/244 [00:14<00:14,  8.32it/s] 50%|####9     | 121/244 [00:14<00:14,  8.31it/s] 50%|#####     | 122/244 [00:14<00:14,  8.32it/s] 50%|#####     | 123/244 [00:14<00:14,  8.32it/s] 51%|#####     | 124/244 [00:15<00:14,  8.35it/s] 51%|#####1    | 125/244 [00:15<00:14,  8.34it/s] 52%|#####1    | 126/244 [00:15<00:14,  8.30it/s] 52%|#####2    | 127/244 [00:15<00:14,  8.28it/s] 52%|#####2    | 128/244 [00:15<00:14,  8.26it/s] 53%|#####2    | 129/244 [00:15<00:14,  8.18it/s] 53%|#####3    | 130/244 [00:15<00:13,  8.20it/s] 54%|#####3    | 131/244 [00:15<00:13,  8.19it/s] 54%|#####4    | 132/244 [00:16<00:13,  8.23it/s] 55%|#####4    | 133/244 [00:16<00:13,  8.24it/s] 55%|#####4    | 134/244 [00:16<00:13,  8.26it/s] 55%|#####5    | 135/244 [00:16<00:13,  8.27it/s] 56%|#####5    | 136/244 [00:16<00:13,  8.28it/s] 56%|#####6    | 137/244 [00:16<00:12,  8.29it/s] 57%|#####6    | 138/244 [00:16<00:12,  8.28it/s] 57%|#####6    | 139/244 [00:16<00:12,  8.30it/s] 57%|#####7    | 140/244 [00:16<00:12,  8.33it/s] 58%|#####7    | 141/244 [00:17<00:12,  8.33it/s] 58%|#####8    | 142/244 [00:17<00:12,  8.33it/s] 59%|#####8    | 143/244 [00:17<00:12,  8.34it/s] 59%|#####9    | 144/244 [00:17<00:12,  8.32it/s] 59%|#####9    | 145/244 [00:17<00:11,  8.33it/s] 60%|#####9    | 146/244 [00:17<00:11,  8.33it/s] 60%|######    | 147/244 [00:17<00:11,  8.34it/s] 61%|######    | 148/244 [00:17<00:11,  8.13it/s] 61%|######1   | 149/244 [00:18<00:11,  8.17it/s] 61%|######1   | 150/244 [00:18<00:11,  8.09it/s] 62%|######1   | 151/244 [00:18<00:11,  8.14it/s] 62%|######2   | 152/244 [00:18<00:11,  8.17it/s] 63%|######2   | 153/244 [00:18<00:11,  8.20it/s] 63%|######3   | 154/244 [00:18<00:10,  8.21it/s] 64%|######3   | 155/244 [00:18<00:10,  8.22it/s] 64%|######3   | 156/244 [00:18<00:10,  8.23it/s] 64%|######4   | 157/244 [00:19<00:10,  8.25it/s] 65%|######4   | 158/244 [00:19<00:10,  8.26it/s] 65%|######5   | 159/244 [00:19<00:10,  8.27it/s] 66%|######5   | 160/244 [00:19<00:10,  8.28it/s] 66%|######5   | 161/244 [00:19<00:09,  8.31it/s] 66%|######6   | 162/244 [00:19<00:09,  8.32it/s] 67%|######6   | 163/244 [00:19<00:09,  8.28it/s] 67%|######7   | 164/244 [00:19<00:09,  8.29it/s] 68%|######7   | 165/244 [00:19<00:09,  8.23it/s] 68%|######8   | 166/244 [00:20<00:09,  8.22it/s] 68%|######8   | 167/244 [00:20<00:09,  8.23it/s] 69%|######8   | 168/244 [00:20<00:09,  8.23it/s] 69%|######9   | 169/244 [00:20<00:09,  8.14it/s] 70%|######9   | 170/244 [00:20<00:09,  8.10it/s] 70%|#######   | 171/244 [00:20<00:08,  8.13it/s] 70%|#######   | 172/244 [00:20<00:08,  8.21it/s] 71%|#######   | 173/244 [00:20<00:08,  8.10it/s] 71%|#######1  | 174/244 [00:21<00:08,  8.17it/s] 72%|#######1  | 175/244 [00:21<00:08,  8.21it/s] 72%|#######2  | 176/244 [00:21<00:08,  8.25it/s] 73%|#######2  | 177/244 [00:21<00:08,  8.30it/s] 73%|#######2  | 178/244 [00:21<00:08,  8.24it/s] 73%|#######3  | 179/244 [00:21<00:07,  8.24it/s] 74%|#######3  | 180/244 [00:21<00:07,  8.27it/s] 74%|#######4  | 181/244 [00:21<00:07,  8.22it/s] 75%|#######4  | 182/244 [00:22<00:07,  8.24it/s] 75%|#######5  | 183/244 [00:22<00:07,  8.27it/s] 75%|#######5  | 184/244 [00:22<00:07,  8.29it/s] 76%|#######5  | 185/244 [00:22<00:07,  8.32it/s] 76%|#######6  | 186/244 [00:22<00:06,  8.32it/s] 77%|#######6  | 187/244 [00:22<00:06,  8.34it/s] 77%|#######7  | 188/244 [00:22<00:06,  8.30it/s] 77%|#######7  | 189/244 [00:22<00:06,  8.15it/s] 78%|#######7  | 190/244 [00:23<00:06,  8.00it/s] 78%|#######8  | 191/244 [00:23<00:06,  7.94it/s] 79%|#######8  | 192/244 [00:23<00:06,  7.81it/s] 79%|#######9  | 193/244 [00:23<00:06,  7.87it/s] 80%|#######9  | 194/244 [00:23<00:06,  7.77it/s] 80%|#######9  | 195/244 [00:23<00:06,  7.83it/s] 80%|########  | 196/244 [00:23<00:06,  7.78it/s] 81%|########  | 197/244 [00:23<00:06,  7.72it/s] 81%|########1 | 198/244 [00:24<00:05,  7.73it/s] 82%|########1 | 199/244 [00:24<00:05,  7.72it/s] 82%|########1 | 200/244 [00:24<00:05,  7.73it/s] 82%|########2 | 201/244 [00:24<00:05,  7.89it/s] 83%|########2 | 202/244 [00:24<00:05,  7.93it/s] 83%|########3 | 203/244 [00:24<00:05,  7.80it/s] 84%|########3 | 204/244 [00:24<00:05,  7.86it/s] 84%|########4 | 205/244 [00:25<00:05,  7.24it/s] 84%|########4 | 206/244 [00:25<00:05,  6.84it/s] 85%|########4 | 207/244 [00:25<00:05,  7.23it/s] 85%|########5 | 208/244 [00:25<00:04,  7.47it/s] 86%|########5 | 209/244 [00:25<00:04,  7.54it/s] 86%|########6 | 210/244 [00:25<00:04,  7.66it/s] 86%|########6 | 211/244 [00:25<00:04,  7.81it/s] 87%|########6 | 212/244 [00:25<00:04,  7.97it/s] 87%|########7 | 213/244 [00:26<00:03,  8.03it/s] 88%|########7 | 214/244 [00:26<00:03,  7.95it/s] 88%|########8 | 215/244 [00:26<00:03,  8.06it/s] 89%|########8 | 216/244 [00:26<00:03,  8.12it/s] 89%|########8 | 217/244 [00:26<00:03,  8.19it/s] 89%|########9 | 218/244 [00:26<00:03,  8.23it/s] 90%|########9 | 219/244 [00:26<00:03,  8.24it/s] 90%|######### | 220/244 [00:26<00:02,  8.27it/s] 91%|######### | 221/244 [00:27<00:02,  8.13it/s] 91%|######### | 222/244 [00:27<00:02,  8.19it/s] 91%|#########1| 223/244 [00:27<00:02,  8.25it/s] 92%|#########1| 224/244 [00:27<00:02,  8.28it/s] 92%|#########2| 225/244 [00:27<00:02,  8.30it/s] 93%|#########2| 226/244 [00:27<00:02,  8.32it/s] 93%|#########3| 227/244 [00:27<00:02,  8.33it/s] 93%|#########3| 228/244 [00:27<00:01,  8.32it/s] 94%|#########3| 229/244 [00:27<00:01,  8.30it/s] 94%|#########4| 230/244 [00:28<00:01,  8.30it/s] 95%|#########4| 231/244 [00:28<00:01,  8.24it/s] 95%|#########5| 232/244 [00:28<00:01,  8.27it/s] 95%|#########5| 233/244 [00:28<00:01,  8.30it/s] 96%|#########5| 234/244 [00:28<00:01,  8.32it/s] 96%|#########6| 235/244 [00:28<00:01,  8.33it/s] 97%|#########6| 236/244 [00:28<00:00,  8.29it/s] 97%|#########7| 237/244 [00:28<00:00,  8.31it/s] 98%|#########7| 238/244 [00:29<00:00,  8.35it/s] 98%|#########7| 239/244 [00:29<00:00,  8.33it/s] 98%|#########8| 240/244 [00:29<00:00,  8.34it/s] 99%|#########8| 241/244 [00:29<00:00,  8.35it/s] 99%|#########9| 242/244 [00:29<00:00,  8.35it/s]100%|#########9| 243/244 [00:29<00:00,  8.36it/s]100%|##########| 244/244 [00:29<00:00,  8.37it/s]100%|##########| 244/244 [00:29<00:00,  8.20it/s]

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>Traffic congestion</th>
      <th>year 1</th>
      <th>kiosk hoarding</th>
      <th>hoarding margin</th>
      <th>geometry</th>
      <th>utilization</th>
      <th>unserviced</th>
      <th>capex</th>
      <th>opex</th>
      <th>margin</th>
      <th>max vehicles</th>
      <th>estimated vehicles</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Mahanagar Gas CNG station</td>
      <td>19.177391</td>
      <td>72.968405</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>15520</td>
      <td>POINT (72.96841 19.17739)</td>
      <td>0.096406</td>
      <td>NaN</td>
      <td>1872000.0</td>
      <td>3.868338e+06</td>
      <td>1.256811e+06</td>
      <td>25.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Gurukrupa Indian Oil Petrol Pump</td>
      <td>19.138905</td>
      <td>73.050305</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>15520</td>
      <td>POINT (73.05031 19.13891)</td>
      <td>0.047681</td>
      <td>NaN</td>
      <td>1872000.0</td>
      <td>3.663296e+06</td>
      <td>1.226055e+06</td>
      <td>25.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Mohan Warehouse</td>
      <td>19.124338</td>
      <td>73.054793</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>15520</td>
      <td>POINT (73.05479 19.12434)</td>
      <td>0.001121</td>
      <td>NaN</td>
      <td>1872000.0</td>
      <td>3.469664e+06</td>
      <td>1.197010e+06</td>
      <td>25.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>HP petrol pump</td>
      <td>19.122921</td>
      <td>73.054856</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>15520</td>
      <td>POINT (73.05486 19.12292)</td>
      <td>0.000676</td>
      <td>NaN</td>
      <td>1872000.0</td>
      <td>3.467810e+06</td>
      <td>1.196732e+06</td>
      <td>25.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Bharat Petroleum Petrol Pump</td>
      <td>19.119491</td>
      <td>73.056161</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>15520</td>
      <td>POINT (73.05616 19.11949)</td>
      <td>0.000549</td>
      <td>NaN</td>
      <td>1872000.0</td>
      <td>3.467281e+06</td>
      <td>1.196653e+06</td>
      <td>25.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>
