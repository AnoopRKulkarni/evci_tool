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

analyze_sites ('chandigarh_leh')
```


    Initial Analysis
    ________________

    Number of sites: 48/48
    Total capex charges = INR Cr 8.99
    Total opex charges = INR Cr 42.25
    Total Margin = INR Cr nan
    confirmed sites with utilization > 20%:  41

      0%|          | 0/48 [00:00<?, ?it/s]  4%|4         | 2/48 [00:00<00:04, 10.65it/s]  8%|8         | 4/48 [00:00<00:04, 10.66it/s] 12%|#2        | 6/48 [00:00<00:04, 10.39it/s] 17%|#6        | 8/48 [00:00<00:03, 10.54it/s] 21%|##        | 10/48 [00:00<00:03, 10.65it/s] 25%|##5       | 12/48 [00:01<00:03, 10.71it/s] 29%|##9       | 14/48 [00:01<00:03, 10.82it/s] 33%|###3      | 16/48 [00:01<00:02, 10.95it/s] 38%|###7      | 18/48 [00:01<00:02, 10.99it/s] 42%|####1     | 20/48 [00:01<00:02, 11.00it/s] 46%|####5     | 22/48 [00:02<00:02, 11.09it/s] 50%|#####     | 24/48 [00:02<00:02, 11.13it/s] 54%|#####4    | 26/48 [00:02<00:01, 11.17it/s] 58%|#####8    | 28/48 [00:02<00:01, 11.18it/s] 62%|######2   | 30/48 [00:02<00:01, 11.20it/s] 67%|######6   | 32/48 [00:02<00:01, 11.17it/s] 71%|#######   | 34/48 [00:03<00:01, 11.19it/s] 75%|#######5  | 36/48 [00:03<00:01, 11.28it/s] 79%|#######9  | 38/48 [00:03<00:00, 11.21it/s] 83%|########3 | 40/48 [00:03<00:00, 11.11it/s] 88%|########7 | 42/48 [00:03<00:00, 11.09it/s] 92%|#########1| 44/48 [00:03<00:00, 11.07it/s] 96%|#########5| 46/48 [00:04<00:00, 11.05it/s]100%|##########| 48/48 [00:04<00:00, 11.02it/s]100%|##########| 48/48 [00:04<00:00, 11.00it/s]

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
      <td>Fish Pond Restaurant</td>
      <td>34.071387</td>
      <td>77.634410</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>POINT (77.63441 34.07139)</td>
      <td>0.647994</td>
      <td>0.188651</td>
      <td>1872000.0</td>
      <td>7.949308e+06</td>
      <td>NaN</td>
      <td>44.0</td>
      <td>29.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Jack &amp; jim's</td>
      <td>34.146624</td>
      <td>77.581010</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>POINT (77.58101 34.14662)</td>
      <td>0.002483</td>
      <td>0.000000</td>
      <td>1872000.0</td>
      <td>3.475824e+06</td>
      <td>NaN</td>
      <td>44.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>DIKI FAST FOOD CORNER</td>
      <td>34.111891</td>
      <td>77.589740</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>POINT (77.58974 34.11189)</td>
      <td>0.781058</td>
      <td>0.322434</td>
      <td>1872000.0</td>
      <td>9.709403e+06</td>
      <td>NaN</td>
      <td>44.0</td>
      <td>34.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Hunger Eye Restaurant</td>
      <td>34.142904</td>
      <td>77.584443</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>POINT (77.58444 34.14290)</td>
      <td>0.350293</td>
      <td>0.050997</td>
      <td>1872000.0</td>
      <td>5.266601e+06</td>
      <td>NaN</td>
      <td>44.0</td>
      <td>15.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Shakya Aryan Restaurant</td>
      <td>34.100226</td>
      <td>77.597219</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>POINT (77.59722 34.10023)</td>
      <td>0.488419</td>
      <td>0.087652</td>
      <td>1872000.0</td>
      <td>6.280517e+06</td>
      <td>NaN</td>
      <td>44.0</td>
      <td>21.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Kathi junction leh</td>
      <td>34.099131</td>
      <td>77.596892</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>POINT (77.59689 34.09913)</td>
      <td>0.498720</td>
      <td>0.089826</td>
      <td>1872000.0</td>
      <td>6.359125e+06</td>
      <td>NaN</td>
      <td>44.0</td>
      <td>22.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Leh Manali Hwy Restaurant</td>
      <td>34.126578</td>
      <td>77.587576</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>POINT (77.58758 34.12658)</td>
      <td>0.677883</td>
      <td>0.206331</td>
      <td>1872000.0</td>
      <td>8.320973e+06</td>
      <td>NaN</td>
      <td>44.0</td>
      <td>30.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Snacks Point Gyasum (Tibetan camp#3)</td>
      <td>34.107448</td>
      <td>77.591596</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>POINT (77.59160 34.10745)</td>
      <td>0.753574</td>
      <td>0.282571</td>
      <td>1872000.0</td>
      <td>9.352604e+06</td>
      <td>NaN</td>
      <td>44.0</td>
      <td>33.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Mom's Square Restaurant</td>
      <td>34.125563</td>
      <td>77.607312</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>POINT (77.60731 34.12556)</td>
      <td>0.891463</td>
      <td>0.439543</td>
      <td>1872000.0</td>
      <td>1.118817e+07</td>
      <td>NaN</td>
      <td>44.0</td>
      <td>39.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Cafe Corner</td>
      <td>34.129785</td>
      <td>77.562230</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>POINT (77.56223 34.12978)</td>
      <td>0.874410</td>
      <td>0.419640</td>
      <td>1872000.0</td>
      <td>1.099428e+07</td>
      <td>NaN</td>
      <td>44.0</td>
      <td>38.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Shangrila Cafe and Restro</td>
      <td>34.113353</td>
      <td>77.557274</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>POINT (77.55727 34.11335)</td>
      <td>0.906086</td>
      <td>0.463619</td>
      <td>1872000.0</td>
      <td>1.136086e+07</td>
      <td>NaN</td>
      <td>44.0</td>
      <td>40.0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Sindh Bar</td>
      <td>34.147046</td>
      <td>77.569907</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>POINT (77.56991 34.14705)</td>
      <td>0.078737</td>
      <td>0.000000</td>
      <td>1872000.0</td>
      <td>3.808218e+06</td>
      <td>NaN</td>
      <td>44.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Lazeez Restaurant -Bijal</td>
      <td>34.025559</td>
      <td>77.680366</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>POINT (77.68037 34.02556)</td>
      <td>0.814300</td>
      <td>0.359673</td>
      <td>1872000.0</td>
      <td>1.014955e+07</td>
      <td>NaN</td>
      <td>44.0</td>
      <td>36.0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Dogra dhaba</td>
      <td>34.145925</td>
      <td>77.567921</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>POINT (77.56792 34.14593)</td>
      <td>0.076440</td>
      <td>0.000000</td>
      <td>1872000.0</td>
      <td>3.798204e+06</td>
      <td>NaN</td>
      <td>44.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Cafe Cloud</td>
      <td>34.045943</td>
      <td>77.670187</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>POINT (77.67019 34.04594)</td>
      <td>0.822955</td>
      <td>0.368635</td>
      <td>1872000.0</td>
      <td>1.026953e+07</td>
      <td>NaN</td>
      <td>44.0</td>
      <td>36.0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Indian Kitchen restaurant</td>
      <td>34.088048</td>
      <td>77.615142</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>POINT (77.61514 34.08805)</td>
      <td>0.774787</td>
      <td>0.314317</td>
      <td>1872000.0</td>
      <td>9.627996e+06</td>
      <td>NaN</td>
      <td>44.0</td>
      <td>34.0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Street Eats</td>
      <td>34.148837</td>
      <td>77.573982</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>POINT (77.57398 34.14884)</td>
      <td>0.226510</td>
      <td>0.014743</td>
      <td>1872000.0</td>
      <td>4.501494e+06</td>
      <td>NaN</td>
      <td>44.0</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Indian Kitchen restaurant</td>
      <td>34.084801</td>
      <td>77.608646</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>POINT (77.60865 34.08480)</td>
      <td>0.838858</td>
      <td>0.384591</td>
      <td>1872000.0</td>
      <td>1.049616e+07</td>
      <td>NaN</td>
      <td>44.0</td>
      <td>37.0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Punjabi Dhaba</td>
      <td>34.073759</td>
      <td>77.639326</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>POINT (77.63933 34.07376)</td>
      <td>0.734750</td>
      <td>0.251538</td>
      <td>1872000.0</td>
      <td>9.099053e+06</td>
      <td>NaN</td>
      <td>44.0</td>
      <td>32.0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Donsa Restaurant</td>
      <td>34.146281</td>
      <td>77.568783</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>POINT (77.56878 34.14628)</td>
      <td>0.045217</td>
      <td>0.000000</td>
      <td>1872000.0</td>
      <td>3.662100e+06</td>
      <td>NaN</td>
      <td>44.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Khangchain noodles</td>
      <td>34.041373</td>
      <td>77.655196</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>POINT (77.65520 34.04137)</td>
      <td>0.875577</td>
      <td>0.421035</td>
      <td>1872000.0</td>
      <td>1.100813e+07</td>
      <td>NaN</td>
      <td>44.0</td>
      <td>39.0</td>
    </tr>
    <tr>
      <th>21</th>
      <td>NamDruk Restaurant and Lounge</td>
      <td>34.021107</td>
      <td>77.683961</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>POINT (77.68396 34.02111)</td>
      <td>0.755674</td>
      <td>0.286051</td>
      <td>1872000.0</td>
      <td>9.379869e+06</td>
      <td>NaN</td>
      <td>44.0</td>
      <td>33.0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Chang La Queen Cafe</td>
      <td>34.076692</td>
      <td>77.626214</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>POINT (77.62621 34.07669)</td>
      <td>0.819772</td>
      <td>0.365288</td>
      <td>1872000.0</td>
      <td>1.022463e+07</td>
      <td>NaN</td>
      <td>44.0</td>
      <td>36.0</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Cafe Cloud</td>
      <td>34.045943</td>
      <td>77.670187</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>POINT (77.67019 34.04594)</td>
      <td>0.822955</td>
      <td>0.368635</td>
      <td>1872000.0</td>
      <td>1.026953e+07</td>
      <td>NaN</td>
      <td>44.0</td>
      <td>36.0</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Indian Kitchen restaurant</td>
      <td>34.084801</td>
      <td>77.608646</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>POINT (77.60865 34.08480)</td>
      <td>0.838858</td>
      <td>0.384591</td>
      <td>1872000.0</td>
      <td>1.049616e+07</td>
      <td>NaN</td>
      <td>44.0</td>
      <td>37.0</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Chamba Restaurant</td>
      <td>34.056664</td>
      <td>77.667471</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>POINT (77.66747 34.05666)</td>
      <td>0.753697</td>
      <td>0.282777</td>
      <td>1872000.0</td>
      <td>9.354200e+06</td>
      <td>NaN</td>
      <td>44.0</td>
      <td>33.0</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Lazeez Restaurant -Bijal</td>
      <td>34.035446</td>
      <td>77.673996</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>POINT (77.67400 34.03545)</td>
      <td>0.823179</td>
      <td>0.368867</td>
      <td>1872000.0</td>
      <td>1.027269e+07</td>
      <td>NaN</td>
      <td>44.0</td>
      <td>36.0</td>
    </tr>
    <tr>
      <th>27</th>
      <td>HOC SHEY</td>
      <td>34.070911</td>
      <td>77.645050</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>POINT (77.64505 34.07091)</td>
      <td>0.733371</td>
      <td>0.249523</td>
      <td>1872000.0</td>
      <td>9.079368e+06</td>
      <td>NaN</td>
      <td>44.0</td>
      <td>32.0</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Lazeez Restaurant -Bijal</td>
      <td>34.025559</td>
      <td>77.680366</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>POINT (77.68037 34.02556)</td>
      <td>0.814300</td>
      <td>0.359673</td>
      <td>1872000.0</td>
      <td>1.014955e+07</td>
      <td>NaN</td>
      <td>44.0</td>
      <td>36.0</td>
    </tr>
    <tr>
      <th>29</th>
      <td>CHAMBA Hotel &amp; Restaurant</td>
      <td>34.053181</td>
      <td>77.665392</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>POINT (77.66539 34.05318)</td>
      <td>0.714726</td>
      <td>0.228657</td>
      <td>1872000.0</td>
      <td>8.806683e+06</td>
      <td>NaN</td>
      <td>44.0</td>
      <td>31.0</td>
    </tr>
    <tr>
      <th>30</th>
      <td>Punjabi Dhaba</td>
      <td>34.073759</td>
      <td>77.639326</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>POINT (77.63933 34.07376)</td>
      <td>0.734750</td>
      <td>0.251538</td>
      <td>1872000.0</td>
      <td>9.099053e+06</td>
      <td>NaN</td>
      <td>44.0</td>
      <td>32.0</td>
    </tr>
    <tr>
      <th>31</th>
      <td>Bus Stand JK SRTC</td>
      <td>34.156756</td>
      <td>77.583231</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>POINT (77.58323 34.15676)</td>
      <td>0.750841</td>
      <td>0.277916</td>
      <td>1872000.0</td>
      <td>9.317133e+06</td>
      <td>NaN</td>
      <td>44.0</td>
      <td>33.0</td>
    </tr>
    <tr>
      <th>32</th>
      <td>Bus Stand JK SRTC</td>
      <td>34.156756</td>
      <td>77.583231</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>POINT (77.58323 34.15676)</td>
      <td>0.750841</td>
      <td>0.277916</td>
      <td>1872000.0</td>
      <td>9.317133e+06</td>
      <td>NaN</td>
      <td>44.0</td>
      <td>33.0</td>
    </tr>
    <tr>
      <th>33</th>
      <td>220kV POWERGRID Phyang Substation</td>
      <td>34.164948</td>
      <td>77.473473</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>POINT (77.47347 34.16495)</td>
      <td>0.910858</td>
      <td>0.474815</td>
      <td>1872000.0</td>
      <td>1.143246e+07</td>
      <td>NaN</td>
      <td>44.0</td>
      <td>40.0</td>
    </tr>
    <tr>
      <th>34</th>
      <td>220kV POWERGRID Phyang Substation</td>
      <td>34.164948</td>
      <td>77.473473</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>POINT (77.47347 34.16495)</td>
      <td>0.910858</td>
      <td>0.474815</td>
      <td>1872000.0</td>
      <td>1.143246e+07</td>
      <td>NaN</td>
      <td>44.0</td>
      <td>40.0</td>
    </tr>
    <tr>
      <th>35</th>
      <td>220kV POWERGRID Phyang Substation</td>
      <td>34.164948</td>
      <td>77.473473</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>POINT (77.47347 34.16495)</td>
      <td>0.910858</td>
      <td>0.474815</td>
      <td>1872000.0</td>
      <td>1.143246e+07</td>
      <td>NaN</td>
      <td>44.0</td>
      <td>40.0</td>
    </tr>
    <tr>
      <th>36</th>
      <td>DELHIVERY center leh</td>
      <td>34.148423</td>
      <td>77.557227</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>POINT (77.55723 34.14842)</td>
      <td>0.809968</td>
      <td>0.355187</td>
      <td>1872000.0</td>
      <td>1.009069e+07</td>
      <td>NaN</td>
      <td>44.0</td>
      <td>36.0</td>
    </tr>
    <tr>
      <th>37</th>
      <td>Otilia logistics</td>
      <td>34.147252</td>
      <td>77.581146</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>POINT (77.58115 34.14725)</td>
      <td>0.147414</td>
      <td>0.000000</td>
      <td>1872000.0</td>
      <td>4.107581e+06</td>
      <td>NaN</td>
      <td>44.0</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>38</th>
      <td>Tsao trading company</td>
      <td>34.132597</td>
      <td>77.588117</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>POINT (77.58812 34.13260)</td>
      <td>0.781039</td>
      <td>0.322409</td>
      <td>1872000.0</td>
      <td>9.709150e+06</td>
      <td>NaN</td>
      <td>44.0</td>
      <td>34.0</td>
    </tr>
    <tr>
      <th>39</th>
      <td>lamdon service reservior</td>
      <td>34.174654</td>
      <td>77.592274</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>POINT (77.59227 34.17465)</td>
      <td>0.905923</td>
      <td>0.463226</td>
      <td>1872000.0</td>
      <td>1.135843e+07</td>
      <td>NaN</td>
      <td>44.0</td>
      <td>40.0</td>
    </tr>
    <tr>
      <th>40</th>
      <td>DELHIVERY center leh</td>
      <td>34.148423</td>
      <td>77.557227</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>POINT (77.55723 34.14842)</td>
      <td>0.809968</td>
      <td>0.355187</td>
      <td>1872000.0</td>
      <td>1.009069e+07</td>
      <td>NaN</td>
      <td>44.0</td>
      <td>36.0</td>
    </tr>
    <tr>
      <th>41</th>
      <td>Otilia logistics</td>
      <td>34.147252</td>
      <td>77.581146</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>POINT (77.58115 34.14725)</td>
      <td>0.147414</td>
      <td>0.000000</td>
      <td>1872000.0</td>
      <td>4.107581e+06</td>
      <td>NaN</td>
      <td>44.0</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>42</th>
      <td>Tsao trading company</td>
      <td>34.132597</td>
      <td>77.588117</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>POINT (77.58812 34.13260)</td>
      <td>0.781039</td>
      <td>0.322409</td>
      <td>1872000.0</td>
      <td>9.709150e+06</td>
      <td>NaN</td>
      <td>44.0</td>
      <td>34.0</td>
    </tr>
    <tr>
      <th>43</th>
      <td>lamdon service reservior</td>
      <td>34.174654</td>
      <td>77.592274</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>POINT (77.59227 34.17465)</td>
      <td>0.905923</td>
      <td>0.463226</td>
      <td>1872000.0</td>
      <td>1.135843e+07</td>
      <td>NaN</td>
      <td>44.0</td>
      <td>40.0</td>
    </tr>
    <tr>
      <th>44</th>
      <td>Otilia logistics</td>
      <td>34.147252</td>
      <td>77.581146</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>POINT (77.58115 34.14725)</td>
      <td>0.147414</td>
      <td>0.000000</td>
      <td>1872000.0</td>
      <td>4.107581e+06</td>
      <td>NaN</td>
      <td>44.0</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>45</th>
      <td>lamdon service reservior</td>
      <td>34.174654</td>
      <td>77.592274</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>POINT (77.59227 34.17465)</td>
      <td>0.905923</td>
      <td>0.463226</td>
      <td>1872000.0</td>
      <td>1.135843e+07</td>
      <td>NaN</td>
      <td>44.0</td>
      <td>40.0</td>
    </tr>
    <tr>
      <th>46</th>
      <td>Tsao trading company</td>
      <td>34.132597</td>
      <td>77.588117</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>POINT (77.58812 34.13260)</td>
      <td>0.781039</td>
      <td>0.322409</td>
      <td>1872000.0</td>
      <td>9.709150e+06</td>
      <td>NaN</td>
      <td>44.0</td>
      <td>34.0</td>
    </tr>
    <tr>
      <th>47</th>
      <td>DELHIVERY center leh</td>
      <td>34.148423</td>
      <td>77.557227</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>POINT (77.55723 34.14842)</td>
      <td>0.809968</td>
      <td>0.355187</td>
      <td>1872000.0</td>
      <td>1.009069e+07</td>
      <td>NaN</td>
      <td>44.0</td>
      <td>36.0</td>
    </tr>
  </tbody>
</table>
</div>
