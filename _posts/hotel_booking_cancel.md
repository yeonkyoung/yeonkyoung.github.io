호텔 예약이 취소되는 원인이 무엇인지 찾아봅시다.  
이 분석은 Dowhy 라이브러리 사이트의 Case Study내용을 발췌하였으며, Antonio, Almeida, Nunes(2019)의 호텔 예약 데이터 셋을 사용합니다.  
데이터는 github의 rfordatascience/tidytuseday 에서 구할 수 있습니다.  

호텔 예약이 취소되는 이유는 여러가지가 있을 수 있습니다.  

예를 들어,  
  - 1. 고객이 호텔이 제공하기 어려운 요청을 하고(ex. 호텔의 주차공간이 부족하고, 고객은 주차공간을 요청), 요청을 거절받은 고객이 예약을 취소할 수 있고,  
  혹은  
  - 2. 고객이 여행 계획을 취소했기 때문에 호텔예약을 취소했을 수 있습니다.  

1번과 같은 경우는, 호텔에서 추가 조치(다른 시설의 주차공간을 확보)를 취할 수 있는 반면, 2번과 같은 경우는 호텔이 취할 수 있는 조치가 없습니다.
어찌됐든, 우리는 예약취소를 유발하는 원인들을 보다 더 자세히 이해하는 것이 목표입니다.

이를 발견하는 가장 좋은 방법은 RCT(Randomized Control Trail)와 같은 실험을 하는 것입니다. 

주차 공간 제공이 호텔 예약 취소에 미치는 정량적 영향도를 알아보겠다면,  
고객을 두 개의 범주로 나눠, 한 그룹에는 주차공간을 할당하고, 나머지 한 그룹에는 주차공간을 할당하지 않습니다.  
그리고 각 그룹 간 호텔 예약 취소율을 비교하면 됩니다.  

물론, 저런 실험이 소문나면 호텔 장사는 다했다고 봐야죠.

과거 데이터와 가설만 있는 상황에서, 우리는 어떻게 답을 찾아야 할까요?  




```python
%reload_ext autoreload
%autoreload 2
```


```python
# Config dict to set the logging level
import logging.config
DEFAULT_LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'loggers': {
        '': {
            'level': 'INFO',
        },
    }
}

logging.config.dictConfig(DEFAULT_LOGGING)
# Disabling warnings output
import warnings
# !pip install sklearn
from sklearn.exceptions import DataConversionWarning, ConvergenceWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)
warnings.filterwarnings(action='ignore', category=UserWarning)

# !pip install dowhy

import dowhy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```


```python
# dataset = pd.read_csv('https://raw.githubusercontent.com/Sid-darthvader/DoWhy-The-Causal-Story-Behind-Hotel-Booking-Cancellations/master/hotel_bookings.csv')
dataset = pd.read_csv('hotel_bookings.csv')
dataset.head()
```




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
      <th>Unnamed: 0</th>
      <th>hotel</th>
      <th>is_canceled</th>
      <th>lead_time</th>
      <th>arrival_date_year</th>
      <th>arrival_date_month</th>
      <th>arrival_date_week_number</th>
      <th>arrival_date_day_of_month</th>
      <th>stays_in_weekend_nights</th>
      <th>stays_in_week_nights</th>
      <th>...</th>
      <th>deposit_type</th>
      <th>agent</th>
      <th>company</th>
      <th>days_in_waiting_list</th>
      <th>customer_type</th>
      <th>adr</th>
      <th>required_car_parking_spaces</th>
      <th>total_of_special_requests</th>
      <th>reservation_status</th>
      <th>reservation_status_date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>342</td>
      <td>2015</td>
      <td>July</td>
      <td>27</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>No Deposit</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>Transient</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>Check-Out</td>
      <td>2015-07-01</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>737</td>
      <td>2015</td>
      <td>July</td>
      <td>27</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>No Deposit</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>Transient</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>Check-Out</td>
      <td>2015-07-01</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>7</td>
      <td>2015</td>
      <td>July</td>
      <td>27</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>No Deposit</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>Transient</td>
      <td>75.0</td>
      <td>0</td>
      <td>0</td>
      <td>Check-Out</td>
      <td>2015-07-02</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>13</td>
      <td>2015</td>
      <td>July</td>
      <td>27</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>No Deposit</td>
      <td>304.0</td>
      <td>NaN</td>
      <td>0</td>
      <td>Transient</td>
      <td>75.0</td>
      <td>0</td>
      <td>0</td>
      <td>Check-Out</td>
      <td>2015-07-02</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>14</td>
      <td>2015</td>
      <td>July</td>
      <td>27</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>...</td>
      <td>No Deposit</td>
      <td>240.0</td>
      <td>NaN</td>
      <td>0</td>
      <td>Transient</td>
      <td>98.0</td>
      <td>0</td>
      <td>1</td>
      <td>Check-Out</td>
      <td>2015-07-03</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 33 columns</p>
</div>




```python
dataset.columns
```




    Index(['Unnamed: 0', 'hotel', 'is_canceled', 'lead_time', 'arrival_date_year',
           'arrival_date_month', 'arrival_date_week_number',
           'arrival_date_day_of_month', 'stays_in_weekend_nights',
           'stays_in_week_nights', 'adults', 'children', 'babies', 'meal',
           'country', 'market_segment', 'distribution_channel',
           'is_repeated_guest', 'previous_cancellations',
           'previous_bookings_not_canceled', 'reserved_room_type',
           'assigned_room_type', 'booking_changes', 'deposit_type', 'agent',
           'company', 'days_in_waiting_list', 'customer_type', 'adr',
           'required_car_parking_spaces', 'total_of_special_requests',
           'reservation_status', 'reservation_status_date'],
          dtype='object')




## Feature Engineering
이제 차원수를 줄이기 위해 의미있는 Feature들을 만들어봅시다.

**Total Stay** = **stays_in_weekend_nights** + **stays_in_weekend_nights**  
**Guests** = **adults** + **children** + **babies**  
**Different_room_assigned** = 예약과 다른 룸을 받았다면 1 아니라면 0 



```python
# Total stay in nights
dataset['total_stay'] = dataset['stays_in_week_nights']+dataset['stays_in_weekend_nights']

# Total number of guests
dataset['guests'] = dataset['adults']+dataset['children'] +dataset['babies']

# Creating the different_room_assigned feature
dataset['different_room_assigned']=0
slice_indices = dataset['reserved_room_type']!=dataset['assigned_room_type']
dataset.loc[slice_indices,'different_room_assigned']=1

# Deleting older features
dataset = dataset.drop(['stays_in_week_nights','stays_in_weekend_nights','adults','children','babies'
                        ,'reserved_room_type','assigned_room_type'],axis=1)
```

결측치가 많거나 Unique value가 많은 컬럼은 본 분석에서는 사용될 일이 적으니, 삭제를 하겠습니다.  
그리고 Country의 경우는, 가장 빈도가 높은 나라를 결측치에 대입하겠습니다.  
**distribution_channel** 도 **market_segemnt** 컬럼과 많이 중복되니 삭제를 하도록 하겠습니다.


```python
dataset.isnull().sum() # Country,Agent,Company contain 488,16340,112593 missing entries
dataset = dataset.drop(['agent','company'],axis=1)
# Replacing missing countries with most freqently occuring countries
dataset['country']= dataset['country'].fillna(dataset['country'].mode()[0])
```


```python
dataset = dataset.drop(['reservation_status','reservation_status_date','arrival_date_day_of_month'],axis=1)
dataset = dataset.drop(['arrival_date_year'],axis=1)
dataset = dataset.drop(['distribution_channel'], axis=1)
```


```python
# Replacing 1 by True and 0 by False for the experiment and outcome variables
dataset['different_room_assigned']= dataset['different_room_assigned'].replace(1,True)
dataset['different_room_assigned']= dataset['different_room_assigned'].replace(0,False)
dataset['is_canceled']= dataset['is_canceled'].replace(1,True)
dataset['is_canceled']= dataset['is_canceled'].replace(0,False)
dataset.dropna(inplace=True)
print(dataset.columns)
dataset.iloc[:, 5:20].head(100)
```

    Index(['Unnamed: 0', 'hotel', 'is_canceled', 'lead_time', 'arrival_date_month',
           'arrival_date_week_number', 'meal', 'country', 'market_segment',
           'is_repeated_guest', 'previous_cancellations',
           'previous_bookings_not_canceled', 'booking_changes', 'deposit_type',
           'days_in_waiting_list', 'customer_type', 'adr',
           'required_car_parking_spaces', 'total_of_special_requests',
           'total_stay', 'guests', 'different_room_assigned'],
          dtype='object')





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
      <th>arrival_date_week_number</th>
      <th>meal</th>
      <th>country</th>
      <th>market_segment</th>
      <th>is_repeated_guest</th>
      <th>previous_cancellations</th>
      <th>previous_bookings_not_canceled</th>
      <th>booking_changes</th>
      <th>deposit_type</th>
      <th>days_in_waiting_list</th>
      <th>customer_type</th>
      <th>adr</th>
      <th>required_car_parking_spaces</th>
      <th>total_of_special_requests</th>
      <th>total_stay</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>27</td>
      <td>BB</td>
      <td>PRT</td>
      <td>Direct</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>No Deposit</td>
      <td>0</td>
      <td>Transient</td>
      <td>0.00</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>27</td>
      <td>BB</td>
      <td>PRT</td>
      <td>Direct</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>No Deposit</td>
      <td>0</td>
      <td>Transient</td>
      <td>0.00</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>27</td>
      <td>BB</td>
      <td>GBR</td>
      <td>Direct</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No Deposit</td>
      <td>0</td>
      <td>Transient</td>
      <td>75.00</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>27</td>
      <td>BB</td>
      <td>GBR</td>
      <td>Corporate</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No Deposit</td>
      <td>0</td>
      <td>Transient</td>
      <td>75.00</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>27</td>
      <td>BB</td>
      <td>GBR</td>
      <td>Online TA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No Deposit</td>
      <td>0</td>
      <td>Transient</td>
      <td>98.00</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>95</th>
      <td>27</td>
      <td>BB</td>
      <td>PRT</td>
      <td>Online TA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No Deposit</td>
      <td>0</td>
      <td>Transient</td>
      <td>73.80</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>96</th>
      <td>27</td>
      <td>BB</td>
      <td>PRT</td>
      <td>Online TA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No Deposit</td>
      <td>0</td>
      <td>Transient</td>
      <td>117.00</td>
      <td>0</td>
      <td>1</td>
      <td>7</td>
    </tr>
    <tr>
      <th>97</th>
      <td>27</td>
      <td>HB</td>
      <td>ESP</td>
      <td>Offline TA/TO</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No Deposit</td>
      <td>0</td>
      <td>Transient</td>
      <td>196.54</td>
      <td>0</td>
      <td>1</td>
      <td>7</td>
    </tr>
    <tr>
      <th>98</th>
      <td>27</td>
      <td>BB</td>
      <td>PRT</td>
      <td>Online TA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No Deposit</td>
      <td>0</td>
      <td>Transient</td>
      <td>99.30</td>
      <td>1</td>
      <td>2</td>
      <td>7</td>
    </tr>
    <tr>
      <th>99</th>
      <td>27</td>
      <td>BB</td>
      <td>DEU</td>
      <td>Direct</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No Deposit</td>
      <td>0</td>
      <td>Transient</td>
      <td>90.95</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 15 columns</p>
</div>




```python
dataset = dataset[dataset.deposit_type=="No Deposit"]
dataset.groupby(['deposit_type','is_canceled']).count()
```




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
      <th></th>
      <th>Unnamed: 0</th>
      <th>hotel</th>
      <th>lead_time</th>
      <th>arrival_date_month</th>
      <th>arrival_date_week_number</th>
      <th>meal</th>
      <th>country</th>
      <th>market_segment</th>
      <th>is_repeated_guest</th>
      <th>previous_cancellations</th>
      <th>previous_bookings_not_canceled</th>
      <th>booking_changes</th>
      <th>days_in_waiting_list</th>
      <th>customer_type</th>
      <th>adr</th>
      <th>required_car_parking_spaces</th>
      <th>total_of_special_requests</th>
      <th>total_stay</th>
      <th>guests</th>
      <th>different_room_assigned</th>
    </tr>
    <tr>
      <th>deposit_type</th>
      <th>is_canceled</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">No Deposit</th>
      <th>False</th>
      <td>74947</td>
      <td>74947</td>
      <td>74947</td>
      <td>74947</td>
      <td>74947</td>
      <td>74947</td>
      <td>74947</td>
      <td>74947</td>
      <td>74947</td>
      <td>74947</td>
      <td>74947</td>
      <td>74947</td>
      <td>74947</td>
      <td>74947</td>
      <td>74947</td>
      <td>74947</td>
      <td>74947</td>
      <td>74947</td>
      <td>74947</td>
      <td>74947</td>
    </tr>
    <tr>
      <th>True</th>
      <td>29690</td>
      <td>29690</td>
      <td>29690</td>
      <td>29690</td>
      <td>29690</td>
      <td>29690</td>
      <td>29690</td>
      <td>29690</td>
      <td>29690</td>
      <td>29690</td>
      <td>29690</td>
      <td>29690</td>
      <td>29690</td>
      <td>29690</td>
      <td>29690</td>
      <td>29690</td>
      <td>29690</td>
      <td>29690</td>
      <td>29690</td>
      <td>29690</td>
    </tr>
  </tbody>
</table>
</div>




```python
dataset_copy = dataset.copy(deep=True)
```

### Calculating Expected Count  

가설을 하나 세워봅시다.
  - *고객은 예약과 다른 방을 배정받으면, 예약을 취소한다.* 

위의 가설에 해당하는 그룹과 그렇지 않은 그룹으로 데이터를 구분 후 *가설에 해당되는 그룹의 인원*을 계산해볼 수 있겠습니다.

**is_cancled**와 **different_room_assigned**가 매우 Imbalance하기 때문에,(different_room_assigned=0: 104,469개, different_room_assigned=1: 14,917개) 

1,000개의 관측치를 랜덤으로 샘플링 후,  

   - **different_room_assigned**변수와 **is_cancled**변수가 같은 값을 가지는 경우가 얼마나 있었는지  
     (i.e. case 1. "예약과 다른 방이 배정" & "예약 취소" 인 경우와 case 2. "예약과 동일한 방이 배정" & "예약 유지"인 경우가 얼마나 있었는지)  
     확인합니다.

그리고 이 프로세스(샘플링하고 갯수세기)를 10,000번 반복하면, *가설에 해당되는 그룹 인원* 기댓값를 계산할 수 있겠네요

계산해보면, *가설에 해당되는 그룹*의 Expected Count는 거의 50%에 가깝습니다.  
(i.e. 두 변수가 무작위로 동일한 값을 얻을 확률)  

풀어서 설명하면, 임의의 고객에게 예약한 방과 다른 방을 배정하면, 예약을 취소할 수도 있고 취소하지 않을 수도 있습니다.  

따라서, 통계적으로는 이 단계에서는 명확한 결론이 없습니다. 



```python
counts_sum=0
for i in range(1,10000):
        counts_i = 0
        rdf = dataset.sample(1000)
        counts_i = rdf[rdf["is_canceled"]== rdf["different_room_assigned"]].shape[0]
        # counts_i = rdf.loc[(rdf["is_canceled"]==1)&(rdf["different_room_assigned"]==1)].shape[0]
        counts_sum+= counts_i
counts_sum/10000
```




$\displaystyle 588.6111$



이제 예약변경 횟수가 0인 집단 중 *가설에 해당되는 그룹*의 Expected Count를 확인하겠습니다. 


```python
# Expected Count when there are no booking changes
counts_sum=0
for i in range(1,10000):
        counts_i = 0
        rdf = dataset[dataset["booking_changes"]==0].sample(1000)
        counts_i = rdf[rdf["is_canceled"]== rdf["different_room_assigned"]].shape[0]
        counts_sum+= counts_i
counts_sum/10000
```




$\displaystyle 572.6134$



두 번째 케이스로, 예약변경이 1회 이상인 집단 중 *가설에 해당되는 그룹*의 Expected Count를 확인하겠습니다.


```python
# Expected Count when there are booking changes = 66.4%
counts_sum=0
for i in range(1,10000):
        counts_i = 0
        rdf = dataset[dataset["booking_changes"]>0].sample(1000)
        counts_i = rdf[rdf["is_canceled"]== rdf["different_room_assigned"]].shape[0]
        counts_sum+= counts_i
counts_sum/10000
```




$\displaystyle 665.7959$



예약 변경횟수가 1보다 큰 경우의 Expected Count(약 600)가 예약 변경횟수가 0인 경우(약 500)보다 훨씬 큰 것을 확인할 수 있습니다.  

우리는 여기서 **Booking Changes** 컬럼이 Confounding variable(교란변수, X와 Y 양쪽에 영향을 미쳐 X, Y 간 인과관계의 크기를 왜곡함)임을 알 수 있습니다.  

하지만, **Booking Changes**가 유일한 Confounding Variable일까요?  

만약 컬럼들 중, 우리가 확인하지 못한 Confounding Variable이 있다면,  

우리는 이전과 같은 주장을 할 수 있을까요?

### Step-1. Create a Causal Graph

예측 모델과 관련된 사전 지식들을 Causal Inference Graph로 먼저 표현해봅시다.  

전체 그래프를 다 그릴 필요는 없으니 큰 걱정은 안하셔도 됩니다.

Causal Inference Graph로 표현할 가정들은 아래와 같습니다.  

* **Market Segment** 컬럼은 2개의 값을 가지고 있음. **TA**는 Travel Agent를 의미하고, **TO**는 Tour Operator임. Market Segment는 LeadTime(예약시점부터 체크인할 때까지의 시간)에 영향을 미칠 것임  
  TA, TO 상세 내용은 링크 참조: https://www.tenontours.com/the-difference-between-tour-operators-and-travel-agents/ 

* **Country**는 고객이 예약을 일찍할지 늦게할지와 고객이 어떤 식사를 좋아할지 판단하는데 도움이 될 것임.  

* **LeadTime**은 **Days in Waitlist**의 크기에 영향을 미칠 것임(예약을 늦게 한다면 남아있는 방이 적겠죠?)  

* **Days in Waitlist**, **Total Stay in nights**, **Guest**의 크기는 예약이 취소될지 유지될지에 영향을 미칠 겁니다  
(손님이 많고 숙박할 날짜가 길다면 다른 호텔을 구하기 쉽지 않겠죠)

* **Previous Booking Retentions**는 고객이 Repeated Guest인지 아닌지에 영향을 미칠겁니다. 그리고, 이 두 변수는 예약이 취소될지 아닐지에도 영향을 줄겁니다.  
(예를들어, 이전에 여러 번 예약을 유지한 고객은 다음번에도 예약을 유지할 가능성이 크고, 예약을 자주 취소했던 고객은 다음번에도 예약을 취소할 가능성이 크겠죠)

* **Booking Changes** 는 (앞에서 보셨다시피) 고객이 different room할지말지(=예약과 다른 방에 배정될지 아닐지)와 예약취소에도 영향을 미칠 겁니다.

* 마지막으로 **Booking Changes**가 우리가 알고자 하는 원인인자변수(다른 방을 배정)를 교란하는 *유일한 변수*일 개연성은 작습니다.(경험적으로)





```python
import pygraphviz
causal_graph = """digraph {
different_room_assigned[label="Different Room Assigned"];
is_canceled[label="Booking Cancelled"];
booking_changes[label="Booking Changes"];
previous_bookings_not_canceled[label="Previous Booking Retentions"];
days_in_waiting_list[label="Days in Waitlist"];
lead_time[label="Lead Time"];
market_segment[label="Market Segment"];
country[label="Country"];
U[label="Unobserved Confounders"];
is_repeated_guest;
total_stay;
guests;
meal;
hotel;
U->different_room_assigned; U->is_canceled;U->required_car_parking_spaces;
market_segment -> lead_time;
lead_time->is_canceled; country -> lead_time;
different_room_assigned -> is_canceled;
country->meal;
lead_time -> days_in_waiting_list;
days_in_waiting_list ->is_canceled;
previous_bookings_not_canceled -> is_canceled;
previous_bookings_not_canceled -> is_repeated_guest;
is_repeated_guest -> is_canceled;
total_stay -> is_canceled;
guests -> is_canceled;
booking_changes -> different_room_assigned; booking_changes -> is_canceled;
hotel -> is_canceled;
required_car_parking_spaces -> is_canceled;
total_of_special_requests -> is_canceled;
country->{hotel, required_car_parking_spaces,total_of_special_requests,is_canceled};
market_segment->{hotel, required_car_parking_spaces,total_of_special_requests,is_canceled};
}"""
```

Treatment는 고객이 예약을 할 때 선택한 방을 배정받았는지(**different_room_assigned**)입니다.  

Outcome은 예약이 취소될지 아닐지(**is_cancled**) 입니다.

Common Cause는 Treatment와 Outcome 둘 모두에 영향을 미치는 변수입니다. **Booking Changes**와 **Unobserved Confounders**(우리가 확인하지 못한 교란변수) 2개가 Common Cause에 해당됩니다.  

만약 우리가 그래프를 명시적으로 지정하지 않는다면(추천하지 않습니다!), 교란변수들은 파라메터로 사용됩니다.




```python
model= dowhy.CausalModel(
        data = dataset,
        graph=causal_graph.replace("\n", " "),
        treatment='different_room_assigned',
        outcome='is_canceled')
model.view_model()
from IPython.display import Image, display
display(Image(filename="causal_model.png"))
```


    
![png](hotel_booking_cancel_files/hotel_booking_cancel_25_0.png)
    


## Step2. Identify the Causal Effect

Treatment변수의 변화가 Outcome변수의 변화만 이끌어낸다면 우리는 Treatment변수가 Outcome변수에 영향을 끼쳤다고 말할수 있습니다.  

이번 step에서는 영향인자를 식별해보도록 하겠습니다.


```python
import statsmodels
model= dowhy.CausalModel(
        data = dataset,
        graph=causal_graph.replace("\n", " "),
        treatment="different_room_assigned",
        outcome='is_canceled')
#Identify the causal effect
identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
print(identified_estimand)
```

    Estimand type: nonparametric-ate
    
    ### Estimand : 1
    Estimand name: backdoor
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|hotel,days_in_waiting_list,
    d[different_room_assigned]                                                    
    
                                                                                  
    booking_changes,market_segment,previous_bookings_not_canceled,meal,guests,coun
                                                                                  
    
                                                                                  
    try,total_of_special_requests,required_car_parking_spaces,lead_time,is_repeate
                                                                                  
    
                        
    d_guest,total_stay))
                        
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,hotel,days_in_waiting_list,booking_changes,market_segment,previous_bookings_not_canceled,meal,guests,country,total_of_special_requests,required_car_parking_spaces,lead_time,is_repeated_guest,total_stay,U) = P(is_canceled|different_room_assigned,hotel,days_in_waiting_list,booking_changes,market_segment,previous_bookings_not_canceled,meal,guests,country,total_of_special_requests,required_car_parking_spaces,lead_time,is_repeated_guest,total_stay)
    
    ### Estimand : 2
    Estimand name: iv
    No such variable found!
    
    ### Estimand : 3
    Estimand name: frontdoor
    No such variable found!
    


## Step3. Estimate the identified estimand


```python
estimate = model.estimate_effect(identified_estimand,
                                 method_name="backdoor.propensity_score_stratification",target_units="ate")
# ATE = Average Treatment Effect
# ATT = Average Treatment Effect on Treated (i.e. those who were assigned a different room)
# ATC = Average Treatment Effect on Control (i.e. those who were not assigned a different room)
print(estimate)
```

    *** Causal Estimate ***
    
    ## Identified estimand
    Estimand type: nonparametric-ate
    
    ### Estimand : 1
    Estimand name: backdoor
    Estimand expression:
                d                                                                 
    ──────────────────────────(Expectation(is_canceled|hotel,days_in_waiting_list,
    d[different_room_assigned]                                                    
    
                                                                                  
    booking_changes,market_segment,previous_bookings_not_canceled,meal,guests,coun
                                                                                  
    
                                                                                  
    try,total_of_special_requests,required_car_parking_spaces,lead_time,is_repeate
                                                                                  
    
                        
    d_guest,total_stay))
                        
    Estimand assumption 1, Unconfoundedness: If U→{different_room_assigned} and U→is_canceled then P(is_canceled|different_room_assigned,hotel,days_in_waiting_list,booking_changes,market_segment,previous_bookings_not_canceled,meal,guests,country,total_of_special_requests,required_car_parking_spaces,lead_time,is_repeated_guest,total_stay,U) = P(is_canceled|different_room_assigned,hotel,days_in_waiting_list,booking_changes,market_segment,previous_bookings_not_canceled,meal,guests,country,total_of_special_requests,required_car_parking_spaces,lead_time,is_repeated_guest,total_stay)
    
    ## Realized estimand
    b: is_canceled~different_room_assigned+hotel+days_in_waiting_list+booking_changes+market_segment+previous_bookings_not_canceled+meal+guests+country+total_of_special_requests+required_car_parking_spaces+lead_time+is_repeated_guest+total_stay
    Target units: ate
    
    ## Estimate
    Mean value: -0.2509265086102207
    


상당히 재밌는 결과가 나왔습니다. 라이브러리가 계산한 영향도를 보면, 예약과 다른 방이 배정됐을 때(**different_room_assign** = 1) , 예약이 취소될 가능성이 더 적을 것이라고 하네요.  

여기서 한번 더 생각을 해보면...이게 올바른 Causal Effect가 맞는 걸까요??  

예약된 객실을 사용할 수 없고, 다른 객실이 배정하는 것이 고객에게 긍정적인 영향을 미칠 수 있을까요?  
  
<br/>
다른 매커니즘이 있을 수도 있습니다. 

예약과 다른 객실을 배정하는 건 체크인 할 때만 발생하고, 이는 고객이 이미 호텔에 도착했다는 뜻이니, 예약을 취소할 가능성이 낮다고 볼 수 있을 것 같네요.  

이 매커니즘이 맞다면, 우리가 가정한 그래프에는 예약과 다른 객실이 "언제" 발생하는 지에 대한 정보가 없습니다.  

예약과 다른 객실이 배정되는 이벤트가 "언제" 발생하는 지 알 수 있다면, 분석을 개선하는데 도움이 될 수 있을 겁니다.

<br/>   
앞서 연관 분석에서 is_cancled와 different_room_assign 사이에 양의 상관관계가 있음을 확인 했지만,  

DoWhy라이브러리를 이용해 인과관계를 추정하면 다른 결과가 나옵니다. 

이는 호텔이 "예약과 다른 객실을 배정하는 행위"의 횟수를 줄이는 결정이 호텔에게 비생산적일 수 있음을 의미합니다.



## Step4. Refute result

인과 자체는 데이터 자체에서 나오는 것이 아닙니다.  데이터는 단순히 통계적 추정에 사용됩니다.  

다시말해 우리의 가설("고객은 예약과 다른 방을 배정받으면, 예약을 취소한다.")이 옳은지 여부를 확인하는 것이 중요합니다.

만약 또다른 common cause가 있다면 어떻게 될까요?  Treatment 변수의 영향도가 플라시보 효과에 의한 거라면 어떻게 될까요?

### Method-1  

**Random Common Cause:** 무작위로 추출한 공변량을 데이터에 추가하고, 분석을 다시 실행하여, 인과 추정치(estimand effect)가 변하는지 여부를 확인합니다.

우리의 가정이 옳았다면, 인과추정치(estimand effect)가 크게 변하지 않아야 합니다.


```python
refute1_results=model.refute_estimate(identified_estimand, estimate,
        method_name="random_common_cause")
print(refute1_results)
```

    Refute: Add a Random Common Cause
    Estimated effect:-0.2509265086102207
    New effect:-0.24891037769504973
    


### Method-2  

**Placebo Treatment Refuter:** 임의의 공변량을 Treatment변수에 할당하고 분석을 다시 실행합니다. 

우리의 가정이 옳았다면, 새로운 추정치는 0이 되어야 합니다.


```python
refute2_results=model.refute_estimate(identified_estimand, estimate,
        method_name="placebo_treatment_refuter")
print(refute2_results)
```

    Refute: Use a Placebo Treatment
    Estimated effect:-0.2509265086102207
    New effect:0.0003681316319065167
    p value:0.43
    


### Method-3  

**Data Subset Refuter:** 데이터 부분집합을 생성하고(cross-validation하듯이), 부분집합 별로 분석을 수행하면서, 추정치가 얼마나 변하는 지 확인합니다.

우리의 가정이 옳았다면, 추정치는 크게 변하지 않아야 합니다.


```python
refute3_results=model.refute_estimate(identified_estimand, estimate,
        method_name="data_subset_refuter")
print(refute3_results)
```

    Refute: Use a subset of data
    Estimated effect:-0.2509265086102207
    New effect:-0.24911870944918946
    p value:0.19
    


총 3가지 방법을 이용해 Refutation test(반박 시험)을 수행했습니다. 이 테스트는 정확성을 증명하지는 않지만, 추정치에 대한 신뢰도가 더 올라갔습니다.
