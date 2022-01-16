# SSDNet-State-Space-Decomposition-Neural-Network-for-Time-Series-Forecasting

## List of Implementations:

Sanyo: http://dkasolarcentre.com.au/source/alice-springs/dka-m4-b-phase

Hanergy: http://dkasolarcentre.com.au/source/alice-springs/dka-m16-b-phase

Solar: https://www.nrel.gov/grid/solar-power-data.html

Electricity: https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014

Exchange: https://github.com/laiguokun/multivariate-time-series-data/tree/master/exchange_rate



## To run:

1. Preprocess the data:
  
   ```bash
   python preprocess_Sanyo.py
   python preprocess_Hanergy.py
   python preprocess_solar.py
   python preprocess_elect.py
   python preprocess_exchange.py
   ```

2. Restore the saved model and make prediction:
   
   ```bash
   python train.py --dataset='Sanyo' --model-name='base_model_Sanyo' --restore-file='best'
   python train.py --dataset='Hanergy' --model-name='base_model_Hanergy' --restore-file='best'
   python train.py --dataset='Solar' --model-name='base_model_Solar' --restore-file='best'
   python train.py --dataset='elect' --model-name='base_model_elect' --restore-file='best'
   python train.py --dataset='exchange' --model-name='base_model_exchange' --restore-file='best'
   ```

3. Train the model:
  
   ```bash
   python train.py --dataset='Sanyo' --model-name='base_model_Sanyo' 
   python train.py --dataset='Hanergy' --model-name='base_model_Hanergy'
   python train.py --dataset='Solar' --model-name='base_model_Solar' 
   python train.py --dataset='elect' --model-name='base_model_elect' 
   python train.py --dataset='exchange' --model-name='base_model_exchange' 
   ```
