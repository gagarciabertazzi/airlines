import streamlit as st
import pandas as pd
import joblib
import sklearn
from streamlit_lottie import st_lottie
import requests
import numpy as np

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

model = joblib.load('rf.pkl')

lottie_airplane = load_lottieurl('https://assets4.lottiefiles.com/packages/lf20_jhu1lqdz.json')
st_lottie(lottie_airplane, speed=1, height=200, key="initial")
st.title('Airline Delayed Predictor')

st.write("Here's our first attempt at using data to create a table:")

# Input Variables
st.sidebar.title('Flight Delay Predictor')

# Input Variables
add_id = st.sidebar.number_input(label='Flight ID', min_value=1, max_value=539383, step=1)

airline_tuple = ('9E', 'AA', 'AS', 'B6', 'CO', 'DL', 'EV', 'F9', 'FL', 'HA', 'MQ', 'OH', 'OO', 'UA', 'US', 'WN', 'XE', 'YV')
add_airline = st.sidebar.selectbox("Airline", airline_tuple)

add_flight = st.sidebar.number_input('Flight', 1, 7814, 1) 

#AirportFrom	
airport_from_tuple = ('ABE', 'ABI', 'ABQ', 'ABR', 'ABY', 'ACT', 'ACV', 'ACY', 'ADK',
        'ADQ', 'AEX', 'AGS', 'ALB', 'AMA', 'ANC', 'ASE', 'ATL', 'ATW',
        'AUS', 'AVL', 'AVP', 'AZO', 'BDL', 'BET', 'BFL', 'BGM', 'BGR',
        'BHM', 'BIL', 'BIS', 'BKG', 'BLI', 'BMI', 'BNA', 'BOI', 'BOS',
        'BQK', 'BQN', 'BRO', 'BRW', 'BTM', 'BTR', 'BTV', 'BUF', 'BUR',
        'BWI', 'BZN', 'CAE', 'CAK', 'CDC', 'CDV', 'CEC', 'CHA', 'CHO',
        'CHS', 'CIC', 'CID', 'CLD', 'CLE', 'CLL', 'CLT', 'CMH', 'CMI',
        'CMX', 'COD', 'COS', 'COU', 'CPR', 'CRP', 'CRW', 'CSG', 'CVG',
        'CWA', 'CYS', 'DAB', 'DAL', 'DAY', 'DBQ', 'DCA', 'DEN', 'DFW',
        'DHN', 'DLH', 'DRO', 'DSM', 'DTW', 'EAU', 'ECP', 'EGE', 'EKO',
        'ELM', 'ELP', 'ERI', 'EUG', 'EVV', 'EWN', 'EWR', 'EYW', 'FAI',
        'FAR', 'FAT', 'FAY', 'FCA', 'FLG', 'FLL', 'FLO', 'FNT', 'FSD',
        'FSM', 'FWA', 'GCC', 'GEG', 'GFK', 'GGG', 'GJT', 'GNV', 'GPT',
        'GRB', 'GRK', 'GRR', 'GSO', 'GSP', 'GTF', 'GTR', 'GUC', 'GUM',
        'HDN', 'HLN', 'HNL', 'HOU', 'HPN', 'HRL', 'HSV', 'HTS', 'IAD',
        'IAH', 'ICT', 'IDA', 'ILM', 'IND', 'IPL', 'ISP', 'ITH', 'ITO',
        'IYK', 'JAC', 'JAN', 'JAX', 'JFK', 'JNU', 'KOA', 'KTN', 'LAN',
        'LAS', 'LAX', 'LBB', 'LCH', 'LEX', 'LFT', 'LGA', 'LGB', 'LIH',
        'LIT', 'LMT', 'LNK', 'LRD', 'LSE', 'LWB', 'LWS', 'LYH', 'MAF',
        'MBS', 'MCI', 'MCO', 'MDT', 'MDW', 'MEI', 'MEM', 'MFE', 'MFR',
        'MGM', 'MHK', 'MHT', 'MIA', 'MKE', 'MKG', 'MLB', 'MLI', 'MLU',
        'MMH', 'MOB', 'MOD', 'MOT', 'MQT', 'MRY', 'MSN', 'MSO', 'MSP',
        'MSY', 'MTJ', 'MYR', 'OAJ', 'OAK', 'OGG', 'OKC', 'OMA', 'OME',
        'ONT', 'ORD', 'ORF', 'OTH', 'OTZ', 'PAH', 'PBI', 'PDX', 'PHF',
        'PHL', 'PHX', 'PIA', 'PIE', 'PIH', 'PIT', 'PLN', 'PNS', 'PSC',
        'PSE', 'PSG', 'PSP', 'PVD', 'PWM', 'RAP', 'RDD', 'RDM', 'RDU',
        'RIC', 'RKS', 'RNO', 'ROA', 'ROC', 'ROW', 'RST', 'RSW', 'SAF',
        'SAN', 'SAT', 'SAV', 'SBA', 'SBN', 'SBP', 'SCC', 'SCE', 'SDF',
        'SEA', 'SFO', 'SGF', 'SGU', 'SHV', 'SIT', 'SJC', 'SJT', 'SJU',
        'SLC', 'SMF', 'SMX', 'SNA', 'SPI', 'SPS', 'SRQ', 'STL', 'STT',
        'STX', 'SUN', 'SWF', 'SYR', 'TEX', 'TLH', 'TOL', 'TPA', 'TRI',
        'TUL', 'TUS', 'TVC', 'TWF', 'TXK', 'TYR', 'TYS', 'UTM', 'VLD',
        'VPS', 'WRG', 'XNA', 'YAK', 'YUM')
add_airport_from = st.sidebar.selectbox("AirPort From", airport_from_tuple)


#AirportTo	
airport_to_tuple = ('ABE', 'ABI', 'ABQ', 'ABR', 'ABY', 'ACT', 'ACV', 'ACY', 'ADK',
        'ADQ', 'AEX', 'AGS', 'ALB', 'AMA', 'ANC', 'ASE', 'ATL', 'ATW',
        'AUS', 'AVL', 'AVP', 'AZO', 'BDL', 'BET', 'BFL', 'BGM', 'BGR',
        'BHM', 'BIL', 'BIS', 'BKG', 'BLI', 'BMI', 'BNA', 'BOI', 'BOS',
        'BQK', 'BQN', 'BRO', 'BRW', 'BTM', 'BTR', 'BTV', 'BUF', 'BUR',
        'BWI', 'BZN', 'CAE', 'CAK', 'CDC', 'CDV', 'CEC', 'CHA', 'CHO',
        'CHS', 'CIC', 'CID', 'CLD', 'CLE', 'CLL', 'CLT', 'CMH', 'CMI',
        'CMX', 'COD', 'COS', 'COU', 'CPR', 'CRP', 'CRW', 'CSG', 'CVG',
        'CWA', 'CYS', 'DAB', 'DAL', 'DAY', 'DBQ', 'DCA', 'DEN', 'DFW',
        'DHN', 'DLH', 'DRO', 'DSM', 'DTW', 'EAU', 'ECP', 'EGE', 'EKO',
        'ELM', 'ELP', 'ERI', 'EUG', 'EVV', 'EWN', 'EWR', 'EYW', 'FAI',
        'FAR', 'FAT', 'FAY', 'FCA', 'FLG', 'FLL', 'FLO', 'FNT', 'FSD',
        'FSM', 'FWA', 'GCC', 'GEG', 'GFK', 'GGG', 'GJT', 'GNV', 'GPT',
        'GRB', 'GRK', 'GRR', 'GSO', 'GSP', 'GTF', 'GTR', 'GUC', 'GUM',
        'HDN', 'HLN', 'HNL', 'HOU', 'HPN', 'HRL', 'HSV', 'HTS', 'IAD',
        'IAH', 'ICT', 'IDA', 'ILM', 'IND', 'IPL', 'ISP', 'ITH', 'ITO',
        'IYK', 'JAC', 'JAN', 'JAX', 'JFK', 'JNU', 'KOA', 'KTN', 'LAN',
        'LAS', 'LAX', 'LBB', 'LCH', 'LEX', 'LFT', 'LGA', 'LGB', 'LIH',
        'LIT', 'LMT', 'LNK', 'LRD', 'LSE', 'LWB', 'LWS', 'LYH', 'MAF',
        'MBS', 'MCI', 'MCO', 'MDT', 'MDW', 'MEI', 'MEM', 'MFE', 'MFR',
        'MGM', 'MHK', 'MHT', 'MIA', 'MKE', 'MKG', 'MLB', 'MLI', 'MLU',
        'MMH', 'MOB', 'MOD', 'MOT', 'MQT', 'MRY', 'MSN', 'MSO', 'MSP',
        'MSY', 'MTJ', 'MYR', 'OAJ', 'OAK', 'OGG', 'OKC', 'OMA', 'OME',
        'ONT', 'ORD', 'ORF', 'OTH', 'OTZ', 'PAH', 'PBI', 'PDX', 'PHF',
        'PHL', 'PHX', 'PIA', 'PIE', 'PIH', 'PIT', 'PLN', 'PNS', 'PSC',
        'PSE', 'PSG', 'PSP', 'PVD', 'PWM', 'RAP', 'RDD', 'RDM', 'RDU',
        'RIC', 'RKS', 'RNO', 'ROA', 'ROC', 'ROW', 'RST', 'RSW', 'SAF',
        'SAN', 'SAT', 'SAV', 'SBA', 'SBN', 'SBP', 'SCC', 'SCE', 'SDF',
        'SEA', 'SFO', 'SGF', 'SGU', 'SHV', 'SIT', 'SJC', 'SJT', 'SJU',
        'SLC', 'SMF', 'SMX', 'SNA', 'SPI', 'SPS', 'SRQ', 'STL', 'STT',
        'STX', 'SUN', 'SWF', 'SYR', 'TEX', 'TLH', 'TOL', 'TPA', 'TRI',
        'TUL', 'TUS', 'TVC', 'TWF', 'TXK', 'TYR', 'TYS', 'UTM', 'VLD',
        'VPS', 'WRG', 'XNA', 'YAK', 'YUM')
add_airport_to = st.sidebar.selectbox("AirPort To", airport_to_tuple)


#DayOfWeek
dow_tuple = ('Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday')
add_dow = st.sidebar.selectbox("Day of Week", dow_tuple)

#Time	
add_time = st.sidebar.slider('Flight time (min)', min_value=10, max_value=1439, value=10, step=1)

#Length	
add_lenght = st.sidebar.slider('Flight Length', min_value=0, max_value=655, value=30, step=1)

#Delay

# Using "with" notation
#delay_tuple = ("Delayed", "On-Time")
#with st.sidebar:
#    add_delay = st.radio("Flight has arrived?", delay_tuple)


airline = airline_tuple.index(add_airline)
airportfrom = airport_from_tuple.index(add_airport_from) 
airportto = airport_to_tuple.index(add_airport_to)
dayofweek = dow_tuple.index(add_dow) + 1
#delay = delay_tuple.index(add_delay)

#print(add_id)
#print(airline)
#print(add_flight)
#print(airportfrom)
#print(airportto)
#print(dayofweek)
#print(add_time)
#print(add_lenght)
#print(delay)

vel = add_lenght/add_time
acc = vel/add_time

data = np.array([[add_id, airline, add_flight, airportfrom, airportto, 
                  dayofweek, add_time, add_lenght, vel, acc]])
y = model.predict(data)

if y[0]:
  st.write('Not Delayed')
else:
  st.write('Delayed')