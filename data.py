'''
smart_1 Read Error Rate (want low)
smart_2 Throughput performance (want high)
smart_3 Spin up time (want low)
smart_4 Start-stop count
! smart_5 Reallocated sectors count (want low)
smart_7 Seek error rate (vendor specific, value not comparable between vendors)
smart_8 Seek time performance (want high)
smart_9 Power on hours
! smart_10 Spin retry count (want low)
smart_11 Recalibration retries (want low)
smart_12 Power cycle count
smart_13 Soft read error rate (uncorrected read errors, want low)
smart_15 unknown
smart_22 Current helium level (want high)
smart_183 SATA downshift error rate (want low)
! smart_184 End-to-end error (want low)
! smart_187 Count of errors that couldn't be recovered (want low)
! smart_188 Command timeout (aborted operations due to timeout) (should be zero)
smart_189 High fly writes (want low)
smart_190 Temp diff (100 minus temperature in celsius)
smart_191 G sense error rate (errors from external shock) (want low)
smart_192 Emergency retract cycle count (want low)
smart_193 Load cycle count (want low)
smart_194 Temperature (want low)
smart_195 Hardware ECC recovered (vendor specific)
! smart_196 Reallocation event count (want low)
! smart_197 Current pending sector count (want low)
! smart_198 Uncorrectable sector count (want low)
smart_199 UltraDMA CRC error count (want low)
smart_200 Multi-zone error rate (want low)
! smart_201 Soft read error rate (want low)
smart_220 Disk shift (want low)
smart_222 Loaded hours (time spent under load)
smart_223 Load/unload retry count
smart_224 Load friction (want low)
smart_225 Load/unload cycles (want low)
smart_226 Load in time
smart_240 Head flying hours
smart_241 Total LBAs written
smart_242 Total LBAs read
smart_250 Read error retry rate (want low)
smart_251 Minimum spares remaining
smart_252 Newly added bad flask block
smart_254 Free fall events (want low)
smart_255 unknown

'''

import numpy as np
import pandas as pd

DATASET_FILENAME = 'harddrive.csv'
TESTING = True if __name__ == "__main__" else False

MANUFACTURERS = {'HG':'HGST',
                 'Hi':'Hitachi',
                 'SA':'Samsung',
                 'ST':'Seagate',
                 'TO':'Toshiba',
                 'WD':'Western Digital'}

KNOWN_INDICATIVE_COLUMNS = ['smart_5_raw','smart_10_raw','smart_184_raw','smart_187_raw',
                            'smart_188_raw','smart_196_raw','smart_197_raw','smart_198_raw',
                            'smart_201_raw']


def get_data(filename=DATASET_FILENAME):
    df = pd.read_csv(filename,parse_dates=['date'],nrows=None if TESTING else None)
    print('read data, processing...')

    #adding manufacturer info to data
    conditions = [df['model'].str[:2] == mfr for mfr in sorted(MANUFACTURERS.keys())]
    outputs  = [MANUFACTURERS[key] for key in sorted(MANUFACTURERS.keys())]
    res = np.select(conditions,outputs,'unknown')
    df['manufacturer'] = pd.Series(res)

    #showing summed failures by manufacturer
    grouped = df.groupby(['manufacturer'])#,'serial_number'])
    print(grouped.failure.sum())

    #adding failure information per serial number
    grouped = df.groupby(['serial_number'])
    failing_serial_numbers = df.loc[df['failure'] == 1]['serial_number']
    df['will_fail'] = df['serial_number'].isin(failing_serial_numbers)
    print(len(df.loc[df['will_fail']==True]))

    # remove data from serial numbers with non-unique dates
    # (possibly not necessary)
    #g.filter(lambda x: x.date.is_unique) #remove

    reduced_df = df[['will_fail','model','manufacturer'] + KNOWN_INDICATIVE_COLUMNS]
    reduced_df = reduced_df.fillna(-0.5) # give numeric value to NaN

    return reduced_df


if __name__ == "__main__":
    get_data()