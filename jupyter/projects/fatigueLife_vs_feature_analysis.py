from additive.utility import *
from additive.features import *
from imports import *

columns = [x.split('\t') for x in """Surface Condition	specimen	Frequency (Hz)	Strain Amplitude (mm/mm)	Fatigue Life (cylces)	Reversal to failure (2Nf)
As-built	V02	5	0.004	45859	91718
As-built	V10	5	0.004	49177	98354
As-built	V04	7.5	0.003	91222	182444
As-built	V16	7.5	0.003	110013	220026
As-built	V08	7.5	0.003	136464	272928
As-built	V12	8	0.0025	192404	384808
As-built	V18	8	0.0025	259128	518256
As-built	V14	8	0.0025	320856	641712
As-built	V06	10	0.002	5000000	10000000
Half-polished	V07	5	0.004	50916	101832
Half-polished	V09	5	0.004	60992	121984
Half-polished	V15	7.5	0.003	132668	265336
Half-polished	V05	7.5	0.003	153540	307080
Half-polished	V03	7.5	0.003	163123	326246
Half-polished	V13	8	0.0025	287061	574122
Half-polished	V11	8	0.0025	291206	582412
Half-polished	V17	8	0.0025	395801	791602
Half-polished	V01	10	0.002	5000000	10000000""".split('\n')]
fatigue = pd.DataFrame(columns[1:], columns=columns[0])
#fatigue['Specimen ID'] = fatigue['Specimen ID']#.str.lower()
for c in ['Frequency (Hz)', 'Strain Amplitude (mm/mm)', 'Fatigue Life (cylces)', 'Reversal to failure (2Nf)']:
    fatigue[c] = pd.to_numeric(fatigue[c])
fatigue['ispolished'] = fatigue['Surface Condition'].str.contains('polish')