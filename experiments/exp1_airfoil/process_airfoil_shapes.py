from AF_NUFT import *

# Load data
data=pd.read_csv('../../data/airfoil_cnn/processed-data/airfoil_data.csv')
print('Read DataFrame.')

# Convert all airfoils in airfoil-data
directory='../../data/airfoil_cnn'    
i=0
start=time.time()
for root, dirs, files in os.walk(directory):
    for airfoil in dirs:
        if not os.path.exists('../../data/airfoil_cnn/'+airfoil+'/seligdatfile'):
            print('WARNING: Selig .dat file for '+airfoil+' not found. Airfoil not converted and saved.')
        else:
            # Get all unique AoA for airfoil
            af_data=data[data['Directory']==airfoil]     
            aoa_data=af_data['AoA'].unique()
            
            # Get NUFT of airfoil at all AoA
            for aoa in aoa_data:
                
                # Create save file name: <airfoil name>_aoa_<p for positive, n for negative>_<AoA in degrees, '.' replaced with '_'>
                aoa_str=str(aoa).replace('.','_')
                aoa_str='aoa_p_'+aoa_str
                aoa_str=aoa_str.replace('aoa_p_-', 'aoa_n_')
                save_file=airfoil+'_'+aoa_str+'.npy'
                
                if not os.path.exists('../../data/airfoil_cnn/processed-data/'+save_file):
                    convertAirfoil(airfoil, aoa, r=224, save_name=save_file)
                i=i+1
                
                if (i+1)%1000==0:
                    end=time.time()
                    print(str(i+1)+' airfoils processed! Time elapsed: '+str(end-start))
                    start=time.time()
                    
print('Processing complete!')
