from airfoil_NUFT import *
import time

print('Now processing airfoil shapes...')

# Convert all airfoils in airfoil-data
directory='data'    
i=0
start=time.time()
for root, dirs, files in os.walk(directory):
    for airfoil in dirs:
        if not os.path.exists('data/'+airfoil+'/seligdatfile'):
            print('WARNING: Selig .dat file for '+airfoil+' not found. Airfoil not converted and saved.')
        else:                   
            # Create save file name
            save_file='processed_data/'+airfoil+'.pt'
            if not os.path.exists(save_file):
                airfoil_phys(airfoil, res=(224,224), device='cpu', save_name=save_file, grad=False)
            i=i+1

            if (i+1)%100==0:
                end=time.time()
                print(str(i+1)+' airfoils processed! Time elapsed: '+str(end-start))
                start=time.time()
                    
print('Processing complete!')
