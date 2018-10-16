from AF_NUFT import *

# Convert an example
name='s1210'
saved_af=convertAirfoil(name, 17.0, r=224, save_name='s1210_aoa_p_17_0')
savefile='s1210_aoa_p_17_0.png'
plotAirfoil(saved_af, name).savefig(savefile)
print('Plotted airfoil saved to '+savefile)
