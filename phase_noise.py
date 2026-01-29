import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

def bubbleSort(arr): 
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j][0] > arr[j+1][0] :
                arr[j], arr[j+1] = arr[j+1], arr[j]

def readData(file):
    f = open(file, "r")
    fulldata = f.read()
    f.close()
    linedata = fulldata.split("\n")
    pointlist = []
    for line in linedata:
        if line != "":
            try:
                line = line.replace("\t",",")
                line = line.replace(" ","")
                pointstring = line.split(",")
                point = [float(pointstring[0]),float(pointstring[1])]
                pointlist.append(point)
            except:
                print("Error reading string: {} in file {}".format(line,file[1]));
    return pointlist

# def generateFitEnvelope(func, range,N, td, h0, fg, sg, hg, nf):
# 	pointlist = []
# 	for f in np.linspace(-range, range, num=2*N + 1):
# 		# pointlist.append((f, 4*np.sin(np.pi*f*td)**2))
# 		if f != 0:
# 			pointlist.append((f, func(f, td, h0, fg, sg, hg, nf)))#test_Si(f,td, h0, fg, sg, hg)))
# 	return pointlist

def generateFitEnvelope(func, range, N, p0, theoretical):
    pointlist = []
    for f in np.linspace(-range, range, num=2*N + 1):
        if f != 0:
            pointlist.append((f, func(f, *p0dict_to_p0array(p0, theoretical))))
    return pointlist

def makeRelHetData(pointlist):
    datalist = []
    f_max = 0
    p_max = -200
    for point in pointlist:
        if point[1] > p_max:
            f_max = point[0]
            p_max = point[1]
    for point in pointlist:
            datalist.append([point[0] - f_max, point[1] - p_max])
    return datalist

def plotData(data, logx=True, logy=True):
    fdata = []
    vdata = []
    cdata = []
    for d in data:
        if len(d) == 3:
            cdata.append((d))
        elif not (d == ("","")):
            cdata.append((d[0], d[1], False))
    for i in range(0,len(cdata)):
        fdata.append([])
        vdata.append([])
        for point in cdata[i][0]:
            fdata[i].append(point[0])
            vdata[i].append(point[1])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(0,len(fdata)):
        j = len(fdata) - i - 1
        if cdata[j][2] == True:
            fmt = 'o'
        else:
            fmt = '-'
        ax.plot(np.array(fdata[j])/1e6,vdata[j], fmt, label = cdata[j][1], linewidth = 2.5, markersize = 1)		# TODO Ensure Hz or MHz
        if logx:
            ax.set_xscale('log')
        if logy:
            ax.set_yscale('log')
    # ax.xaxis.label.set_size(20)
    # ax.yaxis.label.set_size(20)



    # POWER SPECTRUM DENSITY GRAPH SETTINGS
    plt.xlabel(xaxislabel, fontsize = 20)
    plt.ylabel(yaxislabel, fontsize = 20)
    ax.set_yscale('log')
    ax.set_xlim(-0.75, 0.75)	# ZOOMED IN
    plt.title(plotlabel, fontsize = 28)
    plt.grid(axis="both")
    plt.tick_params(labelsize = 15)
    plt.legend(loc='upper right', fontsize = 16)

    fig.set_size_inches(12, 6.8)

    plt.show()

    # fig.savefig('638nm_PSD_ZOOMED.svg', bbox_inches='tight') #, dpi=300)

    # NOTE: If graph isn't working, ensure in ax.plot that data is divided by 1e6


    # # FREQUENCY NOISE GRAPH SETTINGS
    # ax.set_xlim(1e3, 1e6)
    # ax.set_ylim(0.5*1e0, 2*1e4)
    # ax.set_xscale('log')
    # ax.set_yscale('log')
    # plt.xlabel(xaxislabel, fontsize = 20)
    # plt.ylabel(yaxislabel, fontsize = 20)
    # plt.title(plotlabel, fontsize = 28)
    # plt.grid(axis="both")
    # plt.tick_params(labelsize = 15)
    # # plt.legend(loc='upper left', fontsize = 16)
    # fig.set_size_inches(7, 4.8)
    # # Annotating points
    # points_black = [
    # 	[0.43, 0.42, "1"],
    # 	[0.525, 0.45, "2"],
    # 	[0.565, 0.45, "3"],
    # 	[0.69, 0.74, "4"],
    # 	[0.79, 0.845, "5"],
    # ]
    # points_red = [
    # 	[0.87, 0.6, "6"],
    # 	[0.9, 0.67, "7"],
    # 	[0.82, 0.32, "8"]
    # ]
    # for p in points_black:
    # 	ax.text(p[0], p[1], p[2], fontsize=15, color='black', transform=ax.transAxes,
    # 		ha='center')    #0:x, 1:y, 2:text
    # for p in points_red:
    # 	ax.text(p[0], p[1], p[2], fontsize=15, color='red', transform=ax.transAxes,
    # 		ha='center')    #0:x, 1:y, 2:text


    # plt.show()
    # fig.savefig('638nm_self-het-removed_frequency_noise.svg', bbox_inches='tight')

    return


def dBToR(pointdata):
    datalist = []
    for point in pointdata:
        datalist.append((point[0],10**(point[1]/10)))
    return datalist

def RTodB(pointdata):
    datalist = []
    for point in pointdata:
        datalist.append((point[0],np.log10(point[1])*10))
    return datalist

def simpleGainTransfer(pointdata, gain=1):
    datalist = []
    for point in pointdata:
        datalist.append((point[0],point[1]*gain))
    return datalist

def normalizePower(pointdata, BW=300):
    datalist = []
    tot = 0
    last_freq = pointdata[0][0]
    for point in pointdata:
        tot += point[1]*(point[0] - last_freq)
        last_freq = point[0]
    for point in pointdata:
        datalist.append((point[0],point[1]/tot))

    return datalist

def filterEnvelope(pointdata, td= 5.445*10**(-5)):
    datalist = []
    for point in pointdata:
        if 4*np.sin(np.pi*point[0]*td)**2 > 0.3:
            datalist.append((point[0],point[1]/(4*np.sin(np.pi*point[0]*td)**2)))
    return datalist

def phaseToFreqNoise(pointdata):
    datalist = []
    for point in pointdata:
        datalist.append((point[0],point[1]*point[0]**2))
    return datalist

def white_Si(f, td, h0):
    return (2*h0/(f**2+(2*np.pi*h0)**2))*(1-(np.cos(2*np.pi*f*td)-(2/f)*np.pi*h0*np.sin(2*np.pi*f*td))*np.e**(-4*np.pi**2*h0*td))
def white_Sphi(f, h0):
    return h0/f**2
def white_Sdv(f, h0):
    return h0

def bump_Si(f, td, fg, sg, hg):
    return (4*hg*np.sin(np.pi*f*td)**2)/(fg**2)*(np.e**(-(f-fg)**2/(2*sg**2))+np.e**(-(f+fg)**2/(2*sg**2)))
def bump_Sphi(f, fg, sg, hg):
    return  (hg/f**2)*(np.e**(-(f-fg)**2/(2*sg**2))+np.e**(-(f+fg)**2/(2*sg**2)))
def bump_Sdv(f, fg, sg, hg):
    return (hg)*(np.e**(-(f-fg)**2/(2*sg**2))+np.e**(-(f+fg)**2/(2*sg**2)))

def si_peak(f, f0, sp, sig):
    alpha = 5/2
    return sp*sig**(2*alpha - 1)/((f-f0)**2 + np.pi**2*sig**2)**alpha

def test_Si_log(f, td, nf, h0, *bumps):
    white_noise = white_Si(f, td, h0)
    bump_noise = 0
    for i in range(len(bumps)):
        if i % 3 == 0:
            fg = bumps[i]
            sg = bumps[i+1]
            hg = bumps[i+2]
            bump_noise += bump_Si(f, td, fg, sg, hg)
    return 10*np.log10(white_noise + bump_noise + nf)
def test_Si(f, td, nf, h0, *bumps):
    white_noise = white_Si(f, td, h0)
    bump_noise = 0
    for i in range(len(bumps)):
        if i % 3 == 0:
            fg = bumps[i]
            sg = bumps[i+1]
            hg = bumps[i+2]
            bump_noise += bump_Si(f, td, fg, sg, hg)
    return white_noise + bump_noise + nf
def test_Sphi(f, h0, *bumps):
    white_noise = white_Sphi(f, h0)
    bump_noise = 0
    for i in range(len(bumps)):
        if i % 3 == 0:
            fg = bumps[i]
            sg = bumps[i+1]
            hg = bumps[i+2]
            bump_noise += bump_Sphi(f, fg, sg, hg)
    return white_noise + bump_noise
def test_Sdv(f, h0, *bumps):
    white_noise = white_Sdv(f, h0)
    bump_noise = 0
    for i in range(len(bumps)):
        if i % 3 == 0:
            fg = bumps[i]
            sg = bumps[i+1]
            hg = bumps[i+2]
            bump_noise += bump_Sdv(f, fg, sg, hg)
    return white_noise + bump_noise

p0dict_sample = {
    'td' : 5.16732481e-05,
    'noise_floor': 2.7e-11,
    'h0': 2.99502710e-01,
    'bumps': [
    {'f': 81215.87188822053, 's': 22072.18757101612, 'h': 2.1938195400327127}]
}
def p0dict_to_p0array(p0dict, theoretical = False):
    p0array = []
    if not theoretical:
        p0array.append(p0dict['td'])
        p0array.append(p0dict['noise_floor'])
    p0array.append(p0dict['h0'])
    for bump in p0dict['bumps']:
        p0array.append(bump['f'])
        p0array.append(bump['s'])
        p0array.append(bump['h'])
    return p0array

def p0array_to_p0dict(p0array, theoretical = False):
    l = len(p0array)
    p0dict = {}
    if not theoretical:
        p0dict['td'] = p0array[0]
        p0dict['noise_floor'] = p0array[1]
        i = 2
    else:
        p0dict['td'] = 0
        p0dict['noise_floor'] = 0
        i = 0
    p0dict['h0'] = p0array[i]
    i += 1
    p0dict['bumps'] = []
    while i < l:
        p0dict['bumps'].append({'f': p0array[i], 's': p0array[i + 1], 'h': p0array[i + 2]})
        i += 3
    return p0dict

def print_p0dict(p0dict, desc=""):
    print("Noise fit of " + desc)
    print("td: {} ms".format(1000*p0dict['td']))
    print("noise floor: {} = {} dB".format(p0dict['noise_floor'], np.log10(p0dict['noise_floor'])*10))
    print("white noise amp, h0: {} Hz^2/Hz".format(p0dict['h0']))
    bumps = p0dict['bumps']
    n = len(bumps)
    for i in range(n):
        for j in range(0, n-i-1):
            if bumps[j]['f'] > bumps[j+1]['f'] :
                bumps[j], bumps[j+1] = bumps[j+1], bumps[j]
    for i, bump in enumerate(bumps):
        print("{}: ----- \n\tf: {} kHz, \n\ts: {} Hz, \n\th: {} Hz^2/Hz,".format(i+1, bump['f']/1000,bump['s'],bump['h']))
    print("--------------------------------")
    print("Copiable data:")
    print(p0dict)

def fitNoise(pointdata, func, p0, theoretical = False, LF_cutoff = 1000, verbose=False):
    xdata = []
    ydata = []
    for p in pointdata:
        if (p[0] != 0 and np.abs(p[0]) > LF_cutoff):
            xdata.append(p[0])
            ydata.append(p[1])
    popt, pcov, infodict, mesg, ier = opt.curve_fit(func, xdata, ydata, p0dict_to_p0array(p0,theoretical), full_output=True)
    if verbose:
        print(mesg)
    return p0array_to_p0dict(popt,theoretical)

def fitPeak(pointdata, bw=20e3, verbose=False):
    xdata = []
    ydata = []
    for p in pointdata:
        if (p[0] != 0 and np.abs(p[0]) < bw):
            xdata.append(p[0])
            ydata.append(p[1])
    popt, _  = opt.curve_fit(si_peak, xdata, ydata, [0, 1, 300])
    if verbose:
        print(f"sp: {popt[1]}, sig: {popt[2]}")
    return popt

def procData(source, enable, 
        desc = "",
        range = 0.5e6,
        N = int(1e6/600),
        func=test_Si_log,
        p0=p0dict_sample,
        verbose=False,
        theoretical = False
    ):
    if enable == True:
        if(source == 'envelope'):
            pointdata = generateFitEnvelope(func, range, N, p0, theoretical)
            # pointdata = dBToR(pointdata)
            return [(pointdata, desc),]
        data = readData(source)
        pointdata = makeRelHetData(data)
        pointdata = dBToR(pointdata)
        pointdata = normalizePower(pointdata)
        peakopt = fitPeak(pointdata, bw=20e3, verbose=verbose)
        pnorm = peakopt[1]*4/(3*np.pi**4)
        pointdata = simpleGainTransfer(pointdata, gain=1/pnorm)

        pointdata = RTodB(pointdata)

        # fit_popt = pointdata
        fit_popt = fitNoise(pointdata, func, p0=p0, theoretical=False, verbose=verbose)
        if verbose:
            print_p0dict(fit_popt, desc)

        pointdata = dBToR(pointdata)
        # pointdata = filterEnvelope(pointdata, td = fit_popt[0])
        # pointdata = phaseToFreqNoise(pointdata)

        phaseenvelopedata_windowed = generateFitEnvelope(test_Si, range, N, fit_popt, theoretical = False)
        phaseenvelopedata = generateFitEnvelope(test_Sphi, range, N, fit_popt, theoretical = True)
        freqenvelopedata = generateFitEnvelope(test_Sdv, range, N, fit_popt, theoretical = True)

        #peak points:

        peakpl = []
        for f in np.linspace(-range, range, num=2*N + 1):
            if f != 0:
                peakpl.append((f, si_peak(f, *peakopt)))
        peakpl = simpleGainTransfer(peakpl, gain=1/pnorm)



        # pointdata = simpleGainTransfer(pointdata, gain=1/4)
        # phaseenvelopedata_windowed = simpleGainTransfer(phaseenvelopedata_windowed, gain=1/4)


        normalizePower(phaseenvelopedata, BW=1000)

        # phaseenvelopedata_windowed = dBToR(phaseenvelopedata_windowed)
        # envelopedata = filterEnvelope(envelopedata, td = fit_popt[0])
        # envelopedata = phaseToFreqNoise(envelopedata)

        # envelopedata = generateFit_dv_Envelope(test_Si_dv, range,N,h0 = fit_popt[1], fg=fit_popt[2], sg=fit_popt[3], hg=fit_popt[4])

        # return [(phaseenvelopedata_windowed, desc+" windowed phase fit"), (phaseenvelopedata, desc+" phase fit"), (pointdata, desc+" self-het data", True),]
        return [(phaseenvelopedata_windowed, desc+" self-het phase fit"), (phaseenvelopedata, desc+" laser phase noise fit"), (pointdata, desc+" self-het data", True),]

        # return [(phaseenvelopedata_windowed, desc+" windowed phase fit, dB/Hz"), (phaseenvelopedata, desc+" phase fit, dB/Hz"), (freqenvelopedata, desc+" frequency fit, Hz^2/Hz"), (pointdata, desc+", dB/Hz")]
        # return [(phaseenvelopedata_windowed, desc+" windowed phase fit, dB/Hz"), (freqenvelopedata, desc+" frequency fit, Hz^2/Hz"), (pointdata, desc+", dB/Hz")]
        # return [(phaseenvelopedata_windowed, desc+" windowed phase fit, dB/Hz"),  (pointdata, desc+"dB/Hz")]
        # return [(pointdata, desc+"dB/Hz")]

        return [(freqenvelopedata, desc+" frequency fit, Hz^2/Hz"),]

    else:
        return ("","")

plotlabel = ''
xaxislabel = "Frequency [MHz]"
yaxislabel = "Noise Spectral Density [1/Hz]"

# plotlabel = "Self-Het–Removed Frequency Noise PSD"
# xaxislabel = "Fourier Frequency [Hz]"
# yaxislabel = r"$S_{\delta v}$ [$Hz^2$/$Hz$]"

p0dict_1040_solstis_paper = {'td': 5.422889860450032e-05, 'noise_floor': 5.371617037728882e-10, 'h0': 13, 
    'bumps': [
    {'f': 130000, 's': 18000, 'h': 25},
    {'f': 234000, 's': 1500, 'h': 2000}
    ]}

p0dict_685 = {'td':0.045e-03, 'noise_floor': 2.5536463324617545e-11, 'h0':39,
              'bumps': [
                  {'f': 20e3, 's': 1000, 'h': 10e3},
                  {'f': 50e3, 's': 1000, 'h': 1e3},
              ]}


p0dict_638 = {
  'td': 0.01989e-03,
  'noise_floor': 1.833020205593e-09,
  'h0': 3.166,                # white noise
  'bumps': [
      # {'f': 7.5e3, 's': 300, 'h': 200},		# Uncertain bump
      # {'f': 6e3, 's': 700, 'h': 2000},		# phantom bump
      {'f': 19.9e3, 's': 410, 'h': 24.8},		# GOOD
      {'f': 40e3, 's': 527, 'h': 35.28},		# GOOD
      {'f': 46e3, 's': 1229, 'h': 48},		# GOOD
      {'f': 73e3, 's': 778, 'h': 100},		# GOOD	(real h val: ~3.49)
      {'f': 118e3, 's':505, 'h':715},		# GOOD
      {'f': 139e3, 's': 534, 'h': 35},		# GOOD
      {'f': 236e3, 's':470, 'h':3835},		# GOOD
        {'f': 386e3, 's':100e3, 'h':90},		# Uncertain bump
      {'f': 418e3, 's':4090, 'h':164},	# GOOD though uncertain
      {'f': 473e3, 's':484, 'h':371},	# GOOD
      # {'f': 481e3, 's':400, 'h':30}		# PHANTOM bump Small, probably correct servobump

  ]}
#      {'f': 118e3, 's': 5e3, 'h': 0.7e3},
#      {'f': 236e3, 's': 5e3, 'h': 0.3e3},
#  ]
#}


p0dict_459_solstis_paper = {'td': 0.049035657562851e-03, 'noise_floor': 5.371654228034129e-10, 'h0': 2.1354579950565684, 
    'bumps': [
    {'f': 100, 's': 1000.66250896435, 'h': 5},
    {'f': 6200, 's': 1000.66250896435, 'h': 100.047054211920583},
    {'f': 46067.68, 's': 45402, 'h': 1.33786},
    {'f': 146067.68, 's': 45402, 'h': 1.33786}]}
p0dict_1040_solstis_old = {'td': 5.4335065614489123e-05, 'noise_floor': 5.371654228034129e-10, 'h0': 6.641739235307105, 
    'bumps': [
    {'f': 69125.60358008381, 's': 40561.66250896435, 'h': 11.047054211920583},
    {'f': 233664.6665332151, 's': 1317.9584063745058, 'h': 1714.3588429133786}]}
p0dict_1040_solstis = {'td': 5.1781493324844175e-05, 'noise_floor': 2.5536463324617545e-11, 'h0': 0.1308914322275196, 
    'bumps': [{'f': 6764.645175321227, 's': 444.07971622148403, 'h': 1.0050889654239965},
    {'f': 19216.472370078838, 's': 310.56293322635173, 'h': 70.6071741751727},
    {'f': 80255.33647386247, 's': 19710.845582308397, 'h': 2.3051719629189087},
    {'f': 104561.43637095785, 's': 337.44131053095305, 'h': 4.538047357990236}
    ]}
p0dict_1010_vecsel = {'td': 5.1773002792287886e-05, 'noise_floor': 5.6103122049523973e-11, 'h0': 0.42226025080305324, 
    'bumps': [
    {'f': 6858.207750338815, 's': 281.9963561039165, 'h': 1.3160036281794025},
    {'f': 62828.85833564577, 's': 11896.18852187011, 'h': 5.059869190489349},
    {'f': 70608.93124725755, 's': 397.32671775968794, 'h': 90.16715592729244},
    {'f': 82751.09263385143, 's': 976.579278091818, 'h': 34.94339728301235},
    {'f': 89266.50205367006, 's': 724.3559393851034, 'h': 2.045565307590331},
    {'f': 117237.49200196186, 's': -475.7217943584276, 'h': 275.7363228729694},
    {'f': 141157.85766617287, 's': 402.1466594073561, 'h': 23.37122534479351},
    {'f': 234959.75752485916, 's': 391.2226768440656, 'h': 34.55382285356568}
    ]}

def plotNoise():
    data = []
    # data.append(procData("dark noise", 1000, 50, True, "y-Dark Noise"))
    data_sets = []
    # data_sets += procData("envelope", True, "J_Locked_11km_600kHzSpan_300HzBW -- paper data Sdv", p0=p0dict_1040_solstis_paper, func=test_Sdv, theoretical=True)
    # data_sets += procData("envelope", True, "J_Locked_11km_600kHzS/pan_300HzBW -- paper data Si", p0=p0dict_1040_solstis_paper, func=test_Si, theoretical=False)

    # data_sets += procData("./Laser noise characterization/638-narrower-scan_28-7-2025.csv", True, "685nm", range=260e3, p0=p0dict_685)
    data_sets += procData("638-narrower-scan_28-7-2025.csv", True, "638nm", range=1e6, p0=p0dict_638)

    # data_sets += procData("./J_Locked_11km_600kHzSpan_300HzBW.csv", True, "1040nm",range=260e3, p0=p0dict_1040_solstis_paper)
    # data_sets += procData("./Locked_SH_Spectrum_1MHzSpan_300HzBW.csv", True, "1010 Vexcel 1MHzBW 300HzRBW 0.3W", p0=p0dict_1010_vecsel)
    for set in data_sets:
        data.append(set)
        # print(set)
    # data.append(procData("./1040_pre_amp_1MHzBW_300HzRBW.csv", True, "1040 SolsTiS 1MHzBW 300HzRBW"))
    # data.append(procData("./Locked_SH_Spectrum_1MHzSpan_300HzBW.csv", True, "1010 Vexcel 1MHzBW 300HzRBW 0.3W"))
    # a,b = procData("envelope", True, "envelope")
    # data.append(a)
    # data.append(b)
    # data.append(procData("1038 RIN Data", 0.05, 1/0.340, True, "1038 RIN Fractional Noise Spectral Density (measured with a DC amp of 340mV)"))
    plotData(data, logx=False, logy=True)


    return

if __name__ == "__main__":
    plotNoise()