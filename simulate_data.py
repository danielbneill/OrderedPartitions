import numpy as np
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt
import solverSWIG_DP
import solverSWIG_LTSS

def score_fn(g, h, p):
    gsum = np.sum(g[list(p)])
    hsum = np.sum(h[list(p)])
    if gsum > hsum:
        return gsum*np.log(gsum/hsum) + hsum - gsum
    else:
        return 0.

def Poisson_pointset(xMin, xMax, yMin, yMax, lambda0):
    xDelta=xMax-xMin;yDelta=yMax-yMin; #rectangle dimensions
    areaTotal=xDelta*yDelta;
    
    XCenter=.2
    YCenter=.8
    
    numbPoints = scipy.stats.poisson( lambda0*areaTotal ).rvs()#Poisson number of points
    xx = xDelta*scipy.stats.uniform.rvs(0,1,((numbPoints,1)))+xMin#x coordinates of Poisson points
    yy = yDelta*scipy.stats.uniform.rvs(0,1,((numbPoints,1)))+yMin#y coordinates of Poisson points

    return xx, yy

def plot_pointset(xx, yy, xMin, xMax, yMin, yMax, numSplits, lambdas):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1,1,1)
    xAxis = np.linspace(xMin, xMax, numSplits)
    yAxis = np.linspace(yMin, yMax, numSplits)
    # ax.set_xticks(xAxis, minor=False)
    # ax.set_yticks(yAxis, minor=False)
    ax.grid(which='both')
    ax.tick_params(which='both', # Options for both major and minor ticks
                   top='off', # turn off top ticks
                   left='off', # turn off left ticks
                   right='off',  # turn off right ticks
                   bottom='off') # turn off bottom ticks    
    plt.scatter(xx,yy, edgecolor='b', facecolor='none', alpha=0.5 )
    plt.xlabel("x"); plt.ylabel("y")
    plt.title('Simulated Dataset, Intensities:{}'.format(lambdas[::-1]))
    plt.grid('on')
    plt.savefig('PoissonDist.pdf')
    plt.close()
    # plt.pause(1e-3)

def form_location_data(xx, yy, xMin, xMax, yMin, yMax, baseline, num_partitions=4, numSplits=10):
    d = np.concatenate([xx, yy], axis=1)
    xAxis = np.linspace(xMin, xMax, numSplits)
    yAxis = np.linspace(yMin, yMax, numSplits)
    occ = np.zeros((xAxis.shape[0], yAxis.shape[0]))
    for x,y in d:
        xind = np.searchsorted(xAxis, x, side='left')
        yind = np.searchsorted(yAxis, y, side='left')
        occ[xind, yind] +=1

    columns = list()
    data = list()
    for i,_ in enumerate(xAxis):
        for j,_ in enumerate(yAxis):
            columns += ['%s_%s' % (i,j)]
            data += [occ[i,j]]
    df = pd.DataFrame(dict(zip(columns, [[i] for i in data])))
    
    g = pd.Series(data).to_numpy().astype('float')
    h = np.array(([baseline]*len(g))).astype('float')

    # iAxis = list(range(1,21))
    # ires = [solverSWIG_DP.OptimizerSWIG(i, g, h)()[1] for i in iAxis]
    
    all_results = solverSWIG_DP.OptimizerSWIG(num_partitions, g, h)()
    single_result = solverSWIG_LTSS.OptimizerSWIG(g, h)()

    cc = [[columns[c] for c in list(x)] for x in all_results[0]]
    cc = cc[::-1]

    dd = [columns[c] for c in list(single_result[0])]

    colors = list(plt.rcParams['axes.prop_cycle'])
    num_colors = len(colors)
    fig,ax = plt.subplots(**dict(figsize=(8,8)))

    scatters = list()
    # Plot all regions
    for ind,split in enumerate(cc):
        x = list()
        y = list()
        s = list()
        c = list()        
        for coord in split:
            xc,yc = (int(p) for p in coord.split('_'))
            if xc > 0 and yc > 0:
                sze = 100/(ind+1)
                colInd = ind
                x.append(xAxis[xc])
                y.append(yAxis[yc])
                s.append(sze)
                c.append(colors[colInd]['color'])
        scatters.append(ax.scatter(x, y, c=c, s=s, label=c))
    plt.legend(scatters, ['Region {} Score: {}'.format(1+ind, round(score_fn(g, h, split), 2)) for ind,split in enumerate(all_results[0][::-1])],
               loc='lower left',
               )
    plt.title('Simulated Data Subsets: {} All Regions'.format(num_partitions))
    # ax.add_artist(legend1)

    # plt.pause(1e-3)
    plt.savefig('AllRegions.pdf')
    plt.close()
    # plt.close()
    
    fig, axs = plt.subplots(2, 2, figsize=(8, 8))
    fig.suptitle('Simulated Dataset Top 4 Regions')
    
    # Plot first region
    x = list()
    y = list()
    s = list()
    c = list()
    colInd = 0
    for ind,split in enumerate(cc):
        for coord in split:
            xc,yc = (int(p) for p in coord.split('_'))
            if xc > 0 and yc > 0:
                color = colors[ind]['color'] if ind == 0 else 'white'
                sze = 100/(3+1) if ind == 0 else 1
                x.append(xAxis[xc])
                y.append(yAxis[yc])
                s.append(sze)
                c.append(color)
            
    scatter = axs[0,0].scatter(x, y, c=c, s=s, label=c)
    axs[0,0].set_title('Region 1')
    
    # Plot second region
    x = list()
    y = list()
    s = list()
    c = list()
    colInd = 0
    for ind,split in enumerate(cc):
        for coord in split:
            xc,yc = (int(p) for p in coord.split('_'))
            if xc > 0 and yc > 0:
                color = colors[ind]['color'] if ind == 1 else 'white'
                sze = 100/(3+1) if ind == 1 else 1                
                x.append(xAxis[xc])
                y.append(yAxis[yc])
                s.append(sze)
                c.append(color)
            
    scatter = axs[0,1].scatter(x, y, c=c, s=s, label=c)
    axs[0,1].set_title('Region 2')    

    # Plot third region
    x = list()
    y = list()
    s = list()
    c = list()
    colInd = 0
    for ind,split in enumerate(cc):
        for coord in split:
            xc,yc = (int(p) for p in coord.split('_'))
            if xc > 0 and yc > 0:
                color = colors[ind]['color'] if ind == 2 else 'white'
                sze = 100/(3+1) if ind == 2 else 1                
                x.append(xAxis[xc])
                y.append(yAxis[yc])
                s.append(sze)
                c.append(color)

    scatter = axs[1,1].scatter(x, y, c=c, s=s, label=c)
    axs[1,1].set_title('Region 3')    

    # Plot fourth region
    x = list()
    y = list()
    s = list()
    c = list()
    colInd = 0
    for ind,split in enumerate(cc):
        for coord in split:
            xc,yc = (int(p) for p in coord.split('_'))
            if xc > 0 and yc > 0:
                color = colors[ind]['color'] if ind == 3 else 'white'
                sze = 100/(3+1) if ind == 3 else 1                                
                x.append(xAxis[xc])
                y.append(yAxis[yc])
                s.append(sze)
                c.append(color)
            
    scatter = axs[1,0].scatter(x, y, c=c, s=s, label=c)
    axs[1,0].set_title('Region 4')    

    # plt.pause(1e-3)
    plt.savefig('SepRegions.pdf')
    plt.close()
    # plt.close()

    fig,ax = plt.subplots(**dict(figsize=(8,8)))
    # fig = plt.figure(figsize=(8, 8))
    plt.title('Simulated Dataset Single Best')
    
    for coord in dd:
        x,y = (int(p) for p in coord.split('_'))
        sze = 100/(ind+1)
        colInd = ind
        plt.scatter(xAxis[x], yAxis[y], c=colors[colInd]['color'], alpha=0.95, s=sze,
                    label='dd0')

    # plt.pause(1e-3)
    plt.savefig('SingleRegion.pdf')
    plt.close()
    # plt.close()
        

if __name__ == '__main__':
    xMin,xMax,yMin,yMax=0,1,0,1
    base_q = 50*50/4
    q1_baseline,q2_baseline,q3_baseline,q4_baseline=(70000,50000,30000,10000)
    lambdas = (int(q4_baseline),
               int(q3_baseline),
               int(q2_baseline),
               int(q1_baseline))
    numSplits = 50
    num_partitions = 4
    baseline_all = 50000
    baseline = baseline_all/(numSplits*numSplits)
    
    xx1, yy1 = Poisson_pointset(0, .5, 0, .5, lambdas[0])
    xx2, yy2 = Poisson_pointset(.5, 1, 0, .5, lambdas[1])
    xx4, yy4 = Poisson_pointset(.5, 1, .5, 1, lambdas[2])
    xx3, yy3 = Poisson_pointset(0, .5, .5, 1, lambdas[3])
    
    xx = np.concatenate([xx1, xx2, xx3, xx4])
    yy = np.concatenate([yy1, yy2, yy3, yy4])
    plot_pointset(xx, yy, xMin, xMax, yMin, yMax, numSplits, lambdas)

    # XXX
    form_location_data(xx, yy, xMin, xMax, yMin, yMax, baseline,
                       num_partitions=num_partitions, numSplits=numSplits)

