from itertools import chain, combinations
import sys

def subsets(ns):
    return list(chain(*[[[list(x)] for x in combinations(range(ns), i)] for i in range(1,ns+1)]))



if (False):
#### Reconstruct ####
    import numpy as np
    def F(x,y):
        return np.sum(x)**2/np.sum(y)
    X1 = -3.70595
    Y1 = 0.106088
    X2 = 4.87052
    Y2 = 4.58719
    beta = 0.0772588
    alpha = -5.64313
    b = 3.35838
    a = 9.99038

    X1_val = X1
    Y1_val = Y1
    X2_val = X2
    Y2_val = Y2
    X1 = np.array([alpha, X1_val-alpha])
    Y1 = np.array([beta, Y1_val-beta])
    X2 = np.array([X2_val-a, a])
    Y2 = np.array([Y2_val-b, b])
    
    X1_m_alpha = X1[1:]
    Y1_m_beta  = Y1[1:]
    X1_p_a     = np.concatenate([X1, np.array([a])])
    Y1_p_b     = np.concatenate([Y1, np.array([b])])
    X2_p_alpha = np.concatenate([X2, np.array([alpha])])
    Y2_p_beta  = np.concatenate([Y2, np.array([beta])])
    X2_m_a     = X2[:-1]
    Y2_m_b     = Y2[:-1]
    X1_m_alpha_p_a = np.concatenate([X1_m_alpha, np.array([a])])
    Y1_m_beta_p_b = np.concatenate([Y1_m_beta, np.array([b])])
    X2_p_alpha_m_a = np.concatenate([X2[:-1], np.array([alpha])])
    Y2_p_beta_m_b = np.concatenate([Y2[:-1], np.array([beta])])
    
    assert (F(X1_m_alpha,Y1_m_beta)-F(X1,Y1)) > 0
    assert (F(X1_p_a,Y1_p_b)-F(X1,Y1)) < 0
    assert (F(X2_p_alpha,Y2_p_beta)-F(X2,Y2)) < 0
    assert (F(X2_m_a,Y2_m_b)-F(X2,Y2)) > 0
    
    top_row = F(X1_m_alpha,Y1_m_beta)+F(X2_p_alpha,Y2_p_beta)-F(X1,Y1)-F(X2,Y2)
    bot_row = F(X1_p_a,Y1_p_b)+F(X2_m_a,Y2_m_b)-F(X1,Y1)-F(X2,Y2)
    plus_minus = F(X1_m_alpha_p_a,Y1_m_beta_p_b)+F(X2_p_alpha_m_a,Y2_p_beta_m_b)-F(X1,Y1)-F(X2,Y2)

    a0 = np.array([X1[0], X2[0], X2[1], X1[1]])
    b0 = np.array([Y1[0], Y2[0], Y2[1], Y1[1]])

if (False):
    import numpy as np

    # gamma == 2.0, lambda > 1.0
    x = 1e6
    delta = 10.01
    theta = 2.0
    gamma = 2.0
    a0 = np.array([x-delta, delta, x+delta])
    b0 = np.array([x, delta, x])

    sortind = np.argsort(a0**theta/b0)
    a = a0[sortind]
    b = b0[sortind]

    part0 = [[0],[1,2]]
    part1 = [[0,1],[2]]
    part2 = [[0,2],[1]]

    sum([np.sum(a[part])**gamma/np.sum(b[part]) for part in part0])

if (False):
    import numpy as np
    import matplotlib.pyplot as plot
    
    gamma = 2.0
    delta = 1e-1
    C1 = 2+delta
    C2 = 2+delta
    epsilon = delta/2

    def F(x,y,gamma):
        return x**gamma/y + (C1-x)*gamma/(C2-y)

    xaxis = np.linspace(0.01, 2+delta-epsilon, 1000)
    yaxis = np.linspace(0.25, 2+delta-epsilon, 1000)
    X,Y = np.meshgrid(xaxis, yaxis)
    Z = F(X,Y,gamma)

    fig,ax = plot.subplots(1,1)
    cp = ax.contourf(X, Y, Z)
    fig.colorbar(cp)
    plot.show()

if (False):
    import numpy as np

    count = 0
    
    rng = np.random.RandomState(552)
    while True:
        NUM_POINTS = rng.choice(10)+2
        upper_limit_a = rng.uniform(low=0., high=1000.)
        upper_limit_b = rng.uniform(low=0., high=1000.)
        a0 = rng.uniform(low=0.000001, high=upper_limit_a, size=NUM_POINTS)
        b0 = rng.uniform(low=0.000001, high=upper_limit_b, size=NUM_POINTS)

        a0 = np.round(a0,2)
        b0 = np.round(b0,2)
        
        sortind = np.argsort(a0/b0)
        a0 = a0[sortind]
        b0 = b0[sortind]

        # gamma = 2
        # delFdelX = 2*np.sum(a0)/np.sum(b0)
        # delFdelY = -(np.sum(a0)**2/np.sum(b0)**2)
        # gamma = 4
        delFdelX = 4*(np.sum(a0)**3)/np.sum(b0)
        delFdelY = -(np.sum(a0)**4)/(np.sum(b0)**2)
        # gamma = 6
        # delFdelX = 6*(np.sum(a0)**5)/np.sum(b0)
        # delFdelY = -(np.sum(a0)**6)/(np.sum(b0)**2)
        
        if (-a0[-1]*delFdelX - b0[-1]*delFdelY) > 0:
            if (-a0[0]*delFdelX - b0[0]*delFdelY) > 0:
                print('FOUND')
                
        count+=1
        if not count%100000:
            print('count: {}'.format(count))
        
    
if (False):
    import numpy as np
    import matplotlib.pyplot as plot

    gamma = 2.0
    count = 0

    def score(a,b,gamma):
        return np.sum(a)**gamma/np.sum(b)
    def all_scores(a,b,gamma):
        scores = [score(a[range(0,i)], b[range(0,i)], gamma) + score(a[range(i,len(a))],
                                                              b[range(i,len(a))], gamma)
                  for i in range(1,len(a))] + [score(a, b, gamma)]
        return scores
    
    rng = np.random.RandomState(553)
    while True:
        NUM_POINTS = rng.choice(1000)+2 # so that len(a0) >= 2
        upper_limit_a = rng.uniform(low=-1000., high=1000.)
        upper_limit_b = rng.uniform(low=0., high=1000.)
        a0 = rng.uniform(low=0.000001, high=upper_limit_a, size=NUM_POINTS)
        b0 = rng.uniform(low=0.000001, high=upper_limit_b, size=NUM_POINTS)
        
        sortind = np.argsort(a0/b0)
        a0 = a0[sortind]
        b0 = b0[sortind]

        ind = range(0,len(a0))

        # Symmetric case
        scores = all_scores(a0,b0,gamma)

        # plot.plot(scores)
        # plot.pause(1e-3)
        # import pdb
        # pdb.set_trace()
        # plot.close()

        if np.argmax(scores) == len(a0)-1:
            print('FOUND')
            print(a0)
            print(b0)
            
if (False):
    import numpy as np

    count = 0
    
    rng = np.random.RandomState(552)
    while True:
        NUM_POINTS = rng.choice(10)+2
        upper_limit_a = rng.uniform(low=-1000., high=1000.)
        upper_limit_b = rng.uniform(low=0., high=1000.)
        a0 = rng.uniform(low=0.000001, high=upper_limit_a, size=NUM_POINTS)
        b0 = rng.uniform(low=0.000001, high=upper_limit_b, size=NUM_POINTS)

        sortind = np.argsort(a0/b0)
        a0 = a0[sortind]
        b0 = b0[sortind]

        lhs1 = (a0[0]/b0[0])
        lhs2 = (a0[-1]/b0[-1])
        rhs = np.sum(a0)/np.sum(b0)

        # print("( ",lhs,", ",rhs," )")
        
        if (lhs1 > rhs) or (lhs2 < rhs):
            print('FOUND')
            print("( ",lhs,", ",rhs," )")

if (False):
    import numpy as np

    count = 0
    
    rng = np.random.RandomState(552)
    while True:
        NUM_POINTS = 5
        k = 3
        l = 4
        upper_limit_a = rng.uniform(low=0., high=1000.)
        upper_limit_b = rng.uniform(low=0., high=1000.)
        a0 = rng.uniform(low=0.000001, high=upper_limit_a, size=NUM_POINTS)
        b0 = rng.uniform(low=0.000001, high=upper_limit_b, size=NUM_POINTS)

        sortind = np.argsort(a0/b0)
        a0 = a0[sortind]
        b0 = b0[sortind]

        xk = a0[k]
        yk = b0[k]
        Cxl = np.sum(a0[:l])
        Cyl = np.sum(b0[:l])
        Cxkm1 = np.sum(a0[:(k-1)])
        Cykm1 = np.sum(b0[:(k-1)])

        s1 = ((xk/yk)-(Cxl/Cyl))*yk*Cyl
        s2 = ((xk/yk)-(Cxkm1/Cykm1))*yk*Cykm1
        if (s1-s2)>0:
            print(a0)
            print(b0)
                    
if (False):
    import numpy as np

    count = 0

    gamma = 2.0

    def F(a,b,gamma):
        return np.sum(a)**gamma/np.sum(b)

    # def F(a,b,gamma):
    #     return np.log(np.sum(a))/np.sum(b)
    
    rng = np.random.RandomState(847)
    while True:
        NUM_POINTS = 10
        a0 = rng.uniform(low=0., high=100., size=NUM_POINTS)
        b0 = rng.uniform(low=0., high=1., size=NUM_POINTS)

        sortind = np.argsort(a0/b0)
        a0 = a0[sortind]
        b0 = b0[sortind]

        # lhs = F(a0, b0, gamma)
        # rhs = F(a0[0], b0[0], gamma) + F(a0[1], b0[1], gamma)
        lhs = F(a0, b0, gamma)
        rhs = F(a0[0], b0[0], gamma)

        import pdb
        pdb.set_trace()

        if lhs < rhs:
            print('FOUND')
            print("( ",lhs,", ",rhs," )")

        count += 1
        if not count%1000000:
            print('count: {}'.format(count))

if (False):
    import numpy as np

    count = 0
    gamma = 4.0

    def F(a,b,gamma):
        return np.sum(a)**gamma/np.sum(b)

    # def F(a,b,gamma):
    #     return np.log(np.sum(a))/np.sum(b)
    
    rng = np.random.RandomState(87)
    while True:
        NUM_POINTS = 5
        LEN_SUBSET = 3
        subind = rng.choice(NUM_POINTS, LEN_SUBSET, replace=False)
        minind = min(subind)
        maxind = max(subind)
        a0 = rng.uniform(low=-100., high=0., size=NUM_POINTS)
        b0 = rng.uniform(low=0., high=1., size=NUM_POINTS)

        sortind = np.argsort(a0/b0)
        a0 = a0[sortind]
        b0 = b0[sortind]

        lhs = b0[maxind]/a0[maxind]
        rhs = b0[minind]/a0[minind]
        mid = np.sum(b0[subind])/np.sum(a0[subind])

        if (lhs > mid) or (mid > rhs):
            print('FOUND')
            print(a0)
            print(b0)
            print(subind)
            print("( ",lhs,", ",rhs," )")

        count += 1
        if not count%1000000:
            print('count: {}'.format(count))
            
if (False):
    import numpy as np

    def score(a,b,gamma):
        return np.sum(a)**gamma/np.sum(b)
    
    def score_symmetric(x,y,i,gamma):
        Cx,Cy = np.sum(x), np.sum(y)
        return np.sum(x[:i])**gamma/np.sum(y[:i]) + (Cx-np.sum(x[:i]))**gamma/(Cy-np.sum(y[:i]))
    

    count = 0
    gamma = 2.0
    
    rng = np.random.RandomState(552)
    while True:
        NUM_POINTS = 2
        upper_limit_a = rng.uniform(low=-1000., high=1000.)
        upper_limit_b = rng.uniform(low=0., high=1000.)
        a0 = rng.uniform(low=0.000001, high=upper_limit_a, size=NUM_POINTS)
        b0 = rng.uniform(low=0.000001, high=upper_limit_b, size=NUM_POINTS)

        sortind = np.argsort(a0/b0)
        a0 = a0[sortind]
        b0 = b0[sortind]

        # Seed = 552
        # upper_limit_a = rng.uniform(low=-1000., high=1000.)
        # upper_limit_b = rng.uniform(low=0., high=1000.)
        # a0 = rng.uniform(low=0.000001, high=upper_limit_a, size=NUM_POINTS)
        # b0 = rng.uniform(low=0.000001, high=upper_limit_b, size=NUM_POINTS)        
        # a0 = np.array([-769.88725716, -261.39267287])
        # b0 = np.array([435.20909316, 147.76250352])

        if score(a0, b0, gamma) >= score_symmetric(a0,b0,1,gamma):
        # if (score(a0, b0, gamma) - score(a0[:-1], b0[:-1], gamma)) > score(a0[-1], b0[-1], gamma):
            print('FOUND')
            print(a0)
            print(b0)
    
        count+=1
        if not count%1000000:
            print('count: {}'.format(count))

if (False):
    import numpy as np

    def score(a,b,gamma):
        return np.sum(a)**gamma/np.sum(b)
    
    def score_symmetric(x,y,Cx,Cy,gamma):
        if ((np.sum(x) == Cx) and (np.sum(y) == Cy)) or ((np.sum(x) == 0.) and (np.sum(y) == 0.)):
            return Cx**gamma/Cy
        else:
            return np.sum(x)**gamma/np.sum(y) + (Cx-np.sum(x))**gamma/(Cy-np.sum(y))
    

    count = 0
    gamma = 2.0
    
    rng = np.random.RandomState(552)
    while True:
        NUM_POINTS = 3
        upper_limit_a = rng.uniform(low=-1000., high=1000.)
        upper_limit_b = rng.uniform(low=0., high=1000.)
        a0 = rng.uniform(low=0.000001, high=upper_limit_a, size=NUM_POINTS)
        b0 = rng.uniform(low=0.000001, high=upper_limit_b, size=NUM_POINTS)

        sortind = np.argsort(a0/b0)
        a0 = a0[sortind]
        b0 = b0[sortind]

        Cx,Cy = np.sum(a0), np.sum(b0)        

        # Seed = 552
        # upper_limit_a = rng.uniform(low=-1000., high=1000.)
        # upper_limit_b = rng.uniform(low=0., high=1000.)
        # a0 = rng.uniform(low=0.000001, high=upper_limit_a, size=NUM_POINTS)
        # b0 = rng.uniform(low=0.000001, high=upper_limit_b, size=NUM_POINTS)        
        # a0 = np.array([-769.88725716, -261.39267287])
        # b0 = np.array([435.20909316, 147.76250352])

        # Holds for gamma = 1.0
        # if (score(a0[[0,2]], b0[[0,2]], gamma) - score(a0[0], b0[0], gamma)) <= (score(a0, b0, gamma) - score(a0[[0,1]], b0[[0,1]], gamma)):
        if (score_symmetric(a0[[0,2]], b0[[0,2]], Cx, Cy, gamma) - score_symmetric(a0[0], b0[0], Cx, Cy, gamma)) <= (score_symmetric(a0, b0, Cx, Cy, gamma) - score_symmetric(a0[[0,1]], b0[[0,1]], Cx, Cy, gamma)):
            print('FOUND')
            print(a0)
            print(b0)

        count+=1
        if not count%1000000:
            print('count: {}'.format(count))

if (False):
    import numpy as np

    count = 0

    rng = np.random.RandomState(552)
    while True:

        NUM_POINTS = rng.choice(100)+2
        SPLIT_INDEX = np.max([rng.choice(NUM_POINTS-1), 2])
        
        upper_limit_a = rng.uniform(low=-1000., high=1000.)
        upper_limit_b = rng.uniform(low=-0., high=1000.)
        a0 = rng.uniform(low=-1*upper_limit_a, high=upper_limit_a, size=NUM_POINTS)
        b0 = rng.uniform(low=-0., high=upper_limit_b, size=NUM_POINTS)

        sortind = np.argsort(a0/b0)
        a0 = a0[sortind]
        b0 = b0[sortind]

        Cx,Cy = np.sum(a0), np.sum(b0)        

        # if (np.sum(a0[:SPLIT_INDEX])/np.sum(b0[:SPLIT_INDEX]) > np.sum(a0)/np.sum(b0)) or \
        #    (np.sum(a0)/np.sum(b0) > np.sum(a0[SPLIT_INDEX:])/np.sum(b0[SPLIT_INDEX:])):
        if (np.sum(a0[:-1+SPLIT_INDEX])/np.sum(b0[:-1+SPLIT_INDEX]) > np.sum(a0[:SPLIT_INDEX])/np.sum(b0[:SPLIT_INDEX])):
            print('FOUND')
            print(a0)
            print(b0)

        count+=1

        if not count%10000:
            print('count: {}'.format(count))

if (False):
    # The yellow region represents the True portion
    
    import numpy as np
    import matplotlib.pyplot as plt
    import sys

    count = 0
    alpha = 2.00
    beta = 1.0
    seed = 147
    QUADRANT_ONE = False

    USE_HESSIAN = False

    if USE_HESSIAN:
        def hess(x,y,alpha,beta):
            H = np.array([[alpha*(alpha-1)*x**(alpha-2)/y**beta,
                           -alpha*(beta)*x**(alpha-1)/y**(beta+1)],
                          [-alpha*beta*x**(alpha-1)/y**(beta+1),
                           beta*(beta+1)*x**alpha/y**(beta+2)]
                          ])
            eigs = np.linalg.eig(H)[0]
            return all(eigs >= -1.e-6)
 

        k = 10

        def F(a,b,alpha,beta):
            return (a**alpha)/(b**beta)

        def F_sym(a,b,alpha,beta):
            return (a**alpha)/(b**beta) + ((1-a)**alpha)/((1-b)**beta)

        if (QUADRANT_ONE):
            xaxis = np.arange(0.001, 1, .01)
        else:
            xaxis = np.arange(-1., 1., .01)
        yaxis = np.arange(0.001, 1, .001)
        xx,yy = np.meshgrid(xaxis, yaxis)
        ee = np.zeros([xaxis.shape[0], yaxis.shape[0]])
        for i in range(len(xaxis)):
            for j in range(len(yaxis)):
                ee[i,j] = hess(xaxis[i],yaxis[j],alpha,beta)
            print('{} done'.format(xaxis[i]))
            
        np.unique(ee)
     
        z = F_sym(xx,yy,alpha,beta)
        z = z < k
        cmap = plt.cm.RdYlBu            
        h = plt.contourf(xaxis,yaxis,z,cmap=cmap)
        # plt.pause(1e-3)
        plt.show()
    else:
        import numpy as np
        import matplotlib.pyplot as plt
        import sys
        
        count = 0
        alpha = 3.0
        beta = 2.0
        seed = 151
        q = 1.1
        epsilon = .002
        QUADRANT_ONE = False
        
        SEED = 48

        # def F(a,b,alpha,beta):
        #     return ((a/b)**(a))*np.exp(b-a)

        def F(a,b,alpha,beta):
            return (a**alpha)/(b**beta)

        # def F(a,b,alpha,beta):
        #     if a > b:
        #         return a*np.log(a/b) + b - a
        #     else:
        #         return 0

        # def F(a,b,alpha,beta):
        #     if a > b:
        #         return (a-b)**2/2/b
        #     else:
        #         return 0

        # def F(a,b,alpha,beta):
        #     Ca = 1-a
        #     Cb = 1-b
        #     if a/b > Ca/Cb:
        #         return a*np.log(a/b) + Ca*np.log(Ca/Cb) - (a+Ca)*np.log((a+Ca)/(b+Cb))                
        #     else:
        #         return 0

        # def F(a,b,alpha,beta):
        #     Ca=1-a
        #     Cb=1-b
        #     if a/b>Ca/Cb:
        #         return (a**2/b) + (Ca**2/Cb) - ((a+Ca)**2/(b+Cb))
        #     else:
        #         return 0

        #######
        # def F_(a,b,alpha,beta):
        #     return np.exp(-q*b)*np.power(q,a)/np.exp(-b)

        # def F(a,b,alpha,beta):
        #     return np.log(F_(a,b,alpha,beta))

        # def F(a,b,alpha,beta):
        #     return (1-b)

        # def F(a,b,alpha,beta):
        #     return np.max([((1-epsilon)*np.exp(-q*a)*(np.power(q*a,b))), epsilon*np.exp(-b)*(np.power(a,a))])/ \
        #            np.max([((1-epsilon)*np.exp(-q*a)*(np.power(a,b))), epsilon*np.exp(-b)*(np.power(a,a))])

        # def F(a,b,alpha,beta):
        #     return np.log(np.power(q,a)*np.exp(b*(1-q)))

        rng = np.random.RandomState(SEED)
        if QUADRANT_ONE:
            xaxis = np.arange(0.001, 1, .01)
        else:
            xaxis = np.arange(-1., 1., .01)
        yaxis = np.arange(0.001, 1, .001)
        count = 0
        while True:
            x1,x2 = rng.choice(xaxis,size=2,replace=False)
            y1,y2 = rng.choice(yaxis,size=2,replace=False)
            eta = rng.uniform(low=0,high=1)
            xmid=eta*x1+(1-eta)*x2
            ymid=eta*y1+(1-eta)*y2
            lhs = F(xmid,ymid,alpha,beta)
            rhs = eta*F(x1,y1,alpha,beta)+(1-eta)*F(x2,y2,alpha,beta)

            # XXX
            # Don't check convexity
            # if lhs>rhs and not np.isclose(lhs,rhs):
            if (False):
                print('CONVEXITY VIOLATED')
                print('eta: {} p1: ({},{}), p2: ({},{}), mid: ({},{})'.format(eta, x1,y1,x2,y2,xmid,ymid))
                print('F(p1): {} F(p2): {} lhs: {} rhs: {}'.format(F(x1,y1,alpha,beta),
                                                                   F(x2,y2,alpha,beta),
                                                                   F(xmid,ymid,alpha,beta),
                                                                   eta*F(x1,y1,alpha,beta)+(1-eta)*F(x2,y2,alpha,beta)
                                                                   ))
                # import pdb; pdb.set_trace()

            xsum = x1+x2
            ysum = y1+y2
            lhs = F(xsum,ysum,alpha,beta)
            rhs = F(x1,y1,alpha,beta)+F(x2,y2,alpha,beta)
            if lhs>rhs and not np.isclose(lhs,rhs):
                print('SUBADDITIVITY VIOLATED')
                print('p1: ({},{}), p2: ({},{}), psum: ({},{})'.format(x1,y1,x2,y2,xsum,ysum))
                print('F(p1+p2): {}, F(p1): {} F(p2): {} F(p1)+F(p2): {}'.format(F(xsum,ysum,alpha,beta),
                                                                                 F(x1,y1,alpha,beta),
                                                                                 F(x2,y2,alpha,beta),
                                                                                 rhs
                                                                                 ))
                # import pdb; pdb.set_trace()

            count+=1
            if not count%100000:
                print('count: {}'.format(count))
                

if (True):
    import numpy as np

    count = 0
    gamma = 2.0
    alpha = 2.0
    beta =  1.0
    seed = 151
    FIRST_QUADRANT = True
    
    # def F_noy(a,b,gamma):
    #     return np.sum(a)**gamma
    
    # def F(a,b,gamma):
    #     return np.sum(a)**gamma/np.sum(b)

    # def F(a,b,gamma):
    #     return (np.sum(a) + np.sum(b))**alpha

    def F(a,b,gamma):
        return (np.sum(a)**alpha)/(np.sum(b)**beta)

    # def F_sym(a,b,Cx,Cy,gamma):
    #     if (np.sum(a) == Cx) or (np.sum(a) == 0.) or (np.sum(b) == Cy) or (np.sum(b) == 0):
    #         return F(a,b,alpha,beta)
    #     else:
    #         return (np.sum(a)**alpha)/(np.sum(b)**beta) + ((Cx-np.sum(a))**alpha)/((Cy-np.sum(b))**beta)

    # def F(a,b,gamma):
    #     return 1.*np.log((1+np.sum(a))*(1+np.sum(b)))

    # def F(a,b,gamma):
    #     return 1.*np.log(1+np.sum(a))

    # def F_sym(a,b,Cx,Cy,gamma):
    #     if (np.sum(a) == Cx) or (np.sum(a) == 0.) or (np.sum(b) == Cy) or (np.sum(b) == 0):
    #         return np.log(1. + Cx)
    #     else:
    #         return np.log(1. + np.sum(a)) + np.log(1. + (Cx-np.sum(a)))

    rng = np.random.RandomState(seed)
    while True:

        NUM_POINTS = 8

        lower_limit_a = rng.uniform(low=-100., high=100.)
        lower_limit_b = 0.
        upper_limit_a = rng.uniform(low=0., high=100.)
        upper_limit_b = rng.uniform(low=0., high=100.)        
        if FIRST_QUADRANT:
            lower_limit_a = 0.
        a0 = rng.uniform(low=lower_limit_a, high=upper_limit_a, size=NUM_POINTS)
        b0 = rng.uniform(low=lower_limit_b, high=upper_limit_b, size=NUM_POINTS)

        # a0 = np.round(a0, 0)
        # b0 = np.round(b0, 0)
        sortind = np.argsort(a0/b0)
        a0 = a0[sortind]
        b0 = b0[sortind]
        Cx = np.sum(a0)
        Cy = np.sum(b0)

        # STRONG SUBMODULARITY
        # ====================
        # Submodularity : (lhs1+lhs2) >= (rhs1+rhs2) => submodular
        sets = subsets(NUM_POINTS)
        m,n = rng.choice(len(sets), 2, replace=False)
        lset, rset = sets[m][0], sets[n][0]
        l_r_int = list(set(lset).intersection(set(rset)))
        l_r_union = list(set(lset).union(set(rset)))
        lhs1 = F(a0[lset], b0[lset], gamma)
        lhs2 = F(a0[rset], b0[rset], gamma)
        rhs1 = F(a0[l_r_union], b0[l_r_union], gamma)
        rhs2 = F(a0[l_r_int], b0[l_r_int], gamma)
        print('lset: {}'.format(lset))
        print('rset: {}'.format(rset))
        print('l_r_int: {}'.format(l_r_int))
        print('l_r_union: {}'.format(l_r_union))
        print('=====')

        # REAL-VALUED SUBMODULARITY
        # =========================
        # x1,x2 = rng.uniform(low=lower_limit_a, high=upper_limit_a, size=2)
        # y1,y2 = rng.uniform(low=lower_limit_b, high=upper_limit_b, size=2)
        # xmin = np.min([x1,x2])
        # ymin = np.min([y1,y2])
        # xmax = np.max([x1,x2])
        # ymax = np.max([y1,y2])
        # lhs1 = F([x1],[y1], gamma)
        # lhs2 = F([x2],[y2], gamma)
        # rhs1 = F([xmin], [ymin], gamma)
        # rhs2 = F([xmax], [ymax], gamma)

        # CONSECUTIVE SUBMODULARITY
        # =========================
        # Weak submodularity : (lhs1+lhs2) >= (rhs1+rhs2) => weakly submodular
        # XXX
        # inequality flips for gamma=2 when replacing F with F_noy;
        # F is weakly submodular, F_noy is weakly supermodular
        # F is subadditive, hence weakly subadditive, but
        # F_noy is superadditive
        # j,k = np.sort(rng.choice(int(NUM_POINTS+1), 2, replace=False))        
        # lhs1 = F(a0[j:(k+1)],b0[j:(k+1)],gamma)
        # lhs2 = F(a0[(j+1):(k+2)],b0[(j+1):(k+2)],gamma)
        # rhs1 = F(a0[j:(k+2)],b0[j:(k+2)],gamma)
        # rhs2 = F(a0[(j+1):(k+1)],b0[(j+1):(k+1)],gamma)

        # CONSECUTIVE SUBMODULARITY
        # =========================
        # Another version of weak submodularity : (lhs1+lhs2) >= (rhs1+rhs2) => weakly submodular
        # j,k = np.sort(rng.choice(int(NUM_POINTS+1), 2, replace=False))
        # l,m = np.sort(rng.choice(int(NUM_POINTS+1), 2, replace=False))
        # lset, rset = set(range(j,k)), set(range(l,m))
        # l_r_int = lset.intersection(rset)
        # l_r_union = lset.union(rset)
        # lset, rset, l_r_int, l_r_union = list(lset), list(rset), list(l_r_int), list(l_r_union)
        # lhs1 = F(a0[lset], b0[lset], gamma)
        # lhs2 = F(a0[rset], b0[rset], gamma)
        # rhs1 = F(a0[l_r_union], b0[l_r_union], gamma)
        # rhs2 = F(a0[l_r_int], b0[l_r_int], gamma)
        # print('lset: {}'.format(lset))
        # print('rset: {}'.format(rset))
        # print('l_r_int: {}'.format(l_r_int))
        # print('l_r_union: {}'.format(l_r_union))
        # print(lhs1+lhs2,rhs1+rhs2)
        # print(j,k,l,m)
        # print('========')

        # CONSECUTIVE SUBMODULARITY
        # =========================
        # Yet another version of weak submodularity : (lhs1+lhs2) >= (rhs1+rhs2) => weakly submodular
        # j,k,l,m = np.sort(rng.choice(int(NUM_POINTS+1), 4, replace=False))
        # lhs1 = F(a0[j:l], b0[j:l], gamma)
        # lhs2 = F(a0[k:m], b0[k:m], gamma)
        # rhs1 = F(a0[j:m], b0[j:m], gamma)
        # rhs2 = F(a0[k:l], b0[k:l], gamma)
        # print('j,k,l,m: {},{},{}.{}'.format(j,k,l,m))
 
        # CONSECUTIVE NONSPLITTING SUBMODULARITY
        # ======================================
        # i,j = np.sort(rng.choice(int(NUM_POINTS), 2, replace=False))
        # fb1,fb2 = rng.choice(2, 2, replace=True)
        # lset = set(range(0,i+1)) if fb1 else set(range(i,NUM_POINTS))
        # rset = set(range(0,j+1)) if fb2 else set(range(j,NUM_POINTS))
        # l_r_int = lset.intersection(rset)
        # l_r_union = lset.union(rset)
        # lset,rset,l_r_int,l_r_union = list(lset),list(rset),list(l_r_int),list(l_r_union)
        # lhs1 = F(a0[lset], b0[lset], gamma)
        # lhs2 = F(a0[rset], b0[rset], gamma)
        # rhs1 = F(a0[l_r_union], b0[l_r_union], gamma)
        # rhs2 = F(a0[l_r_int], b0[l_r_int], gamma) if l_r_int else 0.
        # print('i,j: ({},{})'.format(i,j))
        # print('lset: {}'.format(lset))
        # print('rset: {}'.format(rset))
        # print('l_r_int: {}'.format(l_r_int))
        # print('l_r_union: {}'.format(l_r_union))
        # print(lhs1+lhs2,rhs1+rhs2)
        # print('=====')                            
                
        # Subadditivity : (lhs1+lhs2) >= (rhs1+rhs2)
        # sets = subsets(NUM_POINTS)
        # m,n = rng.choice(len(sets), 2, replace=False)
        # lset, rset = sets[m][0], sets[n][0]
        # rset = list(set(rset).difference(set(lset)))
        # l_r_union = list(set(lset).union(set(rset)))
        # lhs1 = F(a0[lset], b0[lset], gamma)
        # lhs2 = F(a0[rset], b0[rset], gamma)
        # rhs1 = F(a0[l_r_union], b0[l_r_union], gamma)
        # rhs2 = 0.
        # lhs1 = F_sym(a0[lset], b0[lset], Cx, Cy, gamma)
        # lhs2 = F_sym(a0[rset], b0[rset], Cx, Cy, gamma)
        # rhs1 = F_sym(a0[l_r_union], b0[l_r_union], Cx, Cy, gamma)
        # rhs2 = 0.

        # Weak subadditivity : (lhs1+lhs2) >= (rhs1+rhs2)
        # Subadditivity of F_sym irrelelvant
        # j,k,l = np.sort(rng.choice(int(NUM_POINTS+1), 3, replace=False))
        # lhs1 = F(a0[j:k],b0[j:k],gamma)
        # lhs2 = F(a0[k:l],b0[k:l],gamma)
        # rhs1 = F(a0[j:l],b0[j:l],gamma)
        # rhs2 = 0.

        # Quasiconvexity - note this is not a property of the set function,
        # but of the function defined on R x R+
        # j,k = rng.choice(range(2, int(NUM_POINTS)), 2, replace=False)
        # x1,y1 = np.sum(a0[:j]), np.sum(b0[:j])
        # x2,y2 = np.sum(a0[:k]), np.sum(b0[:k])
        # For F
        # if F(x1,y1,gamma) >= F(x2,y2,gamma):
        #     x1_tmp = x1
        #     y1_tmp = y1
        #     x1 = x2
        #     y1 = y2
        #     x2 = x1_tmp
        #     y2 = y1_tmp
        # lhs1 = (gamma - 1 - gamma*(x1/x2) + (y1/y2))*F(x2,y2,gamma)
        # lhs2 = 0.
        # rhs1 = 0.
        # rhs2 = 0.

        # For F_sym
        # if F_sym(x1,y1,Cx,Cy,gamma) >= F_sym(x2,y2,Cx,Cy,gamma):
        #     x1_tmp = x1
        #     y1_tmp = y1
        #     x1 = x2
        #     y1 = y2
        #     x2 = x1_tmp
        #     y2 = y1_tmp
        # lhs1 = (gamma - 1 - gamma*(x1/x2) + (y1/y2))*F(x2,y2,gamma)
        # lhs2 = (gamma*(x1/(Cx-x2)) - gamma*(x2/(Cx-x2)) + (y2/(Cy-y2)) - (y1/(Cy-y2)))*F(Cx-x2,Cy-y2,gamma)
        # rhs1 = 0.
        # rhs2 = 0.

        # print('lhs: {}'.format(lhs1+lhs2))
        # print('rhs: {}'.format(rhs1+rhs2))
        
        if ((lhs1+lhs2)<(rhs1+rhs2)) and not np.isclose(lhs1+lhs2, rhs1+rhs2):
        # if not np.isclose(lhs1+lhs2, rhs1+rhs2):
            print('FOUND')
            print(lhs1+lhs2)
            print(rhs1+rhs2)
            print('a0: ', a0)
            print('b0: ', b0)
            import pdb
            pdb.set_trace()
            sys.exit()

        count+=1

        if not count%10:
            print('count: {}'.format(count))
    
if (False):
    import numpy as np

    count = 0
    gamma = 4.0
    
    rng = np.random.RandomState(552)
    while True:
        NUM_POINTS = 2
        upper_limit_a = rng.uniform(low=0., high=1000.)
        upper_limit_b = rng.uniform(low=0., high=1000.)
        a0 = rng.uniform(low=0.000001, high=upper_limit_a, size=NUM_POINTS)
        b0 = rng.uniform(low=0.000001, high=upper_limit_b, size=NUM_POINTS)

        sortind = np.argsort(a0/b0)
        a0 = a0[sortind]
        b0 = b0[sortind]

        lhs1 = (a0[0]**gamma/b0[0])
        lhs2 = (a0[-1]**gamma/b0[-1])
        rhs = np.sum(a0)**gamma/np.sum(b0)

        # print("( ",lhs,", ",rhs," )")

        if rhs > (lhs1+lhs2):
            print('FOUND')

import pandas as pd
import numpy as np
rng = np.random.RandomState(552)
NUM_ROWS = 1000
df = pd.DataFrame({'c1': rng.uniform(0., 1., NUM_ROWS), 'c2': rng.choice(list('ABC'),NUM_ROWS)})

def f1(df):
    return df[df['c1'] > 0.5][df.c2 == 'A']

def f2(df):
    return df[(df.c1 > 0.5) & (df.c2 == 'A')]
