"""
.. module:: ni.tools.plot
   :platform: Unix
   :synopsis: Provides common plotting functions

.. moduleauthor:: Jacob Huth <jahuth@uos.de>

"""
import matplotlib.pyplot as pl
from matplotlib.pyplot import figure, plot, imshow, subplot, legend, title, show
import numpy as np
import scipy
try:
    import networkx as nx
    def plotNetwork(con,normalize_rows=True,normalize_cols=True,high_level=0.9,mid_level=0.5,no_connection=0,high_alpha=1,mid_alpha=0.7,low_alpha=0.4):
        """
            Draws a network graph of the network defined by the connections in **con**.
            Connections of nodes onto themselves are not drawn.

            normalize_rows:

                If True, rows are normalized, forcing at least one outgoing edge per node.

            normalize_cols:

                If normalize_rows==False and normalize_cols==True, columns are normalized, forcing at least one incoming edge per node.

            If both flags are False, con should be normalized by the user beforehand.

            The following options specify which connections are drawn as solid (> high), dashed (> mid), dotted (> no_connection).
            Above the `high_level`, the line is drawn with `high_alpha` opacity. Between high and mid as `mid_alpha` and between mid and no as `low_alpha`.

                `high_level`=0.9

                `mid_level`=0.5

                `no_connection`=0

                `high_alpha`=1

                `mid_alpha`=0.7

                `low_alpha`=0.4
        """
        if normalize_rows:
            for i_1 in range(con.shape[0]):
                con[i_1,:] = con[i_1,:] / np.max(con[i_1,:]) 
        elif normalize_cols:
            for i_2 in range(con.shape[1]):
                con[:,i_2] = con[:,i_2] / np.max(con[:,i_2]) 
        if np.isnan(con).any():
            for w in np.where(np.isnan(con)):
                con[w] = 0
        G=nx.DiGraph()
        for i_1 in range(con.shape[0]):
            for i_2 in range(con.shape[1]):
                G.add_edge(i_1,i_2,weight=con[i_1,i_2])
        pos=nx.spring_layout(G)
        ax = pl.gca()
        for (u,v,d) in G.edges(data=True):
            ax.annotate("",
                xy=pos[u], xycoords='data',
                xytext=pos[v], textcoords='data',
                arrowprops=dict(arrowstyle="-|>",shrinkA=12,shrinkB=12,
                                      connectionstyle="arc3,rad=-0.2",
                                      relpos=(1., 0.),alpha=high_alpha if (d['weight'] > high_level) else (mid_alpha if (d['weight'] > mid_level) else (low_alpha if (d['weight'] > no_connection) else 0.0)),
                                      ls='solid' if (d['weight'] > high_level) else ('dashed' if (d['weight'] > mid_level) else 'dotted'),
                                      fc="w"),
                )
        nx.draw_networkx_nodes(G,pos,node_size=500,node_color='w')
        nx.draw_networkx_labels(G,pos,font_size=16,font_family='sans-serif')
        ax = pl.gca()
        ax.spines['right'].set_color('none')
        ax.spines['bottom'].set_color('none')
        ax.spines['left'].set_color('none')
        ax.spines['top'].set_color('none')
        pl.xticks([])
        pl.yticks([])        
        return G
except:
    def plotNetwork(*args):
        raise Warning("The package networkx is not installed")
    pass


def plotConnections(N,con,scale=30):
    """
        Creates a plot of `N`x`N` bubbles, each one being scaled according to `con`.

    """
    plot(range(-1,N+1),range(1,-1*(N+1),-1),'w')
    if np.max(con) <= 0:
        con = con - np.min(con)
    for i in xrange(N):
        for j in xrange(N):
            #print 0.1*mean_variance[i][j]/np.max(mean_variance)
            ms =10+scale*(con[i][j]/np.max(con))
            if ms <= 0:
                ms = 1
            if i == j:
                plot(i,-1*j,'o',color='0.7',markersize=ms)
            else:
                plot(i,-1*j,'o',color='0.2',markersize=ms)
            #plt.Circle((i, j), radius=0.1, color='b')
    ax = pl.gca()
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.xaxis.set_ticks_position('top')
    ax.spines['top'].set_position(('data',0))
    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_position(('data',-1))
    pl.xticks(range(N),range(N))
    pl.yticks(range(0,-1*N,-1),range(N))


def plotHist(a,format='-',markerformat='o',width=20):
    """
        Plots a smoothed histogram 
    """
    n = np.zeros((np.ceil(np.max(a))-np.floor(np.min(a))) + 4*width)
    for i in np.array(a-np.min(a) + 2*width, dtype=np.int64):
        n[i] = n[i] + 1
    filtered = scipy.ndimage.gaussian_filter(n,width)
    pl.plot(range(int(np.floor(np.min(a)-2*width)),int(np.ceil(np.max(a)+2*width))),filtered,format)
    pl.plot([np.mean(a)],[filtered[int(np.round(np.mean(a)-np.min(a)+ 2*width))]],markerformat)
    return (np.mean(a),filtered[int(np.round(np.mean(a)-np.min(a)+ 2*width))])

def ArrowFigure(ax, head_width = 0.05,head_length=0.1):
    ax.set_xlim(-0.1,1.1)
    ax.set_ylim(-0.1,1.1)
    ax.axis('off')
    pl.arrow(0,0,1,0,fc="k",ec="k",head_width = head_width,head_length=head_length)
    pl.arrow(0,0,0,1,fc="k",ec="k",head_width = head_width,head_length=head_length)
    pl.arrow(0,0,0.1,0.1,fc="k",ec="k",head_width = head_width*0.8,head_length=head_length*0.75)

def plotGaussed(data,width,*args,**kwargs):
    """
    .. testcode::

        p2 = ni.model.pointprocess.createPoisson(sin(numpy.array(range(0,200))*0.01)*0.5- 0.2,1000)
        p2.plot()
        p2.plotGaussed(10)
        
    .. image:: _static/p2_out.png

    Another example::

        for i in range(10):
            r = np.array([(1.0 if n < 0.1 else 0.0) for n in rand(200)])
            for w in where(r)[0]:
                plot(w,i*0.01,'k|',alpha=0.5)
            ni.tools.plot.plotGaussed(r,20,'k',alpha=0.5)
    """
    plot(scipy.ndimage.gaussian_filter(data,width),*args,**kwargs)
