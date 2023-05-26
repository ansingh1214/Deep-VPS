from pylab import *
from scipy.optimize import curve_fit
import matplotlib.colors as colors
import matplotlib as mpl
mpl.style.use('classic')

def fit_func(x, a, b,c):
    return exp(-(x-a)**2./2/b)*c


fig1a=1
fig1c=0
fig2=0
fig3=0
fig4=0
fig5=0
fig6=0
fig7=0
fig8=0

if fig1a:
 figure(figsize=(6,4))
 x=arange(-1.5,1.51,0.01)
 X,Y=meshgrid(x,x)
 z=2*(6+4*X**4-6*Y*Y+3*Y**4+10*X*X*(Y*Y-1))
 z[z>20]=20
 contourf(X,Y,z,levels=arange(-.5,22,.1),cmap='bwr')
 contour(X,Y,z,levels=arange(-.5,15,2),colors='k')
 xlabel(r'$x$',size=20)
 ylabel(r'$y$',size=20)
 xticks(size=14)
 yticks(size=14)
 tight_layout()
 savefig('figa.pdf')

if fig1c:
 figure(figsize=(6,7))
 subplot(211)
 da=loadtxt('nn_train.txt')
 da1=loadtxt('1b.txt')
 step=arange(1,21,1)
 plot(da1[:,0],da1[:,1],'-',color='b',ms=14,mew=3,mec='b',mfc='None',lw=3)
 plot(step,-da,'-',color='r',ms=14,mew=3,mec='r',mfc='None',lw=3)
 plot([0,20],[-6.08,-6.08],'-k',lw=3)
 xlabel(r'$n_\mathrm{int}$',size=20)
 ylabel(r'$\mathcal{L}_\lambda$',size=20)
 xticks(size=14)
 yticks([-7,-6,-5,-4,-3],size=14)
 axis([0,20,-7,-3])
 subplot(212)
 da=loadtxt('linear_bases_convergance.txt')
 step=arange(1,21,1)
 errorbar(da[:,0],da[:,1],da[:,2],0,'-o',color='b',ms=14,mew=3,mec='b',mfc='None',lw=3)
 plot([2,12.2],[-6.08,-6.08],'-k',lw=3)
 axis([2,12,-7,-1])
 xlabel(r'$n_\mathrm{b}^{1/3}$',size=20)
 ylabel(r'$\mathcal{L}_\lambda$',size=20)
 xticks(size=14)
 yticks(size=14)
 tight_layout()
 savefig('figb.pdf')

 show()

if fig2:
    figure(figsize=(6,7))
    subplot(211)
    da=loadtxt('ntraj_errors.txt')
    da1=loadtxt('ntraj_errors_1.txt')
    errorbar(da[:,0],da[:,1],da[:,2],0,'o-',color='b',ms=14,mew=3,mec='b',mfc='None',lw=3)
    errorbar(da1[:,0],da1[:,1],da1[:,2],0,'o-',color='r',ms=14,mew=3,mec='r',mfc='None',lw=3)
    plot([9,100],[0,0],'-k',lw=3)
    xlabel(r'$N_\mathrm{t}$',size=20)
    ylabel(r'$\mathcal{L}_\lambda$',size=20)
    axis([9,100,-1,2.5])
    xticks(size=14)
    yticks(size=14)
    subplot(212)
    da=loadtxt('dconf_errors.txt')
    errorbar(da[:,0],da[:,1],da[:,2],0,'o-',color='b',ms=14,mew=3,mec='b',mfc='None',lw=3)
    errorbar(da[:,0],da[:,3],da[:,4],0,'o-',color='r',ms=14,mew=3,mec='r',mfc='None',lw=3)
    plot([0,20],[0,0],'-k',lw=3)
    xlabel(r'$d_c$',size=20)
    ylabel(r'$\mathcal{L}_\lambda$',size=20)
    xscale('log')
    axis([.9,20,-1,2.5])
    xticks([1,2,5,10,20],size=14)
    yticks(size=14)
    tight_layout()
    savefig('fig22.pdf')
    show()

if fig3:
    x=arange(-2,9,0.1)
    figure(figsize=(6,7))
    subplot(211)
    da=loadtxt('3a.txt')
    plot(da[:,0],da[:,1],'-',color='b',ms=10,mew=3,mec='b',mfc='None',lw=2)
    plot(da[:,0],da[:,2],'-',color='r',ms=10,mew=3,mec='r',mfc='None',lw=2)
    plot(da[:,0],da[:,3],'-',color='k',ms=10,mew=3,mec='k',mfc='None',lw=2)
    params, temp = curve_fit(fit_func, da[:,0],da[:,1])
    fill(x,params[2]*exp(-(x-params[0])**2./2./params[1]),'b',alpha=0.5)
    params, temp = curve_fit(fit_func, da[:,0],da[:,2])
    fill(x,params[2]*exp(-(x-params[0])**2./2./params[1]),'r',alpha=0.5)
    params, temp = curve_fit(fit_func, da[:,0],da[:,3])
    fill(x,params[2]*exp(-(x-params[0])**2./2./params[1]),'k',alpha=0.5)
    xlabel(r'$\Delta U_{\lambda^*}^\alpha$',size=20)
    ylabel(r'$P(\Delta U_{\lambda^*}^\alpha)$',size=20)
    axis([-2,9,0,1])
    xticks(size=14)
    yticks(size=14)
    subplot(212)
    da=loadtxt('3b.txt')
    plot(da[:,0],da[:,1],'-',color='b',ms=10,mew=3,mec='b',mfc='None',lw=2)
    params, temp = curve_fit(fit_func, da[:,0],da[:,1])
    fill(x,params[2]*exp(-(x-params[0])**2./2./params[1]),'b',alpha=0.5)
    plot(da[:,0],da[:,2],'-',color='r',ms=10,mew=3,mec='r',mfc='None',lw=2)
    params, temp = curve_fit(fit_func, da[:,0],da[:,2])
    fill(x,params[2]*exp(-(x-params[0])**2./2./params[1]),'r',alpha=0.5)
    plot(da[:,0],da[:,3],'-',color='k',ms=10,mew=3,mec='k',mfc='None',lw=2)
    params, temp = curve_fit(fit_func, da[:,0],da[:,3])
    fill(x,params[2]*exp(-(x-params[0])**2./2./params[1]),'k',alpha=0.5)
    xlabel(r'$\Delta \tilde{U}_{\lambda^*}^{\alpha,\alpha}$',size=20)
    ylabel(r'$P(\Delta U_{\lambda^*}^{\alpha,\alpha})$',size=20)
    axis([-2,9,0,1])
    xticks(size=14)
    yticks(size=14)
    tight_layout()
    savefig('figure3.pdf')
    show()

if fig4:
    figure(figsize=(6,10.))
    subplot(311)
    da=loadtxt('4a.txt')
    plot(da[:,0],da[:,1],'o-',color='b',ms=14,mew=3,mec='b',mfc='None',lw=3)
    #errorbar(da[:,0],da[:,3],da[:,2],0,'o-',color='r',ms=14,mew=3,mec='r',mfc='None',lw=3)
    #plot([9,100],[0,0],'-k',lw=3)
    ylabel(r'$\ln k t_f$',size=20)
    axis([0.1,1.0,-3.9,-2])
    #twinx()
    errorbar(da[:,0],da[:,2],da[:,3],0,'o-',color='r',ms=14,mew=3,mec='r',mfc='None',lw=3)
    #ylabel(r'$\ln k t_f + \Delta U$',size=20)
    #axis([0.1,1.0,-.5,1.5])
    xlabel(r'$\gamma/\gamma^*$',size=20)
    xticks(size=14)
    yticks(size=14)
    subplot(312)
    da=loadtxt('4b.txt')
    x=arange(-5,5,.01)
    plot(x,(x-4)**2*(x+4)**2./260,'-',color='r',ms=14,mew=3,mec='b',mfc='None',lw=4,alpha=.5)
    plot(da[:,0],da[:,2],'-',color='b',ms=14,mew=3,mec='b',mfc='None',lw=3)
    plot(da[:,0],da[:,1],'--',color='b',ms=14,mew=3,mec='b',mfc='None',lw=3)
    plot(da[:,0],da[:,3],'-.',color='b',ms=14,mew=3,mec='b',mfc='None',lw=3)
    axis([-5,5,0,1])
    xlabel(r'$x$',size=20)
    ylabel(r'$q(x,v,t_f/2)$',size=20)
    yticks(size=14)
    xticks(size=14)
    subplot(313)
    da=loadtxt('4c.txt')
    plot(x,(x-4)**2*(x+4)**2./260,'-',color='r',ms=14,mew=3,mec='b',mfc='None',lw=4,alpha=.5)
    plot(da[:,0],da[:,2],'-',color='b',ms=14,mew=3,mec='b',mfc='None',lw=3)
    plot(da[:,0],da[:,1],'--',color='b',ms=14,mew=3,mec='b',mfc='None',lw=3)
    plot(da[:,0],da[:,3],'-.',color='b',ms=14,mew=3,mec='b',mfc='None',lw=3)
    xlabel(r'$x$',size=20)
    ylabel(r'$q(x,v,t_f/2)$',size=20)
    yticks(size=14)
    xticks(size=14)
    axis([-5,5,0,1])
    tight_layout()
    savefig('figure4.pdf')
    show()

if fig5:
    figure(figsize=(6,3.5))
    da=load('ADP_loss_9_1_split.npy')
    plot(da[:,0],da[:,1],'-',color='r',ms=14,mew=3,mec='r',mfc='None',lw=6)
    plot([0,8000],[-10.21,-10.21],'-',color='k',ms=14,mew=3,mec='k',mfc='None',lw=3,alpha=1)
    plot([0,8000],[-10.21,-10.21],'-',color='k',ms=14,mew=3,mec='k',mfc='None',lw=16,alpha=0.5)
    #errorbar(da[:,0],da[:,3],da[:,2],0,'o-',color='r',ms=14,mew=3,mec='r',mfc='None',lw=3)
    #plot([9,100],[0,0],'-k',lw=3)
    xlabel(r'$n_\mathrm{int}$',size=20)
    ylabel(r'$\mathcal{L}_\lambda$',size=20)
    axis([0,8000,-12,0])
    xticks(size=14)
    yticks(size=14)
    tight_layout()
    savefig('figure5_1.pdf')
    show()

if fig7:
    figure(figsize=(6,3.5))
    da=loadtxt('ADP_solv_loss.txt')
    da1=loadtxt('ADP_solv_loss_ests.txt')
    plot(da[:,0],da[:,1],'-',color='r',ms=10,mew=3,mec='r',mfc='None',lw=3)
    plot(da1[::2,0],da1[::2,1],'-o',color='b',ms=6,mew=3,mec='b',mfc='b',lw=3)
    plot(da1[::2,0],da1[::2,3],'-s',color='c',ms=6,mew=3,mec='c',mfc='c',lw=3)
    errorbar(da1[::2,0],da1[::2,1],da1[::2,2],0,'-',color='b',ms=14,mew=3,mec='b',mfc='None',lw=3,alpha=0.5)
    errorbar(da1[::2,0],da1[::2,3],da1[::2,4],0,'-',color='c',ms=14,mew=3,mec='c',mfc='None',lw=3,alpha=0.5)
    plot([0,2500],[-8.145,-8.145],'-',color='k',ms=14,mew=3,mec='k',mfc='None',lw=3,alpha=1)
    plot([0,2500],[-8.145,-8.145],'-',color='k',ms=14,mew=3,mec='k',mfc='None',lw=12,alpha=0.5)
    #errorbar(da[:,0],da[:,3],da[:,2],0,'o-',color='r',ms=14,mew=3,mec='r',mfc='None',lw=3)
    #plot([9,100],[0,0],'-k',lw=3)
    xlabel(r'$n_\mathrm{int}$',size=20)
    ylabel(r'$\mathcal{L}_\lambda$',size=20)
    axis([0,2400,-9,0])
    xticks(size=14)
    yticks(size=14)
    tight_layout()
    savefig('figure7.pdf')
    show()

if fig6:
    figure(figsize=(6,6))
    da=np.load('ADP_gas_lambda2_transformed.npy')
    da1=np.load('ADP_gas_labels_transformed.npy')
    nam=da1[9:,0]
    nam2=[]
    for n in nam: nam2=append(nam2,'$%s_{%s}$' %(n[0],n[1:]))
    da[np.abs(da)<1e-3] = 0
    vmin = np.min(da)
    vmax = np.max(da)
    norm = colors.TwoSlopeNorm(vmin=-20, vcenter=0, vmax=40)
    pcolormesh(da[9:,9:],norm=norm,cmap='seismic')
    xticks(np.arange(0.5,22.51,1),nam2,size=14)
    tick_params(left=False,bottom=False,right=False,top=False)
    yticks(np.arange(0.5,22.51,1),nam2,size=14)
    xlabel(r'$\tilde{r}_j$',size=20)
    ylabel(r'$\tilde{r}_k$',size=20)
    axis([0,23,0,23])
    tight_layout()
    savefig('fig6a.pdf')
    figure(figsize=(6,3.5))
    da=np.load('ADP_gas_lambda2_transformed.npy')
    
    #sumU=zeros(23)
    sumU=sum(da,axis=0)
    newsum=sumU[9:]/sum(sumU)
    namid=argsort(newsum)[::-1]
    newsum=sort(newsum)
    newsum=newsum[::-1]
    bar(arange(23),newsum)
    for i in range(23): text(i-.3,max(newsum[i]+0.02,.02),'%s' %nam2[namid[i]],size=12)
    axis([-.5,22.5,-.1,.65])
    xlabel(r'$\tilde{r}_j$',size=20)
    ylabel(r'$-\langle \Delta \tilde{U}^{\tilde{r}_j}_{\lambda^*} \rangle/\ln k t_f$',size=20)
    tight_layout()
    savefig('fig6b.pdf')

    #colorbar()
    show()


if fig8:
    figure(figsize=(6,7))
    
    daa=np.load('ADP_solv_contribution_transformed.npy')
    subplot(211)
    nam=daa[:,0]
    
    nam2=[]
    newsum=daa[:,2]
    newsum1=zeros(20)
    for i in range(20): newsum1[i]=newsum[i]
    
    for n in nam: nam2=append(nam2,'$%s_{%s}$' %(n[0],n[1:]))

    print(nam2)
    #namid=argsort(newsum1)[::-1]
    #newsum=sort(newsum)
    #newsum=newsum[::-1]
    print(newsum1)
    bar(arange(20)*2.,newsum1,width=1.4)
    for i in range(20): text((i-.45)*2.,newsum1[i]+0.001,r'%s' %nam2[i],size=13)
    axis([-.5*2.,19.75*2.,-0.001,.045])
    xlabel(r'$\tilde{r}_j$',size=20)
    ylabel(r'$-\langle \Delta U^{\tilde{r}_j}_{\lambda^*} \rangle/\ln k t_f$',size=20)
    xticks([],size=14)
    yticks(size=14)
    subplot(212)
    da=np.load('ADP_solv_contribution_original.npy')
    nam=da[:,0]
    print(da)
    nam2=[]
    newsum=da[:,2]
    newsum1=zeros(22)
    for i in range(22): newsum1[i]=newsum[i]
    
    for n in nam: nam2=append(nam2,'$%s_{%s}$' %(n[0],n[1:]))
    
    
    namid=argsort(newsum1)[::-1]
    newsum1=sort(newsum1)
    newsum1=newsum1[::-1]
    
    bar(arange(22),newsum1)
    nn=array([10,5,13,7,3,9,11,12,0,21,2,17,6,15,4,20,1,18,16,8,19,14])
    axis([-.5,21.75,-0.001,.07])
    xticks(np.arange(0.5,21.51,1)-0.5,nn,size=12)
    xlabel(r'$r_j$',size=20)
    ylabel(r'$-\langle \Delta U^{r_j}_{\lambda^*} \rangle/\ln k t_f$',size=20)
    yticks(size=14)
    tight_layout()
    savefig('figure8.pdf')
    show()


