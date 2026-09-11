"""Manuscript plots from archived results; no solver or campaign execution.

The field snapshots predate this revision. Their hashes are recorded when read;
that record is provenance for this visualization, not a retroactive campaign pin.
"""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import figstyle


def start():
    import matplotlib.pyplot as plt
    figstyle.apply_paper_style()
    plt.rcParams.update({'font.size':10, 'axes.labelsize':10, 'legend.fontsize':9,
                         'xtick.labelsize':9, 'ytick.labelsize':9,
                         'axes.spines.top':False, 'axes.spines.right':False})
    return plt


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, metadata={'CreationDate':None, 'ModDate':None})
    import matplotlib.pyplot as plt
    plt.close(fig)


def tag(ax, label):
    ax.text(0.02, 0.97, label, transform=ax.transAxes, va='top', ha='left',
            fontsize=10, bbox={'facecolor':'white','edgecolor':'none','pad':1.5})


def h_axis(ax, hs):
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xticks(hs); ax.set_xticklabels([f'{h:g}' for h in hs], rotation=45, ha='right')
    ax.minorticks_off(); ax.set_xlabel('bandwidth $h$'); ax.set_ylabel('$E_2$')


def figure_bandwidth_clean(bw, figures):
    plt=start(); hs=np.array(bw['bandwidths'])
    fig,axs=plt.subplots(1,2,figsize=(7.2,3.35),layout='constrained',sharex=True,sharey=True)
    for ax,cases,title in zip(axs,[['C1','C2'],['B','H','Z']],
                              ['(a) Cosine data','(b) Secondary data']):
        bandwidth_panel(ax,hs,bw,cases,title)
    axs[1].set_ylabel('')
    axs[0].set_ylim(.002,1.05)
    save(fig,figures/'bandwidth_clean.pdf')
    figure_bandwidth_variable(bw,figures)


def bandwidth_panel(ax, hs, bw, cases, title):
    """Show every sampled value while labeling only four horizontal ticks."""
    for j,case in enumerate(cases):
        vals=np.array([bw['cases'][case]['E2_by_h'][str(h)] for h in hs])
        ax.plot(hs,vals,color=['#2465a7','#c56316','#73508c'][j],
                marker=['o','s','^'][j],linestyle=['-','--',':'][j],
                label=case,markersize=4.5,linewidth=1.7)
    ax.set_xscale('log');ax.set_yscale('log');ax.minorticks_off()
    ax.set_xticks([.005,.01,.02,.04],['0.005','0.010','0.020','0.040'])
    ax.set_xlabel('bandwidth $h$',fontsize=11)
    ax.set_ylabel('relative error $E_2$',fontsize=11)
    ax.set_title(title,loc='left',fontsize=11,pad=32)
    ax.grid(False);ax.yaxis.grid(True,which='major',color='#e5e5e5',linewidth=.6)
    ax.legend(loc='lower left',bbox_to_anchor=(0,1.005),ncol=len(cases),
              frameon=False,borderaxespad=0,handlelength=2,columnspacing=1.25,
              fontsize=10)
    ax.tick_params(axis='both',labelsize=10)


def figure_bandwidth_variable(bw, figures):
    plt=start();hs=np.array(bw['bandwidths'])
    fig,axs=plt.subplots(1,2,figsize=(7.2,3.35),layout='constrained',sharex=True,sharey=True)
    for ax,cases,title in zip(axs,[['B','VB05','VB09'],['H','VH05','VH09']],
                              ['(a) Gaussian data','(b) Mixture data']):
        bandwidth_panel(ax,hs,bw,cases,title)
    axs[1].set_ylabel('');axs[0].set_ylim(.01,1.05)
    save(fig,figures/'bandwidth_variable.pdf')


def figure_noise_window(noise, hs, figures):
    plt=start(); fig,axs=plt.subplots(2,3,figsize=(7.2,5.0),layout='constrained',sharey='row')
    handles=None
    for row,case in enumerate(['C1','B']):
        for col,eta in enumerate([.001,.005,.01]):
            ax=axs[row,col]; c=noise['curves'][f'{case}|{eta:g}|P']
            e=ax.errorbar(hs,c['mean'],yerr=c['std'],color=figstyle.METHOD,
                          marker='o',markersize=4,capsize=2,label='particles')
            t=ax.axhline(noise['blocks'][f'{case}|{eta:g}|P']['tikhonov_oracle']['mean'],
                         color=figstyle.TIKH,ls='--',lw=1.6,label='Tikhonov')
            h_axis(ax,hs); ax.set_xticks([.005,.014,.04]);ax.set_xticklabels(['0.005','0.014','0.040'],rotation=0)
            tag(ax,f'({chr(97+row*3+col)}) {case}, $\\eta={eta:g}$')
            if col:ax.set_ylabel('')
            handles=[e,t]
    fig.legend(handles=handles,labels=['particles, mean and standard deviation','Tikhonov, mean'],
               loc='outside upper center',ncol=2,frameon=False,fontsize=9)
    save(fig,figures/'noise_window.pdf')


def figure_closure(cl, figures):
    plt=start(); keys=list(cl['decomposition'])
    fig,ax=plt.subplots(figsize=(7.2,3.7),layout='constrained')
    x=np.arange(len(keys));width=.16
    names=['transport','closure','score regularization','discretization']
    for j,comp in enumerate(['wrong_transport','closure','score_regularization','particle_discretization']):
        vals=[cl['decomposition'][k]['final']['u'][comp] for k in keys]
        ax.bar(x+(j-1.5)*width,vals,width,color=['#222222','#666666','#aaaaaa','#dddddd'][j],
               hatch=['','//','..','xx'][j],edgecolor='black',linewidth=.5,label=names[j])
    for i,k in enumerate(keys):
        ax.hlines(cl['decomposition'][k]['final']['u']['total'],i-.42,i+.42,
                  color='black',lw=2.2,label='total' if i==0 else None)
        if not cl['decomposition'][k].get('admissible',True):
            ax.axvspan(i-.48,i+.48,color='#eeeeee',zorder=-2)
            ax.text(i,1.8,'excluded',ha='center',va='bottom',fontsize=8)
    labels=[k.split('|')[0]+('*' if not cl['decomposition'][k].get('admissible',True) else '')+
            '\n'+('fixed anchor' if k.endswith('frozen_left') else 'mass') for k in keys]
    ax.set_yscale('log');ax.set_ylim(1e-6,5);ax.set_xticks(x,labels)
    ax.set_ylabel('$L^2$ norm');ax.minorticks_off()
    fig.legend(*ax.get_legend_handles_labels(),loc='outside lower center',ncol=3,frameon=False)
    save(fig,figures/'closure_decomposition.pdf')


def figure_initial_rate(ir, figures):
    plt=start(); fig,axs=plt.subplots(1,2,figsize=(7.2,3.0),layout='constrained')
    x=np.linspace(0,1,800);r=np.sqrt(1-x*x);y=np.sqrt(r*(1-r))
    axs[0].plot(x,y,color=figstyle.TRUTH,lw=1.7)
    xm=np.sqrt(3)/2;xd=np.exp(-.01*np.pi**2)/2
    yd=np.sqrt(np.sqrt(1-xd*xd)*(1-np.sqrt(1-xd*xd)))
    axs[0].plot(xm,.5,'s',color=figstyle.TRUTH,ms=4)
    axs[0].plot(xd,yd,'o',color=figstyle.GLOB,ms=5,label='counterexample datum')
    axs[0].annotate('maximum',xy=(xm,.5),xytext=(.50,.51),fontsize=9,
                    arrowprops={'arrowstyle':'-','color':'#777777'})
    axs[0].set_ylim(-.01,.57);axs[0].set_xlabel('$B/c$');axs[0].set_ylabel('$c_{\\mathrm{rep}}/(\\alpha\\pi^2c)$')
    axs[0].legend(loc='lower right',frameon=False,fontsize=8);tag(axs[0],'(a)')
    taus=np.array(ir['q_level']['taus']);r=np.array(ir['q_level']['ratio_q'])-1
    axs[1].plot(taus,r,'o',color=figstyle.GLOB,ms=4,label='reference values')
    tt=np.linspace(0,taus.max(),100)
    axs[1].plot(tt,ir['q_level']['linear_coefficient_of_ratio_minus_one']*tt,'--',
                color=figstyle.TRUTH,label='linear fit')
    axs[1].set_xlabel('reverse time $\\tau$');axs[1].set_ylabel('$r_q(\\tau)-1$')
    axs[1].legend(loc='upper left',bbox_to_anchor=(0,0.90),frameon=False,fontsize=8);tag(axs[1],'(b)')
    save(fig,figures/'initial_rate.pdf')


def figure_crossover(co, figures):
    plt=start();fig,axs=plt.subplots(1,2,figsize=(7.2,3.2),layout='constrained')
    for kh,ls,mk in [(0.23,':','^'),(0.264,'-','o'),(0.29,'--','s')]:
        pts=[r for r in co['continuum'] if np.isclose(r['kh'],kh)]
        a=[r['a'] for r in pts]
        for ax,key in zip(axs,['ratio','ratio_out']):
            ax.plot(a,[r[key] for r in pts],color=figstyle.METHOD,ls=ls,marker=mk,ms=4,label=f'$kh={kh:g}$')
        axs[0].plot(a,[r['pred_ratio'] for r in pts],color='#888888',ls=ls,lw=1.1,
                    label='fourth-order expansion' if kh==.23 else None)
    for i,ax in enumerate(axs):
        ax.axhline(1,color=figstyle.TRUTH,ls='--',lw=1)
        ax.axvline(.5,color='#bbbbbb',lw=1,zorder=-2)
        ax.set_ylim(0,1.5);ax.set_xlabel('amplitude $a$');tag(ax,f'({chr(97+i)})')
    axs[0].set_ylabel('$|e_{2k}|/|e_k|$');axs[1].set_ylabel('$|e^{\\mathrm{out}}_{2k}|/|e^{\\mathrm{out}}_k|$')
    axs[0].legend(loc='lower right',frameon=False,fontsize=8)
    save(fig,figures/'crossover.pdf')


def figure_evidence(manifest, summary, figures):
    plt=start();directory=REPO/manifest['studies']['closure']['path']/'fields'
    names=['carrier_G1_mass_400.npz','reference_G1_mass_unregularized_3200.npz',
           'reference_G1_mass_regularized_3200.npz']
    fields=[];provenance={}
    for name in names:
        p=directory/name
        with np.load(p,allow_pickle=False) as z:fields.append(z['u_final'].copy())
        provenance[str(p.relative_to(REPO))]=hashlib.sha256(p.read_bytes()).hexdigest()
    particle,wrong,regularized=fields
    x=(np.arange(400)+.5)/400;truth=2+np.cos(np.pi*x);datum=2+np.exp(-.01*np.pi**2)*np.cos(np.pi*x)
    expected=summary['closure']['decomposition']['G1|mass']['final']['u']
    measured={'total':float(np.sqrt(np.mean((particle-truth)**2))),
              'wrong_transport':float(np.sqrt(np.mean((wrong-truth)**2))),
              'particle_discretization':float(np.sqrt(np.mean((particle-regularized)**2)))}
    for key,value in measured.items():
        if not np.isclose(value,expected[key],rtol=1e-9,atol=1e-13):
            raise ValueError(f'Archived field mismatch for {key}: {value} != {expected[key]}')
    fig,axs=plt.subplots(1,2,figsize=(7.2,3.0),layout='constrained')
    axs[0].plot(x,truth,color=figstyle.TRUTH,label='truth $u_0$')
    axs[0].plot(x,datum,color=figstyle.OBS,ls=':',label='terminal datum')
    axs[0].plot(x,wrong,color=figstyle.EXACT,ls='--',label='$U_0$ reference')
    axs[0].plot(x,particle,color=figstyle.GLOB,ls='none',marker='o',markevery=18,ms=3.5,label='gradient particles')
    axs[1].axhline(0,color=figstyle.TRUTH,lw=1)
    axs[1].plot(x,wrong-truth,color=figstyle.EXACT,ls='--',label='$U_0-u_0$')
    axs[1].plot(x,regularized-truth,color=figstyle.OBS,ls='-.',label='$U_{h,\\epsilon}-u_0$')
    axs[1].plot(x,particle-truth,color=figstyle.GLOB,ls='none',marker='o',markevery=18,ms=3.5,label='$U_P-u_0$')
    for i,ax in enumerate(axs):
        ax.set_xlabel('$x$');tag(ax,f'({chr(97+i)})')
    field_handles=axs[0].get_legend_handles_labels()[0]
    reg_handle=axs[1].get_legend_handles_labels()[0][1]
    fig.legend(handles=field_handles+[reg_handle],
               labels=['truth $u_0$','terminal datum','$U_0$ reference','gradient particles','$U_{h,\\epsilon}$ reference'],
               loc='outside upper center',ncol=3,frameon=False,fontsize=8)
    axs[0].set_ylabel('field');axs[1].set_ylabel('pointwise error')
    save(fig,figures/'representation_fields.pdf')

    fig,axs=plt.subplots(1,2,figsize=(7.2,3.0),layout='constrained')
    for case,ls,mk in [('G1','-','o'),('G2','--','s')]:
        r=summary['closure']['carrier_refinement'][case+'|mass']['diffs']
        ms=np.array(sorted(int(m) for m in r));dx=1/ms
        vals=np.array([r[str(m)]['u'] for m in ms])
        axs[0].loglog(dx,vals,color=figstyle.GLOB,ls=ls,marker=mk,label=case)
        b=summary['closure']['h_bridge'][case+'|mass'];h=np.array(b['h']);v=np.array(b['u'])
        axs[1].loglog(h,v,color=figstyle.EXACT,ls=ls,marker=mk,label=case)
        if case=='G1':
            axs[0].loglog(dx,vals[0]*dx/dx[0],':',color='#888888',label='slope 1')
            axs[1].loglog(h,v[0]*(h/h[0])**2,':',color='#888888',label='slope 2')
    for i,ax in enumerate(axs):
        tag(ax,f'({chr(97+i)})');ax.legend(frameon=False,fontsize=8,loc='lower right');ax.minorticks_off()
    axs[0].set_xlabel('$\\Delta x$');axs[0].set_ylabel('$\\|U_P-U_{h,\\epsilon}\\|_2$')
    axs[0].set_xticks([1/800,1/400,1/200],['1/800','1/400','1/200'])
    axs[1].set_xlabel('$h$');axs[1].set_ylabel('$\\|U_{h,\\epsilon}-U_0\\|_2$')
    axs[1].set_xticks([.007,.01,.014,.02],['0.007','0.010','0.014','0.020'])
    save(fig,figures/'representation_convergence.pdf')
    (figures/'revision_figure_provenance.json').write_text(json.dumps({
        'source':'saved field snapshots and pinned analysis rows; no solver execution',
        'field_sha256_at_visualization':provenance,'field_norm_checks':measured},indent=2)+'\n')
