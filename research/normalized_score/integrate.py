"""Full rational Fourier-Galerkin diagnostic in exact rescaled variables.

Decimal arithmetic; no PDE trajectory or source is read from production files.
The denominator geometric-series remainder is bounded separately in output.
"""
from decimal import Decimal as D, getcontext
from pathlib import Path
import argparse, json, time


def pi_value():
    def atan_inv(n):
        x=D(1)/n; power=x; answer=x; j=1
        while True:
            power *= -x*x
            new=answer+power/D(2*j+1)
            if new==answer: return new
            answer=new; j+=1
    return 16*atan_inv(5)-4*atan_inv(239)


ROWS=[[],['2/27'],['1/36','1/12'],['1/24','0','1/8'],
 ['5/12','0','-25/16','25/16'],['1/20','0','0','1/4','1/5'],
 ['-25/108','0','0','125/108','-65/27','125/54'],
 ['31/300','0','0','0','61/225','-2/9','13/900'],
 ['2','0','0','-53/6','704/45','-107/9','67/90','3'],
 ['-91/108','0','0','23/108','-976/135','311/54','-19/60','17/6','-1/12'],
 ['2383/4100','0','0','-341/164','4496/1025','-301/82','2133/4100','45/82','45/164','18/41'],
 ['3/205','0','0','0','0','-6/41','-3/205','-3/41','3/41','6/41','0'],
 ['-1777/4100','0','0','-341/164','4496/1025','-289/82','2193/4100','51/82','33/164','12/41','0','1']]
BS=['0','0','0','0','0','34/105','9/35','9/35','9/280','9/280','0','41/840','41/840']
CS=['0','2/27','1/9','1/6','5/12','1/2','5/6','1/6','2/3','1/3','1','0','1']


def rational(s):
    if '/' in s:
        a,b=s.split('/'); return D(a)/D(b)
    return D(s)


def run(J,P,dt,dps):
    getcontext().prec=dps
    zero=D(0); one=D(1); s=D('1e-8'); k=32*pi_value()
    h=one/(2*k); eps=h*h; m=one/(one+eps); L=D('.01')*k*k
    phi=[(-D(j*j)/8).exp() for j in range(J+1)]
    r=[m*j*j*phi[j] for j in range(J+1)]
    beta=r[2]/2; d=[beta*j-r[j] for j in range(J+1)]
    qcoef=r[1]*(one-m*phi[1])/(4*d[1])
    max_U=zero; max_physical=zero; calls=0
    def base(t):
        e=(-d[1]*t).exp()
        return s*e/2,qcoef*s*s*(one-e*e)
    def conv(a,b,zp):
        out={}
        for i,ai in a.items():
            if not ai: continue
            for j,bj in b.items():
                if not bj: continue
                ij=i+j
                v=ai*bj*zp[abs(i)+abs(j)-abs(ij)]
                out[ij]=out.get(ij,zero)+v
        return out
    def rhs(t,y):
        nonlocal max_U,max_physical,calls
        calls+=1
        b1,b2=base(t)
        up=[zero]+list(y); up[1]+=b1; up[2]+=b2
        z=(-beta*(L-t)).exp()
        zp=[one]
        for j in range(2*(P+2)*J+1): zp.append(zp[-1]*z)
        U=2*sum(abs(v) for v in up)
        physical=2*sum(abs(up[j])*zp[j] for j in range(1,J+1))
        max_U=max(max_U,U); max_physical=max(max_physical,physical)
        if U>=2*s: raise RuntimeError('Numerical iterate left proved bootstrap envelope')
        u={j:up[abs(j)] for j in range(-J,J+1) if j}
        f={j:(one-m*phi[abs(j)])*v for j,v in u.items()}
        g={j:D(j)*phi[abs(j)]*v for j,v in u.items()}
        v={j:-m*phi[abs(j)]*w for j,w in u.items()}
        term=conv(f,g,zp)
        total=[term.get(j,zero) for j in range(J+1)]
        for p in range(1,P+1):
            term=conv(term,v,zp)
            for j in range(1,J+1): total[j]+=term.get(j,zero)
        result=[-d[j]*y[j-1]+m*j*total[j] for j in range(1,J+1)]
        # Remove only the exact known quadratic forcing; all nonlinear terms remain.
        result[1]-=2*r[1]*(one-m*phi[1])*b1*b1
        return result
    A=[[(i,rational(v)) for i,v in enumerate(row) if v!='0'] for row in ROWS]
    B=[(i,rational(v)) for i,v in enumerate(BS) if v!='0']
    C=[rational(v) for v in CS]
    for a,c in zip(A,C):
        assert abs(sum((v for _,v in a),zero)-c)<D(10)**(-dps+3)
    assert abs(sum(v for _,v in B)-one)<D(10)**(-dps+3)
    steps=int((L/D(dt)).to_integral_value(rounding='ROUND_CEILING'))
    step=L/steps; y=[zero]*J; start=time.time(); trajectory=[]
    for it in range(steps):
        t=step*it; stages=[]
        for ai,ci in zip(A,C):
            yy=[y[j]+step*sum((coef*stages[i][j] for i,coef in ai),zero) for j in range(J)]
            stages.append(rhs(t+step*ci,yy))
        y=[y[j]+step*sum((coef*stages[i][j] for i,coef in B),zero) for j in range(J)]
        if it in (0,steps//4,steps//2,3*steps//4,steps-1):
            bt1,bt2=base(step*(it+1)); z=(-beta*(L-step*(it+1))).exp()
            trajectory.append(dict(t=str(step*(it+1)),physical_cos1=str(2*z*(bt1+y[0])),
                physical_cos2=str(2*z*z*(bt2+y[1])),residual_u2=str(y[1])))
    b1,b2=base(L); full=2*phi[2]*(b2+y[1]); quad=2*phi[2]*b2
    a=s*(-(beta-1)*L).exp(); datum=s*(-beta*L).exp()
    Ubound=2*s
    rhs_tail=4*J*Ubound**(P+3)/(1-Ubound)
    # On U<=2s the finite-J nonlinear map is Lipschitz with this generous bound.
    lip=100*J*Ubound/(1-Ubound)**3
    propagated_tail=L*(lip*L).exp()*rhs_tail
    result=dict(J=J,P=P,dt_requested=dt,step=str(step),steps=steps,dps=dps,
      seconds=time.time()-start,rhs_calls=calls,n=32,s=str(s),certified_s0='0.0000000625',
      k=str(k),h=str(h),epsilon=str(eps),L=str(L),beta=str(beta),source_amplitude=str(a),
      exact_data_amplitude=str(datum),source_graph_norm=str((1+(1+k**8)*a*a/2).sqrt()),
      source_minimum=str(1-a),mass='exactly one; zero Fourier coefficient fixed',
      max_weighted_perturbation_norm=str(max_U),max_physical_perturbation_envelope=str(max_physical),
      certified_density_lower_bound=str(1-2*s),certified_denominator_lower_bound=str(1-2*s+eps),
      returned_cos2_full=str(full),returned_cos2_quadratic=str(quad),
      returned_cos2_nonlinear_correction=str(2*phi[2]*y[1]),
      relative_nonlinear_correction=str(y[1]/b2),
      analytic_cos2_remainder_bound=str(60000*s**3),
      denominator_rhs_tail_bound=str(rhs_tail),
      denominator_propagated_l1_tail_bound=str(propagated_tail),
      denominator_tail_note='Finite-Galerkin bound on the common proved U<=2s envelope; does not bound spectral truncation',
      residual_coefficients=[str(v) for v in y],trajectory=trajectory)
    return result


if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--J',type=int,default=4)
    ap.add_argument('--P',type=int,default=4); ap.add_argument('--dt',default='.1')
    ap.add_argument('--dps',type=int,default=50); ap.add_argument('--out',required=True)
    args=ap.parse_args(); result=run(args.J,args.P,args.dt,args.dps)
    Path(args.out).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('J','P','step','dps','seconds',
      'returned_cos2_full','returned_cos2_nonlinear_correction','source_graph_norm')},indent=2))
