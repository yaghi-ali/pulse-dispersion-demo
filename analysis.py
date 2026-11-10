import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def savefig(fig,out):
    fig.tight_layout()
    fig.savefig(out/'preview.png',dpi=160,bbox_inches='tight')
    plt.close(fig)

def propagate(field,dt,beta2,length):
    a=np.asarray(field,complex)
    if a.ndim!=1 or a.size<16 or not np.all(np.isfinite(a)) or not np.all(np.isfinite([dt,beta2,length])) or dt<=0 or length<0:raise ValueError('Invalid envelope, time step or propagation parameters.')
    omega=2*np.pi*np.fft.fftfreq(a.size,dt)
    return np.fft.ifft(np.fft.fft(a)*np.exp(-.5j*beta2*length*omega**2))
def rms_width(t,a):
    w=np.abs(a)**2;mean=np.sum(t*w)/w.sum();return float(np.sqrt(np.sum((t-mean)**2*w)/w.sum()))
def example(out,seed):
    t=np.linspace(-40,40,8192,endpoint=False);dt=t[1]-t[0];a=np.exp(-t*t/2);b=propagate(a,dt,-20,.1)
    lengths=np.linspace(0,.2,41);numeric=np.array([rms_width(t,propagate(a,dt,-20,l)) for l in lengths]);analytic=np.sqrt(1+(-20*lengths)**2)/np.sqrt(2)
    fig,ax=plt.subplots(1,2,figsize=(10,4));ax[0].plot(t,abs(a)**2,label='Input');ax[0].plot(t,abs(b)**2,label='100 m');ax[0].set(xlim=(-8,8),xlabel='Time (ps)',ylabel='Intensity (a.u.)');ax[0].legend()
    ax[1].plot(lengths*1000,analytic,label='Gaussian analytical width');ax[1].plot(lengths[::4]*1000,numeric[::4],'o',label='FFT');ax[1].set(xlabel='Distance (m)',ylabel='Intensity RMS width (ps)');ax[1].legend();savefig(fig,out)
    np.savetxt(out/'broadening.csv',np.column_stack([lengths,numeric,analytic]),delimiter=',',header='length_km,fft_rms_ps,analytical_rms_ps',comments='')
    return {'output_rms_ps':rms_width(t,b),'max_width_error_ps':float(np.max(abs(numeric-analytic))),'relative_energy_error':float(abs(np.sum(abs(b)**2)/np.sum(abs(a)**2)-1))}

def run(out='results',seed=42):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    metrics=example(out,seed);metrics.update(seed=seed,data='synthetic')
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2));return metrics
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default='results');p.add_argument('--seed',type=int,default=42);a=p.parse_args();print(json.dumps(run(a.out,a.seed),indent=2))
