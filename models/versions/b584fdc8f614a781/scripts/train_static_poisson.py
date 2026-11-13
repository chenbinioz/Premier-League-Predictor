"""Fit the independent Poisson baseline on training seasons only."""
import numpy as np
from scipy.optimize import minimize


def fit_static_poisson(frame):
    teams=sorted(set(frame.HomeTeam)|set(frame.AwayTeam));index={t:i for i,t in enumerate(teams)}
    h=frame.HomeTeam.map(index).to_numpy();a=frame.AwayTeam.map(index).to_numpy();n=len(teams)
    gh=frame.FTHG.to_numpy(dtype=float);ga=frame.FTAG.to_numpy(dtype=float)
    import pandas as pd
    dates=pd.to_datetime(frame.Date);days=(dates.max()-dates).dt.days.to_numpy()
    weights=np.exp(-np.log(2)*days/216)
    def objective(parameters):
        attack=parameters[:n]-parameters[:n].mean();defense=parameters[n:2*n];adv=parameters[-1]
        lh=attack[h]+defense[a]+adv;la=attack[a]+defense[h]
        return float(np.mean(weights*(np.exp(lh)-gh*lh+np.exp(la)-ga*la))+.002*np.mean(parameters**2))
    result=minimize(objective,np.zeros(2*n+1),method='L-BFGS-B',bounds=[(-2,2)]*(2*n)+[(-1,1)])
    if not result.success:raise RuntimeError('Static Poisson fit did not converge')
    attack=result.x[:n]-result.x[:n].mean();defense=result.x[n:2*n]
    return {'model_kind':'static-poisson-v2','trained_until':str(frame.Date.max()),
            'attack':dict(zip(teams,np.exp(attack).tolist())), 'defense':dict(zip(teams,np.exp(defense).tolist())),
            'home_adv':float(np.exp(result.x[-1])),'rho':0.0}
