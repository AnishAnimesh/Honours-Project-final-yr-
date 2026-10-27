// Browser equivalent of the frozen empirical joint-error model.
export function calendarIndex(date){const [y,m,d]=date.slice(0,10).split('-').map(Number);const day=m===2?Math.min(d,28):d;return Math.round((Date.UTC(2001,m-1,day)-Date.UTC(2001,0,1))/86400000);}
export function eventRule(values,thresholds,task){const hot=values.map((v,i)=>v>thresholds[i]);return task==='ongoing'?hot.every(Boolean):!hot[0]&&hot.slice(1).every(Boolean);}
export function adjustedRisk(trajectory,thresholds,errors,site,task){if(!trajectory.every(Number.isFinite))return null;let count=0;for(const error of errors){const scenario=trajectory.map((v,i)=>Math.fround(Math.fround(v)+error[site][i]));if(eventRule(scenario,thresholds,task))count++;}return count/errors.length;}
export function addDays(date,n){const d=new Date(date.slice(0,10)+'T00:00:00Z');d.setUTCDate(d.getUTCDate()+n);return d.toISOString().slice(0,10);}
