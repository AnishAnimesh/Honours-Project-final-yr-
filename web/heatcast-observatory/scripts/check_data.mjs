import fs from 'node:fs';
import path from 'node:path';
import assert from 'node:assert/strict';
import {fileURLToPath} from 'node:url';
import {calendarIndex,addDays,eventRule,adjustedRisk} from '../dist/model.mjs';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'../dist');
const read=name=>JSON.parse(fs.readFileSync(path.join(root,'data',name),'utf8'));
const issued=read('issued.json'),models=read('live-model.json'),research=read('research.json');
let checked=0;
for(const [name,e] of Object.entries(issued.experiments)){
 const [task,l]=name.split('_lead'),lead=Number(l),ofs=task==='ongoing'?[-2,-1,0]:[-1,0,1,2],bank=models.banks[name];
 for(let i=0;i<e.dates.length;i++){
  const target=e.dates[i],run=issued.runs.initializations.indexOf(addDays(target,-lead-1));assert.ok(run>=0);
  for(let s=0;s<7;s++){
   const values=ofs.map(o=>{const d=issued.runs.dates[run].indexOf(addDays(target,o));assert.ok(d>=0);return issued.runs.tmax[run][d][s];});
   const threshold=ofs.map(o=>models.referenceTable[calendarIndex(addDays(target,o))][s]);
   const probability=adjustedRisk(values,threshold,bank.errors,s,task);
   assert.equal(probability,e.probabilities[i][s],`${name} ${target} site ${s}: empirical model mismatch`);
   assert.equal(probability>=e.cutoff,e.predicted[i][s]);
   assert.equal(eventRule(values,threshold,task),Boolean(e.raw[i][s]));checked++;
  }
 }
}
for(const e of Object.values(research.experiments))for(let i=0;i<e.dates.length;i++)for(let s=0;s<7;s++)assert.equal(e.probabilities[i][s]>=e.cutoff,e.predicted[i][s]);
assert.equal(calendarIndex('2024-02-29'),calendarIndex('2023-02-28'));
assert.equal(addDays('2026-12-31',1),'2027-01-01');
for(const p of ['index.html','styles.css','app.js','assets/fonts.css','assets/europe.geojson'])assert.ok(fs.existsSync(path.join(root,p)));
console.log(`Verified ${checked.toLocaleString()} browser-model predictions against saved Python outputs, all research classifications, calendar behavior and required assets.`);
