import React, { useState, useCallback, useRef } from 'react';
import axios from 'axios';
import { Bar, Line, Scatter, Radar, Doughnut } from 'react-chartjs-2';
import {
  Chart as ChartJS, CategoryScale, LinearScale, BarElement, LineElement,
  PointElement, RadialLinearScale, ArcElement, Title, Tooltip, Legend, Filler
} from 'chart.js';
import {
  Upload, CheckCircle, AlertTriangle, AlertCircle, Loader, ChevronRight,
  BarChart2, Cpu, FileText, Download, Copy, RefreshCw, Play,
  TrendingUp, Award, Shield, Layers, Eye, EyeOff, Activity,
  Database, Zap, Filter, Info, Heart, GitCompare, BookOpen, HelpCircle,
  LogIn, UserCircle, LogOut, Send, MessageSquare, Mail, User
} from 'lucide-react';

ChartJS.register(
  CategoryScale, LinearScale, BarElement, LineElement, PointElement,
  RadialLinearScale, ArcElement, Title, Tooltip, Legend, Filler
);

// ── Design tokens ────────────────────────────────────────────────
const C = {
  bg:      '#0c0f1a',
  bg2:     '#131722',
  bg3:     '#1a1f2e',
  bg4:     '#212638',
  border:  'rgba(255,255,255,0.07)',
  border2: 'rgba(255,255,255,0.12)',
  text:    '#e8eaf0',
  textMid: '#8b90a0',
  textDim: '#4a5068',
  red:     '#e53935',
  redL:    'rgba(229,57,53,0.12)',
  blue:    '#4d9fff',
  blueL:   'rgba(77,159,255,0.12)',
  green:   '#2dd4a0',
  greenL:  'rgba(45,212,160,0.12)',
  amber:   '#f59e0b',
  amberL:  'rgba(245,158,11,0.12)',
  purple:  '#a78bfa',
  purpleL: 'rgba(167,139,250,0.12)',
  pink:    '#f472b6',
  teal:    '#22d3ee',
  accent:  '#cc0000',
};

const MODEL_COLORS = {
  ridge:         '#8b90a0',
  plsr:          '#64748b',
  random_forest: '#2dd4a0',
  xgboost:       '#4d9fff',
  gpr:           '#a78bfa',
  ann:           '#f87171',
  hybrid:        '#f59e0b',
};

const MODEL_LABELS = {
  ridge:'Ridge',plsr:'PLSR',random_forest:'Random Forest',
  xgboost:'XGBoost+SHAP',gpr:'GPR',ann:'ANN',hybrid:'Hybrid',
};

const ALL_MODELS = [
  {id:'ridge',         name:'Ridge Regression',        desc:'Linear baseline, fully interpretable'},
  {id:'plsr',          name:'PLSR',                    desc:'VIP scores, chemometrics standard'},
  {id:'random_forest', name:'Random Forest',           desc:'Ensemble — robust nonlinear capture'},
  {id:'xgboost',       name:'XGBoost + SHAP',          desc:'Best accuracy + explainability'},
  {id:'gpr',           name:'GPR',                     desc:'Calibrated uncertainty quantification'},
  {id:'ann',           name:'ANN',                     desc:'Deep learning — needs large datasets'},
  {id:'hybrid',        name:'Hybrid (Physics-Informed)',desc:'Domain knowledge + ML fusion'},
];


// ── BMS1 vs BMS2 static reference data (from docx) ──────────────
const BMS1_DATA = [
  {name:'Ridge',        r2_train:0.63,r2_test:0.51,rmse:3.44,mae:2.85},
  {name:'PLSR',         r2_train:0.63,r2_test:0.51,rmse:3.45,mae:2.85},
  {name:'Random Forest',r2_train:0.96,r2_test:0.66,rmse:3.04,mae:2.28},
  {name:'ANN',          r2_train:0.91,r2_test:0.05,rmse:4.80,mae:3.68},
  {name:'GPR',          r2_train:0.87,r2_test:0.75,rmse:2.47,mae:1.97},
  {name:'Hybrid',       r2_train:0.89,r2_test:0.72,rmse:2.76,mae:2.10},
  {name:'XGBoost+SHAP', r2_train:1.00,r2_test:0.74,rmse:2.82,mae:2.16},
];
const BMS2_DATA = [
  {name:'Ridge',        r2_train:0.57,r2_test:0.57,rmse:3.76,mae:3.05},
  {name:'PLSR',         r2_train:0.57,r2_test:0.57,rmse:3.76,mae:3.05},
  {name:'Random Forest',r2_train:0.96,r2_test:0.79,rmse:2.66,mae:2.14},
  {name:'ANN',          r2_train:0.85,r2_test:0.77,rmse:2.74,mae:2.20},
  {name:'GPR',          r2_train:0.79,r2_test:0.79,rmse:2.61,mae:2.09},
  {name:'Hybrid',       r2_train:0.92,r2_test:0.83,rmse:2.40,mae:1.89},
  {name:'XGBoost+SHAP', r2_train:0.93,r2_test:0.83,rmse:2.47,mae:1.96},
];
const COMPARISON_DATA=[
  {model:'Ridge',        bms1:0.51,bms2:0.57,pct:'+12%', interp:'Structural ceiling'},
  {model:'PLSR',         bms1:0.51,bms2:0.57,pct:'+12%', interp:'Structural ceiling'},
  {model:'Random Forest',bms1:0.66,bms2:0.79,pct:'+20%', interp:'Better splits'},
  {model:'XGBoost+SHAP', bms1:0.74,bms2:0.83,pct:'+12%', interp:'More boosting signal'},
  {model:'GPR',          bms1:0.75,bms2:0.79,pct:'+5%',  interp:'Strong kernel prior'},
  {model:'ANN',          bms1:0.05,bms2:0.77,pct:'+1440%',interp:'Data hunger resolved'},
  {model:'Hybrid',       bms1:0.72,bms2:0.83,pct:'+15%', interp:'Physics features scale'},
];
const RANKINGS=[
  {category:'Best Accuracy',        bms1:'GPR (0.75)',   bms2:'XGBoost (0.83)', rec:'XGBoost+SHAP'},
  {category:'Best Uncertainty',     bms1:'GPR',          bms2:'GPR',            rec:'GPR'},
  {category:'Best Interpretability',bms1:'PLSR / Ridge', bms2:'PLSR / Ridge',   rec:'PLSR'},
  {category:'Most Data-Sensitive',  bms1:'ANN (failed)', bms2:'ANN (recovered)',rec:'Avoid small-N'},
  {category:'Best Domain Insight',  bms1:'Hybrid',       bms2:'Hybrid',         rec:'Hybrid features'},
];
const BMS_MODEL_COLORS={
  'Ridge':'#8b90a0','PLSR':'#64748b','Random Forest':'#2dd4a0',
  'ANN':'#f87171','GPR':'#a78bfa','Hybrid':'#f59e0b','XGBoost+SHAP':'#4d9fff',
};

const S = {
  card: {
    background: C.bg2, borderRadius: 14,
    border: `1px solid ${C.border}`, padding: '20px',
    boxShadow: '0 4px 24px rgba(0,0,0,0.3)',
  },
  label: {
    fontSize: 10, fontWeight: 600, letterSpacing: '0.1em',
    textTransform: 'uppercase', color: C.textDim, marginBottom: 6,
  },
  mono: { fontFamily: "'DM Mono', monospace" },
};

const chartBase = {
  responsive: true, maintainAspectRatio: false,
  plugins: {
    legend: { labels: { font:{family:'DM Sans',size:11}, color: C.textMid, padding:14 }},
    tooltip: { backgroundColor:'#1a1f2e', titleFont:{family:'DM Sans',weight:'600'},
               bodyFont:{family:'DM Sans'}, padding:12, cornerRadius:8,
               borderColor:C.border2, borderWidth:1 },
  },
  scales: {
    x: { ticks:{color:C.textMid,font:{family:'DM Mono',size:10}}, grid:{color:'rgba(255,255,255,0.04)'}, border:{color:'rgba(255,255,255,0.06)'} },
    y: { ticks:{color:C.textMid,font:{family:'DM Mono',size:10}}, grid:{color:'rgba(255,255,255,0.04)'}, border:{color:'rgba(255,255,255,0.06)'} },
  },
};

// ── Pipeline Steps ───────────────────────────────────────────────
const STEPS = [
  { id:0, label:'Upload & Cleanse',  icon:Database },
  { id:1, label:'Review Issues',     icon:AlertTriangle },
  { id:2, label:'Select & Train',    icon:Cpu },
  { id:3, label:'Results Dashboard', icon:BarChart2 },
  { id:4, label:'AI Report',         icon:FileText },
];

// ── Helpers ──────────────────────────────────────────────────────
function Badge({text,color=C.blue}){
  return <span style={{display:'inline-block',padding:'2px 9px',borderRadius:20,fontSize:10,fontWeight:600,background:color+'20',color,border:`1px solid ${color}30`}}>{text}</span>;
}

function GlowCard({children,accent=C.blue,style={}}){
  return (
    <div style={{...S.card,...style,boxShadow:`0 4px 24px rgba(0,0,0,0.3), 0 0 0 1px ${accent}20`}}>
      {children}
    </div>
  );
}

function Spinner({text='Processing…',size=20}){
  return(
    <div style={{display:'flex',alignItems:'center',gap:10,color:C.textMid}}>
      <Loader size={size} className="spin" color={C.blue}/>
      <span style={{fontSize:13}}>{text}</span>
    </div>
  );
}

function ProgressBar({value,color=C.blue,height=6}){
  return(
    <div style={{background:'rgba(255,255,255,0.06)',borderRadius:99,height,overflow:'hidden'}}>
      <div style={{width:`${Math.min(100,Math.max(0,value*100))}%`,height:'100%',borderRadius:99,
        background:color,transition:'width 0.8s ease',boxShadow:`0 0 8px ${color}60`}}/>
    </div>
  );
}

function StatPill({label,value,color=C.blue}){
  return(
    <div style={{padding:'8px 14px',borderRadius:10,background:color+'12',border:`1px solid ${color}25`,
      display:'flex',flexDirection:'column',gap:2}}>
      <div style={{fontSize:9,color,fontWeight:700,textTransform:'uppercase',letterSpacing:'0.08em'}}>{label}</div>
      <div style={{fontSize:16,fontWeight:700,color,...S.mono}}>{value}</div>
    </div>
  );
}

// ── Upload Zone ──────────────────────────────────────────────────
function DropZone({onFile,label,hint,accept='.csv',disabled=false}){
  const [drag,setDrag]=useState(false);
  const ref=useRef();
  const handle=f=>{if(f&&f.name.endsWith('.csv'))onFile(f);};
  return(
    <div
      onClick={()=>!disabled&&ref.current.click()}
      onDragOver={e=>{e.preventDefault();setDrag(true);}}
      onDragLeave={()=>setDrag(false)}
      onDrop={e=>{e.preventDefault();setDrag(false);handle(e.dataTransfer.files[0]);}}
      style={{border:`2px dashed ${drag?C.blue:disabled?C.border:C.border2}`,
        borderRadius:14,padding:'40px 24px',textAlign:'center',cursor:disabled?'not-allowed':'pointer',
        transition:'all 0.2s',background:drag?C.blueL:'rgba(255,255,255,0.02)',
        opacity:disabled?0.4:1}}>
      <input ref={ref} type="file" accept={accept} style={{display:'none'}}
        onChange={e=>handle(e.target.files[0])}/>
      <Upload size={32} color={drag?C.blue:C.textDim} style={{margin:'0 auto 12px'}}/>
      <div style={{fontSize:15,fontWeight:600,color:drag?C.blue:C.text,marginBottom:6}}>{label}</div>
      <div style={{fontSize:12,color:C.textMid}}>{hint}</div>
    </div>
  );
}

// ── Step 0+1: Upload & Cleanse ────────────────────────────────────
function UploadCleanse({onDone}){
  const [file,setFile]=useState(null);
  const [loading,setLoading]=useState(false);
  const [result,setResult]=useState(null);
  const [error,setError]=useState('');
  const [tab,setTab]=useState('summary');

  const upload=async(f)=>{
    setFile(f);setLoading(true);setError('');setResult(null);
    const fd=new FormData();fd.append('file',f);
    try{
      const {data}=await axios.post('/api/upload/cleanse',fd);
      // attach the actual File object so ModelSelect can reuse it without re-upload
      data._file = f;
      setResult(data);
    }catch(e){setError(e.response?.data?.detail||e.message);}
    finally{setLoading(false);}
  };

  const qColor=q=>q>=90?C.green:q>=75?C.blue:q>=60?C.amber:C.red;

  return(
    <div style={{display:'flex',flexDirection:'column',gap:20}}>
      {!result&&!loading&&(
        <GlowCard accent={C.blue} className="slide-up">
          <div style={{fontSize:13,color:C.textMid,marginBottom:16,lineHeight:1.6}}>
            Upload your bioprocess dataset CSV. The pipeline will automatically detect missing values,
            outliers, and batch drift — giving you a full data quality report before training.
          </div>
          <DropZone onFile={upload} label="Drop your dataset CSV here" hint="Accepts any CSV with bioprocess features · Fucosylation_pct as target"/>
        </GlowCard>
      )}

      {loading&&(
        <GlowCard accent={C.blue}>
          <div style={{display:'flex',flexDirection:'column',gap:16,alignItems:'center',padding:'20px 0'}}>
            <Spinner text="Running cleansing analysis…" size={28}/>
            <div style={{fontSize:12,color:C.textDim,textAlign:'center'}}>
              Detecting missing values · Outlier detection (IQR + Z-score + IsolationForest) · Batch drift (KS test)
            </div>
          </div>
        </GlowCard>
      )}

      {error&&(
        <GlowCard accent={C.red}>
          <div style={{display:'flex',gap:10}}>
            <AlertCircle size={18} color={C.red} style={{flexShrink:0,marginTop:1}}/>
            <div style={{fontSize:13,color:C.red}}>{error}</div>
          </div>
          <button onClick={()=>{setError('');setFile(null);}} style={{marginTop:12,padding:'7px 16px',borderRadius:8,border:`1px solid ${C.border2}`,background:'transparent',color:C.textMid,cursor:'pointer',fontSize:12}}>Try again</button>
        </GlowCard>
      )}

      {result&&(
        <>
          {/* Quality score header */}
          <GlowCard accent={qColor(result.quality_score)} className="fade-in">
            <div style={{display:'flex',alignItems:'center',gap:20,flexWrap:'wrap'}}>
              {/* Circular score */}
              <div style={{position:'relative',width:90,height:90,flexShrink:0}}>
                <svg width="90" height="90" viewBox="0 0 90 90">
                  <circle cx="45" cy="45" r="36" fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth="9"/>
                  <circle cx="45" cy="45" r="36" fill="none" stroke={qColor(result.quality_score)} strokeWidth="9"
                    strokeDasharray={`${result.quality_score*2.26} 226`} strokeDashoffset="56.5"
                    strokeLinecap="round"/>
                </svg>
                <div style={{position:'absolute',inset:0,display:'flex',flexDirection:'column',alignItems:'center',justifyContent:'center'}}>
                  <div style={{fontSize:20,fontWeight:800,color:qColor(result.quality_score),...S.mono}}>{result.quality_score}</div>
                  <div style={{fontSize:8,color:C.textDim}}>/ 100</div>
                </div>
              </div>
              <div style={{flex:1}}>
                <div style={{display:'flex',alignItems:'center',gap:10,marginBottom:6}}>
                  <div style={{fontSize:17,fontWeight:700,color:C.text}}>{result.filename}</div>
                  <Badge text={result.quality_status.toUpperCase()} color={qColor(result.quality_score)}/>
                </div>
                <div style={{display:'flex',gap:10,flexWrap:'wrap'}}>
                  {[
                    [`${result.n_rows.toLocaleString()} rows`,C.blue],
                    [`${result.n_features} features`,C.purple],
                    [`${result.missing.total_missing} missing`,result.missing.total_missing>0?C.amber:C.green],
                    [`${result.outliers.n_outliers_any} outliers`,result.outliers.n_outliers_any>0?C.red:C.green],
                    [`${result.drift.n_drifted_batches||0} drifted batches`,result.drift.n_drifted_batches>0?C.amber:C.green],
                  ].map(([l,c])=>(
                    <div key={l} style={{display:'flex',alignItems:'center',gap:5,padding:'4px 10px',borderRadius:7,background:c+'15',border:`1px solid ${c}25`}}>
                      <div style={{width:6,height:6,borderRadius:'50%',background:c}}/>
                      <span style={{fontSize:11,color:c,fontWeight:500}}>{l}</span>
                    </div>
                  ))}
                </div>
              </div>
              <button onClick={()=>{setResult(null);setFile(null);}}
                style={{padding:'7px 14px',borderRadius:9,border:`1px solid ${C.border2}`,
                  background:'transparent',color:C.textMid,cursor:'pointer',fontSize:12,flexShrink:0}}>
                Upload different file
              </button>
            </div>
          </GlowCard>

          {/* Tabs */}
          <div style={{display:'flex',borderBottom:`1px solid ${C.border}`,gap:0,overflowX:'auto'}}>
            {[['summary','Summary'],['missing',`Missing (${result.missing.total_missing})`],
              ['outliers',`Outliers (${result.outliers.n_outliers_any})`],
              ['drift',`Drift (${result.drift.n_drifted_batches||0})`],
              ['preview','Data Preview']].map(([id,label])=>(
              <button key={id} onClick={()=>setTab(id)}
                style={{padding:'9px 16px',border:'none',background:'none',cursor:'pointer',
                  fontSize:12,fontWeight:tab===id?600:400,fontFamily:'DM Sans,sans-serif',
                  color:tab===id?C.blue:C.textMid,
                  borderBottom:tab===id?`2px solid ${C.blue}`:'2px solid transparent',
                  transition:'all 0.15s',whiteSpace:'nowrap'}}>
                {label}
              </button>
            ))}
          </div>

          {/* Summary tab */}
          {tab==='summary'&&(
            <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:14}} className="fade-in">
              <GlowCard accent={C.purple}>
                <div style={{...S.label}}>Missing Values by Column</div>
                {result.missing.col_summary.filter(c=>c.missing>0).length===0
                  ?<div style={{fontSize:13,color:C.green}}>✓ No missing values detected</div>
                  :result.missing.col_summary.filter(c=>c.missing>0).map(col=>(
                  <div key={col.column} style={{marginBottom:10}}>
                    <div style={{display:'flex',justifyContent:'space-between',marginBottom:4}}>
                      <span style={{fontSize:11,...S.mono,color:C.text}}>{col.column}</span>
                      <span style={{fontSize:11,...S.mono,color:col.severity==='high'?C.red:col.severity==='medium'?C.amber:C.blue}}>
                        {col.missing} ({col.missing_pct}%) · {col.pattern}
                      </span>
                    </div>
                    <ProgressBar value={col.missing_pct/100} color={col.severity==='high'?C.red:col.severity==='medium'?C.amber:C.blue}/>
                  </div>
                ))}
              </GlowCard>
              <GlowCard accent={C.red}>
                <div style={{...S.label}}>Outlier Detection Summary</div>
                {[['IQR Fence (1.5×)',result.outliers.n_outliers_iqr,C.amber],
                  ['Z-Score |z|>3',  result.outliers.n_outliers_z,  C.red],
                  ['IsolationForest',result.outliers.n_outliers_iso, C.purple],
                  ['Any Method',     result.outliers.n_outliers_any, C.red]].map(([l,n,c])=>(
                  <div key={l} style={{display:'flex',justifyContent:'space-between',alignItems:'center',marginBottom:10}}>
                    <span style={{fontSize:12,color:C.textMid}}>{l}</span>
                    <div style={{display:'flex',alignItems:'center',gap:8}}>
                      <ProgressBar value={n/result.n_rows} color={c} height={5}/>
                      <span style={{fontSize:12,...S.mono,color:c,width:50,textAlign:'right'}}>{n}</span>
                    </div>
                  </div>
                ))}
              </GlowCard>
              {Object.keys(result.dataset_stats).length>0&&(
                <GlowCard accent={C.green} style={{gridColumn:'1/-1'}}>
                  <div style={{...S.label}}>Feature Statistics</div>
                  <div style={{overflowX:'auto'}}>
                    <table style={{width:'100%',borderCollapse:'collapse',fontSize:11}}>
                      <thead><tr style={{borderBottom:`1px solid ${C.border}`}}>
                        {['Feature','Mean','Std','Min','Max'].map(h=>(
                          <th key={h} style={{padding:'7px 12px',textAlign:'left',color:C.textDim,fontWeight:600,fontSize:10,textTransform:'uppercase',letterSpacing:'0.07em'}}>{h}</th>
                        ))}
                      </tr></thead>
                      <tbody>
                        {Object.entries(result.dataset_stats).map(([col,s])=>(
                          <tr key={col} style={{borderBottom:`1px solid ${C.border}`}}
                            onMouseEnter={e=>e.currentTarget.style.background='rgba(255,255,255,0.03)'}
                            onMouseLeave={e=>e.currentTarget.style.background='transparent'}>
                            <td style={{padding:'7px 12px',...S.mono,color:C.text,fontSize:11}}>{col}</td>
                            {[s.mean,s.std,s.min,s.max].map((v,i)=>(
                              <td key={i} style={{padding:'7px 12px',...S.mono,color:C.textMid,fontSize:11}}>{v}</td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </GlowCard>
              )}
            </div>
          )}

          {/* Missing tab */}
          {tab==='missing'&&(
            <GlowCard accent={C.amber} className="fade-in">
              <div style={{...S.label}}>Flagged Rows — Missing Values (no imputation applied)</div>
              {result.missing.flagged_rows.length===0
                ?<div style={{fontSize:13,color:C.green,padding:'20px 0'}}>✓ No missing values in any row</div>
                :<div style={{overflowX:'auto',maxHeight:400,overflowY:'auto'}}>
                  <table style={{width:'100%',borderCollapse:'collapse',fontSize:11}}>
                    <thead style={{position:'sticky',top:0,background:C.bg2}}>
                      <tr>{['Row','Batch','# Missing','Missing Columns'].map(h=>(
                        <th key={h} style={{padding:'8px 12px',textAlign:'left',color:C.textDim,fontWeight:600,fontSize:10,textTransform:'uppercase',letterSpacing:'0.07em',borderBottom:`1px solid ${C.border}`}}>{h}</th>
                      ))}</tr>
                    </thead>
                    <tbody>
                      {result.missing.flagged_rows.slice(0,100).map(r=>(
                        <tr key={r.row_index} style={{borderBottom:`1px solid ${C.border}`}}
                          onMouseEnter={e=>e.currentTarget.style.background='rgba(255,255,255,0.03)'}
                          onMouseLeave={e=>e.currentTarget.style.background='transparent'}>
                          <td style={{padding:'7px 12px',...S.mono,color:C.text}}>{r.row_index}</td>
                          <td style={{padding:'7px 12px',...S.mono,color:C.textMid}}>{r.batch_id??'—'}</td>
                          <td style={{padding:'7px 12px'}}><Badge text={r.n_missing} color={r.n_missing>3?C.red:C.amber}/></td>
                          <td style={{padding:'7px 12px',fontSize:10,...S.mono,color:C.textMid}}>{r.missing_cols.join(', ')}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>}
            </GlowCard>
          )}

          {/* Outliers tab */}
          {tab==='outliers'&&(
            <GlowCard accent={C.red} className="fade-in">
              <div style={{...S.label}}>Flagged Rows — Outliers</div>
              <div style={{display:'flex',gap:8,marginBottom:14,flexWrap:'wrap'}}>
                <Badge text={`${result.outliers.n_outliers_any} total outliers`} color={C.red}/>
                <Badge text={`${result.outliers.outlier_pct}% of data`} color={C.amber}/>
                <Badge text="IQR + Z-score + IsolationForest" color={C.blue}/>
              </div>
              <div style={{overflowX:'auto',maxHeight:380,overflowY:'auto'}}>
                <table style={{width:'100%',borderCollapse:'collapse',fontSize:11}}>
                  <thead style={{position:'sticky',top:0,background:C.bg2}}>
                    <tr>{['Row','Batch','Methods','IQR','Z','IsoForest','Flagged Cols'].map(h=>(
                      <th key={h} style={{padding:'8px 12px',textAlign:'left',color:C.textDim,fontWeight:600,fontSize:10,textTransform:'uppercase',letterSpacing:'0.07em',borderBottom:`1px solid ${C.border}`}}>{h}</th>
                    ))}</tr>
                  </thead>
                  <tbody>
                    {result.outliers.flagged_rows.slice(0,100).map(r=>(
                      <tr key={r.row_index}
                        style={{borderBottom:`1px solid ${C.border}`,background:r.n_methods===3?'rgba(229,57,53,0.06)':'transparent'}}
                        onMouseEnter={e=>e.currentTarget.style.background='rgba(255,255,255,0.04)'}
                        onMouseLeave={e=>e.currentTarget.style.background=r.n_methods===3?'rgba(229,57,53,0.06)':'transparent'}>
                        <td style={{padding:'7px 12px',...S.mono,color:C.text}}>{r.row_index}</td>
                        <td style={{padding:'7px 12px',...S.mono,color:C.textMid}}>{r.batch_id??'—'}</td>
                        <td style={{padding:'7px 12px'}}><Badge text={`${r.n_methods}/3`} color={r.n_methods===3?C.red:r.n_methods===2?C.amber:C.blue}/></td>
                        <td style={{padding:'7px 12px',textAlign:'center',color:r.iqr_outlier?C.amber:C.textDim}}>{r.iqr_outlier?'✓':'—'}</td>
                        <td style={{padding:'7px 12px',textAlign:'center',color:r.z_outlier?C.red:C.textDim}}>{r.z_outlier?'✓':'—'}</td>
                        <td style={{padding:'7px 12px',textAlign:'center',color:r.iso_outlier?C.purple:C.textDim}}>{r.iso_outlier?'✓':'—'}</td>
                        <td style={{padding:'7px 12px',fontSize:10,...S.mono,color:C.textMid}}>{r.flagged_cols.join(', ')||'—'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </GlowCard>
          )}

          {/* Drift tab */}
          {tab==='drift'&&(
            <GlowCard accent={C.purple} className="fade-in">
              <div style={{...S.label}}>Batch Drift Analysis — KS Test (α=0.05)</div>
              {!result.drift.n_batches
                ?<div style={{fontSize:13,color:C.textMid}}>No Batch_ID column — drift analysis not available</div>
                :<>
                  <div style={{display:'flex',gap:10,marginBottom:16,flexWrap:'wrap'}}>
                    <StatPill label="Total Batches"   value={result.drift.n_batches}            color={C.blue}/>
                    <StatPill label="Drifted Batches" value={result.drift.n_drifted_batches}     color={result.drift.n_drifted_batches>0?C.amber:C.green}/>
                    <StatPill label="% Drifted"       value={`${result.drift.pct_batches_drifted}%`} color={result.drift.pct_batches_drifted>20?C.red:C.green}/>
                  </div>
                  <div style={{fontSize:12,color:C.textMid,marginBottom:12}}>Feature drift frequency (batches where p&lt;0.05):</div>
                  {result.drift.feature_drift_summary.map(f=>(
                    <div key={f.feature} style={{marginBottom:10}}>
                      <div style={{display:'flex',justifyContent:'space-between',marginBottom:4}}>
                        <span style={{fontSize:11,...S.mono,color:C.text}}>{f.feature}</span>
                        <span style={{fontSize:11,color:f.pct_batches>30?C.red:f.pct_batches>10?C.amber:C.green}}>
                          {f.n_batches_drifted}/{result.drift.n_batches} batches ({f.pct_batches}%)
                        </span>
                      </div>
                      <ProgressBar value={f.pct_batches/100} color={f.pct_batches>30?C.red:f.pct_batches>10?C.amber:C.green}/>
                    </div>
                  ))}
                </>}
            </GlowCard>
          )}

          {/* Preview tab */}
          {tab==='preview'&&(
            <GlowCard accent={C.teal} className="fade-in">
              <div style={{...S.label}}>First 5 Rows — {result.feature_cols.length} features · target: {result.target_col}</div>
              <div style={{overflowX:'auto'}}>
                <table style={{width:'100%',borderCollapse:'collapse',fontSize:11}}>
                  <thead><tr style={{borderBottom:`1px solid ${C.border}`}}>
                    {result.columns.map(h=>(
                      <th key={h} style={{padding:'8px 12px',textAlign:'left',color:C.textDim,fontWeight:600,fontSize:10,textTransform:'uppercase',letterSpacing:'0.06em',whiteSpace:'nowrap'}}>{h}</th>
                    ))}
                  </tr></thead>
                  <tbody>
                    {result.preview.map((row,i)=>(
                      <tr key={i} style={{borderBottom:`1px solid ${C.border}`}}>
                        {result.columns.map(col=>(
                          <td key={col} style={{padding:'7px 12px',...S.mono,color:col===result.target_col?C.green:C.textMid,fontSize:11,whiteSpace:'nowrap'}}>
                            {typeof row[col]==='number'?row[col].toFixed(3):String(row[col]??'—')}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </GlowCard>
          )}

          {/* CTA */}
          <div style={{display:'flex',justifyContent:'flex-end'}}>
            <button onClick={()=>onDone(result, result._file || file)}
              style={{display:'flex',alignItems:'center',gap:10,padding:'13px 28px',borderRadius:12,
                border:'none',background:`linear-gradient(135deg,${C.accent},#c026d3)`,
                color:'#fff',fontSize:14,fontWeight:700,cursor:'pointer',
                fontFamily:'DM Sans,sans-serif',boxShadow:'0 4px 20px rgba(204,0,0,0.35)',
                transition:'all 0.2s'}}>
              Continue to Model Selection <ChevronRight size={18}/>
            </button>
          </div>
        </>
      )}
    </div>
  );
}

// ── Step 2: Model Selection & Training ───────────────────────────
function ModelSelect({cleanseResult,uploadedFile,onDone}){
  const [selectedModels,setSelectedModels]=useState(new Set(['ridge','random_forest','xgboost','gpr']));
  const [altFile,setAltFile]=useState(null);   // only needed if user wants a different file
  const [loading,setLoading]=useState(false);
  const [error,setError]=useState('');

  // Use the file from cleansing step by default; altFile overrides it
  const activeFile = altFile || uploadedFile;

  const toggle=id=>{
    const s=new Set(selectedModels);
    s.has(id)?s.delete(id):s.add(id);
    setSelectedModels(s);
  };

  const train=async()=>{
    if(selectedModels.size===0){setError('Select at least one model.');return;}
    if(!activeFile){setError('No dataset available. Please go back and upload your CSV first.');return;}

    setLoading(true);setError('');
    const fd=new FormData();
    fd.append('file',activeFile);
    fd.append('models',JSON.stringify([...selectedModels]));
    try{
      const {data}=await axios.post('/api/upload/train',fd);
      onDone(data);
    }catch(e){setError(e.response?.data?.detail||e.message);}
    finally{setLoading(false);}
  };

  return(
    <div style={{display:'flex',flexDirection:'column',gap:20}}>
      <GlowCard accent={C.green} className="slide-up">
        <div style={{display:'flex',alignItems:'center',gap:12,padding:'14px 16px',borderRadius:10,
          background:C.greenL,border:`1px solid ${C.green}30`}}>
          <CheckCircle size={18} color={C.green}/>
          <div style={{flex:1}}>
            <div style={{fontSize:13,fontWeight:600,color:C.green}}>
              {activeFile?activeFile.name:cleanseResult?.filename||'Dataset ready'}
            </div>
            <div style={{fontSize:11,color:C.textMid}}>
              {activeFile?(activeFile.size/1024).toFixed(1)+' KB · ':''}
              {cleanseResult?.n_rows?.toLocaleString()} rows · {cleanseResult?.n_features} features · carried over from cleansing step
            </div>
          </div>
          <button onClick={()=>setAltFile(null)}
            style={{padding:'5px 12px',borderRadius:7,border:`1px solid ${C.border2}`,
              background:'transparent',color:C.textMid,cursor:'pointer',fontSize:11,
              display:altFile?'block':'none'}}>
            Reset to original
          </button>
        </div>
        <div style={{marginTop:12,fontSize:12,color:C.textDim}}>
          Want to use a different file instead?{' '}
          <label style={{color:C.blue,cursor:'pointer',textDecoration:'underline'}}>
            Upload alternative
            <input type="file" accept=".csv" style={{display:'none'}}
              onChange={e=>e.target.files[0]&&setAltFile(e.target.files[0])}/>
          </label>
        </div>
      </GlowCard>

      <GlowCard accent={C.purple}>
        <div style={{...S.label,marginBottom:14}}>Select Models to Train</div>
        <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:10}}>
          {ALL_MODELS.map(m=>{
            const sel=selectedModels.has(m.id);
            const col=MODEL_COLORS[m.id]||C.blue;
            return(
              <div key={m.id} onClick={()=>toggle(m.id)}
                style={{padding:'14px 16px',borderRadius:11,cursor:'pointer',
                  border:`1px solid ${sel?col:C.border}`,
                  background:sel?col+'15':'rgba(255,255,255,0.02)',
                  transition:'all 0.15s',display:'flex',alignItems:'flex-start',gap:10}}>
                <div style={{width:18,height:18,borderRadius:4,border:`2px solid ${sel?col:C.textDim}`,
                  background:sel?col:'transparent',display:'flex',alignItems:'center',justifyContent:'center',
                  flexShrink:0,marginTop:1}}>
                  {sel&&<CheckCircle size={11} color="#fff"/>}
                </div>
                <div>
                  <div style={{fontSize:13,fontWeight:600,color:sel?col:C.text}}>{m.name}</div>
                  <div style={{fontSize:11,color:C.textMid,marginTop:2}}>{m.desc}</div>
                </div>
              </div>
            );
          })}
        </div>
        <div style={{marginTop:12,fontSize:11,color:C.textDim}}>
          {selectedModels.size} model{selectedModels.size!==1?'s':''} selected
          {selectedModels.has('ann')&&<span style={{color:C.amber}}> · ⚠ ANN requires N&gt;5,000 for reliable results</span>}
          {selectedModels.has('gpr')&&<span style={{color:C.purple}}> · GPR subsampled to 300 rows for scalability</span>}
        </div>
      </GlowCard>

      {error&&(
        <GlowCard accent={C.red}>
          <div style={{display:'flex',gap:10}}>
            <AlertCircle size={16} color={C.red} style={{flexShrink:0,marginTop:1}}/>
            <div style={{fontSize:13,color:C.red}}>{error}</div>
          </div>
        </GlowCard>
      )}

      <div style={{display:'flex',justifyContent:'flex-end'}}>
        <button onClick={train} disabled={loading||selectedModels.size===0||!activeFile}
          style={{display:'flex',alignItems:'center',gap:10,padding:'13px 28px',borderRadius:12,
            border:'none',background:loading||selectedModels.size===0||!activeFile?C.bg4:`linear-gradient(135deg,${C.accent},#c026d3)`,
            color:loading||selectedModels.size===0||!activeFile?C.textDim:'#fff',
            fontSize:14,fontWeight:700,cursor:loading||selectedModels.size===0||!activeFile?'not-allowed':'pointer',
            fontFamily:'DM Sans,sans-serif',transition:'all 0.2s',
            boxShadow:loading||selectedModels.size===0||!activeFile?'none':'0 4px 20px rgba(204,0,0,0.35)'}}>
          {loading?<><Loader size={16} className="spin"/>Training models…</> :<><Play size={16}/>Train {selectedModels.size} Models</>}
        </button>
      </div>
    </div>
  );
}

// ── Step 3: Results Dashboard (PowerBI style) ────────────────────
function ResultsDashboard({trainResult,onNext}){
  const [visibleModels,setVisibleModels]=useState(new Set(trainResult.results.map(r=>r.model_id)));
  const [activeTab,setActiveTab]=useState('overview');
  const [selectedModel,setSelectedModel]=useState(trainResult.results[0]?.model_id);

  const visible=trainResult.results.filter(r=>visibleModels.has(r.model_id));
  const best=trainResult.results.reduce((a,b)=>b.r2_test>a.r2_test?b:a);
  const ds=trainResult.dataset;

  const toggleModel=id=>{
    const s=new Set(visibleModels);
    if(s.has(id)){if(s.size>1)s.delete(id);}else s.add(id);
    setVisibleModels(s);
  };

  // ── Chart data ──────────────────────────────────────────────
  const r2ChartData={
    labels:visible.map(r=>r.model_name),
    datasets:[
      {label:'R² Train',data:visible.map(r=>r.r2_train),
       backgroundColor:visible.map(r=>(MODEL_COLORS[r.model_id]||C.blue)+'40'),
       borderColor:visible.map(r=>MODEL_COLORS[r.model_id]||C.blue),borderWidth:2,borderRadius:6},
      {label:'R² Test', data:visible.map(r=>r.r2_test),
       backgroundColor:visible.map(r=>(MODEL_COLORS[r.model_id]||C.blue)+'aa'),
       borderColor:visible.map(r=>MODEL_COLORS[r.model_id]||C.blue),borderWidth:2,borderRadius:6},
    ],
  };

  const selModel=trainResult.results.find(r=>r.model_id===selectedModel);

  const scatterData={
    datasets:[{
      label:'Actual vs Predicted',
      data:selModel?.scatter||[],
      backgroundColor:(MODEL_COLORS[selectedModel]||C.blue)+'80',
      borderColor:MODEL_COLORS[selectedModel]||C.blue,
      pointRadius:5,pointHoverRadius:7,
    }],
  };

  const lcData={
    labels:(trainResult.lc_sizes||[]).map(s=>s>=1000?`${(s/1000).toFixed(1)}k`:String(s)),
    datasets:Object.entries(trainResult.learning_curve||{})
      .filter(([k])=>visibleModels.has(k))
      .map(([k,v])=>({
        label:MODEL_LABELS[k]||k,data:v,
        borderColor:MODEL_COLORS[k]||C.blue,backgroundColor:'transparent',
        tension:0.4,pointRadius:3,fill:false,
        borderDash:k==='ridge'||k==='plsr'?[4,4]:undefined,
      })),
  };

  const radarData={
    labels:['Accuracy','Interpretability','Uncertainty','Speed','Regulatory'],
    datasets:visible.slice(0,5).map(r=>({
      label:r.model_name,
      data:[
        r.r2_test*5,
        {ridge:5,plsr:5,random_forest:3,xgboost:4,gpr:3,ann:2,hybrid:4}[r.model_id]||3,
        {ridge:2,plsr:2,random_forest:3,xgboost:2,gpr:5,ann:1,hybrid:3}[r.model_id]||2,
        {ridge:5,plsr:5,random_forest:4,xgboost:4,gpr:2,ann:3,hybrid:3}[r.model_id]||3,
        {ridge:4,plsr:5,random_forest:3,xgboost:4,gpr:5,ann:2,hybrid:4}[r.model_id]||3,
      ],
      borderColor:MODEL_COLORS[r.model_id]||C.blue,
      backgroundColor:(MODEL_COLORS[r.model_id]||C.blue)+'18',
      borderWidth:2,pointRadius:4,
    })),
  };

  // SHAP / feature importance for selected model
  const featImp=selModel?.shap_importance||selModel?.feature_importance||
                selModel?.vip_scores||{};
  const featEntries=Object.entries(featImp).sort((a,b)=>b[1]-a[1]).slice(0,10);
  const featColor=MODEL_COLORS[selectedModel]||C.blue;

  const featChartData={
    labels:featEntries.map(([k])=>k),
    datasets:[{
      label:'Importance',
      data:featEntries.map(([,v])=>v),
      backgroundColor:featEntries.map((_,i)=>i<3?featColor+'cc':featColor+'55'),
      borderColor:featColor,borderWidth:1,borderRadius:5,
    }],
  };

  const tabs=[
    {id:'overview',    label:'Overview'},
    {id:'r2',          label:'R² Comparison'},
    {id:'scatter',     label:'Actual vs Predicted'},
    {id:'features',    label:'Feature Importance'},
    {id:'radar',       label:'Multi-Criteria'},
    {id:'learning',    label:'Learning Curves'},
  ];

  return(
    <div style={{display:'flex',flexDirection:'column',gap:16}}>
      {/* KPI Row */}
      <div style={{display:'grid',gridTemplateColumns:'repeat(4,1fr)',gap:12}}>
        {[
          {label:'Best Model',     value:best.model_name,       sub:`R²=${best.r2_test.toFixed(4)}`, color:C.green},
          {label:'Best R² Test',   value:best.r2_test.toFixed(4),sub:`RMSE=${best.rmse.toFixed(4)}`, color:C.blue},
          {label:'Dataset',        value:`N=${ds.n_complete.toLocaleString()}`, sub:`${ds.n_features} features`,color:C.purple},
          {label:'Target Range',   value:`${ds.target_min.toFixed(1)}–${ds.target_max.toFixed(1)}%`,sub:`μ=${ds.target_mean.toFixed(2)} σ=${ds.target_std.toFixed(2)}`,color:C.amber},
        ].map(s=>(
          <div key={s.label} style={{...S.card,background:s.color+'10',border:`1px solid ${s.color}25`}}>
            <div style={{...S.label,color:s.color+'aa'}}>{s.label}</div>
            <div style={{fontSize:20,fontWeight:800,color:s.color,...S.mono,lineHeight:1.2,marginBottom:2}}>{s.value}</div>
            <div style={{fontSize:11,color:C.textDim}}>{s.sub}</div>
          </div>
        ))}
      </div>

      {/* Model visibility toggles */}
      <GlowCard accent={C.blue}>
        <div style={{...S.label,marginBottom:10}}>Model Visibility — click to show/hide</div>
        <div style={{display:'flex',gap:8,flexWrap:'wrap'}}>
          {trainResult.results.map(r=>{
            const on=visibleModels.has(r.model_id);
            const col=MODEL_COLORS[r.model_id]||C.blue;
            return(
              <button key={r.model_id} onClick={()=>toggleModel(r.model_id)}
                style={{display:'flex',alignItems:'center',gap:7,padding:'7px 14px',borderRadius:9,
                  border:`1px solid ${on?col:C.border}`,
                  background:on?col+'18':'transparent',cursor:'pointer',
                  fontSize:12,fontWeight:on?600:400,color:on?col:C.textMid,
                  fontFamily:'DM Sans,sans-serif',transition:'all 0.15s'}}>
                <div style={{width:8,height:8,borderRadius:'50%',background:on?col:C.textDim}}/>
                {r.model_name}
                <span style={{fontSize:10,...S.mono,opacity:0.7}}>{r.r2_test.toFixed(2)}</span>
                {on?<Eye size={12}/>:<EyeOff size={12}/>}
              </button>
            );
          })}
        </div>
      </GlowCard>

      {/* Tab navigation */}
      <div style={{display:'flex',borderBottom:`1px solid ${C.border}`,gap:0,overflowX:'auto'}}>
        {tabs.map(t=>(
          <button key={t.id} onClick={()=>setActiveTab(t.id)}
            style={{padding:'9px 18px',border:'none',background:'none',cursor:'pointer',
              fontSize:12,fontWeight:activeTab===t.id?600:400,fontFamily:'DM Sans,sans-serif',
              color:activeTab===t.id?C.blue:C.textMid,
              borderBottom:activeTab===t.id?`2px solid ${C.blue}`:'2px solid transparent',
              transition:'all 0.15s',whiteSpace:'nowrap'}}>
            {t.label}
          </button>
        ))}
      </div>

      {/* Overview */}
      {activeTab==='overview'&&(
        <div className="fade-in" style={{display:'flex',flexDirection:'column',gap:12}}>
          <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:12}}>
            <GlowCard accent={C.green}>
              <div style={{...S.label,marginBottom:14}}>All Models — R² Test Score</div>
              <div style={{height:220}}><Bar data={r2ChartData} options={{...chartBase,
                scales:{...chartBase.scales,y:{...chartBase.scales.y,min:0,max:1.05}}}}/></div>
            </GlowCard>
            <GlowCard accent={C.purple}>
              <div style={{...S.label,marginBottom:14}}>Multi-Criteria Radar</div>
              <div style={{height:220}}><Radar data={radarData} options={{responsive:true,maintainAspectRatio:false,
                plugins:{legend:{labels:{font:{family:'DM Sans',size:10},color:C.textMid}}},
                scales:{r:{min:0,max:5,ticks:{stepSize:1,color:C.textDim,backdropColor:'transparent',font:{size:8}},
                  grid:{color:'rgba(255,255,255,0.05)'},pointLabels:{font:{family:'DM Sans',size:10},color:C.textMid}}}}}/></div>
            </GlowCard>
          </div>
          {/* Model cards grid */}
          <div style={{display:'grid',gridTemplateColumns:'repeat(auto-fill,minmax(200px,1fr))',gap:10}}>
            {trainResult.results.sort((a,b)=>b.r2_test-a.r2_test).map((r,i)=>{
              const col=MODEL_COLORS[r.model_id]||C.blue;
              return(
                <div key={r.model_id} style={{...S.card,border:`1px solid ${r.model_id===best.model_id?col:C.border}`,
                  background:r.model_id===best.model_id?col+'10':C.bg2,cursor:'pointer',transition:'all 0.2s'}}
                  onClick={()=>{setSelectedModel(r.model_id);setActiveTab('scatter');}}>
                  <div style={{display:'flex',alignItems:'center',justifyContent:'space-between',marginBottom:8}}>
                    <div style={{width:10,height:10,borderRadius:'50%',background:col}}/>
                    {r.model_id===best.model_id&&<Badge text="BEST" color={col}/>}
                    {i===0&&r.r2_test<0.1&&<Badge text="LOW R²" color={C.red}/>}
                  </div>
                  <div style={{fontSize:12,fontWeight:600,color:C.text,marginBottom:6}}>{r.model_name}</div>
                  <div style={{fontSize:24,fontWeight:800,color:col,...S.mono,lineHeight:1}}>{r.r2_test.toFixed(3)}</div>
                  <div style={{fontSize:10,color:C.textDim,marginBottom:8}}>R² test</div>
                  <ProgressBar value={Math.max(0,r.r2_test)} color={col}/>
                  <div style={{display:'flex',justifyContent:'space-between',marginTop:8}}>
                    <span style={{fontSize:10,...S.mono,color:C.textDim}}>RMSE {r.rmse.toFixed(3)}</span>
                    <span style={{fontSize:10,...S.mono,color:C.textDim}}>{r.train_time}s</span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* R² comparison */}
      {activeTab==='r2'&&(
        <GlowCard accent={C.green} className="fade-in">
          <div style={{...S.label,marginBottom:14}}>R² Train vs Test — {visible.length} models</div>
          <div style={{height:300}}>
            <Bar data={r2ChartData} options={{...chartBase,
              scales:{...chartBase.scales,y:{...chartBase.scales.y,min:0,max:1.05,
                title:{display:true,text:'R² Score',color:C.textMid,font:{size:11}}}}}}/>
          </div>
          <div style={{overflowX:'auto',marginTop:16}}>
            <table style={{width:'100%',borderCollapse:'collapse',fontSize:11}}>
              <thead><tr style={{borderBottom:`1px solid ${C.border}`}}>
                {['Model','R² Train','R² Test','RMSE','MAE','Train Time'].map(h=>(
                  <th key={h} style={{padding:'8px 12px',textAlign:'left',color:C.textDim,fontWeight:600,fontSize:10,textTransform:'uppercase',letterSpacing:'0.07em'}}>{h}</th>
                ))}
              </tr></thead>
              <tbody>
                {visible.sort((a,b)=>b.r2_test-a.r2_test).map(r=>{
                  const col=MODEL_COLORS[r.model_id]||C.blue;
                  return(
                    <tr key={r.model_id} style={{borderBottom:`1px solid ${C.border}`,
                      background:r.model_id===best.model_id?col+'08':'transparent'}}
                      onMouseEnter={e=>e.currentTarget.style.background='rgba(255,255,255,0.04)'}
                      onMouseLeave={e=>e.currentTarget.style.background=r.model_id===best.model_id?col+'08':'transparent'}>
                      <td style={{padding:'9px 12px',fontWeight:600,color:col}}>
                        <div style={{display:'flex',alignItems:'center',gap:7}}>
                          <div style={{width:8,height:8,borderRadius:'50%',background:col}}/>
                          {r.model_name}
                          {r.model_id===best.model_id&&<Badge text="★" color={col}/>}
                        </div>
                      </td>
                      {[r.r2_train,r.r2_test].map((v,vi)=>(
                        <td key={vi} style={{padding:'9px 12px'}}>
                          <div style={{display:'flex',alignItems:'center',gap:8}}>
                            <div style={{width:50,background:'rgba(255,255,255,0.05)',borderRadius:3,height:5}}>
                              <div style={{width:`${Math.max(0,v)*100}%`,height:'100%',borderRadius:3,background:col}}/>
                            </div>
                            <span style={{...S.mono,color:v<0.2?C.red:C.text}}>{v.toFixed(4)}</span>
                          </div>
                        </td>
                      ))}
                      <td style={{padding:'9px 12px',...S.mono,color:C.textMid}}>{r.rmse.toFixed(4)}</td>
                      <td style={{padding:'9px 12px',...S.mono,color:C.textMid}}>{r.mae.toFixed(4)}</td>
                      <td style={{padding:'9px 12px',...S.mono,color:C.textDim}}>{r.train_time}s</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </GlowCard>
      )}

      {/* Scatter */}
      {activeTab==='scatter'&&(
        <GlowCard accent={MODEL_COLORS[selectedModel]||C.blue} className="fade-in">
          <div style={{display:'flex',alignItems:'center',gap:12,marginBottom:14}}>
            <div style={{...S.label,margin:0}}>Actual vs Predicted —</div>
            <div style={{display:'flex',gap:8,flexWrap:'wrap'}}>
              {trainResult.results.map(r=>(
                <button key={r.model_id} onClick={()=>setSelectedModel(r.model_id)}
                  style={{padding:'5px 12px',borderRadius:7,border:`1px solid ${r.model_id===selectedModel?(MODEL_COLORS[r.model_id]||C.blue):C.border}`,
                    background:r.model_id===selectedModel?(MODEL_COLORS[r.model_id]||C.blue)+'20':'transparent',
                    cursor:'pointer',fontSize:11,color:r.model_id===selectedModel?(MODEL_COLORS[r.model_id]||C.blue):C.textMid,
                    fontFamily:'DM Sans,sans-serif',transition:'all 0.15s'}}>
                  {r.model_name}
                </button>
              ))}
            </div>
          </div>
          {selModel&&(
            <>
              <div style={{display:'flex',gap:10,marginBottom:14,flexWrap:'wrap'}}>
                <StatPill label="R² Test"  value={selModel.r2_test.toFixed(4)} color={MODEL_COLORS[selectedModel]||C.blue}/>
                <StatPill label="RMSE"     value={selModel.rmse.toFixed(4)}    color={C.amber}/>
                <StatPill label="MAE"      value={selModel.mae.toFixed(4)}     color={C.purple}/>
                <StatPill label="N Test"   value={selModel.n_test}             color={C.teal}/>
                {selModel.ci_coverage_95&&<StatPill label="95% CI" value={`${selModel.ci_coverage_95}%`} color={C.green}/>}
                {selModel.total_params&&<StatPill label="Params" value={selModel.total_params.toLocaleString()} color={selModel.overparameterised?C.red:C.green}/>}
              </div>
              <div style={{height:300}}>
                <Scatter data={scatterData} options={{...chartBase,
                  plugins:{...chartBase.plugins,tooltip:{...chartBase.plugins.tooltip,
                    callbacks:{label:(ctx)=>`Actual: ${ctx.parsed.x.toFixed(3)}, Predicted: ${ctx.parsed.y.toFixed(3)}`}}},
                  scales:{
                    x:{...chartBase.scales.x,title:{display:true,text:'Actual Fucosylation (%)',color:C.textMid,font:{size:11}}},
                    y:{...chartBase.scales.y,title:{display:true,text:'Predicted Fucosylation (%)',color:C.textMid,font:{size:11}}},
                  }}}/>
              </div>
              {selModel.overparameterised&&(
                <div style={{marginTop:12,padding:'10px 14px',background:C.redL,border:`1px solid ${C.red}30`,
                  borderRadius:9,fontSize:12,color:C.red}}>
                  ⚠ ANN is overparameterised — {selModel.total_params?.toLocaleString()} parameters vs {selModel.n_train} training samples
                  (ratio={selModel.param_to_sample_ratio}×). Consider increasing dataset size or reducing architecture.
                </div>
              )}
            </>
          )}
        </GlowCard>
      )}

      {/* Feature importance */}
      {activeTab==='features'&&(
        <GlowCard accent={MODEL_COLORS[selectedModel]||C.blue} className="fade-in">
          <div style={{display:'flex',alignItems:'center',gap:10,marginBottom:14}}>
            <div style={{...S.label,margin:0}}>Feature Importance —</div>
            <div style={{display:'flex',gap:8,flexWrap:'wrap'}}>
              {trainResult.results.filter(r=>{
                const m=r.shap_importance||r.feature_importance||r.vip_scores;
                return m&&Object.keys(m).length>0;
              }).map(r=>(
                <button key={r.model_id} onClick={()=>setSelectedModel(r.model_id)}
                  style={{padding:'5px 12px',borderRadius:7,
                    border:`1px solid ${r.model_id===selectedModel?(MODEL_COLORS[r.model_id]||C.blue):C.border}`,
                    background:r.model_id===selectedModel?(MODEL_COLORS[r.model_id]||C.blue)+'20':'transparent',
                    cursor:'pointer',fontSize:11,color:r.model_id===selectedModel?(MODEL_COLORS[r.model_id]||C.blue):C.textMid,
                    fontFamily:'DM Sans,sans-serif',transition:'all 0.15s'}}>
                  {r.model_name}
                </button>
              ))}
            </div>
          </div>
          {featEntries.length>0?(
            <>
              <div style={{fontSize:11,color:C.textMid,marginBottom:14}}>
                {selectedModel==='xgboost'?'SHAP mean |value| — theoretically grounded (Lundberg & Lee 2017)':
                 selectedModel==='plsr'?'VIP scores — >1.0 = significant (Wold et al. 2001)':
                 'Mean Decrease in Impurity — feature importance'}
              </div>
              <div style={{height:260}}>
                <Bar data={featChartData} options={{...chartBase,indexAxis:'y',
                  scales:{...chartBase.scales,
                    x:{...chartBase.scales.x,title:{display:true,text:'Importance',color:C.textMid,font:{size:10}}},
                    y:{...chartBase.scales.y,ticks:{...chartBase.scales.y.ticks,font:{family:'DM Mono',size:9}}}}}}/>
              </div>
              {selectedModel==='plsr'&&selModel?.extras?.vip_scores&&(
                <div style={{marginTop:10,fontSize:11,color:C.textMid}}>
                  VIP &gt; 1.0 = significant driver. Top 3: {featEntries.filter(([,v])=>v>1.0).slice(0,3).map(([k,v])=>`${k} (${v.toFixed(2)})`).join(', ')}
                </div>
              )}
            </>
          ):(
            <div style={{fontSize:13,color:C.textMid,padding:'20px 0'}}>
              No feature importance available for this model. Select XGBoost, RF, PLSR, or Hybrid.
            </div>
          )}
        </GlowCard>
      )}

      {/* Radar */}
      {activeTab==='radar'&&(
        <GlowCard accent={C.purple} className="fade-in">
          <div style={{...S.label,marginBottom:14}}>Multi-Criteria Evaluation — Top 5 visible models</div>
          <div style={{height:350}}>
            <Radar data={radarData} options={{responsive:true,maintainAspectRatio:false,
              plugins:{legend:{labels:{font:{family:'DM Sans',size:11},color:C.textMid,padding:14}}},
              scales:{r:{min:0,max:5,ticks:{stepSize:1,color:C.textDim,backdropColor:'transparent',font:{size:9}},
                grid:{color:'rgba(255,255,255,0.06)'},pointLabels:{font:{family:'DM Sans',size:11},color:C.textMid}}}}}/>
          </div>
        </GlowCard>
      )}

      {/* Learning curves */}
      {activeTab==='learning'&&(
        <GlowCard accent={C.teal} className="fade-in">
          <div style={{...S.label,marginBottom:14}}>Learning Curves — estimated from actual R² values</div>
          <div style={{height:300}}>
            <Line data={lcData} options={{...chartBase,
              scales:{...chartBase.scales,
                y:{...chartBase.scales.y,min:0,max:1,title:{display:true,text:'R² Test',color:C.textMid,font:{size:11}}},
                x:{...chartBase.scales.x,title:{display:true,text:'Training Size (approx.)',color:C.textMid,font:{size:11}}}}}}/>
          </div>
          <div style={{fontSize:11,color:C.textDim,marginTop:10}}>
            Note: curves are estimated from final R² values. Actual learning curves require multiple training runs at varying data sizes.
          </div>
        </GlowCard>
      )}

      <div style={{display:'flex',justifyContent:'flex-end'}}>
        <button onClick={onNext}
          style={{display:'flex',alignItems:'center',gap:10,padding:'13px 28px',borderRadius:12,
            border:'none',background:`linear-gradient(135deg,${C.accent},#c026d3)`,
            color:'#fff',fontSize:14,fontWeight:700,cursor:'pointer',
            fontFamily:'DM Sans,sans-serif',boxShadow:'0 4px 20px rgba(204,0,0,0.35)'}}>
          Generate AI Report <ChevronRight size={18}/>
        </button>
      </div>
    </div>
  );
}

// ── Step 4: AI Report ────────────────────────────────────────────
function AIReport({trainResult}){
  const [reportType,setReportType]=useState('scientific');
  const [report,setReport]=useState('');
  const [loading,setLoading]=useState(false);
  const [error,setError]=useState('');
  const [words,setWords]=useState(0);

  const REPORT_TYPES=[
    {id:'scientific', label:'Scientific Report',
     desc:'Detailed thesis-style report — model analysis, SHAP features, literature citations, ICH Q8/Q9 context. ~800–1000 words.'},
    {id:'executive',  label:'Executive Report',
     desc:'Business briefing for leadership — plain language, no jargon, ROI framing, deployment recommendation. ~400–500 words.'},
    {id:'summary',    label:'Quick Summary',
     desc:'Snapshot for team meetings — headline R² numbers, top features, one recommendation. ~200–250 words.'},
  ];

  const generate=async()=>{
    setLoading(true);setError('');setReport('');
    try{
      const {data}=await axios.post('/api/generate-report',{
        results:trainResult.results,
        dataset_info:trainResult.dataset,
        report_type:reportType,
      });
      setReport(data.report);
      setWords(data.report.split(/\s+/).length);
    }catch(e){setError(e.response?.data?.detail||e.message);}
    finally{setLoading(false);}
  };

  const exportPDF=()=>{
    const ds=trainResult.dataset;
    const best=trainResult.results.reduce((a,b)=>b.r2_test>a.r2_test?b:a);
    const rows=trainResult.results.sort((a,b)=>b.r2_test-a.r2_test).map(r=>`
      <tr class="${r.model_id===best.model_id?'best':''}">
        <td>${r.model_name}${r.model_id===best.model_id?' ★':''}</td>
        <td>${r.r2_train.toFixed(4)}</td><td>${r.r2_test.toFixed(4)}</td>
        <td>${r.rmse.toFixed(4)}</td><td>${r.mae.toFixed(4)}</td>
        <td>${r.train_time}s</td>
      </tr>`).join('');
    const html=`<!DOCTYPE html><html><head><meta charset="UTF-8"/>
<title>BMS ML Report</title>
<style>
  @page{size:A4;margin:22mm 18mm}
  *{box-sizing:border-box;margin:0;padding:0}
  body{font-family:'Times New Roman',serif;font-size:11pt;color:#111;line-height:1.7}
  .header{border-bottom:3px solid #cc0000;padding-bottom:12px;margin-bottom:18px;display:flex;justify-content:space-between;align-items:flex-end}
  h1{font-size:17pt;color:#cc0000;font-weight:700}
  h2{font-size:9pt;color:#555;font-weight:400;margin-top:3px}
  .meta{font-size:8.5pt;color:#777;text-align:right;line-height:1.5}
  .kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:20px}
  .kpi{background:#f8f8f8;border:1px solid #e0e0e0;border-radius:6px;padding:10px}
  .kpi-label{font-size:8pt;color:#cc0000;font-weight:700;text-transform:uppercase;letter-spacing:.06em;margin-bottom:3px}
  .kpi-value{font-size:13pt;font-weight:700;font-family:'Courier New',monospace}
  table{width:100%;border-collapse:collapse;margin:10px 0;font-size:9.5pt}
  th{background:#cc0000;color:#fff;padding:7px 10px;text-align:left;font-weight:600;font-size:9pt}
  td{padding:7px 10px;border-bottom:1px solid #e8e8e8}
  tr:nth-child(even) td{background:#fafafa}
  .best td{font-weight:700;color:#cc0000}
  .section{font-size:12pt;font-weight:700;color:#cc0000;border-bottom:1px solid #e0e0e0;padding-bottom:6px;margin:20px 0 10px}
  .body{font-size:11pt;line-height:1.85;white-space:pre-wrap;text-align:justify}
  .footer{margin-top:24px;padding-top:10px;border-top:1px solid #e0e0e0;display:flex;justify-content:space-between;font-size:8pt;color:#aaa}
  .badge{background:#cc0000;color:#fff;font-size:7.5pt;font-weight:700;padding:2px 7px;border-radius:3px;margin-left:6px}
</style></head><body>
<div class="header">
  <div><h1>mAb Fucosylation ML Pipeline Report <span class="badge">BMS HACKATHON</span></h1>
  <h2>MS Data Science · Rutgers University × Bristol Myers Squibb · Team Data Minds</h2></div>
  <div class="meta">Generated: ${new Date().toLocaleDateString('en-US',{year:'numeric',month:'long',day:'numeric'})}<br/>
  Dataset: ${ds.filename}<br/>Words: ${words.toLocaleString()}</div>
</div>
<div class="kpis">
  <div class="kpi"><div class="kpi-label">Dataset</div><div class="kpi-value">${ds.filename}</div></div>
  <div class="kpi"><div class="kpi-label">N Complete</div><div class="kpi-value">${ds.n_complete.toLocaleString()}</div></div>
  <div class="kpi"><div class="kpi-label">Best Model</div><div class="kpi-value">${best.model_name}</div></div>
  <div class="kpi"><div class="kpi-label">Best R²</div><div class="kpi-value">${best.r2_test.toFixed(4)}</div></div>
</div>
<div class="section">Model Performance Summary</div>
<table><thead><tr><th>Model</th><th>R² Train</th><th>R² Test</th><th>RMSE</th><th>MAE</th><th>Time</th></tr></thead>
<tbody>${rows}</tbody></table>
<div class="section">Scientific Summary — AI Generated</div>
<div class="body">${report}</div>
<div class="footer">
  <span>BMS Hackathon · Team Data Minds (Sanjith Ganesh, Pranav Senthilkumaran) · Rutgers University</span>
  <span>Gemini 2.5 Flash · ICH Q8/Q9 Format</span>
</div></body></html>`;
    const w=window.open('','_blank');
    w.document.write(html);w.document.close();
    setTimeout(()=>w.print(),400);
  };

  return(
    <div style={{display:'flex',flexDirection:'column',gap:20}}>
      <GlowCard accent={C.purple} className="slide-up">
        <div style={{display:'flex',alignItems:'center',gap:12,marginBottom:20}}>
          <div style={{fontSize:17,fontWeight:700,color:C.text}}>AI Report Generator</div>
          <span style={{fontSize:11,padding:'3px 10px',borderRadius:20,background:C.green+'18',
            color:C.green,border:`1px solid ${C.green}30`}}>Live results · Your CSV</span>
        </div>

        {/* Report type selector */}
        <div style={{...S.label,marginBottom:10}}>Select Report Type</div>
        <div style={{display:'flex',flexDirection:'column',gap:8,marginBottom:20}}>
          {REPORT_TYPES.map(rt=>{
            const active=reportType===rt.id;
            const colors={scientific:C.blue,executive:C.amber,summary:C.green};
            const col=colors[rt.id]||C.blue;
            return(
              <div key={rt.id} onClick={()=>setReportType(rt.id)}
                style={{padding:'14px 16px',borderRadius:11,cursor:'pointer',
                  border:`2px solid ${active?col:C.border}`,
                  background:active?col+'12':'rgba(255,255,255,0.02)',
                  transition:'all 0.15s',display:'flex',alignItems:'flex-start',gap:12}}>
                <div style={{width:18,height:18,borderRadius:'50%',border:`2px solid ${active?col:C.textDim}`,
                  background:active?col:'transparent',display:'flex',alignItems:'center',
                  justifyContent:'center',flexShrink:0,marginTop:1}}>
                  {active&&<div style={{width:7,height:7,borderRadius:'50%',background:'#fff'}}/>}
                </div>
                <div>
                  <div style={{fontSize:13,fontWeight:600,color:active?col:C.text,marginBottom:3}}>{rt.label}</div>
                  <div style={{fontSize:11,color:C.textMid,lineHeight:1.5}}>{rt.desc}</div>
                </div>
              </div>
            );
          })}
        </div>

        {/* Model summary pills */}
        <div style={{display:'flex',gap:8,flexWrap:'wrap',marginBottom:20}}>
          {[...trainResult.results].sort((a,b)=>b.r2_test-a.r2_test).map(r=>{
            const col=MODEL_COLORS[r.model_id]||C.blue;
            return(
              <div key={r.model_id} style={{padding:'5px 12px',borderRadius:8,
                background:col+'15',border:`1px solid ${col}25`}}>
                <span style={{fontSize:10,fontWeight:600,color:col}}>{r.model_name}</span>
                <span style={{fontSize:10,...S.mono,color:C.textDim}}> R²={r.r2_test.toFixed(3)}</span>
              </div>
            );
          })}
        </div>

        <button onClick={generate} disabled={loading}
          style={{display:'flex',alignItems:'center',gap:10,padding:'13px 28px',borderRadius:12,
            border:'none',background:loading?C.bg4:`linear-gradient(135deg,${C.accent},#c026d3)`,
            color:loading?C.textDim:'#fff',fontSize:14,fontWeight:700,
            cursor:loading?'not-allowed':'pointer',fontFamily:'DM Sans,sans-serif',
            boxShadow:loading?'none':'0 4px 20px rgba(204,0,0,0.35)',transition:'all 0.2s'}}>
          {loading?<><Loader size={16} className="spin"/>Generating detailed report…</>:<><FileText size={16}/>Generate AI Report</>}
        </button>

        {error&&(
          <div style={{marginTop:12,padding:'10px 14px',background:C.redL,
            border:`1px solid ${C.red}30`,borderRadius:9,display:'flex',gap:9}}>
            <AlertCircle size={15} color={C.red} style={{flexShrink:0,marginTop:1}}/>
            <div style={{fontSize:12,color:C.red}}>{error}</div>
          </div>
        )}
      </GlowCard>

      {report&&(
        <>
          <GlowCard accent={C.green} className="fade-in">
            <div style={{display:'flex',alignItems:'center',justifyContent:'space-between',marginBottom:14}}>
              <div>
                <div style={{fontSize:16,fontWeight:700,color:C.text}}>Generated Report</div>
                <div style={{fontSize:11,color:C.textDim,marginTop:2}}>
                  gemini-2.5-flash · {REPORT_TYPES.find(r=>r.id===reportType)?.label||'Report'} · {words.toLocaleString()} words
                </div>
              </div>
              <div style={{display:'flex',gap:8}}>
                <button onClick={()=>navigator.clipboard.writeText(report)}
                  style={{display:'flex',alignItems:'center',gap:5,padding:'7px 14px',borderRadius:8,
                    border:`1px solid ${C.border2}`,background:'transparent',cursor:'pointer',
                    fontSize:12,color:C.textMid,fontFamily:'DM Sans,sans-serif'}}>
                  <Copy size={13}/>Copy
                </button>
                <button onClick={generate}
                  style={{display:'flex',alignItems:'center',gap:5,padding:'7px 14px',borderRadius:8,
                    border:`1px solid ${C.border2}`,background:'transparent',cursor:'pointer',
                    fontSize:12,color:C.textMid,fontFamily:'DM Sans,sans-serif'}}>
                  <RefreshCw size={13}/>Regenerate
                </button>
              </div>
            </div>
            <div style={{background:C.bg3,borderRadius:12,padding:24,
              fontSize:14,lineHeight:1.95,color:C.text,whiteSpace:'pre-wrap',
              fontFamily:'Georgia,serif',borderLeft:`4px solid ${C.accent}`}}>
              {report}
            </div>
          </GlowCard>

          {/* Export */}
          <GlowCard accent={C.accent} className="fade-in"
            style={{display:'flex',alignItems:'center',justifyContent:'space-between',flexWrap:'wrap',gap:16,
              background:`linear-gradient(135deg,rgba(204,0,0,0.1),rgba(192,38,211,0.1))`,
              border:`1px solid rgba(204,0,0,0.3)`}}>
            <div style={{display:'flex',alignItems:'flex-start',gap:14}}>
              <div style={{width:44,height:44,borderRadius:11,
                background:`linear-gradient(135deg,${C.accent},#c026d3)`,
                display:'flex',alignItems:'center',justifyContent:'center',flexShrink:0}}>
                <Download size={20} color="#fff"/>
              </div>
              <div>
                <div style={{fontSize:15,fontWeight:700,color:C.text,marginBottom:4}}>Report Export</div>
                <div style={{fontSize:13,color:C.textMid,lineHeight:1.5}}>
                  Download regulatory-style PDF summary — model performance table,
                  feature importance rankings, and full AI narrative.
                  Formatted for ICH Q8/Q9 regulatory submissions.
                </div>
                <div style={{display:'flex',gap:8,marginTop:10,flexWrap:'wrap'}}>
                  {[
                    [`${trainResult.results.length} Models`,'Performance metrics'],
                    ['Live Results','From your uploaded CSV'],
                    [`${words.toLocaleString()} words`,'AI narrative'],
                    ['ICH Q8/Q9','Regulatory format'],
                  ].map(([l,v])=>(
                    <div key={l} style={{padding:'4px 10px',borderRadius:6,
                      background:'rgba(204,0,0,0.12)',border:'1px solid rgba(204,0,0,0.2)'}}>
                      <span style={{fontSize:10,fontWeight:700,color:C.red}}>{l}</span>
                      <span style={{fontSize:10,color:C.textMid}}> · {v}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
            <div style={{display:'flex',flexDirection:'column',gap:8,flexShrink:0}}>
              <button onClick={exportPDF}
                style={{display:'flex',alignItems:'center',gap:9,padding:'12px 22px',borderRadius:11,
                  border:'none',background:`linear-gradient(135deg,${C.accent},#c026d3)`,
                  color:'#fff',fontSize:13,fontWeight:700,cursor:'pointer',
                  fontFamily:'DM Sans,sans-serif',boxShadow:'0 4px 16px rgba(204,0,0,0.35)'}}>
                <Download size={16}/>Download PDF
              </button>
              <button onClick={()=>navigator.clipboard.writeText(report)}
                style={{display:'flex',alignItems:'center',gap:9,padding:'9px 22px',borderRadius:11,
                  border:`1px solid ${C.border2}`,background:'transparent',color:C.textMid,
                  fontSize:12,fontWeight:500,cursor:'pointer',fontFamily:'DM Sans,sans-serif'}}>
                <Copy size={14}/>Copy Text
              </button>
            </div>
          </GlowCard>
        </>
      )}
    </div>
  );
}

// ── About & FAQ ──────────────────────────────────────────────────
function AboutFAQ(){
  const [tab,setTab]=useState('about');
  const [openFAQ,setOpenFAQ]=useState(null);

  const FAQS=[
    {q:'What is fucosylation and why does it matter?',
     a:'Fucosylation is the addition of a fucose sugar molecule to the antibody during production in CHO cells. It directly controls ADCC — the mechanism by which the antibody recruits immune cells to destroy cancer or pathogen targets. Low fucosylation generally means higher ADCC potency. Because it affects drug efficacy, fucosylation is classified as a Critical Quality Attribute (CQA) by regulatory agencies and must be measured and controlled in every manufacturing batch.'},
    {q:'What does R² actually mean?',
     a:'R² (R-squared) measures how much of the variation in fucosylation % the model can explain. An R² of 1.0 means the model predicts perfectly. An R² of 0.0 means it does no better than always predicting the dataset average. In bioprocessing, R² above 0.75 is generally considered strong. Our best model (XGBoost+SHAP) achieved R²=0.83 on the BMS2 dataset.'},
    {q:'Why did the ANN fail at N=500 but work at N=10,000?',
     a:'The ANN (artificial neural network) has 12,225 learnable parameters — internal connections that need to be tuned during training. At N=500, after the 80/20 split, only 340 samples are available for training. This means the model has 36 times more "dials to adjust" than it has data points to learn from. It memorises the training data (noise and all) and fails completely on new batches (R²=0.05). At N=10,000 with 8,000 training samples, the ratio drops below 2x and the model learns real patterns (R²=0.77). This is why neural networks are risky in small pharmaceutical datasets.'},
    {q:'What is SHAP and how should I interpret it?',
     a:'SHAP (SHapley Additive exPlanations) values measure exactly how much each input variable contributed to each individual prediction. If a batch had a high GDP-Fucose measurement and SHAP assigned it a value of +0.18, it means that specific measurement pushed the predicted fucosylation level up by 0.18% compared to the baseline. Unlike traditional feature importance, SHAP gives you a contribution per batch, not just an average. This makes it possible to audit individual predictions and explain to regulators why a specific batch was predicted as at-risk.'},
    {q:'What is the difference between RMSE and MAE?',
     a:'Both measure average prediction error in the same units as your target (fucosylation %). RMSE (Root Mean Squared Error) penalises large errors more heavily because it squares the errors before averaging. MAE (Mean Absolute Error) treats all errors equally. If your model makes a few very large prediction errors, RMSE will be much higher than MAE. A model with RMSE=2.5% and MAE=1.9% is performing well — predictions are typically within about 2% of the true value.'},
    {q:'What is ICH Q8/Q9 and how does this dashboard support it?',
     a:'ICH Q8 (Pharmaceutical Development) and ICH Q9 (Quality Risk Management) are international regulatory guidelines that require manufacturers to demonstrate deep understanding of the factors that control product quality — a concept called Quality by Design (QbD). Regulators expect you to know which process parameters affect your CQAs and by how much. This dashboard directly supports ICH Q8/Q9 compliance by providing interpretable models (PLSR VIP scores, XGBoost SHAP values) that document which bioprocess variables drive fucosylation and by how much in each batch.'},
    {q:'Why do I need to upload the file twice?',
     a:'Browsers have a security restriction that prevents web applications from storing and reusing local files between sessions. The file is uploaded once for cleansing (to get the quality report) and once for training (to send to the backend for model fitting). The cleansed file is automatically carried over from Step 1 to Step 2 in memory — you should not see a re-upload prompt. If you do, it means the file reference was lost; simply select the same file again.'},
    {q:'Can I use my own dataset that is not the BMS dataset?',
     a:'Yes. The pipeline works with any CSV file. The only requirement is a numerical target column — by default it looks for "Fucosylation_pct" but will use the last column if that name is not found. All other columns are treated as input features. The Hybrid model will only engineer physics-based features if it recognises relevant column name keywords (gdp, fucose, mn_, temp, ph, lact, pco2, etc.).'},
    {q:'Which model should I deploy in production?',
     a:'For large datasets (N > 5,000): XGBoost+SHAP — best accuracy (R²=0.83) with full SHAP explainability for regulatory submissions. For small datasets (N < 1,000): GPR — strong performance even with limited data, and it provides calibrated confidence intervals for risk-based batch release decisions. For regulatory simplicity: PLSR — chemometrics gold standard, VIP scores are widely accepted in ICH Q8 submissions, linear and interpretable. Avoid ANN unless N > 5,000.'},
    {q:'How long does training take?',
     a:'Ridge and PLSR: under 1 second. Random Forest: 5–15 seconds. XGBoost: 10–30 seconds. ANN: 20–60 seconds (uses early stopping). GPR: 5–20 seconds (capped at 300 training samples regardless of dataset size). Hybrid: 10–30 seconds. Total for all 7 models on N=10,000: roughly 2–3 minutes.'},
  ];

  return(
    <div style={{display:'flex',flexDirection:'column',gap:0}} className="fade-in">
      {/* Tab bar */}
      <div style={{display:'flex',borderBottom:`1px solid ${C.border}`,marginBottom:24}}>
        {[{id:'about',label:'About',icon:BookOpen},{id:'faq',label:'FAQ',icon:HelpCircle}].map(t=>(
          <button key={t.id} onClick={()=>setTab(t.id)}
            style={{display:'flex',alignItems:'center',gap:7,padding:'10px 20px',border:'none',
              background:'none',cursor:'pointer',fontSize:13,fontWeight:tab===t.id?600:400,
              fontFamily:'DM Sans,sans-serif',color:tab===t.id?C.blue:C.textMid,
              borderBottom:tab===t.id?`2px solid ${C.blue}`:'2px solid transparent',transition:'all 0.15s'}}>
            <t.icon size={14}/>{t.label}
          </button>
        ))}
      </div>

      {/* ABOUT TAB */}
      {tab==='about'&&(
        <div style={{display:'flex',flexDirection:'column',gap:20}} className="fade-in">
          {/* Hero */}
          <div style={{...S.card,background:`linear-gradient(135deg,rgba(204,0,0,0.12),rgba(192,38,211,0.12))`,
            border:'1px solid rgba(204,0,0,0.25)',padding:'32px'}}>
            <div style={{fontSize:26,fontWeight:800,color:C.text,marginBottom:8,letterSpacing:'-0.4px'}}>
              mAb Fucosylation · ML Prediction Pipeline
            </div>
            <div style={{fontSize:14,color:C.textMid,lineHeight:1.8,maxWidth:760}}>
              A full-stack machine learning platform built for the <strong style={{color:C.text}}>MS Data Science BMS Hackathon</strong> at Rutgers University in collaboration with Bristol Myers Squibb. The dashboard predicts monoclonal antibody (mAb) fucosylation — a Critical Quality Attribute directly affecting drug efficacy — from upstream CHO cell culture bioprocess variables, in real time.
            </div>
            <div style={{display:'flex',gap:10,marginTop:18,flexWrap:'wrap'}}>
              {[['BMS HACKATHON',C.red],['MS Data Science',C.blue],['Rutgers × BMS',C.purple],['Team Data Minds',C.green]].map(([l,c])=>(
                <span key={l} style={{padding:'4px 12px',borderRadius:20,fontSize:11,fontWeight:600,background:c+'18',color:c,border:`1px solid ${c}28`}}>{l}</span>
              ))}
            </div>
          </div>

          {/* Why it matters */}
          <GlowCard accent={C.blue}>
            <div style={{fontSize:15,fontWeight:700,color:C.text,marginBottom:12}}>Why Fucosylation Prediction Matters</div>
            <div style={{fontSize:13,color:C.textMid,lineHeight:1.85}}>
              In biopharmaceutical manufacturing, monoclonal antibodies (mAbs) are produced by genetically engineered Chinese Hamster Ovary (CHO) cells in large bioreactors. During production, the cell attaches sugar molecules to the antibody in a process called glycosylation. One of these sugars — fucose — has a direct and measurable effect on how well the antibody performs therapeutically.
              <br/><br/>
              Low-fucosylation antibodies show significantly higher ADCC (Antibody-Dependent Cell-mediated Cytotoxicity) — the mechanism by which the antibody recruits natural killer cells to destroy tumour targets. Because of this, fucosylation is classified as a <strong style={{color:C.text}}>Critical Quality Attribute (CQA)</strong> and must be tightly controlled in every batch.
              <br/><br/>
              The challenge: fucosylation is typically measured at the <em>end</em> of a batch run, using offline analytical methods. By the time you know the result, the batch is already complete. If it is out of specification, the entire run — days of process time and thousands of dollars of materials — may be lost.
              <br/><br/>
              This dashboard solves that by training ML models on historical bioprocess data (temperature, pH, dissolved oxygen, carbon dioxide, viable cell density, metabolite concentrations) to predict the final fucosylation level <em>during</em> the run — giving operators time to adjust parameters and prevent failed batches.
            </div>
          </GlowCard>

          {/* Pipeline overview */}
          <GlowCard accent={C.green}>
            <div style={{fontSize:15,fontWeight:700,color:C.text,marginBottom:16}}>How the Pipeline Works</div>
            <div style={{display:'flex',flexDirection:'column',gap:12}}>
              {[
                {step:'01', color:C.red,    title:'Upload & Cleanse',   body:'Upload any bioprocess CSV. The backend runs missing value detection (MCAR/MNAR classification), three-method outlier detection (IQR fence + Z-score + IsolationForest), and KS-test batch drift analysis. Quality score out of 100. Nothing is modified — inspection only.'},
                {step:'02', color:C.blue,   title:'Select & Train',     body:'Your file carries over automatically. Pick any of the 7 ML models, click Train. The backend splits 80/20, scales features on training data only (no leakage), and trains each model on your actual data. Results are computed from real test-set predictions.'},
                {step:'03', color:C.green,  title:'Results Dashboard',  body:'PowerBI-style dark dashboard with KPI cards, model visibility toggles, R² comparison chart, actual vs predicted scatter per model, SHAP/VIP/MDI feature importance, multi-criteria radar, and learning curves. All from your data.'},
                {step:'04', color:C.purple, title:'AI Report',          body:'Three report types — Scientific (800-1000 words, thesis-style), Executive (400-500 words, plain language for leadership), Quick Summary (200-250 words, team meeting snapshot). All generated from your actual trained model results using Gemini 2.5 Flash. Export as PDF.'},
                {step:'05', color:C.amber,  title:'BMS1 vs BMS2',       body:"Static reference page showing the core hackathon experiment: 7 models trained on BMS1 (N=500) and BMS2 (N=10,000) — identical features, same noise, only dataset size changed. The ANN's +1440% recovery is the headline finding."},
              ].map(({step,color,title,body})=>(
                <div key={step} style={{display:'flex',gap:14,alignItems:'flex-start'}}>
                  <div style={{width:36,height:36,borderRadius:10,background:color+'20',border:`1px solid ${color}35`,
                    display:'flex',alignItems:'center',justifyContent:'center',flexShrink:0,
                    fontSize:11,fontWeight:700,color,fontFamily:'DM Mono,monospace'}}>{step}</div>
                  <div>
                    <div style={{fontSize:13,fontWeight:600,color:C.text,marginBottom:3}}>{title}</div>
                    <div style={{fontSize:12,color:C.textMid,lineHeight:1.65}}>{body}</div>
                  </div>
                </div>
              ))}
            </div>
          </GlowCard>

          {/* Team + models */}
          <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:16}}>
            <GlowCard accent={C.teal}>
              <div style={{fontSize:15,fontWeight:700,color:C.text,marginBottom:12}}>7 ML Models</div>
              {[
                ['Ridge',         C.textDim, 'Linear baseline'],
                ['PLSR',          C.textDim, 'VIP scores, regulatory'],
                ['Random Forest', '#2dd4a0',  '200 trees ensemble'],
                ['XGBoost+SHAP',  '#4d9fff',  'Best accuracy + explainability'],
                ['GPR',           '#a78bfa',  'Calibrated uncertainty'],
                ['ANN',           '#f87171',  'Needs N > 5,000'],
                ['Hybrid',        '#f59e0b',  'Physics-informed features'],
              ].map(([name,color,desc])=>(
                <div key={name} style={{display:'flex',justifyContent:'space-between',alignItems:'center',
                  marginBottom:7,paddingBottom:7,borderBottom:`1px solid ${C.border}`}}>
                  <div style={{display:'flex',alignItems:'center',gap:7}}>
                    <div style={{width:7,height:7,borderRadius:'50%',background:color}}/>
                    <span style={{fontSize:12,color:C.text,fontWeight:500}}>{name}</span>
                  </div>
                  <span style={{fontSize:11,color:C.textMid}}>{desc}</span>
                </div>
              ))}
            </GlowCard>
          </div>

        </div>
      )}

      {/* FAQ TAB */}
      {tab==='faq'&&(
        <div style={{display:'flex',flexDirection:'column',gap:8}} className="fade-in">
          <div style={{fontSize:13,color:C.textMid,marginBottom:8}}>
            {FAQS.length} frequently asked questions about the dashboard, the models, and the science.
          </div>
          {FAQS.map((faq,i)=>{
            const open=openFAQ===i;
            return(
              <div key={i}
                style={{...S.card,padding:0,border:`1px solid ${open?C.blue:C.border}`,
                  transition:'border 0.2s',overflow:'hidden'}}>
                <button onClick={()=>setOpenFAQ(open?null:i)}
                  style={{width:'100%',display:'flex',alignItems:'center',justifyContent:'space-between',
                    padding:'16px 20px',border:'none',background:'transparent',cursor:'pointer',
                    textAlign:'left',fontFamily:'DM Sans,sans-serif'}}>
                  <div style={{display:'flex',alignItems:'center',gap:12}}>
                    <div style={{width:24,height:24,borderRadius:6,background:open?C.blue+'20':C.bg4,
                      display:'flex',alignItems:'center',justifyContent:'center',flexShrink:0,transition:'all 0.2s'}}>
                      <span style={{fontSize:10,fontWeight:700,color:open?C.blue:C.textDim,...S.mono}}>
                        {String(i+1).padStart(2,'0')}
                      </span>
                    </div>
                    <span style={{fontSize:13,fontWeight:open?600:400,color:open?C.text:C.textMid,lineHeight:1.4}}>
                      {faq.q}
                    </span>
                  </div>
                  <ChevronRight size={16} color={open?C.blue:C.textDim}
                    style={{flexShrink:0,marginLeft:12,transform:open?'rotate(90deg)':'none',transition:'transform 0.2s'}}/>
                </button>
                {open&&(
                  <div style={{padding:'0 20px 18px 56px',fontSize:13,color:C.textMid,lineHeight:1.8,
                    borderTop:`1px solid ${C.border}`}} className="fade-in">
                    <div style={{paddingTop:14}}>{faq.a}</div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

// ── BMS1 vs BMS2 Static Comparison Page ─────────────────────────
function BMSComparison(){
  const [tab,setTab]=useState('overview');
  const [ds,setDs]=useState('BMS2');
  const models=ds==='BMS1'?BMS1_DATA:BMS2_DATA;
  const best=[...models].sort((a,b)=>b.r2_test-a.r2_test)[0];

  const r2ChartData={
    labels:models.map(m=>m.name),
    datasets:[
      {label:'R² Train',data:models.map(m=>m.r2_train),
       backgroundColor:models.map(m=>(BMS_MODEL_COLORS[m.name]||'#8b90a0')+'44'),
       borderColor:models.map(m=>BMS_MODEL_COLORS[m.name]||'#8b90a0'),borderWidth:2,borderRadius:6},
      {label:'R² Test', data:models.map(m=>m.r2_test),
       backgroundColor:models.map(m=>(BMS_MODEL_COLORS[m.name]||'#8b90a0')+'bb'),
       borderColor:models.map(m=>BMS_MODEL_COLORS[m.name]||'#8b90a0'),borderWidth:2,borderRadius:6},
    ],
  };

  const compChartData={
    labels:COMPARISON_DATA.map(r=>r.model),
    datasets:[
      {label:'BMS1 (N=500)',    data:COMPARISON_DATA.map(r=>r.bms1),
       backgroundColor:COMPARISON_DATA.map(r=>(BMS_MODEL_COLORS[r.model]||'#8b90a0')+'55'),
       borderColor:COMPARISON_DATA.map(r=>BMS_MODEL_COLORS[r.model]||'#8b90a0'),borderWidth:2,borderRadius:5},
      {label:'BMS2 (N=10,000)',data:COMPARISON_DATA.map(r=>r.bms2),
       backgroundColor:COMPARISON_DATA.map(r=>(BMS_MODEL_COLORS[r.model]||'#8b90a0')+'cc'),
       borderColor:COMPARISON_DATA.map(r=>BMS_MODEL_COLORS[r.model]||'#8b90a0'),borderWidth:2,borderRadius:5},
    ],
  };

  const lcData={
    labels:['500','1k','2k','3k','4k','5k','6k','8k','10k'],
    datasets:[
      {label:'XGBoost+SHAP',data:[0.74,0.77,0.79,0.80,0.81,0.82,0.82,0.83,0.83],borderColor:'#4d9fff',tension:0.4,pointRadius:4,fill:false},
      {label:'GPR',         data:[0.75,0.76,0.77,0.78,0.78,0.79,0.79,0.79,0.79],borderColor:'#a78bfa',tension:0.4,pointRadius:4,fill:false},
      {label:'ANN',         data:[0.05,0.05,0.42,0.58,0.65,0.73,0.75,0.77,0.77],borderColor:'#f87171',tension:0.4,pointRadius:4,fill:false,borderDash:[5,5]},
      {label:'Ridge/PLSR',  data:[0.51,0.53,0.54,0.55,0.56,0.56,0.57,0.57,0.57],borderColor:'#8b90a0',tension:0.4,pointRadius:3,fill:false,borderDash:[3,3]},
    ],
  };

  const tabs=[
    {id:'overview',   label:'Overview'},
    {id:'r2',         label:'R² by Dataset'},
    {id:'comparison', label:'BMS1 vs BMS2'},
    {id:'learning',   label:'Learning Curves'},
    {id:'rankings',   label:'Final Rankings'},
  ];

  return(
    <div style={{display:'flex',flexDirection:'column',gap:16}} className="fade-in">
      {/* Headline banner */}
      <div style={{...S.card,background:`linear-gradient(135deg,rgba(229,57,53,0.15),rgba(167,139,250,0.15))`,
        border:`1px solid rgba(229,57,53,0.3)`}}>
        <div style={{fontSize:22,fontWeight:800,color:C.red,marginBottom:6}}>
          ANN: R²=0.05 → R²=0.77 &nbsp;(+1440%)
        </div>
        <div style={{fontSize:13,color:C.textMid,lineHeight:1.7}}>
          Two identical datasets — BMS1 (N=500) and BMS2 (N=10,000) — same features, same noise, same ground truth.
          Only dataset size changed. This isolates <strong style={{color:C.text}}>data volume as the single experimental variable</strong> across all 7 models.
          ANN failed completely at N=500 (12,225 params vs 340 training samples = 36× overparameterised)
          and fully recovered at N=10,000.
        </div>
        <div style={{display:'flex',gap:10,marginTop:12,flexWrap:'wrap'}}>
          {[['BMS1 · N=500','10 batches · mean=87.5% · std=5.59%',C.purple],
            ['BMS2 · N=10,000','50 batches · mean=88.0% · std=5.73%',C.blue],
            ['Features','GDP-Fucose, Mn, pH, Temp, DO, pCO2, Uridine, VCD, Osmolality, Lactate',C.green],
            ['Split','80% train / 20% test — identical across both',C.amber],
          ].map(([l,v,c])=>(
            <div key={l} style={{padding:'6px 12px',borderRadius:8,background:c+'15',border:`1px solid ${c}25`}}>
              <div style={{fontSize:9,color:c,fontWeight:700,textTransform:'uppercase',letterSpacing:'0.07em'}}>{l}</div>
              <div style={{fontSize:11,color:C.textMid,marginTop:2}}>{v}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Dataset switcher + KPIs */}
      <div style={{display:'flex',gap:8,alignItems:'center'}}>
        {['BMS1','BMS2'].map(d=>(
          <button key={d} onClick={()=>setDs(d)}
            style={{padding:'8px 20px',borderRadius:10,
              border:`2px solid ${ds===d?C.red:C.border}`,
              background:ds===d?C.red:'transparent',
              color:ds===d?'#fff':C.textMid,
              fontWeight:600,fontSize:13,cursor:'pointer',fontFamily:'DM Sans,sans-serif',transition:'all 0.2s'}}>
            {d} · N={d==='BMS1'?'500':'10,000'}
          </button>
        ))}
      </div>

      <div style={{display:'grid',gridTemplateColumns:'repeat(4,1fr)',gap:12}}>
        {[
          {label:'Best R² Test', value:best.r2_test.toFixed(2), sub:best.name, color:C.green},
          {label:'Best RMSE',    value:best.rmse.toFixed(2),    sub:'lowest error', color:C.blue},
          {label:'ANN R² Test',  value:models.find(m=>m.name==='ANN')?.r2_test.toFixed(2)||'—',
           sub:ds==='BMS1'?'⚠ Failed — overparameterised':'✓ Recovered',
           color:ds==='BMS1'?C.red:C.green},
          {label:'Dataset',      value:`N=${ds==='BMS1'?'500':'10k'}`, sub:`${ds==='BMS1'?10:50} batches`, color:C.purple},
        ].map(s=>(
          <div key={s.label} style={{...S.card,background:s.color+'10',border:`1px solid ${s.color}25`}}>
            <div style={{fontSize:9,color:s.color,fontWeight:700,textTransform:'uppercase',letterSpacing:'0.09em',marginBottom:4}}>{s.label}</div>
            <div style={{fontSize:22,fontWeight:800,color:s.color,...S.mono,lineHeight:1.1}}>{s.value}</div>
            <div style={{fontSize:11,color:C.textDim,marginTop:2}}>{s.sub}</div>
          </div>
        ))}
      </div>

      {/* Tabs */}
      <div style={{display:'flex',borderBottom:`1px solid ${C.border}`,overflowX:'auto'}}>
        {tabs.map(t=>(
          <button key={t.id} onClick={()=>setTab(t.id)}
            style={{padding:'9px 18px',border:'none',background:'none',cursor:'pointer',
              fontSize:12,fontWeight:tab===t.id?600:400,fontFamily:'DM Sans,sans-serif',
              color:tab===t.id?C.blue:C.textMid,
              borderBottom:tab===t.id?`2px solid ${C.blue}`:'2px solid transparent',
              transition:'all 0.15s',whiteSpace:'nowrap'}}>
            {t.label}
          </button>
        ))}
      </div>

      {/* Overview */}
      {tab==='overview'&&(
        <div className="fade-in" style={{display:'flex',flexDirection:'column',gap:12}}>
          <GlowCard accent={C.green}>
            <div style={{...S.label,marginBottom:12}}>Model Performance — {ds} · R² Test Score</div>
            <div style={{height:260}}><Bar data={r2ChartData} options={{...chartBase,scales:{...chartBase.scales,y:{...chartBase.scales.y,min:0,max:1.05}}}}/></div>
          </GlowCard>
          <div style={{display:'grid',gridTemplateColumns:'repeat(auto-fill,minmax(190px,1fr))',gap:10}}>
            {[...models].sort((a,b)=>b.r2_test-a.r2_test).map(m=>{
              const col=BMS_MODEL_COLORS[m.name]||C.blue;
              return(
                <div key={m.name} style={{...S.card,border:`1px solid ${m.name===best.name?col:C.border}`,
                  background:m.name===best.name?col+'10':C.bg2}}>
                  <div style={{display:'flex',justifyContent:'space-between',marginBottom:6}}>
                    <div style={{width:9,height:9,borderRadius:'50%',background:col,marginTop:3}}/>
                    {m.name===best.name&&<span style={{fontSize:9,fontWeight:700,padding:'2px 7px',borderRadius:10,background:col+'25',color:col}}>BEST</span>}
                  </div>
                  <div style={{fontSize:12,fontWeight:600,color:C.text,marginBottom:4}}>{m.name}</div>
                  <div style={{fontSize:22,fontWeight:800,color:col,...S.mono,lineHeight:1}}>{m.r2_test.toFixed(3)}</div>
                  <div style={{fontSize:10,color:C.textDim,marginBottom:6}}>R² test</div>
                  <ProgressBar value={Math.max(0,m.r2_test)} color={col}/>
                  <div style={{display:'flex',justifyContent:'space-between',marginTop:6}}>
                    <span style={{fontSize:10,...S.mono,color:C.textDim}}>RMSE {m.rmse.toFixed(2)}</span>
                    <span style={{fontSize:10,...S.mono,color:C.textDim}}>MAE {m.mae.toFixed(2)}</span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* R² table */}
      {tab==='r2'&&(
        <GlowCard accent={C.blue} className="fade-in">
          <div style={{...S.label,marginBottom:14}}>Full Metrics Table — {ds}</div>
          <div style={{overflowX:'auto'}}>
            <table style={{width:'100%',borderCollapse:'collapse',fontSize:12}}>
              <thead><tr style={{borderBottom:`1px solid ${C.border}`}}>
                {['Model','R² Train','R² Test','RMSE','MAE'].map(h=>(
                  <th key={h} style={{padding:'8px 12px',textAlign:'left',color:C.textDim,fontSize:10,fontWeight:600,textTransform:'uppercase',letterSpacing:'0.07em'}}>{h}</th>
                ))}
              </tr></thead>
              <tbody>
                {[...models].sort((a,b)=>b.r2_test-a.r2_test).map((m,i)=>{
                  const col=BMS_MODEL_COLORS[m.name]||C.blue;
                  return(
                    <tr key={m.name} style={{borderBottom:`1px solid ${C.border}`,background:m.name===best.name?col+'08':'transparent'}}
                      onMouseEnter={e=>e.currentTarget.style.background='rgba(255,255,255,0.04)'}
                      onMouseLeave={e=>e.currentTarget.style.background=m.name===best.name?col+'08':'transparent'}>
                      <td style={{padding:'9px 12px',fontWeight:600,color:col}}>
                        <div style={{display:'flex',alignItems:'center',gap:7}}>
                          <div style={{width:8,height:8,borderRadius:'50%',background:col}}/>
                          {m.name}{m.name===best.name&&<span style={{fontSize:9,padding:'1px 6px',borderRadius:8,background:col+'25',color:col}}>★ BEST</span>}
                        </div>
                      </td>
                      {[m.r2_train,m.r2_test].map((v,vi)=>(
                        <td key={vi} style={{padding:'9px 12px'}}>
                          <div style={{display:'flex',alignItems:'center',gap:8}}>
                            <div style={{width:44,background:'rgba(255,255,255,0.05)',borderRadius:3,height:5}}>
                              <div style={{width:`${Math.max(0,v)*100}%`,height:'100%',borderRadius:3,background:v<0.2?C.red:col}}/>
                            </div>
                            <span style={{...S.mono,fontSize:12,color:v<0.2?C.red:C.text}}>{v.toFixed(4)}</span>
                          </div>
                        </td>
                      ))}
                      <td style={{padding:'9px 12px',...S.mono,color:C.textMid,fontSize:12}}>{m.rmse.toFixed(4)}</td>
                      <td style={{padding:'9px 12px',...S.mono,color:C.textMid,fontSize:12}}>{m.mae.toFixed(4)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </GlowCard>
      )}

      {/* BMS1 vs BMS2 comparison */}
      {tab==='comparison'&&(
        <div className="fade-in" style={{display:'flex',flexDirection:'column',gap:12}}>
          <GlowCard accent={C.red}>
            <div style={{...S.label,marginBottom:14}}>R² Test — BMS1 (N=500) vs BMS2 (N=10,000)</div>
            <div style={{height:260}}><Bar data={compChartData} options={{...chartBase,scales:{...chartBase.scales,y:{...chartBase.scales.y,min:0,max:1.05}}}}/></div>
          </GlowCard>
          <GlowCard accent={C.amber}>
            <div style={{...S.label,marginBottom:10}}>Cross-Experiment Delta Table</div>
            <div style={{overflowX:'auto'}}>
              <table style={{width:'100%',borderCollapse:'collapse',fontSize:12}}>
                <thead><tr style={{borderBottom:`1px solid ${C.border}`}}>
                  {['Model','BMS1 R²','BMS2 R²','Change','Interpretation'].map(h=>(
                    <th key={h} style={{padding:'8px 12px',textAlign:'left',color:C.textDim,fontSize:10,fontWeight:600,textTransform:'uppercase',letterSpacing:'0.07em'}}>{h}</th>
                  ))}
                </tr></thead>
                <tbody>
                  {COMPARISON_DATA.map((r,i)=>{
                    const col=BMS_MODEL_COLORS[r.model]||C.blue;
                    return(
                      <tr key={r.model} style={{borderBottom:`1px solid ${C.border}`,background:r.model==='ANN'?C.amberL:'transparent'}}
                        onMouseEnter={e=>e.currentTarget.style.background='rgba(255,255,255,0.04)'}
                        onMouseLeave={e=>e.currentTarget.style.background=r.model==='ANN'?C.amberL:'transparent'}>
                        <td style={{padding:'8px 12px',fontWeight:600,color:col}}>
                          <div style={{display:'flex',alignItems:'center',gap:7}}>
                            <div style={{width:8,height:8,borderRadius:2,background:col}}/>
                            {r.model}
                          </div>
                        </td>
                        <td style={{padding:'8px 12px',...S.mono,color:C.textMid}}>{r.bms1.toFixed(2)}</td>
                        <td style={{padding:'8px 12px',...S.mono,fontWeight:700,color:C.text}}>{r.bms2.toFixed(2)}</td>
                        <td style={{padding:'8px 12px'}}>
                          <span style={{padding:'2px 8px',borderRadius:10,fontSize:10,fontWeight:700,
                            background:r.pct==='+1440%'?C.red+'25':C.green+'20',
                            color:r.pct==='+1440%'?C.red:C.green}}>{r.pct}</span>
                        </td>
                        <td style={{padding:'8px 12px',fontSize:11,color:C.textMid}}>{r.interp}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </GlowCard>
        </div>
      )}

      {/* Learning curves */}
      {tab==='learning'&&(
        <GlowCard accent={C.teal} className="fade-in">
          <div style={{...S.label,marginBottom:6}}>Learning Curves — R² vs Training Dataset Size</div>
          <div style={{fontSize:12,color:C.textMid,marginBottom:14}}>
            ANN stagnates at N=500 (R²=0.05 — 36× overparameterised) and fully recovers above N=5,000.
            Linear models (Ridge/PLSR) hit a structural ceiling regardless of data volume.
          </div>
          <div style={{height:300}}><Line data={lcData} options={{...chartBase,scales:{...chartBase.scales,
            y:{...chartBase.scales.y,min:0,max:1,title:{display:true,text:'R² Test',color:C.textMid,font:{size:11}}},
            x:{...chartBase.scales.x,title:{display:true,text:'Training Set Size',color:C.textMid,font:{size:11}}}}}}/></div>
        </GlowCard>
      )}

      {/* Rankings */}
      {tab==='rankings'&&(
        <GlowCard accent={C.purple} className="fade-in">
          <div style={{...S.label,marginBottom:14}}>Final Rankings — BMS1 vs BMS2</div>
          <div style={{overflowX:'auto'}}>
            <table style={{width:'100%',borderCollapse:'collapse',fontSize:12}}>
              <thead><tr style={{borderBottom:`1px solid ${C.border}`}}>
                {['Category','Best at N=500','Best at N=10,000','Recommendation'].map(h=>(
                  <th key={h} style={{padding:'8px 12px',textAlign:'left',color:C.textDim,fontSize:10,fontWeight:600,textTransform:'uppercase',letterSpacing:'0.07em'}}>{h}</th>
                ))}
              </tr></thead>
              <tbody>
                {RANKINGS.map((r,i)=>(
                  <tr key={r.category} style={{borderBottom:`1px solid ${C.border}`,background:i%2===0?'rgba(255,255,255,0.02)':'transparent'}}
                    onMouseEnter={e=>e.currentTarget.style.background='rgba(255,255,255,0.05)'}
                    onMouseLeave={e=>e.currentTarget.style.background=i%2===0?'rgba(255,255,255,0.02)':'transparent'}>
                    <td style={{padding:'9px 12px',fontWeight:600,color:C.text}}>{r.category}</td>
                    <td style={{padding:'9px 12px',color:C.textMid}}>{r.bms1}</td>
                    <td style={{padding:'9px 12px',color:C.textMid}}>{r.bms2}</td>
                    <td style={{padding:'9px 12px'}}>
                      <span style={{padding:'3px 10px',borderRadius:10,fontSize:10,fontWeight:700,background:C.blue+'20',color:C.blue}}>{r.rec}</span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </GlowCard>
      )}
    </div>
  );
}

// ── Auth helpers ─────────────────────────────────────────────────
const getToken = () => localStorage.getItem('bms_token');
const getStoredUser = () => {
  try {
    const raw = localStorage.getItem('bms_user');
    console.log('[BMS] bms_user raw:', raw ? raw.slice(0,80) : 'null');
    const parsed = JSON.parse(raw || 'null');
    console.log('[BMS] bms_user parsed:', parsed);
    return parsed;
  } catch(e) {
    console.error('[BMS] getStoredUser error:', e);
    return null;
  }
};
const authFetch = (url, opts={}) => {
  const token = getToken();
  return fetch(url, {
    ...opts,
    headers: { ...(opts.headers||{}), ...(token ? {'Authorization': `Bearer ${token}`} : {}) },
  });
};

// ── Login Page ────────────────────────────────────────────────────
function LoginPage({onLogin}){
  const [email,setEmail]=useState('');
  const [otp,setOtp]=useState('');
  const [otpSent,setOtpSent]=useState(false);
  const [loading,setLoading]=useState(false);
  const [error,setError]=useState('');
  const [info,setInfo]=useState('');

  React.useEffect(()=>{
    const p=new URLSearchParams(window.location.search);
    if(p.get('auth')==='google'){
      const token=p.get('token'), userStr=p.get('user');
      if(token&&userStr){
        try{
          const user=JSON.parse(decodeURIComponent(userStr));
          localStorage.setItem('bms_token',token);
          localStorage.setItem('bms_user',JSON.stringify(user));
          window.history.replaceState({},'','/');
          onLogin(user); return;
        }catch(e){ setError('Login error. Please try again.'); window.history.replaceState({},'','/'); return; }
      }
    }
    const ae=p.get('auth_error');
    if(ae){ setError('Google sign-in failed. Please try again.'); window.history.replaceState({},'','/'); return; }
    const u=getStoredUser(), t=getToken();
    if(u&&t&&u.email){ onLogin(u); }
  },[]);

  const sendOTP=async()=>{
    if(!email.trim()||!email.includes('@')){setError('Enter a valid email.');return;}
    setLoading(true);setError('');setInfo('');
    try{
      const r=await fetch('/auth/otp/send',{method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({email:email.trim().toLowerCase()})});
      const d=await r.json();
      if(r.ok){setOtpSent(true);setInfo('Code sent — check your inbox and spam folder.');}
      else setError(d.detail||'Failed to send code.');
    }catch{setError('Cannot reach backend. Is uvicorn running on port 8000?');}
    finally{setLoading(false);}
  };

  const verifyOTP=async()=>{
    if(otp.trim().length!==6){setError('Enter the 6-digit code.');return;}
    setLoading(true);setError('');
    try{
      const r=await fetch('/auth/otp/verify',{method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({email:email.trim().toLowerCase(),otp:otp.trim()})});
      const d=await r.json();
      if(r.ok){
        localStorage.setItem('bms_token',d.token);
        localStorage.setItem('bms_user',JSON.stringify(d.user));
        onLogin(d.user);
      } else setError(d.detail||'Incorrect or expired code.');
    }catch{setError('Network error. Please try again.');}
    finally{setLoading(false);}
  };

  const inp={width:'100%',padding:'12px 16px',borderRadius:11,border:`1px solid rgba(255,255,255,0.12)`,
    background:'rgba(255,255,255,0.05)',color:C.text,fontSize:14,fontFamily:'DM Sans,sans-serif',
    outline:'none',boxSizing:'border-box',transition:'border 0.2s'};

  return(
    <div style={{minHeight:'100vh',background:`radial-gradient(ellipse at 60% 0%, rgba(204,0,0,0.15) 0%, transparent 60%), ${C.bg}`,
      display:'flex',alignItems:'center',justifyContent:'center',padding:24}}>
      <div style={{width:'100%',maxWidth:440}}>

        {/* Card */}
        <div style={{background:'rgba(20,20,30,0.85)',backdropFilter:'blur(24px)',
          borderRadius:22,border:'1px solid rgba(255,255,255,0.09)',padding:'40px 40px 34px',
          boxShadow:'0 32px 80px rgba(0,0,0,0.7), 0 0 0 1px rgba(204,0,0,0.08)'}}>

          {/* Logo */}
          <div style={{display:'flex',justifyContent:'center',marginBottom:28}}>
            <div style={{background:'#fff',borderRadius:12,padding:'10px 22px',display:'inline-flex',alignItems:'center',
              boxShadow:'0 4px 20px rgba(0,0,0,0.3)'}}>
              <img src="/bms-logo.png" alt="BMS" style={{height:32,objectFit:'contain'}}
                onError={e=>{e.target.style.display='none';e.target.nextSibling.style.display='block';}}/>
              <span style={{display:'none',fontSize:16,fontWeight:800,color:'#cc0000',letterSpacing:1}}>BMS</span>
            </div>
          </div>

          {/* Heading */}
          <div style={{textAlign:'center',marginBottom:28}}>
            <div style={{fontSize:11,fontWeight:700,letterSpacing:'0.12em',textTransform:'uppercase',
              color:C.red,marginBottom:10,opacity:0.85}}>
              BMS HACKATHON · Team Data Minds
            </div>
            <div style={{fontSize:22,fontWeight:800,color:C.text,letterSpacing:'-0.5px',marginBottom:6}}>
              mAb Fucosylation
            </div>
            <div style={{fontSize:14,fontWeight:600,color:C.text,opacity:0.75,marginBottom:10}}>
              ML Prediction Pipeline
            </div>
            <div style={{width:36,height:2,background:`linear-gradient(90deg,#cc0000,#c026d3)`,
              borderRadius:2,margin:'0 auto 10px'}}/>
            <div style={{fontSize:12,color:C.textMid,lineHeight:1.65}}>
              Predicting Critical Quality Attributes in<br/>CHO cell culture bioprocessing
            </div>
          </div>

          {/* Google Button */}
          <a href="http://localhost:8000/auth/google"
            style={{display:'flex',alignItems:'center',justifyContent:'center',gap:12,
              padding:'13px 20px',borderRadius:12,border:'1px solid rgba(255,255,255,0.15)',
              background:'#fff',color:'#3c4043',textDecoration:'none',
              fontSize:14,fontWeight:600,fontFamily:'DM Sans,sans-serif',
              boxShadow:'0 2px 12px rgba(0,0,0,0.4)',transition:'all 0.2s',marginBottom:20}}
            onMouseEnter={e=>{e.currentTarget.style.transform='translateY(-2px)';e.currentTarget.style.boxShadow='0 6px 24px rgba(0,0,0,0.5)';}}
            onMouseLeave={e=>{e.currentTarget.style.transform='none';e.currentTarget.style.boxShadow='0 2px 12px rgba(0,0,0,0.4)';}}>
            <svg width="20" height="20" viewBox="0 0 48 48">
              <path fill="#EA4335" d="M24 9.5c3.54 0 6.71 1.22 9.21 3.6l6.85-6.85C35.9 2.38 30.47 0 24 0 14.62 0 6.51 5.38 2.56 13.22l7.98 6.19C12.43 13.72 17.74 9.5 24 9.5z"/>
              <path fill="#4285F4" d="M46.98 24.55c0-1.57-.15-3.09-.38-4.55H24v9.02h12.94c-.58 2.96-2.26 5.48-4.78 7.18l7.73 6c4.51-4.18 7.09-10.36 7.09-17.65z"/>
              <path fill="#FBBC05" d="M10.53 28.59c-.48-1.45-.76-2.99-.76-4.59s.27-3.14.76-4.59l-7.98-6.19C.92 16.46 0 20.12 0 24c0 3.88.92 7.54 2.56 10.78l7.97-6.19z"/>
              <path fill="#34A853" d="M24 48c6.48 0 11.93-2.13 15.89-5.81l-7.73-6c-2.15 1.45-4.92 2.3-8.16 2.3-6.26 0-11.57-4.22-13.47-9.91l-7.98 6.19C6.51 42.62 14.62 48 24 48z"/>
            </svg>
            Continue with Google
          </a>

          {/* Divider */}
          <div style={{display:'flex',alignItems:'center',gap:12,marginBottom:20}}>
            <div style={{flex:1,height:1,background:'rgba(255,255,255,0.08)'}}/>
            <span style={{fontSize:11,color:C.textDim,letterSpacing:'0.05em'}}>OR</span>
            <div style={{flex:1,height:1,background:'rgba(255,255,255,0.08)'}}/>
          </div>

          {/* Email OTP */}
          {!otpSent?(
            <div>
              <input value={email} onChange={e=>setEmail(e.target.value)}
                placeholder="Enter your email address" type="email" style={{...inp,marginBottom:12}}
                onKeyDown={e=>e.key==='Enter'&&sendOTP()}/>
              <button onClick={sendOTP} disabled={loading}
                style={{width:'100%',padding:'12px',borderRadius:11,border:'none',
                  background:loading?'rgba(255,255,255,0.05)':`linear-gradient(135deg,#cc0000,#c026d3)`,
                  color:loading?C.textDim:'#fff',fontSize:14,fontWeight:600,
                  cursor:loading?'not-allowed':'pointer',fontFamily:'DM Sans,sans-serif',
                  boxShadow:loading?'none':'0 4px 16px rgba(204,0,0,0.3)',transition:'all 0.2s'}}>
                {loading?'Sending code…':'Send login code'}
              </button>
            </div>
          ):(
            <div>
              <div style={{textAlign:'center',marginBottom:14}}>
                <div style={{fontSize:13,color:C.green,fontWeight:600,marginBottom:2}}>✓ Code sent to {email}</div>
                <div style={{fontSize:11,color:C.textMid}}>Check inbox and spam · valid 10 minutes</div>
              </div>
              <input value={otp} onChange={e=>setOtp(e.target.value.replace(/[^0-9]/g,'').slice(0,6))}
                placeholder="6-digit code" maxLength={6}
                style={{...inp,fontSize:24,fontWeight:700,letterSpacing:10,textAlign:'center',
                  fontFamily:'DM Mono,monospace',marginBottom:12}}
                onKeyDown={e=>e.key==='Enter'&&verifyOTP()}/>
              <button onClick={verifyOTP} disabled={loading||otp.length!==6}
                style={{width:'100%',padding:'12px',borderRadius:11,border:'none',
                  background:otp.length===6?`linear-gradient(135deg,#cc0000,#c026d3)`:'rgba(255,255,255,0.05)',
                  color:otp.length===6?'#fff':C.textDim,fontSize:14,fontWeight:600,
                  cursor:otp.length===6&&!loading?'pointer':'not-allowed',fontFamily:'DM Sans,sans-serif',
                  boxShadow:otp.length===6?'0 4px 16px rgba(204,0,0,0.3)':'none',transition:'all 0.2s'}}>
                {loading?'Verifying…':'Verify & Sign In'}
              </button>
              <button onClick={()=>{setOtpSent(false);setOtp('');setError('');setInfo('');}}
                style={{display:'block',width:'100%',marginTop:10,padding:'8px',border:'none',
                  background:'transparent',color:C.textDim,fontSize:12,cursor:'pointer',fontFamily:'DM Sans,sans-serif'}}>
                ← Use a different email
              </button>
            </div>
          )}

          {error&&<div style={{marginTop:14,padding:'10px 14px',borderRadius:9,
            background:'rgba(229,57,53,0.12)',border:'1px solid rgba(229,57,53,0.3)',
            fontSize:12,color:'#f87171',lineHeight:1.5}}>{error}</div>}
          {info&&!error&&<div style={{marginTop:14,padding:'10px 14px',borderRadius:9,
            background:'rgba(45,212,160,0.08)',border:'1px solid rgba(45,212,160,0.25)',
            fontSize:12,color:C.green}}>{info}</div>}
        </div>

        {/* Footer */}
        <div style={{textAlign:'center',marginTop:20,display:'flex',alignItems:'center',
          justifyContent:'center',gap:5}}>
          <span style={{fontSize:10,color:C.textDim}}>Built with</span>
          <Heart size={10} color='#f472b6' fill='#f472b6'/>
          <span style={{fontSize:10,color:'#f472b6',fontWeight:600}}>Team Data Minds</span>
          <span style={{fontSize:10,color:C.textDim}}>· Sanjith · Pranav</span>
        </div>
      </div>
    </div>
  );
}

// ── Profile Page ──────────────────────────────────────────────────
function ProfilePage({user,onLogout}){
  const [loading,setLoading]=useState(false);
  const [editing,setEditing]=useState(false);
  const [name,setName]=useState(user.name||'');
  const [photoFile,setPhotoFile]=useState(null);
  const [photoPreview,setPhotoPreview]=useState(user.picture||null);
  const photoRef=React.useRef();

  const initials=(name||'?').split(' ').map(w=>w[0]).join('').toUpperCase().slice(0,2);

  const saveProfile=()=>{
    const updated={...user, name, picture:photoPreview};
    localStorage.setItem('bms_user',JSON.stringify(updated));
    window.location.reload(); // refresh to pick up new name in header
  };

  const handlePhoto=e=>{
    const f=e.target.files[0];
    if(!f) return;
    const reader=new FileReader();
    reader.onload=ev=>{ setPhotoPreview(ev.target.result); setPhotoFile(f); };
    reader.readAsDataURL(f);
  };

  const logout=()=>{
    setLoading(true);
    localStorage.removeItem('bms_token');
    localStorage.removeItem('bms_user');
    onLogout();
  };

  return(
    <div style={{display:'flex',flexDirection:'column',gap:20,maxWidth:600,margin:'0 auto'}} className="fade-in">
      <GlowCard accent={C.blue}>
        {/* Avatar + photo upload */}
        <div style={{display:'flex',alignItems:'flex-start',gap:20,flexWrap:'wrap'}}>
          <div style={{position:'relative',flexShrink:0}}>
            {photoPreview
              ?<img src={photoPreview} alt="avatar" referrerPolicy="no-referrer"
                  style={{width:88,height:88,borderRadius:'50%',border:`3px solid ${C.blue}`,objectFit:'cover'}}/>
              :<div style={{width:88,height:88,borderRadius:'50%',background:C.blue+'30',
                  border:`3px solid ${C.blue}`,display:'flex',alignItems:'center',justifyContent:'center',
                  fontSize:28,fontWeight:700,color:C.blue}}>{initials}</div>
            }
            <button onClick={()=>photoRef.current.click()}
              style={{position:'absolute',bottom:0,right:0,width:26,height:26,borderRadius:'50%',
                background:`linear-gradient(135deg,#cc0000,#c026d3)`,border:'2px solid '+C.bg2,
                display:'flex',alignItems:'center',justifyContent:'center',cursor:'pointer'}}>
              <span style={{color:'#fff',fontSize:13,lineHeight:1}}>+</span>
            </button>
            <input ref={photoRef} type="file" accept="image/*" style={{display:'none'}} onChange={handlePhoto}/>
          </div>

          <div style={{flex:1}}>
            {editing?(
              <div>
                <div style={{...S.label,marginBottom:5}}>Display Name</div>
                <input value={name} onChange={e=>setName(e.target.value)}
                  style={{width:'100%',padding:'10px 14px',borderRadius:10,border:`1px solid ${C.blue}`,
                    background:'rgba(77,159,255,0.08)',color:C.text,fontSize:15,fontWeight:600,
                    fontFamily:'DM Sans,sans-serif',outline:'none',boxSizing:'border-box',marginBottom:12}}/>
                <div style={{display:'flex',gap:8}}>
                  <button onClick={()=>{saveProfile();setEditing(false);}}
                    style={{padding:'8px 18px',borderRadius:9,border:'none',
                      background:`linear-gradient(135deg,#cc0000,#c026d3)`,color:'#fff',
                      fontSize:12,fontWeight:600,cursor:'pointer',fontFamily:'DM Sans,sans-serif'}}>
                    Save
                  </button>
                  <button onClick={()=>{setEditing(false);setName(user.name||'');}}
                    style={{padding:'8px 18px',borderRadius:9,border:`1px solid ${C.border2}`,
                      background:'transparent',color:C.textMid,fontSize:12,cursor:'pointer',fontFamily:'DM Sans,sans-serif'}}>
                    Cancel
                  </button>
                </div>
              </div>
            ):(
              <div>
                <div style={{display:'flex',alignItems:'center',gap:10,marginBottom:4}}>
                  <div style={{fontSize:20,fontWeight:800,color:C.text}}>{name||user.name}</div>
                  <button onClick={()=>setEditing(true)}
                    style={{padding:'3px 10px',borderRadius:7,border:`1px solid ${C.border2}`,
                      background:'transparent',color:C.textMid,fontSize:11,cursor:'pointer',fontFamily:'DM Sans,sans-serif'}}>
                    Edit
                  </button>
                </div>
                <div style={{display:'flex',alignItems:'center',gap:7,marginBottom:10}}>
                  <Mail size={12} color={C.textDim}/>
                  <span style={{fontSize:13,color:C.textMid}}>{user.email}</span>
                </div>
                <div style={{display:'flex',gap:7}}>
                  <span style={{padding:'3px 10px',borderRadius:20,fontSize:10,fontWeight:600,
                    background:C.green+'20',color:C.green,border:`1px solid ${C.green}30`}}>
                    ✓ {user.provider==='google'?'Google':'Email'} Account
                  </span>
                  <span style={{padding:'3px 10px',borderRadius:20,fontSize:10,fontWeight:600,
                    background:C.blue+'20',color:C.blue,border:`1px solid ${C.blue}30`}}>
                    BMS Hackathon
                  </span>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Account info */}
        <div style={{marginTop:20,paddingTop:16,borderTop:`1px solid ${C.border}`}}>
          {[
            ['Email',       user.email],
            ['Last login',  user.login_at ? new Date(user.login_at+'Z').toLocaleString() : '—'],
            ['Sign-in method', user.provider==='google' ? 'Google' : 'Email OTP'],
          ].map(([l,v])=>(
            <div key={l} style={{display:'flex',justifyContent:'space-between',
              padding:'8px 0',borderBottom:`1px solid ${C.border}`}}>
              <span style={{fontSize:12,color:C.textDim}}>{l}</span>
              <span style={{fontSize:12,color:C.text,fontWeight:500}}>{v}</span>
            </div>
          ))}
        </div>

        {/* Sign out */}
        <button onClick={logout} disabled={loading}
          style={{display:'flex',alignItems:'center',gap:8,marginTop:20,padding:'11px 22px',
            borderRadius:10,border:`1px solid rgba(229,57,53,0.4)`,background:'rgba(229,57,53,0.08)',
            color:C.red,fontSize:13,fontWeight:600,cursor:loading?'not-allowed':'pointer',
            fontFamily:'DM Sans,sans-serif',transition:'all 0.2s'}}>
          {loading?<Loader size={14} className="spin"/>:<LogOut size={14}/>}
          Sign Out
        </button>
      </GlowCard>
    </div>
  );
}

// ── Support Page ──────────────────────────────────────────────────
function SupportPage({user}){
  const [form,setForm]=useState({
    name:user?.name||'', email:user?.email||'',
    subject:'', message:'', category:'question',
  });
  const [loading,setLoading]=useState(false);
  const [status,setStatus]=useState(null);
  const [errMsg,setErrMsg]=useState('');
  const [tickets,setTickets]=useState(()=>{
    try{ return JSON.parse(localStorage.getItem('bms_tickets')||'[]'); }catch{ return []; }
  });

  const set=k=>e=>setForm(f=>({...f,[k]:e.target.value}));

  const submit=async()=>{
    if(!form.subject.trim()||!form.message.trim()){
      setStatus('error');setErrMsg('Please fill in subject and message.');return;
    }
    setLoading(true);setStatus(null);
    try{
      const r=await authFetch('/auth/support',{method:'POST',
        headers:{'Content-Type':'application/json'},body:JSON.stringify(form)});
      const d=await r.json();
      if(r.ok){
        const ticket={id:Date.now(),subject:form.subject,category:form.category,
          status:'open',date:new Date().toLocaleDateString()};
        const updated=[ticket,...tickets];
        setTickets(updated);
        localStorage.setItem('bms_tickets',JSON.stringify(updated));
        setStatus('success');
        setForm(f=>({...f,subject:'',message:''}));
      } else{ setStatus('error');setErrMsg(d.detail||'Failed to send.'); }
    }catch{ setStatus('error');setErrMsg('Network error — is the backend running?'); }
    finally{ setLoading(false); }
  };

  const markResolved=id=>{
    const updated=tickets.map(t=>t.id===id?{...t,status:'resolved'}:t);
    setTickets(updated);
    localStorage.setItem('bms_tickets',JSON.stringify(updated));
  };

  const inp={width:'100%',padding:'10px 14px',borderRadius:10,border:`1px solid ${C.border2}`,
    background:C.bg3,color:C.text,fontSize:13,fontFamily:'DM Sans,sans-serif',
    outline:'none',boxSizing:'border-box'};

  return(
    <div style={{display:'flex',flexDirection:'column',gap:20}} className="fade-in">
      <GlowCard accent={C.blue}>
        <div style={{fontSize:17,fontWeight:700,color:C.text,marginBottom:4}}>Support</div>
        <div style={{fontSize:13,color:C.textMid,marginBottom:20}}>
          Questions, bugs, or feature requests — we typically reply within 1–2 business days.
        </div>

        <div style={{...S.label,marginBottom:8}}>Category</div>
        <div style={{display:'flex',gap:8,marginBottom:16,flexWrap:'wrap'}}>
          {[['question','Question'],['bug','Bug Report'],['feature','Feature Request'],['general','General']].map(([v,l])=>(
            <button key={v} onClick={()=>setForm(f=>({...f,category:v}))}
              style={{padding:'6px 16px',borderRadius:20,fontSize:12,fontWeight:500,cursor:'pointer',
                border:`1px solid ${form.category===v?C.blue:C.border}`,
                background:form.category===v?C.blueL:'transparent',
                color:form.category===v?C.blue:C.textMid,fontFamily:'DM Sans,sans-serif',transition:'all 0.15s'}}>
              {l}
            </button>
          ))}
        </div>

        <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:12,marginBottom:12}}>
          <div>
            <div style={{...S.label,marginBottom:5}}>Name</div>
            <input value={form.name} onChange={set('name')} style={inp} placeholder="Your name" disabled={!!user}/>
          </div>
          <div>
            <div style={{...S.label,marginBottom:5}}>Email</div>
            <input value={form.email} onChange={set('email')} style={inp} placeholder="your@email.com" disabled={!!user}/>
          </div>
        </div>

        <div style={{...S.label,marginBottom:5}}>Subject</div>
        <input value={form.subject} onChange={set('subject')} style={{...inp,marginBottom:12}}
          placeholder="Briefly describe your issue"/>

        <div style={{...S.label,marginBottom:5}}>Message</div>
        <textarea value={form.message} onChange={set('message')} rows={5}
          style={{...inp,resize:'vertical',marginBottom:16,lineHeight:1.6}}
          placeholder="Include error messages, steps to reproduce, or feature ideas."/>

        {status==='success'&&(
          <div style={{padding:'12px 16px',borderRadius:10,background:C.greenL,
            border:`1px solid ${C.green}30`,fontSize:13,color:C.green,marginBottom:14,
            display:'flex',gap:9,alignItems:'center'}}>
            <CheckCircle size={15} color={C.green}/>
            Ticket submitted — we will reply to <strong>{form.email||user?.email}</strong>.
          </div>
        )}
        {status==='error'&&(
          <div style={{padding:'12px 16px',borderRadius:10,background:C.redL,
            border:`1px solid ${C.red}30`,fontSize:13,color:C.red,marginBottom:14,
            display:'flex',gap:9,alignItems:'center'}}>
            <AlertCircle size={15} color={C.red}/>{errMsg}
          </div>
        )}

        <button onClick={submit} disabled={loading}
          style={{display:'flex',alignItems:'center',gap:10,padding:'12px 24px',borderRadius:12,
            border:'none',background:loading?C.bg4:`linear-gradient(135deg,${C.accent},#c026d3)`,
            color:loading?C.textDim:'#fff',fontSize:14,fontWeight:700,
            cursor:loading?'not-allowed':'pointer',fontFamily:'DM Sans,sans-serif',
            boxShadow:loading?'none':'0 4px 20px rgba(204,0,0,0.3)',transition:'all 0.2s'}}>
          {loading?<><Loader size={15} className="spin"/>Sending…</>:<><Send size={15}/>Submit Ticket</>}
        </button>
      </GlowCard>

      {/* Ticket tracker */}
      {tickets.length>0&&(
        <GlowCard accent={C.purple}>
          <div style={{...S.label,marginBottom:14}}>Your Tickets</div>
          {tickets.map(t=>(
            <div key={t.id} style={{display:'flex',alignItems:'center',justifyContent:'space-between',
              padding:'10px 0',borderBottom:`1px solid ${C.border}`,flexWrap:'wrap',gap:8}}>
              <div>
                <div style={{fontSize:13,fontWeight:500,color:C.text,marginBottom:2}}>{t.subject}</div>
                <div style={{fontSize:11,color:C.textDim}}>{t.category} · {t.date}</div>
              </div>
              <div style={{display:'flex',alignItems:'center',gap:8}}>
                <span style={{padding:'3px 10px',borderRadius:20,fontSize:10,fontWeight:700,
                  background:t.status==='resolved'?C.green+'20':C.amber+'20',
                  color:t.status==='resolved'?C.green:C.amber,
                  border:`1px solid ${t.status==='resolved'?C.green:C.amber}30`}}>
                  {t.status==='resolved'?'Resolved':'Open'}
                </span>
                {t.status!=='resolved'&&(
                  <button onClick={()=>markResolved(t.id)}
                    style={{padding:'4px 10px',borderRadius:7,border:`1px solid ${C.border2}`,
                      background:'transparent',color:C.textMid,fontSize:11,cursor:'pointer',fontFamily:'DM Sans,sans-serif'}}>
                    Mark resolved
                  </button>
                )}
              </div>
            </div>
          ))}
        </GlowCard>
      )}
    </div>
  );
}

// ── Main App ─────────────────────────────────────────────────────
export default function App(){
  const [user,setUser]=useState(()=>{
    const u=getStoredUser();
    console.log('[BMS App init] user from localStorage:', u ? u.email : 'null');
    return u;
  });
  const [step,setStep]=useState(0);
  const [cleanseResult,setCleanseResult]=useState(null);
  const [trainResult,setTrainResult]=useState(null);
  const [uploadedFile,setUploadedFile]=useState(null);

  const handleCleanseDone=(result,file)=>{setCleanseResult(result);setUploadedFile(file);setStep(1);};
  const handleTrainDone=result=>{setTrainResult(result);setStep(2);};
  const handleNext=()=>setStep(3);
  const handleLogin=u=>{setUser(u);setStep(0);};
  const handleLogout=()=>{
    localStorage.removeItem('bms_token');
    localStorage.removeItem('bms_user');
    setUser(null);setStep(0);
  };

  // Watch for localStorage changes (e.g. after Google OAuth redirect)
  React.useEffect(()=>{
    if(!user){
      const u=getStoredUser();
      const t=getToken();
      if(u && t && u.email) setUser(u);
    }
  },[user]);

  // Show login if not authenticated
  if(!user) return <LoginPage onLogin={handleLogin}/>;

  // steps 4,5,6,7 always accessible
  const canGo=s=>{
    if([0,4,5,6,7].includes(s))return true;
    if(s===1)return!!cleanseResult;
    if(s===2)return!!trainResult;
    if(s===3)return!!trainResult;
    return false;
  };

  // step 6 = Profile, step 7 = Support
  const UI_STEPS=[
    {id:0,label:'Upload & Cleanse', icon:Database},
    {id:1,label:'Select & Train',   icon:Cpu},
    {id:2,label:'Results',          icon:BarChart2},
    {id:3,label:'AI Report',        icon:FileText},
    {id:4,label:'BMS1 vs BMS2',     icon:GitCompare},
    {id:5,label:'About & FAQ',      icon:BookOpen},
  ];

  return(
    <div style={{minHeight:'100vh',background:C.bg,display:'flex',flexDirection:'column'}}>
      {/* Header */}
      <div style={{background:C.bg2,borderBottom:`1px solid ${C.border}`,
        padding:'11px 28px',display:'flex',alignItems:'center',gap:14,flexWrap:'wrap'}}>

        {/* Logo */}
        <div style={{background:'#fff',borderRadius:8,padding:'5px 10px',display:'inline-flex',alignItems:'center',flexShrink:0}}>
          <img src="/bms-logo.png" alt="BMS" style={{height:26,objectFit:'contain'}}
            onError={e=>{e.target.style.display='none';}}/>
        </div>

        {/* Title */}
        <div style={{flexShrink:0}}>
          <div style={{fontSize:15,fontWeight:700,color:C.text,letterSpacing:'-0.2px'}}>
            mAb Fucosylation · ML Pipeline
          </div>
          <div style={{display:'flex',alignItems:'center',gap:7,marginTop:2}}>
            <span style={{fontSize:9,fontWeight:700,padding:'2px 7px',borderRadius:4,
              background:`linear-gradient(90deg,${C.accent},#c026d3)`,color:'#fff',letterSpacing:'0.05em'}}>
              BMS HACKATHON
            </span>
            <span style={{fontSize:10,color:C.textDim}}>MS Data Science · Rutgers × BMS</span>
          </div>
        </div>

        {/* Built with love */}
        <div style={{display:'flex',alignItems:'center',gap:5,padding:'5px 12px',borderRadius:20,
          background:'rgba(244,114,182,0.08)',border:'1px solid rgba(244,114,182,0.2)',flexShrink:0}}>
          <span style={{fontSize:10,color:'#f472b6',fontWeight:500}}>Built with</span>
          <Heart size={11} color='#f472b6' fill='#f472b6'/>
          <span style={{fontSize:10,color:'#f472b6',fontWeight:500}}>by Team Data Minds</span>
        </div>

        {/* Step nav */}
        <div style={{marginLeft:'auto',display:'flex',alignItems:'center',gap:0,flexWrap:'wrap'}}>
          {UI_STEPS.map((s,i)=>{
            const done=step>s.id&&s.id<4;
            const active=step===s.id;
            const Icon=s.icon;
            const accessible=canGo(s.id);
            const isBMS=s.id===4;
            const isAbout=s.id===5;
            const activeColor=isAbout?C.teal:isBMS?C.purple:C.blue;
            const activeLight=isAbout?C.teal+'18':isBMS?C.purpleL:C.blueL;
            return(
              <React.Fragment key={s.id}>
                {(i===4||i===5)&&<div style={{width:1,height:20,background:C.border,margin:'0 4px'}}/>}
                <button onClick={()=>accessible&&setStep(s.id)} disabled={!accessible}
                  style={{display:'flex',alignItems:'center',gap:6,padding:'6px 11px',borderRadius:8,
                    border:`1px solid ${active?activeColor:done?C.green:C.border}`,
                    background:active?activeLight:done?C.greenL:'transparent',
                    cursor:accessible?'pointer':'not-allowed',transition:'all 0.2s',
                    fontSize:11,fontWeight:active?600:400,
                    color:active?activeColor:done?C.green:C.textDim,
                    fontFamily:'DM Sans,sans-serif',opacity:accessible?1:0.4}}>
                  {done?<CheckCircle size={12} color={C.green}/>:<Icon size={12}/>}
                  {s.label}
                </button>
                {i<3&&<ChevronRight size={12} color={C.textDim} style={{margin:'0 1px'}}/>}
              </React.Fragment>
            );
          })}
          {/* Divider */}
          <div style={{width:1,height:20,background:C.border,margin:'0 8px'}}/>
          {/* Profile button */}
          <button onClick={()=>setStep(6)}
            style={{display:'flex',alignItems:'center',gap:7,padding:'5px 5px 5px 10px',borderRadius:20,
              border:`1px solid ${step===6?C.blue:C.border}`,
              background:step===6?C.blueL:'rgba(255,255,255,0.04)',
              cursor:'pointer',transition:'all 0.2s',fontFamily:'DM Sans,sans-serif'}}>
            {user.picture
              ?<img src={user.picture} referrerPolicy="no-referrer"
                  style={{width:22,height:22,borderRadius:'50%',border:`1px solid ${C.blue}`}} alt=""/>
              :<UserCircle size={18} color={step===6?C.blue:C.textDim}/>
            }
            <span style={{fontSize:11,color:step===6?C.blue:C.textMid,fontWeight:step===6?600:400}}>
              {user.given_name||user.name?.split(' ')[0]||'Profile'}
            </span>
          </button>
          {/* Support button */}
          <button onClick={()=>setStep(7)}
            style={{display:'flex',alignItems:'center',gap:6,padding:'6px 11px',marginLeft:4,borderRadius:8,
              border:`1px solid ${step===7?C.amber:C.border}`,
              background:step===7?C.amberL:'transparent',
              cursor:'pointer',transition:'all 0.2s',fontSize:11,
              color:step===7?C.amber:C.textDim,fontFamily:'DM Sans,sans-serif',fontWeight:step===7?600:400}}>
            <MessageSquare size={12}/>Support
          </button>
        </div>
      </div>

      {/* Content */}
      <div style={{flex:1,maxWidth:1200,width:'100%',margin:'0 auto',padding:'28px 32px'}}>
        {step===0&&<UploadCleanse onDone={handleCleanseDone}/>}
        {step===1&&<ModelSelect cleanseResult={cleanseResult} uploadedFile={uploadedFile} onDone={handleTrainDone}/>}
        {step===2&&trainResult&&<ResultsDashboard trainResult={trainResult} onNext={handleNext}/>}
        {step===3&&trainResult&&<AIReport trainResult={trainResult}/>}
        {step===4&&<BMSComparison/>}
        {step===5&&<AboutFAQ/>}
        {step===6&&<ProfilePage user={user} onLogout={handleLogout}/>}
        {step===7&&<SupportPage user={user}/>}
      </div>

      {/* Footer */}
      <div style={{borderTop:`1px solid ${C.border}`,padding:'10px 32px',
        display:'flex',alignItems:'center',justifyContent:'space-between',background:C.bg2}}>
        <span style={{fontSize:11,color:C.textDim}}>
          BMS Hackathon · MS Data Science · Rutgers University × Bristol Myers Squibb
        </span>
        <div style={{display:'flex',alignItems:'center',gap:5}}>
          <span style={{fontSize:11,color:C.textDim}}>Built with</span>
          <Heart size={11} color='#f472b6' fill='#f472b6'/>
          <span style={{fontSize:11,color:'#f472b6',fontWeight:600}}>Team Data Minds</span>
          <span style={{fontSize:11,color:C.textDim}}>· Sanjith Ganesh & Pranav Senthilkumaran</span>
        </div>
      </div>
    </div>
  );
}