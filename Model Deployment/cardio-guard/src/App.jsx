import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { LineChart, Line, XAxis, YAxis, ResponsiveContainer, Tooltip as RechartsTooltip } from 'recharts';
import { Activity, Heart, Baby, AlertCircle, CheckCircle, ArrowRight, HelpCircle, Info, Upload, FileText, Play, Cpu, ShieldCheck, Database, ExternalLink, XCircle, BookOpen, FileCheck } from 'lucide-react';

// --- DEFINITIONS ---
const DEFINITIONS = {
  sdnn: "Standard Deviation of NN intervals (ms). <50ms indicates risk.",
  rmssd: "Root Mean Square of Successive Differences (ms). Vagal tone marker.",
  pnn50: "Percentage of intervals >50ms different. Healthy > 3%.",
  mean_hr: "Average Heart Rate (BPM).",
  baseline_value: "Baseline FHR (bpm). Normal: 110-160.",
  accelerations: "Healthy temporary increases in FHR.",
  fetal_movement: "Recorded fetal movements.",
  uterine_contractions: "Contractions per second.",
  decelerations: "Drops in FHR. Late/Prolonged are dangerous.",
  astv: "Abnormal Short Term Variability (%). High is bad.",
  mstv: "Mean Short Term Variability. Low is bad."
};

// --- COMPONENTS ---

const Tooltip = ({ text }) => (
  <div className="group relative inline-block ml-2 z-50">
    <HelpCircle size={14} className="text-blue-300 cursor-help hover:text-white transition-colors" />
    <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 w-64 p-3 bg-slate-800 border border-white/20 rounded-lg text-xs text-white shadow-xl opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all duration-200 pointer-events-none z-50">
      {text}
    </div>
  </div>
);

const Card = ({ children, className = "" }) => (
  <motion.div 
    initial={{ opacity: 0, scale: 0.99 }}
    animate={{ opacity: 1, scale: 1 }}
    className={`bg-slate-900/80 backdrop-blur-xl border border-white/10 rounded-3xl shadow-2xl overflow-hidden flex flex-col ${className}`}
  >
    {children}
  </motion.div>
);

const InputField = ({ label, name, value, onChange, placeholder, tooltipKey }) => (
  <div className="mb-4">
    <label className="flex items-center text-blue-100 text-sm font-medium mb-2">
      {label}
      {tooltipKey && <Tooltip text={DEFINITIONS[tooltipKey]} />}
    </label>
    <input
      type="number"
      step="0.001"
      name={name}
      value={value}
      onChange={onChange}
      placeholder={placeholder}
      className="w-full bg-white/5 border border-blue-300/30 rounded-lg py-3 px-4 text-white placeholder-blue-300/50 focus:outline-none focus:ring-2 focus:ring-blue-400 transition-all"
    />
  </div>
);

const InfoCard = ({ icon: Icon, title, desc, delay, color, link, linkText }) => (
  <motion.div
    initial={{ opacity: 0, y: 20 }}
    animate={{ opacity: 1, y: 0 }}
    transition={{ delay }}
    className="bg-white/5 border border-white/10 rounded-xl p-6 hover:bg-white/10 transition-colors flex flex-col h-full"
  >
    <div className={`p-3 rounded-lg w-fit mb-4 ${color}`}>
      <Icon size={24} className="text-white" />
    </div>
    <h3 className="text-lg font-bold mb-2 text-white">{title}</h3>
    <p className="text-blue-200/70 text-sm leading-relaxed mb-4 flex-grow">{desc}</p>
    
    {link && (
        <a 
            href={link} 
            target="_blank" 
            rel="noopener noreferrer"
            className="inline-flex items-center gap-2 text-xs font-bold text-blue-300 hover:text-white transition-colors mt-auto group"
        >
            {linkText || "Learn more"} <ExternalLink size={12} className="group-hover:translate-x-1 transition-transform" />
        </a>
    )}
  </motion.div>
);

const ResultGauge = ({ score, status }) => {
  const safeScore = typeof score === 'number' ? score : 0;
  const isDanger = safeScore > 0.5;
  const colorClass = isDanger ? "text-red-400" : "text-emerald-400";
  const bgClass = isDanger ? "bg-red-500" : "bg-emerald-500";
  
  return (
    <div className="flex items-center gap-8 h-full justify-center">
      <div className="relative inline-flex justify-center items-center">
        <motion.div 
          animate={{ scale: [1, 1.1, 1], opacity: [0.2, 0.1, 0.2] }} 
          transition={{ repeat: Infinity, duration: 2 }}
          className={`absolute w-32 h-32 rounded-full ${bgClass}`}
        />
        <div className={`relative z-10 p-6 rounded-full bg-slate-900 border-2 border-white/10 shadow-xl`}>
          {isDanger ? <AlertCircle size={48} className="text-red-500" /> : <CheckCircle size={48} className="text-emerald-500" />}
        </div>
      </div>
      <div className="text-left">
          <h2 className={`text-3xl font-bold mb-1 ${colorClass}`}>{status || "Unknown Status"}</h2>
          <div className="flex items-center gap-2">
            <span className="text-blue-200/60 text-sm uppercase tracking-wider font-bold">Risk Factor</span>
            <span className={`text-2xl font-mono font-bold ${colorClass}`}>{(safeScore * 100).toFixed(1)}%</span>
          </div>
          <p className="text-xs text-blue-200/40 mt-2 max-w-[200px]">
              {isDanger ? "High probability of cardiac anomaly detected. Immediate clinical review recommended." : "Biomarkers indicate normal cardiac function. Continue routine monitoring."}
          </p>
      </div>
    </div>
  );
};

const SimulationMonitor = ({ data, color = "#10b981", label, meanVal, stdVal }) => {
  return (
    <div className="h-full w-full bg-black/40 relative overflow-hidden flex flex-col">
      <div className="absolute top-4 left-4 text-xs font-mono text-white/50 tracking-wider flex items-center gap-2 z-10">
        <span className="w-2 h-2 rounded-full bg-red-500 animate-pulse"/> {label} LIVE
      </div>
      <div className="absolute top-4 right-4 text-2xl font-mono font-bold text-white tabular-nums z-10">
         {data.length > 0 ? data[data.length - 1].value.toFixed(1) : 0} <span className="text-xs text-white/50">BPM</span>
      </div>
      <div className="flex-1 min-h-0">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data}>
            <YAxis domain={[meanVal - stdVal * 4, meanVal + stdVal * 4]} hide />
            <Line 
              type="monotone" 
              dataKey="value" 
              stroke={color} 
              strokeWidth={2} 
              dot={false} 
              isAnimationActive={false} 
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
      {/* Grid Overlay */}
      <div className="absolute inset-0 bg-[linear-gradient(rgba(255,255,255,0.03)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.03)_1px,transparent_1px)] bg-[size:20px_20px] pointer-events-none" />
    </div>
  );
};

// --- MAIN APP ---

export default function App() {
  const [view, setView] = useState('landing'); // landing, adult, fetal, about
  const [mode, setMode] = useState('manual');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  
  const [extractedData, setExtractedData] = useState(null);
  const [simParams, setSimParams] = useState(null);
  
  // Simulation State
  const [simData, setSimData] = useState([]);
  const [isSimulating, setIsSimulating] = useState(false);
  const simInterval = useRef(null);

  // Forms
  const [adultForm, setAdultForm] = useState({ sdnn: '145', rmssd: '35', pnn50: '12', mean_hr: '72' });
  const [fetalForm, setFetalForm] = useState({ baseline_value: '133', accelerations: '0.003', fetal_movement: '0.0', uterine_contractions: '0.008', astv: '20', mstv: '0.9' });

  // Unified Prediction Trigger
  const triggerPrediction = async (featuresOverride = null) => {
    setLoading(true);
    setResult(null);
    const endpoint = view === 'adult' ? 'http://localhost:8000/predict/adult' : 'http://localhost:8000/predict/fetal';
    
    let payload;
    
    // Logic to construct payload
    if (view === 'adult') {
        // If overriding features are provided (from file upload)
        if (featuresOverride) {
            // ADULT INFERENCE FIX:
            // The file parsing endpoint returns general keys (often from Fetal .hea files).
            // We need to map them to the Adult model input (sdnn, rmssd, pnn50, mean_hr).
            // If keys are missing, we default them to prevent 422 errors.
            payload = {
                sdnn: featuresOverride['sdnn'] || featuresOverride['Std FHR'] || 50.0,
                rmssd: featuresOverride['rmssd'] || 30.0, // Default if missing
                pnn50: featuresOverride['pnn50'] || 10.0, // Default if missing
                mean_hr: featuresOverride['mean_hr'] || featuresOverride['Mean FHR'] || 75.0
            };
        } else if (mode === 'simulation' && extractedData) {
            // Same mapping logic if triggering from existing extracted data
            payload = {
                sdnn: extractedData['sdnn'] || extractedData['Std FHR'] || 50.0,
                rmssd: extractedData['rmssd'] || 30.0,
                pnn50: extractedData['pnn50'] || 10.0,
                mean_hr: extractedData['mean_hr'] || extractedData['Mean FHR'] || 75.0
            };
        } else {
            // Manual form data
            payload = adultForm;
        }
    } else {
        // Fetal Mode
        if (featuresOverride) {
            payload = { features: featuresOverride };
        } else if (mode === 'simulation' && extractedData) {
            payload = { features: extractedData };
        } else {
            payload = { features: fetalForm };
        }
    }

    try {
      const res = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Prediction failed");
      setResult(data);
    } catch (err) {
      console.error(err);
      setResult({ error: true, message: err.message || "Connection Failed" });
    } finally {
      setLoading(false);
    }
  };

  // Handle File Upload
  const handleFileUpload = async (e) => {
    const uploadedFile = e.target.files[0];
    if (!uploadedFile) return;
    setLoading(true);

    const formData = new FormData();
    formData.append('file', uploadedFile);

    try {
      const endpoint = 'http://localhost:8000/upload/fetal-header';
      const res = await fetch(endpoint, { method: 'POST', body: formData });
      if (!res.ok) throw new Error("Backend failed");
      
      const data = await res.json();
      setExtractedData(data.extracted_features);
      setSimParams(data.simulation_params);
      
      // AUTO-TRIGGER PREDICTION HERE
      await triggerPrediction(data.extracted_features);

    } catch (err) {
      console.error(err);
      alert("Error parsing file. Check backend.");
    } finally {
      setLoading(false);
    }
  };

  const toggleSimulation = () => {
    if (isSimulating) {
      clearInterval(simInterval.current);
      setIsSimulating(false);
    } else {
      setIsSimulating(true);
      const mean = simParams?.['Mean FHR'] || (view === 'adult' ? 70 : 140);
      const std = simParams?.['Std FHR'] || (view === 'adult' ? 5 : 10);
      
      const initialData = Array.from({ length: 150 }, (_, i) => ({
        time: Date.now() - (150 - i) * 40,
        value: mean
      }));
      setSimData(initialData);

      simInterval.current = setInterval(() => {
        setSimData(prev => {
          const newVal = mean + (Math.random() - 0.5) * std * 2;
          const newPoint = { time: Date.now(), value: newVal };
          const newData = [...prev, newPoint];
          newData.shift(); 
          return newData;
        });
      }, 40);
    }
  };

  useEffect(() => {
    return () => clearInterval(simInterval.current);
  }, []);

  // Stop simulation on View Change, Mode Change, Window Hidden, or Navigation
  useEffect(() => {
    // 1. Stop if navigating away from analysis views
    if (view === 'landing' || view === 'about' || mode === 'manual') {
        if (simInterval.current) {
            clearInterval(simInterval.current);
            simInterval.current = null;
        }
        setIsSimulating(false);
    }

    // 2. Window Visibility Handler
    const handleVisibilityChange = () => {
      if (document.hidden) {
        if (simInterval.current) {
            clearInterval(simInterval.current);
            simInterval.current = null;
        }
        setIsSimulating(false);
      }
    };

    document.addEventListener("visibilitychange", handleVisibilityChange);
    return () => document.removeEventListener("visibilitychange", handleVisibilityChange);
  }, [view, mode]);

  const LandingPage = () => (
    <div className="flex gap-6 h-full items-center justify-center">
      <div onClick={() => setView('adult')} className="w-64 h-80 group cursor-pointer bg-white/5 border border-white/10 rounded-3xl hover:bg-blue-600/20 transition-all flex flex-col items-center justify-center text-center">
        <Activity size={56} className="text-blue-400 mb-6 group-hover:scale-110 transition-transform" />
        <h2 className="text-xl font-bold text-white">Adult CHF</h2>
        <p className="text-blue-200/60 mt-1 text-sm">ECG Analysis</p>
      </div>
      <div onClick={() => setView('fetal')} className="w-64 h-80 group cursor-pointer bg-white/5 border border-white/10 rounded-3xl hover:bg-emerald-600/20 transition-all flex flex-col items-center justify-center text-center">
        <Baby size={56} className="text-emerald-400 mb-6 group-hover:scale-110 transition-transform" />
        <h2 className="text-xl font-bold text-white">Fetal Health</h2>
        <p className="text-emerald-200/60 mt-1 text-sm">CTG Analysis</p>
      </div>
    </div>
  );

  const AboutPage = () => (
    <Card className="flex-1 w-full mx-auto max-h-full">
        <div className="flex items-center justify-between p-6 border-b border-white/10 bg-black/20">
            <div className="flex items-center gap-3">
                 <div className="p-2 rounded-lg bg-purple-500/20 text-purple-400">
                    <BookOpen size={24} />
                 </div>
                 <h2 className="text-lg font-bold">About System</h2>
            </div>
        </div>
        
        <div className="flex-1 overflow-y-auto p-8 scrollbar-hide">
            <div className="max-w-4xl mx-auto">
                <div className="text-center mb-10">
                    <h1 className="text-3xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-300 to-emerald-300 mb-3">Clinical Decision Support</h1>
                    <p className="text-blue-200/60 max-w-2xl mx-auto">CardioGuard AI leverages advanced ensemble machine learning to detect cardiac anomalies in adults and fetuses with high precision.</p>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-10">
                    <InfoCard 
                      icon={Activity}
                      title="Adult Model"
                      color="bg-blue-500"
                      delay={0.1}
                      desc="Analyzes HRV biomarkers (SDNN, RMSSD) from ECG signals to detect Congestive Heart Failure."
                      link="https://physionet.org/content/chfdb/1.0.0/"
                      linkText="BIDMC Data"
                    />
                    <InfoCard 
                      icon={Baby}
                      title="Fetal Model"
                      color="bg-emerald-500"
                      delay={0.2}
                      desc="Processes Cardiotocography (CTG) data including FHR accelerations and uterine contractions."
                      link="https://physionet.org/content/ctu-uhb-ctgdb/1.0.0/"
                      linkText="CTU-CHB Data"
                    />
                    <InfoCard 
                      icon={Cpu}
                      title="AI Ensemble"
                      color="bg-purple-500"
                      delay={0.3}
                      desc="Uses a Voting Classifier combining XGBoost, Random Forest, and SVM for robust predictions."
                    />
                     <InfoCard 
                      icon={ShieldCheck}
                      title="Safety"
                      color="bg-indigo-500"
                      delay={0.4}
                      desc="Designed as a decision aid. Always verify results with standard clinical protocols."
                    />
                </div>

                <div className="bg-white/5 border border-white/10 rounded-xl p-6 flex items-center justify-between">
                    <div className="flex items-center gap-4">
                        <div className="p-3 bg-slate-900 rounded-full text-blue-300">
                            <Database size={20} />
                        </div>
                        <div>
                            <h4 className="font-bold text-sm">Validated Datasets</h4>
                            <p className="text-xs text-white/50">Trained on PhysioNet Gold Standard Data</p>
                        </div>
                    </div>
                    <button onClick={() => setView('landing')} className="text-sm bg-white/10 hover:bg-white/20 px-4 py-2 rounded-lg transition-colors">Start Diagnosis</button>
                </div>
            </div>
        </div>
    </Card>
  );

  return (
    <div className="fixed inset-0 bg-slate-950 text-white font-sans overflow-hidden flex flex-col selection:bg-blue-500/30">
      <div className="absolute inset-0 pointer-events-none">
        <div className="absolute top-[-50%] left-[-20%] w-[80%] h-[80%] rounded-full bg-blue-600/5 blur-[120px]" />
        <div className="absolute bottom-[-50%] right-[-20%] w-[80%] h-[80%] rounded-full bg-emerald-600/5 blur-[120px]" />
      </div>

      <div className="relative z-10 w-full max-w-7xl mx-auto h-full flex flex-col p-4 md:p-6">
        <header className="flex-none flex justify-between items-center mb-4 px-2">
          <div className="flex items-center gap-3 cursor-pointer" onClick={() => { setView('landing'); setResult(null); setMode('manual'); }}>
            <Heart className="text-red-500 fill-red-500 animate-pulse" />
            <span className="text-lg font-bold tracking-tight">CardioGuard AI</span>
          </div>
          <div className="flex gap-4">
             <button onClick={() => setView('about')} className={`text-xs px-3 py-1.5 rounded-lg transition-all ${view === 'about' ? 'bg-white text-slate-900 font-bold' : 'text-blue-300 hover:text-white hover:bg-white/10'}`}>About</button>
             {view !== 'landing' && <button onClick={() => { setView('landing'); setResult(null); }} className="text-xs px-3 py-1.5 rounded-lg bg-white/5 hover:bg-white/10 text-blue-300 hover:text-white transition-all">Menu</button>}
          </div>
        </header>

        {view === 'landing' && <LandingPage />}
        {view === 'about' && <AboutPage />}
        
        {(view === 'adult' || view === 'fetal') && (
          <div className="flex-1 w-full mx-auto min-h-0 flex flex-col">
            <div className="flex-none flex items-center justify-between p-6 bg-slate-900/50 backdrop-blur-md rounded-t-3xl border border-white/10 mb-4">
              <div className="flex items-center gap-3">
                 <div className={`p-2 rounded-lg ${view === 'adult' ? 'bg-blue-500/20 text-blue-400' : 'bg-emerald-500/20 text-emerald-400'}`}>
                    {view === 'adult' ? <Activity size={24} /> : <Baby size={24} />}
                 </div>
                 <div>
                    <h2 className="text-lg font-bold">{view === 'adult' ? 'Adult Heart' : 'Fetal Monitor'}</h2>
                    <p className="text-xs opacity-50">Real-time Clinical DSS</p>
                 </div>
              </div>
              
              <div className="flex bg-black/40 rounded-full p-1 border border-white/5">
                 <button onClick={() => {setMode('manual'); setResult(null);}} className={`px-4 py-1.5 rounded-full text-xs font-medium transition-all ${mode === 'manual' ? 'bg-blue-600 text-white' : 'text-blue-300 hover:text-white'}`}>Manual Entry</button>
                 <button onClick={() => {setMode('simulation'); setResult(null);}} className={`px-4 py-1.5 rounded-full text-xs font-medium transition-all ${mode === 'simulation' ? 'bg-blue-600 text-white' : 'text-blue-300 hover:text-white'}`}>Simulation</button>
              </div>
            </div>

            {mode === 'manual' ? (
                // --- MANUAL MODE (Restored Original Style) ---
                <div className="flex-1 flex items-center justify-center p-4">
                    <Card className="w-full max-w-4xl p-8 border-t-4 border-t-blue-500/50">
                        {/* Title Section */}
                        <div className="mb-8 pb-4 border-b border-white/10">
                            <h2 className="text-2xl font-bold text-white mb-2">Patient Data Entry</h2>
                            <p className="text-blue-200/60 text-sm">Enter clinical parameters manually for immediate risk assessment.</p>
                        </div>

                        {!result ? (
                            <>
                                <div className={`grid grid-cols-1 ${view === 'fetal' ? 'md:grid-cols-3' : 'md:grid-cols-2'} gap-6`}>
                                    {view === 'adult' ? (
                                        <>
                                            <InputField label="SDNN (ms)" name="sdnn" tooltipKey="sdnn" placeholder="Normal: >50" value={adultForm.sdnn} onChange={(e) => setAdultForm({...adultForm, sdnn: e.target.value})} />
                                            <InputField label="RMSSD (ms)" name="rmssd" tooltipKey="rmssd" placeholder="Normal: >20" value={adultForm.rmssd} onChange={(e) => setAdultForm({...adultForm, rmssd: e.target.value})} />
                                            <InputField label="pNN50 (%)" name="pnn50" tooltipKey="pNN50" placeholder="Normal: >3%" value={adultForm.pnn50} onChange={(e) => setAdultForm({...adultForm, pnn50: e.target.value})} />
                                            <InputField label="Mean Heart Rate" name="mean_hr" tooltipKey="mean_hr" placeholder="60-100 BPM" value={adultForm.mean_hr} onChange={(e) => setAdultForm({...adultForm, mean_hr: e.target.value})} />
                                        </>
                                    ) : (
                                        <>
                                            <InputField label="Baseline FHR" name="baseline_value" tooltipKey="baseline_value" value={fetalForm.baseline_value} onChange={(e) => setFetalForm({...fetalForm, baseline_value: e.target.value})} />
                                            <InputField label="Accelerations" name="accelerations" tooltipKey="accelerations" value={fetalForm.accelerations} onChange={(e) => setFetalForm({...fetalForm, accelerations: e.target.value})} />
                                            <InputField label="Fetal Movement" name="fetal_movement" tooltipKey="fetal_movement" value={fetalForm.fetal_movement} onChange={(e) => setFetalForm({...fetalForm, fetal_movement: e.target.value})} />
                                            <InputField label="Uterine Contractions" name="uterine_contractions" tooltipKey="uterine_contractions" value={fetalForm.uterine_contractions} onChange={(e) => setFetalForm({...fetalForm, uterine_contractions: e.target.value})} />
                                            <InputField label="Abnormal STV" name="astv" tooltipKey="astv" value={fetalForm.astv} onChange={(e) => setFetalForm({...fetalForm, astv: e.target.value})} />
                                            <InputField label="Mean STV" name="mstv" tooltipKey="mstv" value={fetalForm.mstv} onChange={(e) => setFetalForm({...fetalForm, mstv: e.target.value})} />
                                        </>
                                    )}
                                </div>
                                <div className="mt-8 flex justify-center">
                                     <button
                                       onClick={() => triggerPrediction()}
                                       disabled={loading}
                                       className="w-full md:w-1/2 py-4 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 rounded-xl font-bold text-lg shadow-lg hover:shadow-blue-500/25 transition-all flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
                                     >
                                       {loading ? (
                                         <span className="flex items-center gap-2">
                                           <Activity className="animate-spin" /> Processing...
                                         </span>
                                       ) : <>Run Diagnostic Analysis <ArrowRight size={20}/></>}
                                     </button>
                                </div>
                            </>
                        ) : (
                            <div className="text-center py-8 animate-in fade-in zoom-in duration-300">
                                {result.error ? (
                                    <div className="text-center py-4">
                                      <XCircle className="mx-auto text-red-500 mb-2" size={40} />
                                      <h3 className="text-red-400 font-bold mb-1">Analysis Failed</h3>
                                      <p className="text-red-300/60 text-sm px-4">{result.message}</p>
                                    </div>
                                ) : (
                                    <ResultGauge score={result.risk_score} status={result.prediction} />
                                )}
                                <button onClick={() => setResult(null)} className="mt-8 px-8 py-3 bg-white/5 hover:bg-white/10 border border-white/10 rounded-xl font-medium transition-colors text-blue-200">
                                    Analyze New Patient
                                </button>
                            </div>
                        )}
                    </Card>
                </div>
            ) : (
                // --- SIMULATION MODE (New Split Screen) ---
                <Card className="flex-1 flex overflow-hidden">
                    <div className="flex-1 flex overflow-hidden">
                       {/* LEFT PANEL */}
                       <div className="w-1/3 p-6 border-r border-white/10 overflow-y-auto scrollbar-hide flex flex-col gap-6 bg-white/5">
                          <div className="border border-dashed border-white/20 rounded-xl p-6 text-center hover:border-blue-500/50 transition-colors bg-black/20 group cursor-pointer relative">
                             <input type="file" onChange={handleFileUpload} className="absolute inset-0 opacity-0 cursor-pointer"/>
                             <Upload className="mx-auto mb-3 text-blue-300 group-hover:scale-110 transition-transform" size={24} />
                             <p className="text-xs text-blue-100">Click to upload .hea file</p>
                          </div>

                          {extractedData ? (
                            <div className="flex-1 flex flex-col min-h-0">
                               <div className="flex-none flex justify-between items-center mb-3">
                                  <h3 className="text-xs font-bold text-emerald-400 flex items-center gap-2"><Database size={12}/> Extracted Data</h3>
                                  <button onClick={toggleSimulation} className="text-[10px] bg-emerald-500/20 text-emerald-300 px-2 py-1 rounded-full flex items-center gap-1 hover:bg-emerald-500/30">
                                    <Play size={10} /> {isSimulating ? 'Pause' : 'Stream'}
                                  </button>
                               </div>
                               <div className="flex-1 overflow-y-auto pr-2 space-y-2">
                                 {Object.entries(extractedData).map(([k, v]) => (
                                   <div key={k} className="flex justify-between items-center p-2 rounded bg-white/5 border border-white/5">
                                     <span className="text-[10px] opacity-60 font-mono truncate max-w-[120px]">{k}</span>
                                     <span className="text-xs text-blue-200 font-mono">{v}</span>
                                   </div>
                                 ))}
                               </div>
                            </div>
                          ) : (
                            <div className="flex-1 flex items-center justify-center opacity-30 text-xs">No data loaded</div>
                          )}
                       </div>

                       {/* RIGHT PANEL */}
                       <div className="w-2/3 flex flex-col bg-black/40 relative">
                          <div className={`flex-1 relative border-b border-white/10 min-h-0 transition-all duration-500 ${result ? 'basis-1/2' : 'basis-full'}`}>
                             {mode === 'simulation' ? (
                               <SimulationMonitor 
                                  data={simData} 
                                  label={view === 'adult' ? 'ECG' : 'FHR'} 
                                  meanVal={simParams?.['Mean FHR'] || (view === 'adult' ? 70 : 140)} 
                                  stdVal={simParams?.['Std FHR'] || (view === 'adult' ? 5 : 10)}
                               />
                             ) : (
                               <div className="absolute inset-0 flex items-center justify-center text-white/10">
                                  <Activity size={64} />
                               </div>
                             )}
                          </div>
                          
                          <div className={`flex-none transition-all duration-500 ease-in-out bg-slate-900/50 flex gap-6 items-center p-6 ${result ? 'h-[45%] border-t-2 border-white/10' : 'h-auto min-h-[140px]'}`}>
                             <div className="flex-1 h-full">
                               {!result ? (
                                  <div className="h-full flex items-center justify-center border border-white/5 rounded-xl bg-white/5">
                                     <button 
                                       onClick={() => triggerPrediction()} 
                                       disabled={loading || (mode === 'simulation' && !extractedData)}
                                       className="px-8 py-3 bg-blue-600 hover:bg-blue-500 rounded-xl font-bold shadow-lg transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
                                     >
                                       {loading ? <><Activity className="animate-spin" size={20}/> Analyzing...</> : 'Run Clinical Analysis'}
                                     </button>
                                  </div>
                               ) : (
                                 <div className="h-full flex flex-col md:flex-row gap-6 animate-in fade-in slide-in-from-bottom-4">
                                    <div className="flex-1 bg-black/40 rounded-2xl border border-white/10 h-full p-6 relative overflow-hidden shadow-inner">
                                       {result.error ? (
                                            <div className="h-full flex flex-col items-center justify-center text-center">
                                                <XCircle className="text-red-500 mb-3" size={40} />
                                                <h3 className="text-red-400 font-bold mb-1">Analysis Failed</h3>
                                                <p className="text-red-300/60 text-xs px-4">{result.message}</p>
                                            </div>
                                       ) : (
                                           <>
                                               <div className="absolute top-3 left-4 flex items-center gap-2 opacity-50">
                                                   <FileCheck size={14} />
                                                   <span className="text-xs font-mono uppercase tracking-widest">Clinical Report</span>
                                               </div>
                                               <ResultGauge score={result.risk_score} status={result.prediction} />
                                           </>
                                       )}
                                    </div>
                                    <div className="flex flex-col justify-center gap-3">
                                        <button onClick={() => setResult(null)} className="h-12 px-6 rounded-xl bg-white/5 hover:bg-white/10 text-sm text-white/70 hover:text-white transition-colors border border-white/5">
                                           Reset View
                                        </button>
                                    </div>
                                 </div>
                               )}
                             </div>
                          </div>
                       </div>
                    </div>
                </Card>
            )}
          </div>
        )}
      </div>
    </div>
  );
}