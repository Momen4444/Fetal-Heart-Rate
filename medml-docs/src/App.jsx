import React, { useState } from 'react';
import { 
  Activity, 
  Database, 
  GitBranch, 
  Brain, 
  ChevronRight, 
  Code, 
  BarChart2, 
  Layers,
  PieChart,
  TrendingUp,
  Zap,
  Info,
  HeartPulse,
  BookOpen,
  X,
  Thermometer,
  Clock,
  CheckCircle,
  AlertCircle,
  Server // Imported Server icon for Deployment step
} from 'lucide-react';

// --- VISUALIZATION COMPONENTS ---

const SimplePieChart = ({ data, size = 160, hole = 80, title, description }) => {
  let accumulatedAngle = 0;
  const radius = size / 2;
  const center = size / 2;

  return (
    <div className="flex flex-col items-center p-4 bg-white rounded-xl border border-gray-100 shadow-sm h-full">
      <h4 className="text-sm font-bold text-gray-600 mb-4 uppercase tracking-wide">{title}</h4>
      <div className="relative">
        <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} className="transform -rotate-90">
          {data.map((slice, index) => {
            const angle = (slice.value / 100) * 360;
            const x1 = center + radius * Math.cos((Math.PI * accumulatedAngle) / 180);
            const y1 = center + radius * Math.sin((Math.PI * accumulatedAngle) / 180);
            const x2 = center + radius * Math.cos((Math.PI * (accumulatedAngle + angle)) / 180);
            const y2 = center + radius * Math.sin((Math.PI * (accumulatedAngle + angle)) / 180);
            
            const largeArcFlag = angle > 180 ? 1 : 0;
            const pathData = `M ${center} ${center} L ${x1} ${y1} A ${radius} ${radius} 0 ${largeArcFlag} 1 ${x2} ${y2} Z`;
            
            accumulatedAngle += angle;

            return (
              <path
                key={index}
                d={pathData}
                fill={slice.color}
                stroke="white"
                strokeWidth="2"
                className="transition-all duration-300 hover:opacity-90"
              />
            );
          })}
          <circle cx={center} cy={center} r={hole / 2} fill="white" />
        </svg>
        {/* Center Label (Optional) */}
        <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
           <PieChart className="w-6 h-6 text-gray-300" />
        </div>
      </div>
      
      {/* Legend */}
      <div className="mt-6 w-full space-y-2 flex-grow">
        {data.map((slice, idx) => (
          <div key={idx} className="flex items-center justify-between text-xs">
            <div className="flex items-center space-x-2">
              <span className="w-3 h-3 rounded-full shadow-sm" style={{ backgroundColor: slice.color }}></span>
              <span className="text-gray-600 font-medium">{slice.label}</span>
            </div>
            <span className="font-bold text-gray-800">{slice.value}%</span>
          </div>
        ))}
      </div>

      {/* Description */}
      {description && (
        <div className="mt-6 pt-4 border-t border-gray-100 w-full">
          <p className="text-xs text-gray-500 leading-relaxed text-center">{description}</p>
        </div>
      )}
    </div>
  );
};

const RangeBar = ({ min, q1, median, q3, max, label, unit, color }) => {
  const range = max - min;
  const scale = (val) => ((val - min) / range) * 100;

  return (
    <div className="mb-8 select-none px-2 pt-4">
      <div className="flex justify-between text-xs text-gray-500 mb-1">
        <span className="font-bold text-gray-700">{label}</span>
        <span>Median: {median} {unit}</span>
      </div>
      <div className="relative h-12 w-full bg-gray-50 rounded-lg border border-gray-200 mt-2">
        {/* Full Range Line */}
        <div 
          className="absolute top-1/2 left-0 h-0.5 bg-gray-300 transform -translate-y-1/2 rounded"
          style={{ left: `${scale(min)}%`, width: `${scale(max) - scale(min)}%` }}
        ></div>
        
        {/* IQR Box */}
        <div 
          className={`absolute top-1/4 h-1/2 bg-${color}-200 border border-${color}-400 rounded-sm opacity-60`}
          style={{ left: `${scale(q1)}%`, width: `${scale(q3) - scale(q1)}%` }}
        ></div>
        
        {/* Median Line */}
        <div 
          className={`absolute top-1 h-10 w-1 bg-${color}-600 rounded-full z-10 shadow-sm`}
          style={{ left: `${scale(median)}%` }}
        ></div>
        
        {/* Labels */}
        <div className="absolute -bottom-6 text-[10px] text-gray-400 font-mono transform -translate-x-1/2" style={{ left: `${scale(min)}%` }}>{min}</div>
        <div className="absolute -bottom-6 text-[10px] text-gray-400 font-mono transform -translate-x-1/2" style={{ left: `${scale(q1)}%` }}>{q1}</div>
        <div className="absolute -bottom-6 text-[10px] text-gray-400 font-mono transform -translate-x-1/2" style={{ left: `${scale(q3)}%` }}>{q3}</div>
        <div className="absolute -bottom-6 text-[10px] text-gray-400 font-mono transform -translate-x-1/2" style={{ left: `${scale(max)}%` }}>{max}</div>
      </div>
    </div>
  );
};

const StatCard = ({ title, value, subtext, icon: Icon, colorClass }) => (
  <div className="bg-white p-5 rounded-xl border border-gray-100 shadow-sm transition-all duration-300 hover:shadow-md">
    <div className="flex justify-between items-start">
      <div>
        <p className="text-xs font-bold text-gray-400 uppercase tracking-wider">{title}</p>
        <h3 className="text-2xl font-extrabold text-gray-900 mt-1">{value}</h3>
        {subtext && <p className="text-sm text-gray-500 mt-1">{subtext}</p>}
      </div>
      <div className={`p-2 rounded-lg ${colorClass} bg-opacity-10`}>
        <Icon className={`w-5 h-5 ${colorClass.replace('bg-', 'text-')}`} />
      </div>
    </div>
  </div>
);

// --- DATA CONTENT ---

const INSIGHTS_DATA = {
  ctu: {
    title: "Fetal Health (CTU-CHB)",
    color: "pink",
    metrics: [
      { 
        title: "Baseline Heart Rate", 
        value: "140 bpm", 
        subtext: "Median FHR", 
        icon: HeartPulse, 
        color: "bg-pink-500",
      },
      { 
        title: "HR Variability", 
        value: "18.5 bpm", 
        subtext: "Standard Deviation", 
        icon: Activity, 
        color: "bg-purple-500",
      },
      { 
        title: "Active Labor", 
        value: "54.2%", 
        subtext: "Contractions > 20u", 
        icon: Zap, 
        color: "bg-orange-500",
      },
      { 
        title: "Peak Contraction", 
        value: "127 units", 
        subtext: "Max Uterine Pressure", 
        icon: TrendingUp, 
        color: "bg-red-500",
      }
    ],
    extraMetrics: [
      { 
        title: "Bradycardia Events", 
        value: "11.4%", 
        subtext: "HR < 110 bpm", 
        icon: AlertCircle, 
        color: "bg-amber-500",
      },
      { 
        title: "Signal Noise", 
        value: "21.97", 
        subtext: "Contraction Std Dev", 
        icon: Activity, 
        color: "bg-gray-500",
      }
    ],
    charts: {
      hrZones: {
        data: [
          { label: "Normal (110-160)", value: 84.4, color: "#10B981" },
          { label: "Bradycardia (<110)", value: 9.1, color: "#F59E0B" },
          { label: "Tachycardia (>160)", value: 6.5, color: "#EF4444" },
        ],
        description: "Distribution of Fetal Heart Rate (FHR) across clinical zones. The majority (84.4%) falls within the normal range, with a small percentage indicating potential distress (Bradycardia/Tachycardia)."
      },
      labor: {
        data: [
          { label: "Active Contraction", value: 55.1, color: "#EC4899" },
          { label: "Resting Tone", value: 44.9, color: "#9CA3AF" },
        ],
        description: "Comparison of uterine activity. 'Active Contraction' indicates periods where uterine pressure exceeds 20 units, reflecting the intense phase of labor versus resting tone."
      },
      boxPlot: { 
        data: { min: 51.75, q1: 132, median: 140, q3: 148.5, max: 193 },
        description: "Statistical spread of Fetal Heart Rate (bpm). The median is centered at 140 bpm, with a tight Interquartile Range (IQR) indicating stable heart variability."
      }
    }
  },
  bidmc: {
    title: "Heart Failure (BIDMC)",
    color: "blue",
    metrics: [
      { 
        title: "Median Voltage", 
        value: "-0.360 mV", 
        subtext: "Electrical Baseline", 
        icon: Activity, 
        color: "bg-blue-500",
      },
      { 
        title: "Voltage Spread", 
        value: "1.66 mV", 
        subtext: "5th to 95th Percentile", 
        icon: BarChart2, 
        color: "bg-indigo-500",
      },
      { 
        title: "Recording Duration", 
        value: "~20 hrs", 
        subtext: "Per Subject", 
        icon: Clock, 
        color: "bg-cyan-500",
      },
      { 
        title: "Sampling Rate", 
        value: "250 Hz", 
        subtext: "High Resolution", 
        icon: Database, 
        color: "bg-teal-500",
      }
    ],
    extraMetrics: [
      { 
        title: "Negative Polarity", 
        value: "82.9%", 
        subtext: "Dominant Wave Direction", 
        icon: TrendingUp, 
        color: "bg-blue-800",
      },
      { 
        title: "Total Samples", 
        value: "268M", 
        subtext: "Data Points Analyzed", 
        icon: Layers, 
        color: "bg-gray-600",
      }
    ],
    charts: {
      polarity: {
        data: [
          { label: "Negative Deflection", value: 85.9, color: "#1E3A8A" }, 
          { label: "Positive Deflection", value: 14.1, color: "#60A5FA" }, 
        ],
        description: "Dominance of signal direction in Lead I. The heavy negative skew (85.9%) is characteristic of the specific electrode placement used in this study."
      },
      beats: {
        data: [
          { label: "Normal Beats", value: 85, color: "#3B82F6" },
          { label: "Premature (PVC)", value: 10, color: "#F59E0B" },
          { label: "Artifacts", value: 5, color: "#9CA3AF" },
        ],
        description: "Breakdown of detected heartbeats. While most are normal, the presence of Premature Ventricular Contractions (PVCs) is common in heart failure patients."
      },
      boxPlot: {
        data: { min: -1.025, q1: -0.525, median: -0.360, q3: -0.180, max: 0.635 },
        description: "Voltage amplitude distribution (mV). The narrow range and negative median reflect the attenuated electrical signals often seen in failing hearts."
      }
    }
  }
};

const PIPELINE_STEPS = {
  ctu: [
    {
      id: 'data-load',
      title: 'Data Loading & Cleaning',
      icon: <Database />,
      description: 'Loads CSV data and removes non-predictive identifiers.',
      details: "The pipeline starts by loading 'final.csv'. Crucially, it drops 'Unnamed: 0' and 'ID' columns. These are unique identifiers that could cause the model to memorize specific patients rather than learning general patterns.",
      code: `df = pd.read_csv(DATA_FILE)\n\n# Drop identifiers\ndf.drop(['Unnamed: 0', 'ID'], axis=1, inplace=True)`
    },
    {
      id: 'feature-eng',
      title: 'Feature Engineering',
      icon: <GitBranch />,
      description: 'Transforms continuous Apgar scores into clinical categories.',
      details: "Raw Apgar scores (0-10) are binned into categories: 'Excellent', 'Moderately Abnormal', and 'Attention'. This helps the model find non-linear patterns and matches how doctors assess risk.",
      code: `def categorize_apgar(x):\n    if 6.5 < x <= 10: return 'Excellent'\n    elif 4 < x <= 6.5: return 'ModeratelyAbnormal'\n    elif x <= 4: return 'Attention'\n\ndf['Apgar1'] = df['Apgar1'].apply(categorize_apgar)`,
    },
    {
      id: 'modeling',
      title: 'Model Training',
      icon: <Brain />,
      description: 'Trains XGBoost, Random Forest, and SVM models.',
      details: "Three distinct classifiers are trained. XGBoost is tuned with specific hyperparameters (learning_rate=0.05, max_depth=5).",
      code: `xgb_model = xgb.XGBClassifier(\n    learning_rate=0.05,\n    max_depth=5,\n    n_estimators=150,\n    eval_metric='mlogloss'\n)`
    },
    {
      id: 'deployment',
      title: 'Model Deployment',
      icon: <Server />,
      description: 'Serving the predictive model via FastAPI.',
      details: "The trained XGBoost model is serialized and exposed via a REST API. The API uses Pydantic for data validation, ensuring that the clinical features (Age, pH, Apgar) are correctly typed before inference. The backend also handles feature scaling using the saved StandardScaler artifact.",
      code: `class FetalCTUInput(BaseModel):\n    features: Dict[str, Any]\n\n@app.post("/predict/fetal")\ndef predict_fetal(data: FetalCTUInput):\n    # Load & Scale Features\n    X_scaled = MODELS["fetal"]["scaler"].transform(df)\n    \n    # Inference\n    prob = float(MODELS["fetal"]["model"].predict_proba(X_scaled)[0][1])\n\n    return {\n        "prediction": "Pathological" if prob > 0.5 else "Normal",\n        "risk_score": prob\n    }`
    }
  ],
  bidmc: [
    {
      id: 'signal',
      title: 'Raw Signal Ingestion',
      icon: <Activity />,
      description: 'Reads .dat and .hea files using WFDB.',
      details: "Ingests raw voltage signals from the PhysioNet BIDMC database. It uses the `wfdb` library to parse the binary signal files.",
      code: `import wfdb\n\nrecord = wfdb.rdrecord(record_path)\nsig = record.p_signal[:, 0] # Lead I`
    },
    {
      id: 'r-peak',
      title: 'R-Peak Detection',
      icon: <BarChart2 />,
      description: 'Detects heartbeats using the XQRS algorithm.',
      details: "The model cannot understand a raw wave. We must convert the continuous signal into discrete events (heartbeats). The XQRS algorithm scans for the sharp 'R' spike in the QRS complex.",
      code: `from wfdb import processing\n\n# Detect R-peaks in the signal\nqrs_inds = processing.xqrs_detect(\n    sig=sig_segment, \n    fs=fs\n)`
    },
    {
      id: 'ensemble',
      title: 'Ensemble Modeling',
      icon: <Brain />,
      description: 'Combines RF, XGBoost, and SVM via Voting.',
      details: "Instead of trusting a single model, this pipeline uses a VotingClassifier with 'soft' voting. It averages the probabilities from three independently tuned models.",
      code: `voting_clf = VotingClassifier(\n    estimators=[\n        ('rf', best_rf),\n        ('xgb', best_xgb),\n        ('svm', best_svm)\n    ],\n    voting='soft'\n)`
    },
    {
      id: 'deployment',
      title: 'Model Deployment',
      icon: <Server />,
      description: 'Real-time HRV inference API.',
      details: "The ensemble model (VotingClassifier) is deployed using FastAPI. The endpoint accepts raw Heart Rate Variability (HRV) metrics computed from the signal processing stage. It returns a binary classification (Healthy vs. Heart Failure) along with a 'Critical' or 'Normal' status flag.",
      code: `class AdultHRVInput(BaseModel):\n    sdnn: float\n    rmssd: float\n    pnn50: float\n    mean_hr: float\n\n@app.post("/predict/adult")\ndef predict_adult(data: AdultHRVInput):\n    # Features vector\n    features = [[data.sdnn, data.rmssd, data.pnn50, data.mean_hr]]\n    \n    # Inference\n    pred = int(MODELS["adult"]["model"].predict(features)[0])\n    \n    return {\n        "prediction": "Heart Failure" if pred == 1 else "Healthy",\n        "status": "Critical" if pred == 1 else "Normal"\n    }`
    }
  ]
};

// --- MAIN COMPONENTS ---

const InsightsView = ({ dataset }) => {
  const data = INSIGHTS_DATA[dataset];
  const isCtu = dataset === 'ctu';

  return (
    <div className="space-y-6 animate-fadeIn">
      
      {/* Introduction Banner */}
      <div className={`p-4 rounded-xl border ${isCtu ? 'bg-pink-50 border-pink-100 text-pink-800' : 'bg-blue-50 border-blue-100 text-blue-800'}`}>
        <div className="flex items-start space-x-3">
          <BookOpen className="w-5 h-5 mt-0.5 flex-shrink-0" />
          <div>
            <h4 className="font-bold text-sm uppercase mb-1">Clinical Context</h4>
            <p className="text-sm leading-relaxed opacity-90">
              {isCtu 
                ? "This dataset captures the most critical period of childbirth: labor. By analyzing Fetal Heart Rate (FHR) and Uterine Contractions (UC), we aim to detect 'Acidosis' (oxygen deprivation) early. The goal is to predict when a C-Section is medically necessary to prevent brain injury." 
                : "This dataset focuses on patients with advanced Heart Failure (NYHA Class 3-4). The key to analysis here is 'Heart Rate Variability' (HRV). A healthy heart constantly speeds up and slows down; a failing heart often becomes ominously regular (metronomic) or chaotic."
              }
            </p>
          </div>
        </div>
      </div>

      {/* Main Grid Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        
        {/* Left Column: Key Vitals (Stacked) */}
        <div className="lg:col-span-1 space-y-4">
           <h3 className="text-sm font-bold text-gray-400 uppercase tracking-wider pl-1">Key Vitals</h3>
           {data.metrics.map((m, idx) => (
            <StatCard 
              key={idx}
              {...m}
              colorClass={data.color === 'pink' ? m.color.replace('blue', 'pink') : m.color}
            />
          ))}
           {/* Extra Metrics in same column for vertical flow */}
           {data.extraMetrics.map((em, idx) => (
              <StatCard 
                key={`extra-${idx}`}
                {...em}
                colorClass="bg-gray-100 text-gray-600"
              />
            ))}
        </div>

        {/* Center/Right Column: Charts */}
        <div className="lg:col-span-3 space-y-6">
          
          {/* Pie Charts Row */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <SimplePieChart 
              title={isCtu ? "Heart Rate Zones" : "Signal Polarity"}
              data={isCtu ? data.charts.hrZones.data : data.charts.polarity.data} 
              description={isCtu ? data.charts.hrZones.description : data.charts.polarity.description}
              size={200} 
              hole={100} 
            />
            <SimplePieChart 
              title={isCtu ? "Labor State" : "Beat Classification"}
              data={isCtu ? data.charts.labor.data : data.charts.beats.data} 
              description={isCtu ? data.charts.labor.description : data.charts.beats.description}
              size={200} 
              hole={100} 
            />
          </div>

          {/* Analysis Box with Embedded Range Bar */}
          <div className="bg-white p-6 rounded-xl border border-gray-200 shadow-sm">
             <div className="flex items-center space-x-2 mb-6">
                <BarChart2 className="w-5 h-5 text-gray-400" />
                <h3 className="text-lg font-bold text-gray-900">Statistical Distribution</h3>
             </div>
             
             <RangeBar 
                min={data.charts.boxPlot.data.min}
                q1={data.charts.boxPlot.data.q1}
                median={data.charts.boxPlot.data.median}
                q3={data.charts.boxPlot.data.q3}
                max={data.charts.boxPlot.data.max}
                label={isCtu ? "Heart Rate Spread (bpm)" : "Voltage Spread (mV)"}
                unit={isCtu ? "bpm" : "mV"}
                color={isCtu ? "pink" : "blue"}
             />

             {/* Description for Range Bar */}
             <div className="mt-4 pt-4 border-t border-gray-100">
               <p className="text-xs text-gray-500 leading-relaxed text-center">{data.charts.boxPlot.description}</p>
             </div>
             
             <div className="mt-6 p-4 bg-slate-50 rounded-lg border border-slate-100">
               <h4 className="text-sm font-bold text-slate-700 mb-2">AI Diagnostic Interpretation</h4>
               <p className="text-slate-600 text-sm leading-relaxed">
                 {isCtu 
                   ? "Based on the aggregate data, the dataset represents a mostly healthy population with a 9% incidence of Bradycardia. The baseline FHR of 140 bpm is ideal. The 54% Active Labor statistic confirms that these are intrapartum (during birth) recordings, which are the most noisy and difficult to analyze, validating the need for robust ML models like XGBoost."
                   : "The voltage data shows a heavily skewed negative distribution (median -0.36 mV), which is consistent with the placement of leads in the specific Holter monitors used for the BIDMC study. The extremely high sample count (268M) provides high confidence. The 'Signal Noise' is low, meaning the data is high quality despite being long-term ambulatory recordings."
                 }
               </p>
             </div>
          </div>
        </div>

      </div>
    </div>
  );
};

const PipelineView = ({ steps, color }) => {
  const [activeStep, setActiveStep] = useState(steps[0].id);
  
  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
      {/* Navigation */}
      <div className="lg:col-span-4 space-y-4">
         <div className="space-y-2">
           {steps.map((step, idx) => (
             <button
               key={step.id}
               onClick={() => setActiveStep(step.id)}
               className={`w-full flex items-center space-x-3 p-3 rounded-lg transition-colors text-left ${
                 activeStep === step.id 
                   ? 'bg-white shadow-md ring-1 ring-gray-200' 
                   : 'hover:bg-gray-100'
               }`}
             >
               <div className={`w-8 h-8 rounded-full flex items-center justify-center text-xs font-bold border-2 ${
                 activeStep === step.id 
                   ? `border-${color}-500 text-${color}-600 bg-white` 
                   : 'border-gray-300 text-gray-400 bg-gray-50'
               }`}>
                 {idx + 1}
               </div>
               <span className={`font-medium ${activeStep === step.id ? 'text-gray-900' : 'text-gray-500'}`}>
                 {step.title}
               </span>
             </button>
           ))}
         </div>
      </div>

      {/* Content */}
      <div className="lg:col-span-8">
        {steps.map((step) => (
          step.id === activeStep && (
            <div key={step.id} className="bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden animate-slideUp">
               <div className="p-6">
                 <div className="flex items-center space-x-3 mb-4">
                   <div className={`p-2 rounded-lg bg-${color}-100 text-${color}-600`}>
                     {step.icon}
                   </div>
                   <h2 className="text-xl font-bold text-gray-900">{step.title}</h2>
                 </div>
                 <p className="text-gray-600 mb-6">{step.description}</p>
                 
                 <div className="bg-gray-50 rounded-lg p-4 border border-gray-100">
                    <h4 className="text-xs font-bold text-gray-400 uppercase tracking-wider mb-2">Technical Detail</h4>
                    <p className="text-sm text-gray-700 leading-relaxed mb-4">{step.details}</p>
                    <div className="bg-slate-900 rounded-md p-4 overflow-x-auto">
                      <pre className="text-xs text-blue-300 font-mono"><code>{step.code}</code></pre>
                    </div>
                 </div>
               </div>
            </div>
          )
        ))}
      </div>
    </div>
  );
};

export default function App() {
  const [dataset, setDataset] = useState('ctu');
  const [view, setView] = useState('insights'); // 'pipeline' or 'insights'

  return (
    <div className="min-h-screen bg-gray-50 font-sans text-gray-900 pb-12">
      
      {/* Navbar */}
      <nav className="bg-white border-b border-gray-200 sticky top-0 z-40 shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-16">
            <div className="flex items-center space-x-3">
              <div className="bg-indigo-600 p-2 rounded-lg">
                <Code className="text-white w-5 h-5" />
              </div>
              <span className="font-bold text-xl tracking-tight text-gray-900">MedML<span className="text-indigo-600">Docs</span></span>
            </div>
            
            <div className="flex items-center space-x-6">
               {/* Dataset Toggle */}
               <div className="bg-gray-100 p-1 rounded-lg flex text-sm font-medium">
                  <button 
                    onClick={() => setDataset('ctu')}
                    className={`px-3 py-1.5 rounded-md transition-all ${dataset === 'ctu' ? 'bg-white text-pink-600 shadow-sm' : 'text-gray-500 hover:text-gray-700'}`}
                  >
                    Fetal
                  </button>
                  <button 
                    onClick={() => setDataset('bidmc')}
                    className={`px-3 py-1.5 rounded-md transition-all ${dataset === 'bidmc' ? 'bg-white text-blue-600 shadow-sm' : 'text-gray-500 hover:text-gray-700'}`}
                  >
                    Cardiac
                  </button>
               </div>
            </div>
          </div>
        </div>
      </nav>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        
        {/* Header */}
        <div className="mb-8 flex flex-col md:flex-row md:items-center justify-between">
           <div>
             <h1 className={`text-3xl font-extrabold ${dataset === 'ctu' ? 'text-pink-600' : 'text-blue-600'}`}>
               {dataset === 'ctu' ? 'CTU-CHB Intrapartum CTG' : 'BIDMC Heart Failure Database'}
             </h1>
             <p className="text-gray-500 mt-2 max-w-2xl">
               {dataset === 'ctu' 
                 ? "Predicting fetal hypoxia during labor using Machine Learning." 
                 : "Analyzing Heart Rate Variability (HRV) in severe heart failure patients."
               }
             </p>
           </div>
        </div>

        {/* View Tabs */}
        <div className="border-b border-gray-200 mb-8">
          <nav className="-mb-px flex space-x-8">
            <button
              onClick={() => setView('pipeline')}
              className={`whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm flex items-center ${
                view === 'pipeline'
                  ? `border-${dataset === 'ctu' ? 'pink' : 'blue'}-500 text-${dataset === 'ctu' ? 'pink' : 'blue'}-600`
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              <GitBranch className="w-4 h-4 mr-2" />
              Pipeline Documentation
            </button>
            <button
              onClick={() => setView('insights')}
              className={`whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm flex items-center ${
                view === 'insights'
                  ? `border-${dataset === 'ctu' ? 'pink' : 'blue'}-500 text-${dataset === 'ctu' ? 'pink' : 'blue'}-600`
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              <BarChart2 className="w-4 h-4 mr-2" />
              Data Insights
            </button>
          </nav>
        </div>

        {/* Dynamic Content */}
        {view === 'pipeline' ? (
          <PipelineView steps={PIPELINE_STEPS[dataset]} color={dataset === 'ctu' ? 'pink' : 'blue'} />
        ) : (
          <InsightsView dataset={dataset} />
        )}

      </main>

      {/* Global CSS for Animations */}
      <style>{`
        @keyframes slideUp {
          from { opacity: 0; transform: translateY(10px); }
          to { opacity: 1; transform: translateY(0); }
        }
        .animate-slideUp {
          animation: slideUp 0.4s ease-out forwards;
        }
        @keyframes fadeIn {
          from { opacity: 0; }
          to { opacity: 1; }
        }
        .animate-fadeIn {
          animation: fadeIn 0.3s ease-out forwards;
        }
      `}</style>
    </div>
  );
}