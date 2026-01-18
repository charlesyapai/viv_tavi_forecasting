
  const TitleSlide = () => (
    <div className="flex flex-col items-center justify-center h-full bg-gradient-to-br from-blue-900 to-slate-900 text-white p-12 text-center animate-slide-enter">
      <div className="mb-6 p-4 bg-blue-500/20 rounded-full border border-blue-400/30">
        <Icons.Activity />
      </div>
      <h1 className="text-5xl font-bold mb-6 tracking-tight leading-tight">
        Forecasting ViV TAVI in Korea <br/> (2025-2035)
      </h1>
      <p className="text-xl text-blue-200 mb-8 max-w-2xl">
        A Demography-Anchored Monte Carlo Simulation to estimate clinical demand, correcting for post-COVID anomalies.
      </p>
      <div className="flex gap-4 text-sm text-slate-400 font-mono border-t border-slate-700 pt-6 mt-4">
        <span>HYUNJIN AHN</span>
        <span>•</span>
        <span>CHARLES YAP</span>
      </div>
      <div className="mt-8 px-4 py-2 bg-green-500/20 text-green-400 rounded border border-green-500/30 text-sm">
        Status: Results Validated (Nov 2024)
      </div>
    </div>
  );

  const ProblemSlide = () => (
    <div className="p-10 h-full flex flex-col animate-slide-enter">
      <h2 className="text-3xl font-bold text-slate-800 mb-6 flex items-center gap-3">
        <span className="text-red-500"><Icons.AlertTriangle/></span>
        Why the Original Approach Failed
      </h2>
      <div className="grid grid-cols-2 gap-8 flex-1">
        <div className="bg-red-50 p-6 rounded-xl border border-red-100 flex flex-col justify-center">
          <h3 className="text-xl font-bold text-red-800 mb-4">1. The 2023 "Backlog Spike"</h3>
          <p className="text-slate-700 mb-4">
            Korean registry data showed a massive, anomalous spike in 2023 procedures (Post-COVID).
          </p>
          <div className="h-32 w-full flex items-end justify-between gap-2 px-4 border-b border-slate-300 pb-2">
            <div className="w-1/5 bg-slate-300 h-1/4 rounded-t"></div>
            <div className="w-1/5 bg-slate-300 h-1/3 rounded-t"></div>
            <div className="w-1/5 bg-slate-300 h-1/2 rounded-t"></div>
            <div className="w-1/5 bg-red-500 h-full rounded-t relative group">
              <span className="absolute -top-6 left-1/2 -translate-x-1/2 text-xs font-bold text-red-600">Spike</span>
            </div>
            <div className="w-1/5 bg-slate-300 h-2/3 rounded-t opacity-50 border-t-2 border-dashed border-slate-500"></div>
          </div>
          <p className="text-xs text-red-600 mt-2 italic text-center">Linear extrapolation from here creates infinite growth.</p>
        </div>
        <div className="bg-slate-50 p-6 rounded-xl border border-slate-200 flex flex-col justify-center">
          <h3 className="text-xl font-bold text-slate-800 mb-4">2. The JACC Critique</h3>
          <blockquote className="italic border-l-4 border-blue-500 pl-4 text-slate-600 mb-4">
            "Extrapolations... potentially introducing bias... significant adjustments were made... clinical factors were not accounted for."
          </blockquote>
          <div className="mt-4 bg-white p-4 rounded shadow-sm text-sm text-slate-700">
            <strong>Conclusion:</strong> We cannot blindly apply US/Japan growth rates (Ohno et al.) to Korea's unique aging demographic.
          </div>
        </div>
      </div>
    </div>
  );

  const ConceptSlide = () => (
    <div className="p-10 h-full flex flex-col animate-slide-enter">
      <h2 className="text-3xl font-bold text-slate-800 mb-8">The Pivot: A Demography-Anchored Approach</h2>
      <div className="flex flex-col gap-6">
        <div className="flex items-center gap-6 p-6 bg-slate-100 rounded-xl opacity-60 grayscale">
          <div className="w-24 text-right font-bold text-slate-500">OLD MODEL</div>
          <div className="flex-1 border-l-2 border-slate-300 pl-6">
            <h4 className="font-bold text-lg">Linear Extrapolation</h4>
            <p className="text-slate-600">"Procedures will grow by X% every year based on history."</p>
          </div>
        </div>

        <div className="flex flex-col items-center justify-center py-2 text-slate-400">
          <div className="h-8 w-0.5 bg-slate-300"></div>
          <span className="text-xs uppercase tracking-widest bg-white px-2 py-1">Transformed Into</span>
          <div className="h-8 w-0.5 bg-slate-300"></div>
        </div>

        <div className="flex items-center gap-6 p-8 bg-blue-50 rounded-xl border-2 border-blue-500 shadow-lg">
          <div className="w-24 text-right font-bold text-blue-600">NEW MODEL (v9)</div>
          <div className="flex-1 border-l-2 border-blue-400 pl-6">
            <h4 className="font-bold text-2xl text-blue-900 mb-2 flex items-center gap-2">
              Risk &times; Demography
            </h4>
            <p className="text-blue-800 text-lg italic">
              "Given the stable risk profile of a Korean patient, how many failures occur as the population ages?"
            </p>
          </div>
          <div className="bg-white p-4 rounded-lg shadow text-center min-w-[120px]">
            <Icons.Users />
            <div className="text-xs font-bold text-blue-500 mt-2">Pop Structure</div>
          </div>
        </div>
      </div>
    </div>
  );

  const MethodologySlide = () => (
    <div className="p-10 h-full flex flex-col animate-slide-enter">
      <h2 className="text-3xl font-bold text-slate-800 mb-8">Simulation Architecture (model_v9)</h2>
      <div className="grid grid-cols-3 gap-4 h-64">
        {/* Step 1 */}
        <div className="bg-slate-800 text-white p-6 rounded-xl flex flex-col relative">
          <div className="absolute -top-3 -right-3 w-8 h-8 bg-blue-500 rounded-full flex items-center justify-center font-bold">1</div>
          <h3 className="font-bold text-lg mb-2">Pre-Compute</h3>
          <p className="text-slate-400 text-sm mb-4">Ingest Data Sources</p>
          <ul className="text-sm space-y-2 text-slate-300">
            <li>• HIRA Registry (2015-2024)</li>
            <li>• Ministry Pop Stats</li>
            <li>• Age/Sex Bands</li>
          </ul>
        </div>
        {/* Step 2 */}
        <div className="bg-slate-800 text-white p-6 rounded-xl flex flex-col relative">
          <div className="absolute -top-3 -right-3 w-8 h-8 bg-blue-500 rounded-full flex items-center justify-center font-bold">2</div>
          <h3 className="font-bold text-lg mb-2">Risk Calculation</h3>
          <p className="text-slate-400 text-sm mb-4">Establish Anchor</p>
          <div className="bg-slate-700 p-3 rounded text-center font-mono text-sm border border-slate-600 mb-2">
            Risk = Obs / Pop
          </div>
          <p className="text-xs text-slate-400 italic">Generates stable risk profiles per age-band based on 2023-24 data.</p>
        </div>
        {/* Step 3 */}
        <div className="bg-blue-900 text-white p-6 rounded-xl flex flex-col relative border border-blue-500 shadow-[0_0_15px_rgba(59,130,246,0.3)]">
          <div className="absolute -top-3 -right-3 w-8 h-8 bg-white text-blue-900 rounded-full flex items-center justify-center font-bold">3</div>
          <h3 className="font-bold text-lg mb-2 flex items-center gap-2"><Icons.Server size={16}/> Monte Carlo</h3>
          <p className="text-blue-200 text-sm mb-4">The Engine</p>
          <ul className="text-sm space-y-2 text-blue-100">
            <li>• 100 Runs per Scenario</li>
            <li>• Durability vs Survival</li>
            <li>• Stochastic Jitter (5-10%)</li>
          </ul>
        </div>
      </div>
      <div className="mt-8 bg-slate-100 p-4 rounded-lg border border-slate-200 flex items-center justify-between">
        <div className="text-sm text-slate-600">
          <strong>Filters Applied:</strong> Redo-SAVR subtraction applied. Speculative penetration (60-80%) removed for conservative floor.
        </div>
      </div>
    </div>
  );

  const MechanismSlide = () => (
    <div className="p-10 h-full flex flex-col animate-slide-enter">
      <h2 className="text-3xl font-bold text-slate-800 mb-6">The Core Logic: "The Race"</h2>
      <p className="text-slate-600 mb-8 max-w-3xl">
        Inside the Monte Carlo Engine, every simulated patient runs a "Race" between two dates.
        A patient only becomes a candidate if their valve fails while they are still alive.
      </p>

      <div className="bg-white p-8 rounded-xl shadow-lg border border-slate-200 space-y-8">
        
        {/* Scenario A */}
        <div>
          <div className="flex justify-between mb-2">
            <span className="font-bold text-green-600 flex items-center gap-2"><div className="w-3 h-3 bg-green-500 rounded-full"></div> Scenario A: ViV Candidate</span>
            <span className="text-xs font-mono text-slate-400">Fail_Year &le; Death_Year</span>
          </div>
          <div className="relative h-12 w-full bg-slate-100 rounded-full flex items-center px-2 overflow-hidden">
            {/* Life Line */}
            <div className="absolute left-2 top-4 h-1 bg-blue-300 w-[80%]"></div>
            {/* Durability Line */}
            <div className="absolute left-2 top-7 h-1 bg-yellow-400 w-[50%] z-10"></div>
            
            <div className="absolute left-2 w-4 h-4 bg-slate-800 rounded-full z-20" title="Index Procedure"></div>
            
            {/* Failure Point */}
            <div className="absolute left-[50%] top-2 flex flex-col items-center z-30">
               <div className="w-1 h-8 bg-yellow-500/50 border-l border-dashed border-yellow-600"></div>
               <span className="text-[10px] font-bold bg-yellow-100 text-yellow-800 px-1 rounded -mt-10">Valve Fails</span>
            </div>

            {/* Death Point */}
             <div className="absolute left-[80%] top-2 flex flex-col items-center z-30">
               <div className="w-0.5 h-8 bg-slate-400"></div>
               <span className="text-[10px] text-slate-500 -mt-10">Death</span>
            </div>

            {/* Window */}
            <div className="absolute left-[50%] w-[30%] h-full bg-green-100/50 border-x border-green-200 flex items-center justify-center text-xs font-bold text-green-700 tracking-wide">
              ELIGIBLE WINDOW
            </div>
          </div>
        </div>

        {/* Scenario B */}
        <div>
          <div className="flex justify-between mb-2">
            <span className="font-bold text-red-500 flex items-center gap-2"><div className="w-3 h-3 bg-red-500 rounded-full"></div> Scenario B: Died First</span>
             <span className="text-xs font-mono text-slate-400">Fail_Year &gt; Death_Year</span>
          </div>
          <div className="relative h-12 w-full bg-slate-100 rounded-full flex items-center px-2 overflow-hidden opacity-70">
            {/* Life Line */}
            <div className="absolute left-2 top-4 h-1 bg-blue-300 w-[40%]"></div>
            {/* Durability Line */}
            <div className="absolute left-2 top-7 h-1 bg-yellow-400 w-[70%] z-10"></div>
            
            <div className="absolute left-2 w-4 h-4 bg-slate-800 rounded-full z-20"></div>
            
             {/* Death Point */}
             <div className="absolute left-[40%] top-2 flex flex-col items-center z-30">
               <div className="w-0.5 h-8 bg-slate-600"></div>
               <span className="text-[10px] text-slate-700 font-bold -mt-10">Death</span>
            </div>

            {/* Failure Point */}
            <div className="absolute left-[70%] top-2 flex flex-col items-center z-30 opacity-50">
               <div className="w-1 h-8 bg-yellow-500/50 border-l border-dashed border-yellow-600"></div>
               <span className="text-[10px] bg-yellow-50 text-yellow-800 px-1 rounded -mt-10">Valve Fails</span>
            </div>
             
             <div className="absolute right-4 text-red-400 font-bold border-2 border-red-200 p-1 px-2 rounded -rotate-12 text-xs">
               NOT ELIGIBLE
             </div>
          </div>
        </div>

      </div>
    </div>
  );

  const LogicSlide = () => (
    <div className="p-10 h-full flex flex-col animate-slide-enter">
      <h2 className="text-3xl font-bold text-slate-800 mb-6">Sampling Distributions</h2>
      <div className="grid grid-cols-2 gap-8 h-full">
        <div className="bg-yellow-50 p-6 rounded-xl border border-yellow-100">
          <h3 className="text-xl font-bold text-yellow-800 mb-4 flex items-center gap-2"><Icons.Activity/> Durability</h3>
          <p className="text-sm text-slate-700 mb-4">Bimodal Mixture Model: Valves fail in two clusters (Early Manufacturing vs Late Wear).</p>
          <div className="flex items-end justify-center h-32 gap-1 border-b border-yellow-300/50 pb-1">
            {/* Early Failure */}
            <div className="w-4 h-4 bg-yellow-300 rounded-t"></div>
            <div className="w-4 h-8 bg-yellow-400 rounded-t"></div>
            <div className="w-4 h-3 bg-yellow-300 rounded-t mr-8"></div>
            
            {/* Late Failure */}
            <div className="w-4 h-6 bg-yellow-300 rounded-t"></div>
            <div className="w-4 h-12 bg-yellow-400 rounded-t"></div>
            <div className="w-4 h-24 bg-yellow-500 rounded-t"></div>
            <div className="w-4 h-16 bg-yellow-400 rounded-t"></div>
            <div className="w-4 h-8 bg-yellow-300 rounded-t"></div>
          </div>
          <div className="flex justify-between text-xs text-slate-500 mt-2 font-mono">
             <span>4 Years (20%)</span>
             <span>11.5 Years (80%)</span>
          </div>
        </div>

        <div className="bg-blue-50 p-6 rounded-xl border border-blue-100">
          <h3 className="text-xl font-bold text-blue-800 mb-4 flex items-center gap-2"><Icons.TrendingUp/> Survival</h3>
          <p className="text-sm text-slate-700 mb-4">Actuarial curves adjusted for risk category (Low/Int/High).</p>
          <div className="h-32 w-full relative border-l border-b border-blue-300">
            <svg className="absolute inset-0 w-full h-full" preserveAspectRatio="none">
               {/* Low Risk */}
               <path d="M0,0 Q150,10 300,100" fill="none" stroke="#3b82f6" strokeWidth="3" />
               <text x="200" y="40" fill="#3b82f6" fontSize="10">Low Risk</text>
               {/* High Risk */}
               <path d="M0,0 Q100,80 300,120" fill="none" stroke="#ef4444" strokeWidth="3" strokeDasharray="5,5" />
               <text x="100" y="90" fill="#ef4444" fontSize="10">High Risk</text>
            </svg>
          </div>
          <div className="text-xs text-slate-500 mt-2 font-mono text-center">
             15 Year Horizon
          </div>
        </div>
      </div>
    </div>
  );