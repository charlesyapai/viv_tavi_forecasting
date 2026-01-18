// --- ICONS ---
const Icons = {
  Activity: () => (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M22 12h-4l-3 9L9 3l-3 9H2" />
    </svg>
  ),
  ChevronRight: () => (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <polyline points="9 18 15 12 9 6" />
    </svg>
  ),
  ChevronLeft: () => (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <polyline points="15 18 9 12 15 6" />
    </svg>
  ),
  Alert: () => (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <circle cx="12" cy="12" r="10" />
      <line x1="12" y1="8" x2="12" y2="12" />
      <line x1="12" y1="16" x2="12.01" y2="16" />
    </svg>
  ),
  Target: () => (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <circle cx="12" cy="12" r="10" />
      <circle cx="12" cy="12" r="6" />
      <circle cx="12" cy="12" r="2" />
    </svg>
  ),
  Code: () => (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <polyline points="16 18 22 12 16 6" />
      <polyline points="8 6 2 12 8 18" />
    </svg>
  ),
  Users: () => (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2" />
      <circle cx="9" cy="7" r="4" />
      <path d="M23 21v-2a4 4 0 0 0-3-3.87" />
      <path d="M16 3.13a4 4 0 0 1 0 7.75" />
    </svg>
  ),
  TrendingUp: () => (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <polyline points="23 6 13.5 15.5 8.5 10.5 1 18" />
      <polyline points="17 6 23 6 23 12" />
    </svg>
  ),
  AlertTriangle: () => (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
      <line x1="12" y1="9" x2="12" y2="13" />
      <line x1="12" y1="17" x2="12.01" y2="17" />
    </svg>
  ),
  Server: () => (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <rect x="2" y="2" width="20" height="8" rx="2" ry="2" />
      <rect x="2" y="14" width="20" height="8" rx="2" ry="2" />
      <line x1="6" y1="6" x2="6.01" y2="6" />
      <line x1="6" y1="18" x2="6.01" y2="18" />
    </svg>
  ),
  Play: () => (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <polygon points="5 3 19 12 5 21 5 3" />
    </svg>
  ),
  GitBranch: () => (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <line x1="6" y1="3" x2="6" y2="15" />
      <circle cx="18" cy="6" r="3" />
      <circle cx="6" cy="18" r="3" />
      <path d="M18 9a9 9 0 0 1-9 9" />
    </svg>
  ),
  Layers: () => (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <polygon points="12 2 2 7 12 12 22 7 12 2" />
      <polyline points="2 17 12 22 22 17" />
      <polyline points="2 12 12 17 22 12" />
    </svg>
  ),
  Shuffle: () => (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <polyline points="16 3 21 3 21 8" />
      <line x1="4" y1="20" x2="21" y2="3" />
      <polyline points="21 16 21 21 16 21" />
      <line x1="15" y1="15" x2="21" y2="21" />
      <line x1="4" y1="4" x2="9" y2="9" />
    </svg>
  ),
  BarChart3: () => (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <line x1="18" y1="20" x2="18" y2="10" />
      <line x1="12" y1="20" x2="12" y2="4" />
      <line x1="6" y1="20" x2="6" y2="14" />
    </svg>
  ),
  Globe2: () => (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <circle cx="12" cy="12" r="10" />
      <line x1="2" y1="12" x2="22" y2="12" />
      <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z" />
    </svg>
  ),
  FileText: () => (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
      <polyline points="14 2 14 8 20 8" />
      <line x1="16" y1="13" x2="8" y2="13" />
      <line x1="16" y1="17" x2="8" y2="17" />
      <polyline points="10 9 9 9 8 9" />
    </svg>
  ),
};



window.V6_DIAGRAMS = {
  // Prior work / replication
  genereuxUSForecast: "images/other_papers/generuex_viv_prediction.jpg",
  ohnoJapanUSComparison: "images/other_papers/genereux_ohno_viv_prediction.png",
  koreaFirstPassVivForecast: "images/baseline_adoption/projected_adoption.png",

  // Korea input patterns
  koreaIndexVolumes2015_2024: "images/baseline_adoption/historical_adoption.png",
  koreaPopulationByAgeSex: "images/population_projections_korea/2022-2050/demography_summary_trends.png",
  koreaPerCapitaRiskByAgeSex: "images/savr_bar_avg_risks.png",
  koreaIndexProjectionFigure4: "images/baseline_adoption/projected_adoption.png",

  // Core Korean ViV result figures
  koreaVivCandidatesVsRealisedImageD:
    "images/realised_viv/image_D_viv_candidates_vs_realized_2022_2070.png",
  koreaRealisedVivUSPenetrationImageC:
    "images/realised_viv/image_C_viv_pretty_2022_2070.png",
  waterfallExamplesFigure6: "images/waterfall_pathways/waterfall_TAVR-in-SAVR_2035.png",

  // Singapore / UN projections (future figures)
  singaporePopulationByAgeSex:
    "images/population_projections_singapore/2022-2050/age_projection_lines_Men.png",

  // Conceptual / schematic diagrams
  pipelineOverviewSchematic: "images/waterfall_pathways/waterfall_TAVR-in-SAVR_2025.png",
  sigmoidAdoptionCurveSchematic:
    "images/baseline_adoption/projected_adoption.png",

  // New 2nd Dec Images
  baselineAdoptionHistorical: "images/baseline_adoption/historical_adoption.png",
  baselineAdoptionProjected: "images/baseline_adoption/projected_adoption.png",
  popProjMen: "images/population_projections_korea/2022-2050/age_projection_lines_Men.png",
  popProjWomen: "images/population_projections_korea/2022-2050/age_projection_lines_Women.png",
  popProjTrends: "images/population_projections_korea/2022-2050/demography_summary_trends.png",
  realisedVivPretty: "images/realised_viv/image_C_viv_pretty_2022_2070.png",
  realisedVivCandidates: "images/realised_viv/image_D_viv_candidates_vs_realized_2022_2070.png",
  waterfall2025: "images/waterfall_pathways/waterfall_TAVR-in-SAVR_2025.png",
  waterfall2030: "images/waterfall_pathways/waterfall_TAVR-in-SAVR_2030.png",
  waterfall2035: "images/waterfall_pathways/waterfall_TAVR-in-SAVR_2035.png",
  koreaPopProjection: "images/population_projections_korea/2022-2050/age_projection_lines_Men.png",
  singaporePopProjection: "images/population_projections_singapore/2022-2050/age_projection_lines_Men.png",
  
  // Slide 4 images
  genereuxOhnoVivPrediction: "images/other_papers/genereux_ohno_viv_prediction.png",
  genereuxVivPrediction: "images/other_papers/generuex_viv_prediction.jpg",
  criticismText: "images/other_papers/criticism_text.png",
  criticismFactors: "images/other_papers/criticism_table_factors.png",
  unPopulationData: "images/other_papers/un_population_data_screenshot.png",
};



const V6 = {
  // 0. CONTENTS & SECTIONS --------------------------------------------------
  ContentsSlide: () => (
    <div className="flex flex-col h-full bg-white text-slate-900 p-12 animate-slide-enter">
      <h2 className="text-3xl font-bold text-slate-900 mb-12">Presentation Overview</h2>
      <div className="grid gap-4">
        <div className="flex items-center gap-4 p-4 rounded-xl bg-blue-50 border border-blue-100">
          <div className="w-10 h-10 rounded-full bg-blue-100 text-blue-600 flex items-center justify-center font-bold text-lg">1</div>
          <div className="text-lg font-semibold text-slate-800">Introduction & Plan</div>
        </div>
        <div className="flex items-center gap-4 p-4 rounded-xl bg-slate-50 border border-slate-200">
          <div className="w-10 h-10 rounded-full bg-slate-200 text-slate-600 flex items-center justify-center font-bold text-lg">2</div>
          <div className="text-lg font-semibold text-slate-800">Initial Findings</div>
        </div>
        <div className="flex items-center gap-4 p-4 rounded-xl bg-amber-50 border border-amber-100">
          <div className="w-10 h-10 rounded-full bg-amber-100 text-amber-600 flex items-center justify-center font-bold text-lg">3</div>
          <div className="text-lg font-semibold text-slate-800">The Pivot</div>
        </div>
        <div className="flex items-center gap-4 p-4 rounded-xl bg-indigo-50 border border-indigo-100">
          <div className="w-10 h-10 rounded-full bg-indigo-100 text-indigo-600 flex items-center justify-center font-bold text-lg">4</div>
          <div className="text-lg font-semibold text-slate-800">New Approach</div>
        </div>
        <div className="flex items-center gap-4 p-4 rounded-xl bg-emerald-50 border border-emerald-100">
          <div className="w-10 h-10 rounded-full bg-emerald-100 text-emerald-600 flex items-center justify-center font-bold text-lg">5</div>
          <div className="text-lg font-semibold text-slate-800">Results</div>
        </div>
      </div>
    </div>
  ),

  Section_Intro: () => (
    <div className="flex flex-col items-center justify-center h-full bg-blue-600 text-white p-12 text-center animate-slide-enter">
      <div className="w-24 h-24 rounded-full bg-white/20 flex items-center justify-center mb-8 backdrop-blur-sm">
        <span className="text-5xl font-bold">1</span>
      </div>
      <h2 className="text-5xl font-bold mb-4">Introduction & Plan</h2>
      <p className="text-xl text-blue-100 max-w-2xl">
        Where we are, what we set out to do, and the original blueprint.
      </p>
    </div>
  ),

  Section_Findings: () => (
    <div className="flex flex-col items-center justify-center h-full bg-slate-800 text-white p-12 text-center animate-slide-enter">
      <div className="w-24 h-24 rounded-full bg-white/20 flex items-center justify-center mb-8 backdrop-blur-sm">
        <span className="text-5xl font-bold">2</span>
      </div>
      <h2 className="text-5xl font-bold mb-4">Initial Findings</h2>
      <p className="text-xl text-slate-300 max-w-2xl">
        First pass at Korean data and the "Old Forecast".
      </p>
    </div>
  ),

  Section_Pivot: () => (
    <div className="flex flex-col items-center justify-center h-full bg-amber-500 text-white p-12 text-center animate-slide-enter">
      <div className="w-24 h-24 rounded-full bg-white/20 flex items-center justify-center mb-8 backdrop-blur-sm">
        <span className="text-5xl font-bold">3</span>
      </div>
      <h2 className="text-5xl font-bold mb-4">The Pivot</h2>
      <p className="text-xl text-amber-100 max-w-2xl">
        Why the old model failed and the new demography-anchored concept.
      </p>
    </div>
  ),

  Section_Approach: () => (
    <div className="flex flex-col items-center justify-center h-full bg-indigo-600 text-white p-12 text-center animate-slide-enter">
      <div className="w-24 h-24 rounded-full bg-white/20 flex items-center justify-center mb-8 backdrop-blur-sm">
        <span className="text-5xl font-bold">4</span>
      </div>
      <h2 className="text-5xl font-bold mb-4">New Approach</h2>
      <p className="text-xl text-indigo-100 max-w-2xl">
        Methodology, Monte Carlo engine, and the demography anchor.
      </p>
    </div>
  ),

  Section_Results: () => (
    <div className="flex flex-col items-center justify-center h-full bg-emerald-600 text-white p-12 text-center animate-slide-enter">
      <div className="w-24 h-24 rounded-full bg-white/20 flex items-center justify-center mb-8 backdrop-blur-sm">
        <span className="text-5xl font-bold">5</span>
      </div>
      <h2 className="text-5xl font-bold mb-4">Results</h2>
      <p className="text-xl text-emerald-100 max-w-2xl">
        The new 3× forecast, comparisons, and next steps.
      </p>
    </div>
  ),

  // 1. TITLE / OPENING -------------------------------------------------------
  Slide1: () => (
    <div className="flex flex-col items-center justify-center h-full bg-white text-slate-900 p-12 text-center animate-slide-enter">
      <div className="mb-6 p-4 bg-blue-100 text-blue-600 rounded-full shadow-lg shadow-blue-200">
        <Icons.Activity />
      </div>

      <div className="text-xs font-mono text-blue-600 mb-3 tracking-[0.25em] uppercase">
        VIV–TAVR FORECASTING · VERSION 6
      </div>

      <h1 className="text-4xl md:text-5xl font-bold mb-4 text-slate-900">
        Demography‑Anchored Forecasting of ViV‑TAVR
      </h1>

      <p className="text-lg md:text-xl text-slate-600 mb-8 max-w-3xl">
        Korea today, Singapore next: a refreshed Monte Carlo framework grounded
        in registry data and population demography, with adoption modelled on
        top rather than baked in.
      </p>

      <div className="border-t border-slate-200 pt-6 w-full max-w-3xl flex flex-col md:flex-row items-center md:justify-between gap-2 text-sm text-slate-500">
        <span>Hyunjin Ahn · Charles Yap</span>
        <span>Department of Cardiology / Cardiac Surgery</span>
        <span className="text-emerald-600 font-medium">
          Progress update, methodological pivot &amp; next steps
        </span>
      </div>
    </div>
  ),

  // 2. CONTEXT / WHERE WE ARE -----------------------------------------------
  Slide2: () => (
    <div className="flex flex-col h-full bg-white text-slate-900 p-10 md:p-14 animate-slide-enter">
      <div className="flex items-center gap-3 mb-6">
        <div className="p-2 rounded-full bg-blue-100 text-blue-600">
          <Icons.Target className="w-5 h-5" />
        </div>
        <h2 className="text-2xl md:text-3xl font-semibold text-slate-900">
          Where We Are in the Project
        </h2>
      </div>

      <p className="text-slate-600 max-w-3xl mb-6">
        Written updates have already outlined our first Korea‑only ViV‑TAVR
        forecast. Today we focus on why we pivoted the modelling strategy, what
        the new demography‑anchored results look like, and how we will extend
        this to Singapore.
      </p>

      <div className="mt-4 grid md:grid-cols-3 gap-4 text-sm">
        <div className="border border-slate-200 rounded-xl p-4 bg-slate-50">
          <div className="flex items-center gap-2 mb-2">
            <span className="inline-flex h-6 w-6 items-center justify-center rounded-full bg-emerald-100 text-emerald-700 text-xs font-semibold">
              1
            </span>
            <span className="font-semibold text-slate-900">Korea model v9</span>
          </div>
          <p className="text-slate-600">
            Monte Carlo engine implemented; demography‑anchored index volumes;
            ViV candidates vs realised separated.
          </p>
        </div>

        <div className="border border-slate-200 rounded-xl p-4 bg-white shadow-sm">
          <div className="flex items-center gap-2 mb-2">
            <span className="inline-flex h-6 w-6 items-center justify-center rounded-full bg-blue-100 text-blue-700 text-xs font-semibold">
              2
            </span>
            <span className="font-semibold text-slate-900">Current focus</span>
          </div>
          <p className="text-slate-600">
            Explain the pivot away from adoption‑driven extrapolation, show the
            &quot;3× not 8×&quot; Korean forecast, and link this to editorial
            critiques.
          </p>
        </div>

        <div className="border border-slate-200 rounded-xl p-4 bg-slate-50">
          <div className="flex items-center gap-2 mb-2">
            <span className="inline-flex h-6 w-6 items-center justify-center rounded-full bg-slate-200 text-slate-700 text-xs font-semibold">
              3
            </span>
            <span className="font-semibold text-slate-900">Next steps</span>
          </div>
          <p className="text-slate-600">
            Apply the same framework to Singapore with UN projections, then
            build explicit adoption curves on top for cross‑country comparison.
          </p>
        </div>
      </div>
    </div>
  ),

  // 3. CLINICAL & MODELLING OBJECTIVE ---------------------------------------
  Slide3: () => (
    <div className="flex flex-col h-full bg-white text-slate-900 p-10 md:p-14 animate-slide-enter">
      <div className="flex items-center gap-3 mb-6">
        <div className="p-2 rounded-full bg-blue-100 text-blue-600">
          <Icons.Activity className="w-5 h-5" />
        </div>
        <h2 className="text-2xl md:text-3xl font-semibold text-slate-900">
          Clinical &amp; Modelling Objective
        </h2>
      </div>

      <div className="mb-6">
        <p className="text-slate-700 text-lg mb-2">
          <span className="font-semibold">One‑sentence goal:</span> estimate
          future volumes of ViV‑TAVR in Korea (and later Singapore), split into
          TAVR‑in‑SAVR and TAVR‑in‑TAVR, using registry data plus a
          patient‑level Monte Carlo model anchored to demography.
        </p>
      </div>

      <div className="grid md:grid-cols-2 gap-6 mb-6 text-sm">
        <div className="border border-slate-200 rounded-xl p-4 bg-slate-50">
          <h3 className="font-semibold mb-3 text-slate-900">
            What the model must deliver
          </h3>
          <ul className="space-y-2 text-slate-600">
            <li>Annual ViV‑TAVR volumes over the next 10–25 years.</li>
            <li>Clear separation of TAVR‑in‑SAVR vs TAVR‑in‑TAVR trajectories.</li>
            <li>
              Linkage to index TAVR/SAVR/redo‑SAVR volumes and population
              structure.
            </li>
            <li>Transparent assumptions suitable for publication and critique.</li>
          </ul>
        </div>

        <div className="border border-slate-200 rounded-xl p-4 bg-white shadow-sm">
          <h3 className="font-semibold mb-3 text-slate-900">
            Two paths to ViV‑TAVR (schematic)
          </h3>
          <div className="space-y-3 text-slate-600">
            <div className="flex items-start gap-3">
              <div className="mt-1 h-8 w-8 rounded-full bg-slate-100 text-slate-600 flex items-center justify-center text-xs font-bold border border-slate-200">
                SAVR
              </div>
              <div>
                <p className="font-medium text-slate-900 mb-1">
                  TAVR‑in‑SAVR path
                </p>
                <p className="text-xs">
                  Surgical bioprosthesis → structural valve degeneration
                  (durability curve) → ViV‑TAVR in suitable survivors.
                </p>
              </div>
            </div>

            <div className="flex items-start gap-3">
              <div className="mt-1 h-8 w-8 rounded-full bg-slate-100 text-slate-600 flex items-center justify-center text-xs font-bold border border-slate-200">
                TAVR
              </div>
              <div>
                <p className="font-medium text-slate-900 mb-1">
                  TAVR‑in‑TAVR path
                </p>
                <p className="text-xs">
                  Index TAVR → bimodal durability (early + late mode) → ViV‑TAVR
                  when failure occurs before death within the horizon.
                </p>
              </div>
            </div>

            <p className="text-xs text-slate-500 border-t border-slate-200 pt-3">
              A Monte Carlo engine combines index volumes, survival curves, and
              durability distributions to estimate how many patients reach these
              ViV stages each year.
            </p>
          </div>
        </div>
      </div>
    </div>
  ),

  // 4. PRIOR BLUEPRINT: GENEREUX / OHNO -------------------------------------
  // 4a. PRIOR BLUEPRINT: GENEREUX (US) -------------------------------------
  Slide4a: () => (
    <div className="flex flex-col h-full bg-white text-slate-900 p-10 md:p-14 animate-slide-enter">
      <div className="flex items-center gap-3 mb-6">
        <div className="p-2 rounded-full bg-slate-100 text-slate-600">
          <Icons.GitBranch className="w-5 h-5" />
        </div>
        <h2 className="text-2xl md:text-3xl font-semibold text-slate-900">
          Prior Blueprint: Genereux US Model
        </h2>
      </div>

      <div className="grid md:grid-cols-2 gap-6 text-sm flex-1 min-h-0">
        <div className="border border-slate-200 rounded-xl p-6 bg-slate-50 flex flex-col justify-center">
          <h3 className="font-semibold mb-4 text-slate-900 text-lg">
            Shared core methodology
          </h3>
          <ol className="space-y-4 text-slate-600 list-decimal list-inside text-base">
            <li>Use historical index TAVR/SAVR volumes by age.</li>
            <li>Extrapolate index volumes into the future.</li>
            <li>Run Monte Carlo for durability, survival, and failure.</li>
            <li>Apply assumed ViV penetration curves (e.g. 10%→60%).</li>
          </ol>
          <p className="mt-6 text-slate-500 text-sm border-t border-slate-200 pt-4">
            We reused the patient‑level Monte Carlo logic, but later changed how
            index volumes and penetration are handled for Korea.
          </p>
        </div>

        <div className="border border-slate-200 rounded-xl p-4 bg-white shadow-sm flex flex-col gap-4">
          <h3 className="font-semibold text-slate-900">
            Genereux et al. – US TVT ViV forecast
          </h3>

          <div className="flex flex-col gap-4 flex-1 min-h-0">
            <div className="flex-1 min-h-0 flex flex-col gap-2">
              <div className="rounded-lg overflow-hidden border border-slate-200 bg-white flex-1 flex items-center justify-center p-2">
                <img
                  src={V6_DIAGRAMS.genereuxVivPrediction}
                  alt="Genereux US ViV forecast"
                  className="max-h-full max-w-full object-contain"
                />
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  ),

  // 4b. PRIOR BLUEPRINT: OHNO (JAPAN) ---------------------------------------
  Slide4b: () => (
    <div className="flex flex-col h-full bg-white text-slate-900 p-10 md:p-14 animate-slide-enter">
      <div className="flex items-center gap-3 mb-6">
        <div className="p-2 rounded-full bg-slate-100 text-slate-600">
          <Icons.GitBranch className="w-5 h-5" />
        </div>
        <h2 className="text-2xl md:text-3xl font-semibold text-slate-900">
          Prior Blueprint: Ohno Japan Model
        </h2>
      </div>

      <div className="grid md:grid-cols-2 gap-6 text-sm flex-1 min-h-0">
        <div className="border border-slate-200 rounded-xl p-6 bg-slate-50 flex flex-col justify-center">
          <h3 className="font-semibold mb-4 text-slate-900 text-lg">
            Shared core methodology
          </h3>
          <ol className="space-y-4 text-slate-600 list-decimal list-inside text-base">
            <li>Use historical index TAVR/SAVR volumes by age.</li>
            <li>Extrapolate index volumes into the future.</li>
            <li>Run Monte Carlo for durability, survival, and failure.</li>
            <li>Apply assumed ViV penetration curves (e.g. 10%→60%).</li>
          </ol>
          <p className="mt-6 text-slate-500 text-sm border-t border-slate-200 pt-4">
            We reused the patient‑level Monte Carlo logic, but later changed how
            index volumes and penetration are handled for Korea.
          </p>
        </div>

        <div className="border border-slate-200 rounded-xl p-4 bg-white shadow-sm flex flex-col gap-4">
          <h3 className="font-semibold text-slate-900">
            Ohno &amp; Genereux – JACC Asia US vs Japan
          </h3>

          <div className="flex flex-col gap-4 flex-1 min-h-0">
            <div className="flex-1 min-h-0 flex flex-col gap-2">
              <div className="rounded-lg overflow-hidden border border-slate-200 bg-white flex-1 flex items-center justify-center p-2">
                <img
                  src={V6_DIAGRAMS.genereuxOhnoVivPrediction}
                  alt="Ohno Japan vs US ViV comparison"
                  className="max-h-full max-w-full object-contain"
                />
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  ),

  // 5. EXTERNAL CRITIQUE & ADOPTION PROBLEMS -------------------------------
  Slide5: () => (
    <div className="flex flex-col h-full bg-white text-slate-900 p-8 animate-slide-enter">
      <div className="flex items-center gap-3 mb-4">
        <div className="p-2 rounded-full bg-amber-100 text-amber-600">
          <Icons.AlertTriangle className="w-5 h-5" />
        </div>
        <h2 className="text-2xl font-semibold text-slate-900">
          Editorial Critique &amp; The Case for a Pivot
        </h2>
      </div>

      <div className="flex-1 grid grid-rows-[auto_1fr] gap-4 min-h-0">
        {/* Top: Images - Increased height for portrait clarity */}
        <div className="grid grid-cols-[1fr_1.5fr] gap-4 h-80">
             <div className="rounded-xl overflow-hidden border border-slate-200 bg-white flex items-center justify-center p-1">
                <img
                  src={V6_DIAGRAMS.criticismText}
                  alt="JACC Asia Editorial Text"
                  className="max-h-full max-w-full object-contain"
                />
             </div>
             <div className="rounded-xl overflow-hidden border border-slate-200 bg-white flex items-center justify-center p-1">
                <img
                  src={V6_DIAGRAMS.criticismFactors}
                  alt="Factors influencing ViV-TAVR"
                  className="max-h-full max-w-full object-contain"
                />
             </div>
        </div>

        {/* Bottom: Text */}
        <div className="grid md:grid-cols-2 gap-6 text-sm overflow-y-auto">
          <div className="border border-amber-200 bg-amber-50 rounded-xl p-4">
            <h3 className="font-semibold mb-2 text-amber-800">
              Critique of Extrapolation (Ohno et al.)
            </h3>
            <ul className="space-y-2 text-amber-900/80 list-disc list-inside text-xs">
              <li>
                <strong>Over-projection:</strong> Predicted 4-fold increase in Japan by 2035, but relied heavily on assumptions.
              </li>
              <li>
                <strong>Fragile Baselines:</strong> SAVR volumes were often forced to plateau rather than decline naturally.
              </li>
              <li>
                <strong>Missing Context:</strong> Failed to account for Asian-specific clinical/anatomical factors, technology maturation, and socioeconomic shifts.
              </li>
            </ul>
          </div>

          <div className="border border-emerald-200 bg-emerald-50 rounded-xl p-4">
            <h3 className="font-semibold mb-2 text-emerald-800">
              Our Methodological Pivot
            </h3>
             <ul className="space-y-2 text-emerald-900/80 list-disc list-inside text-xs">
              <li>
                <strong>Adoption is Volatile:</strong> Pure adoption curves are sensitive to shocks (e.g., reimbursement, pandemics) and hard to calibrate from early data.
              </li>
              <li>
                <strong>Demography is Stable:</strong> We anchor volumes to population structure (ageing) and recent practice patterns.
              </li>
              <li>
                <strong>Scenario Layering:</strong> Adoption becomes an explicit scenario layer on top, not a hidden driver of the baseline.
              </li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  ),


  // 5b. PROBLEM DETAIL: WHY IT FAILED ---------------------------------------
  Slide_Problem: () => (
    <div className="flex flex-col h-full bg-white text-slate-900 p-10 md:p-14 animate-slide-enter">
      <div className="flex items-center gap-3 mb-6">
        <div className="p-2 rounded-full bg-red-100 text-red-600">
          <Icons.AlertTriangle className="w-6 h-6" />
        </div>
        <h2 className="text-3xl font-bold text-slate-900">
          Why the Original Approach Failed
        </h2>
      </div>

      <div className="grid grid-cols-2 gap-8 flex-1 min-h-0">
        <div className="bg-red-50 p-6 rounded-xl border border-red-200 flex flex-col justify-center">
          <h3 className="text-xl font-bold text-red-700 mb-4">1. The 2023 "Backlog Spike"</h3>
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
            <div className="w-1/5 bg-slate-300 h-2/3 rounded-t opacity-50 border-t-2 border-dashed border-slate-400"></div>
          </div>
          <p className="text-xs text-red-600 mt-2 italic text-center">Linear extrapolation from here creates infinite growth.</p>
        </div>

        <div className="bg-slate-50 p-6 rounded-xl border border-slate-200 flex flex-col justify-center">
          <h3 className="text-xl font-bold text-slate-900 mb-4">2. The JACC Critique</h3>
          <blockquote className="italic border-l-4 border-blue-500 pl-4 text-slate-600 mb-4">
            "Extrapolations... potentially introducing bias... significant adjustments were made... clinical factors were not accounted for."
          </blockquote>
          <div className="mt-4 bg-white p-4 rounded border border-slate-200 text-sm text-slate-700 shadow-sm">
            <strong className="text-slate-900">Conclusion:</strong> We cannot blindly apply US/Japan growth rates (Ohno et al.) to Korea's unique aging demographic.
          </div>
        </div>
      </div>
    </div>
  ),

  // 5c. CONCEPT: THE PIVOT --------------------------------------------------
  Slide_Concept: () => (
    <div className="flex flex-col h-full bg-white text-slate-900 p-10 md:p-14 animate-slide-enter">
      <div className="flex items-center gap-3 mb-6">
        <div className="p-2 rounded-full bg-blue-100 text-blue-600">
           <Icons.Shuffle className="w-6 h-6" />
        </div>
        <h2 className="text-3xl font-bold text-slate-900">The Pivot: A Demography-Anchored Approach</h2>
      </div>

      <div className="flex flex-col gap-6 flex-1 justify-center">
        <div className="flex items-center gap-6 p-6 bg-slate-50 rounded-xl border border-slate-200 opacity-60 grayscale">
          <div className="w-24 text-right font-bold text-slate-500">OLD MODEL</div>
          <div className="flex-1 border-l-2 border-slate-300 pl-6">
            <h4 className="font-bold text-lg text-slate-500">Linear Extrapolation</h4>
            <p className="text-slate-500">"Procedures will grow by X% every year based on history."</p>
          </div>
        </div>

        <div className="flex flex-col items-center justify-center py-2 text-slate-400">
          <div className="h-8 w-0.5 bg-slate-300"></div>
          <span className="text-xs uppercase tracking-widest bg-slate-100 px-2 py-1 border border-slate-200 rounded">Transformed Into</span>
          <div className="h-8 w-0.5 bg-slate-300"></div>
        </div>

        <div className="flex items-center gap-6 p-8 bg-blue-50 rounded-xl border-2 border-blue-200 shadow-lg shadow-blue-100">
          <div className="w-24 text-right font-bold text-blue-600">NEW MODEL (v9)</div>
          <div className="flex-1 border-l-2 border-blue-200 pl-6">
            <h4 className="font-bold text-2xl text-blue-800 mb-2 flex items-center gap-2">
              Risk &times; Demography
            </h4>
            <p className="text-blue-700 text-lg italic">
              "Given the stable risk profile of a Korean patient, how many failures occur as the population ages?"
            </p>
          </div>
          <div className="bg-white p-4 rounded-lg border border-slate-200 shadow text-center min-w-[120px]">
            <Icons.Users className="w-6 h-6 mx-auto text-blue-600" />
            <div className="text-xs font-bold text-blue-600 mt-2">Pop Structure</div>
          </div>
        </div>
      </div>
    </div>
  ),

  // 6a. KOREA DATA: INDEX VOLUMES -------------------------------------------
  Slide6a: () => (
    <div className="flex flex-col h-full bg-white text-slate-900 p-10 md:p-14 animate-slide-enter">
      <div className="flex items-center gap-3 mb-4">
        <div className="p-2 rounded-full bg-rose-100 text-rose-600">
          <Icons.TrendingUp className="w-5 h-5" />
        </div>
        <h2 className="text-2xl md:text-3xl font-semibold text-slate-900">
          Korea Data: Index Volumes 2015–2024
        </h2>
      </div>

      <div className="flex flex-col flex-1 min-h-0 gap-4">
        <div className="flex-1 min-h-0 rounded-lg overflow-hidden border border-slate-200 bg-slate-50 flex items-center justify-center">
          <img
            src={V6_DIAGRAMS.koreaIndexVolumes2015_2024}
            alt="Index TAVR and SAVR volumes 2015–2024 with 2023 spike"
            className="max-h-full max-w-full object-contain"
          />
        </div>
        <p className="text-slate-600 text-sm">
          Sharp spike in 2023 with partial regression in 2024 (still above
          2022), likely reflecting post‑COVID backlog clearance and insurance
          changes rather than smooth organic growth.
        </p>
      </div>
    </div>
  ),

  // 6b. KOREA DATA: FIRST PASS FORECAST -------------------------------------
  Slide6b: () => (
    <div className="flex flex-col h-full bg-white text-slate-900 p-10 md:p-14 animate-slide-enter">
      <div className="flex items-center gap-3 mb-4">
        <div className="p-2 rounded-full bg-rose-100 text-rose-600">
          <Icons.TrendingUp className="w-5 h-5" />
        </div>
        <h2 className="text-2xl md:text-3xl font-semibold text-slate-900">
          First‑pass ViV Forecast (Genereux/Ohno replication)
        </h2>
      </div>

      <div className="flex flex-col flex-1 min-h-0 gap-4">
        <div className="flex-1 min-h-0 rounded-lg overflow-hidden border border-slate-200 bg-slate-50 flex items-center justify-center">
          <img
            src={V6_DIAGRAMS.koreaFirstPassVivForecast}
            alt="First-pass Korean ViV forecast using extrapolated volumes"
            className="max-h-full max-w-full object-contain"
          />
        </div>
        <ul className="space-y-2 text-slate-600 text-sm">
          <li>
            Extrapolating volumes through the 2023 spike bakes in a transient
            anomaly.
          </li>
          <li>
            Combined with aggressive penetration curves, this yields forecasts
            closer to 8× growth.
          </li>
          <li>
            For Korea’s rapidly ageing population, this felt misaligned with
            both data and editorial concerns.
          </li>
        </ul>
      </div>
    </div>
  ),

  // 7. DESIGN GOALS FOR THE NEW MODEL ---------------------------------------
  Slide7: () => (
    <div className="flex flex-col h-full bg-white text-slate-900 p-10 md:p-14 animate-slide-enter">
      <div className="flex items-center gap-3 mb-6">
        <div className="p-2 rounded-full bg-emerald-100 text-emerald-600">
          <Icons.Layers className="w-5 h-5" />
        </div>
        <h2 className="text-2xl md:text-3xl font-semibold text-slate-900">
          Design Goals for the Demography‑Anchored Model
        </h2>
      </div>

      <div className="grid md:grid-cols-3 gap-5 text-sm">
        <div className="border border-slate-200 rounded-xl p-4 bg-slate-50">
          <h3 className="font-semibold mb-2 text-slate-900">
            Goal 1 · Anchor to Demography
          </h3>
          <p className="text-slate-600">
            Make age and sex structure explicit. Drive index TAVR/SAVR volumes
            using:
          </p>
          <p className="mt-2 text-emerald-600 font-semibold">
            per‑capita risk × population
          </p>
          <p className="mt-2 text-slate-500 text-xs">
            Avoid arbitrary regression lines; let population ageing dictate the
            trend.
          </p>
        </div>

        <div className="border border-slate-200 rounded-xl p-4 bg-white shadow-sm">
          <h3 className="font-semibold mb-2 text-slate-900">
            Goal 2 · Minimise Speculative Inputs
          </h3>
          <ul className="space-y-2 text-slate-600">
            <li>Report ViV candidates first as the primary output.</li>
            <li>
              Treat ViV penetration as an optional scenario layer on top (e.g.
              US‑style ramp).
            </li>
            <li>
              Make every assumption modular so it can be swapped or stress‑
              tested.
            </li>
          </ul>
        </div>

        <div className="border border-slate-200 rounded-xl p-4 bg-slate-50">
          <h3 className="font-semibold mb-2 text-slate-900">
            Goal 3 · Transparent Mechanics
          </h3>
          <p className="text-slate-600">
            Explicitly model the "Race":
          </p>
          <div className="flex items-center gap-2 mt-2 justify-center py-2 bg-white rounded border border-slate-200 shadow-sm">
            <span className="text-rose-600 font-mono">Death</span>
            <span className="text-slate-400 text-xs">vs</span>
            <span className="text-amber-600 font-mono">Valve Failure</span>
          </div>
          <p className="mt-2 text-slate-500 text-xs">
            Patients are dropped if they die before failure. No "ghost patients"
            getting ViV.
          </p>
        </div>
      </div>
    </div>
  ),



  // 7b. METHODOLOGY: ARCHITECTURE -------------------------------------------
  Slide_Methodology: () => (
    <div className="flex flex-col h-full bg-white text-slate-900 p-10 md:p-14 animate-slide-enter">
      <div className="flex items-center gap-3 mb-6">
        <div className="p-2 rounded-full bg-purple-100 text-purple-600">
           <Icons.Layers className="w-6 h-6" />
        </div>
        <h2 className="text-3xl font-bold text-slate-900">Simulation Architecture (model_v9)</h2>
      </div>

      <div className="grid grid-cols-3 gap-4 h-64 mb-8">
        {/* Step 1 */}
        <div className="bg-slate-50 p-6 rounded-xl border border-slate-200 flex flex-col relative">
          <div className="absolute -top-3 -right-3 w-8 h-8 bg-blue-600 rounded-full flex items-center justify-center font-bold text-white">1</div>
          <h3 className="font-bold text-lg mb-2 text-slate-800">Pre-Compute</h3>
          <p className="text-slate-500 text-sm mb-4">Ingest Data Sources</p>
          <ul className="text-sm space-y-2 text-slate-600">
            <li>• HIRA Registry (2015-2024)</li>
            <li>• Ministry Pop Stats</li>
            <li>• Age/Sex Bands</li>
          </ul>
        </div>
        {/* Step 2 */}
        <div className="bg-slate-50 p-6 rounded-xl border border-slate-200 flex flex-col relative">
          <div className="absolute -top-3 -right-3 w-8 h-8 bg-blue-600 rounded-full flex items-center justify-center font-bold text-white">2</div>
          <h3 className="font-bold text-lg mb-2 text-slate-800">Risk Calculation</h3>
          <p className="text-slate-500 text-sm mb-4">Establish Anchor</p>
          <div className="bg-white p-3 rounded text-center font-mono text-sm border border-slate-200 mb-2 text-emerald-600">
            Risk = Obs / Pop
          </div>
          <p className="text-xs text-slate-500 italic">Generates stable risk profiles per age-band based on 2023-24 data.</p>
        </div>
        {/* Step 3 */}
        <div className="bg-blue-50 text-slate-900 p-6 rounded-xl flex flex-col relative border border-blue-200 shadow-[0_0_15px_rgba(59,130,246,0.1)]">
          <div className="absolute -top-3 -right-3 w-8 h-8 bg-white text-blue-600 border border-blue-200 rounded-full flex items-center justify-center font-bold">3</div>
          <h3 className="font-bold text-lg mb-2 flex items-center gap-2 text-blue-700"><Icons.Server size={16}/> Monte Carlo</h3>
          <p className="text-blue-600/80 text-sm mb-4">The Engine</p>
          <ul className="text-sm space-y-2 text-blue-800/80">
            <li>• 100 Runs per Scenario</li>
            <li>• Durability vs Survival</li>
            <li>• Stochastic Jitter (5-10%)</li>
          </ul>
        </div>
      </div>
      <div className="mt-auto bg-slate-50 p-4 rounded-lg border border-slate-200 flex items-center justify-between">
        <div className="text-sm text-slate-500">
          <strong>Filters Applied:</strong> Redo-SAVR subtraction applied. Speculative penetration (60-80%) removed for conservative floor.
        </div>
      </div>
    </div>
  ),

  // 7c. MECHANISM: THE RACE -------------------------------------------------
  Slide_Mechanism: () => (
    <div className="flex flex-col h-full bg-white text-slate-900 p-10 md:p-14 animate-slide-enter">
      <div className="flex items-center gap-3 mb-6">
        <div className="p-2 rounded-full bg-orange-100 text-orange-600">
           <Icons.Activity className="w-6 h-6" />
        </div>
        <h2 className="text-3xl font-bold text-slate-900">The Core Logic: "The Race"</h2>
      </div>

      <p className="text-slate-600 mb-8 max-w-3xl">
        Inside the Monte Carlo Engine, every simulated patient runs a "Race" between two dates.
        A patient only becomes a candidate if their valve fails while they are still alive.
      </p>

      <div className="bg-slate-50 p-8 rounded-xl border border-slate-200 space-y-8">
        
        {/* Scenario A */}
        <div>
          <div className="flex justify-between mb-2">
            <span className="font-bold text-green-600 flex items-center gap-2"><div className="w-3 h-3 bg-green-500 rounded-full"></div> Scenario A: ViV Candidate</span>
            <span className="text-xs font-mono text-slate-500">Fail_Year &le; Death_Year</span>
          </div>
          <div className="relative h-12 w-full bg-slate-200 rounded-full flex items-center px-2 overflow-hidden">
            {/* Life Line */}
            <div className="absolute left-2 top-4 h-1 bg-blue-500 w-[80%]"></div>
            {/* Durability Line */}
            <div className="absolute left-2 top-7 h-1 bg-yellow-500 w-[50%] z-10"></div>
            
            <div className="absolute left-2 w-4 h-4 bg-white border border-slate-300 rounded-full z-20" title="Index Procedure"></div>
            
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
            <div className="absolute left-[50%] w-[30%] h-full bg-green-50 border-x border-green-200 flex items-center justify-center text-xs font-bold text-green-700 tracking-wide">
              ELIGIBLE WINDOW
            </div>
          </div>
        </div>

        {/* Scenario B */}
        <div>
          <div className="flex justify-between mb-2">
            <span className="font-bold text-red-600 flex items-center gap-2"><div className="w-3 h-3 bg-red-500 rounded-full"></div> Scenario B: Died First</span>
             <span className="text-xs font-mono text-slate-500">Fail_Year &gt; Death_Year</span>
          </div>
          <div className="relative h-12 w-full bg-slate-200 rounded-full flex items-center px-2 overflow-hidden opacity-70">
            {/* Life Line */}
            <div className="absolute left-2 top-4 h-1 bg-blue-500 w-[40%]"></div>
            {/* Durability Line */}
            <div className="absolute left-2 top-7 h-1 bg-yellow-500 w-[70%] z-10"></div>
            
            <div className="absolute left-2 w-4 h-4 bg-white border border-slate-300 rounded-full z-20"></div>
            
            {/* Death Point */}
             <div className="absolute left-[40%] top-2 flex flex-col items-center z-30">
               <div className="w-0.5 h-8 bg-slate-400"></div>
               <span className="text-[10px] text-slate-500 font-bold -mt-10">Death</span>
            </div>

            {/* Failure Point */}
            <div className="absolute left-[70%] top-2 flex flex-col items-center z-30 opacity-50">
               <div className="w-1 h-8 bg-yellow-500/50 border-l border-dashed border-yellow-600"></div>
               <span className="text-[10px] bg-yellow-100 text-yellow-800 px-1 rounded -mt-10">Valve Fails</span>
            </div>
             
             <div className="absolute right-4 text-red-600 font-bold border-2 border-red-200 p-1 px-2 rounded -rotate-12 text-xs">
               NOT ELIGIBLE
             </div>
          </div>
        </div>

      </div>
    </div>
  ),

  // 7d. LOGIC: DISTRIBUTIONS ------------------------------------------------
  Slide_Logic: () => (
    <div className="flex flex-col h-full bg-white text-slate-900 p-10 md:p-14 animate-slide-enter">
      <div className="flex items-center gap-3 mb-6">
        <div className="p-2 rounded-full bg-yellow-100 text-yellow-600">
           <Icons.BarChart3 className="w-6 h-6" />
        </div>
        <h2 className="text-3xl font-bold text-slate-900">Sampling Distributions</h2>
      </div>

      <div className="grid grid-cols-2 gap-8 h-full">
        <div className="bg-yellow-50 p-6 rounded-xl border border-yellow-200">
          <h3 className="text-xl font-bold text-yellow-700 mb-4 flex items-center gap-2"><Icons.Activity/> Durability</h3>
          <p className="text-sm text-slate-600 mb-4">Bimodal Mixture Model: Valves fail in two clusters (Early Manufacturing vs Late Wear).</p>
          <div className="flex items-end justify-center h-32 gap-1 border-b border-yellow-200 pb-1">
            {/* Early Failure */}
            <div className="w-4 h-4 bg-yellow-600 rounded-t"></div>
            <div className="w-4 h-8 bg-yellow-500 rounded-t"></div>
            <div className="w-4 h-3 bg-yellow-600 rounded-t mr-8"></div>
            
            {/* Late Failure */}
            <div className="w-4 h-6 bg-yellow-600 rounded-t"></div>
            <div className="w-4 h-12 bg-yellow-500 rounded-t"></div>
            <div className="w-4 h-24 bg-yellow-400 rounded-t"></div>
            <div className="w-4 h-16 bg-yellow-500 rounded-t"></div>
            <div className="w-4 h-8 bg-yellow-600 rounded-t"></div>
          </div>
          <div className="flex justify-between text-xs text-slate-500 mt-2 font-mono">
             <span>4 Years (20%)</span>
             <span>11.5 Years (80%)</span>
          </div>
        </div>

        <div className="bg-blue-50 p-6 rounded-xl border border-blue-200">
          <h3 className="text-xl font-bold text-blue-700 mb-4 flex items-center gap-2"><Icons.TrendingUp/> Survival</h3>
          <p className="text-sm text-slate-600 mb-4">Actuarial curves adjusted for risk category (Low/Int/High).</p>
          <div className="h-32 w-full relative border-l border-b border-blue-200">
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
  ),

  // 10b. RESULTS VISUAL -----------------------------------------------------
  Slide_Results: () => (
    <div className="flex flex-col h-full bg-white text-slate-900 p-10 md:p-14 animate-slide-enter">
      <div className="flex items-center gap-3 mb-6">
        <div className="p-2 rounded-full bg-emerald-100 text-emerald-600">
           <Icons.BarChart3 className="w-6 h-6" />
        </div>
        <h2 className="text-3xl font-bold text-slate-900">Model Comparison (2035)</h2>
      </div>

      <div className="grid grid-cols-2 gap-12 flex-1 items-end pb-8">
        
        {/* Our Model */}
        <div className="flex flex-col items-center">
           <div className="text-lg font-bold text-blue-600 mb-2">Our Model (Realized)</div>
           <div className="flex items-end space-x-2 h-64 w-full justify-center border-b border-slate-200 pb-1">
              <div className="w-12 bg-blue-600 h-[30%] rounded-t animate-[pulse_3s_ease-in-out_infinite]"></div>
              <div className="w-12 bg-blue-600 h-[50%] rounded-t"></div>
              <div className="w-12 bg-blue-600 h-[75%] rounded-t"></div>
              <div className="w-12 bg-blue-500 h-[90%] rounded-t relative">
                <span className="absolute -top-8 left-1/2 -translate-x-1/2 font-bold text-blue-600 text-xl">~3x</span>
              </div>
           </div>
           <p className="text-center text-sm text-slate-600 mt-4 px-4">
             Anchored to demographic probability. Realistic growth curve.
           </p>
        </div>

        {/* Old Model */}
        <div className="flex flex-col items-center opacity-50 grayscale hover:grayscale-0 transition-all duration-500">
           <div className="text-lg font-bold text-slate-500 mb-2">Ohno / Genereux (Linear)</div>
           <div className="flex items-end space-x-2 h-64 w-full justify-center border-b border-slate-200 pb-1">
              <div className="w-12 bg-slate-400 h-[30%] rounded-t"></div>
              <div className="w-12 bg-slate-400 h-[60%] rounded-t"></div>
              <div className="w-12 bg-slate-400 h-[100%] rounded-t"></div>
              <div className="w-12 bg-red-400/60 h-[140%] rounded-t relative overflow-visible">
                 <span className="absolute -top-8 left-1/2 -translate-x-1/2 font-bold text-red-500 text-xl w-32 text-center">7-9x</span>
              </div>
           </div>
           <p className="text-center text-sm text-slate-500 mt-4 px-4">
             Unchecked linear extrapolation creates improbable demand.
           </p>
        </div>

      </div>
    </div>
  ),

  // 7b. DATA SOURCE: UN POPULATION ------------------------------------------
  Slide_UN_Data: () => (
    <div className="flex flex-col h-full bg-white text-slate-900 p-10 md:p-14 animate-slide-enter">
      <div className="flex items-center gap-3 mb-6">
        <div className="p-2 rounded-full bg-indigo-100 text-indigo-600">
          <Icons.Globe2 className="w-6 h-6" />
        </div>
        <h2 className="text-3xl font-bold text-slate-900">
          Data Source: UN Population Projections
        </h2>
      </div>

      <div className="grid md:grid-cols-2 gap-8 flex-1 min-h-0">
        {/* Left: Image */}
        <div className="rounded-xl overflow-hidden border border-slate-200 bg-slate-50 flex items-center justify-center p-2 shadow-sm">
          <img
            src={V6_DIAGRAMS.unPopulationData}
            alt="UN World Population Prospects Website"
            className="max-h-full max-w-full object-contain"
          />
        </div>

        {/* Right: Text */}
        <div className="flex flex-col justify-center space-y-6">
          <div className="bg-indigo-50 border border-indigo-100 p-5 rounded-xl">
            <h3 className="font-bold text-indigo-900 mb-2 flex items-center gap-2">
              <Icons.FileText className="w-4 h-4" />
              UN World Population Prospects
            </h3>
            <p className="text-indigo-800/80 text-sm">
              We utilize the <strong>'Median' Probabilistic Projection</strong> tables, providing a robust central forecast for the next 75+ years.
            </p>
          </div>

          <ul className="space-y-4 text-slate-600">
            <li className="flex items-start gap-3">
              <div className="mt-1 min-w-[20px] h-5 rounded-full bg-slate-200 text-slate-600 flex items-center justify-center text-xs font-bold">1</div>
              <div>
                <strong className="text-slate-900 block">Granular Resolution</strong>
                Unlike KOSIS (Korean Gov) data which often groups by 5-year intervals, the UN dataset provides <strong>year-by-year</strong> projections, essential for our annual Monte Carlo simulation.
              </div>
            </li>
            <li className="flex items-start gap-3">
              <div className="mt-1 min-w-[20px] h-5 rounded-full bg-slate-200 text-slate-600 flex items-center justify-center text-xs font-bold">2</div>
              <div>
                <strong className="text-slate-900 block">Consistency</strong>
                The projections align closely with national statistics but offer the extended horizon and detail needed for long-term ViV forecasting.
              </div>
            </li>
          </ul>
        </div>
      </div>
    </div>
  ),

  // 8a. DEMOGRAPHY FORMULA --------------------------------------------------
  Slide8a: () => (
    <div className="flex flex-col h-full bg-white text-slate-900 p-10 md:p-14 animate-slide-enter justify-center">
      <div className="flex items-center gap-3 mb-6">
        <div className="p-2 rounded-full bg-blue-100 text-blue-600">
          <Icons.Users className="w-5 h-5" />
        </div>
        <h2 className="text-2xl md:text-3xl font-semibold text-slate-900">
          Demography as the Driver: The Formula
        </h2>
      </div>

      <div className="border border-slate-200 rounded-xl p-8 bg-slate-50 max-w-4xl w-full mx-auto">
        <h3 className="font-semibold mb-6 text-slate-900 text-xl">
          Core projection formula
        </h3>
        <p className="text-slate-600 mb-6 text-lg">
          For each year <span className="font-mono">y ≥ 2025</span>, sex, and
          age band:
        </p>
        <p className="text-emerald-600 font-mono text-2xl md:text-3xl mb-8 text-center p-6 bg-white rounded-lg border border-emerald-200 shadow-sm">
          IndexVolume<sub>y,sex,age</sub> = risk<sub>2023–24,sex,age</sub> ×
          population<sub>y,sex,age</sub>
        </p>
        <ul className="space-y-4 text-slate-600 text-base">
          <li>
            Risks are computed from 2023–2024 HIRA registry counts for TAVR,
            SAVR, and redo‑SAVR.
          </li>
          <li>
            Population projections are taken from national tables, later to be
            standardised to UN WPP for Korea vs Singapore.
          </li>
          <li>
            Summing over age and sex gives total index TAVR/SAVR volumes per
            year.
          </li>
        </ul>
        <p className="mt-6 text-sm text-slate-500 italic border-t border-slate-200 pt-4">
          Intuition: instead of saying “TAVR grows by X% per year”, we ask:
          “If each age–sex group is treated at 2023–24 rates, what happens as
          the population ages?”
        </p>
      </div>
    </div>
  ),

  // 8b. DEMOGRAPHY: POPULATION TRENDS ---------------------------------------
  Slide8b: () => (
    <div className="flex flex-col h-full bg-white text-slate-900 p-10 md:p-14 animate-slide-enter">
      <div className="flex items-center gap-3 mb-4">
        <div className="p-2 rounded-full bg-blue-100 text-blue-600">
          <Icons.Users className="w-5 h-5" />
        </div>
        <h2 className="text-2xl md:text-3xl font-semibold text-slate-900">
          Korean Population Trends
        </h2>
      </div>

      <div className="flex flex-col flex-1 min-h-0 gap-4">
        <div className="flex-1 min-h-0 rounded-lg overflow-hidden border border-slate-200 bg-slate-50 flex items-center justify-center">
          <img
            src={V6_DIAGRAMS.koreaPopulationByAgeSex}
            alt="Projected Korean population by age band and sex"
            className="max-h-full max-w-full object-contain"
          />
        </div>
        <ul className="space-y-1 text-slate-600 text-sm">
          <li>Younger cohorts stable or shrinking.</li>
          <li>Strong growth in 75–79, 80–84, and ≥85, especially women.</li>
          <li>
            With risks held constant, this alone yields ~2.5–3× ViV candidates
            between 2024 and 2035.
          </li>
        </ul>
      </div>
    </div>
  ),

  // 9. MONTE CARLO ENGINE & VIV DEFINITIONS ---------------------------------
  Slide9: () => (
    <div className="flex flex-col h-full bg-white text-slate-900 p-10 md:p-14 animate-slide-enter">
      <div className="flex items-center gap-3 mb-6">
        <div className="p-2 rounded-full bg-slate-100 text-slate-600">
          <Icons.Shuffle className="w-5 h-5" />
        </div>
        <h2 className="text-2xl md:text-3xl font-semibold text-slate-900">
          Monte Carlo Engine &amp; ViV Candidate Flow
        </h2>
      </div>

      <div className="grid md:grid-cols-[1.2fr_1fr] gap-6 text-sm">
        <div className="border border-slate-200 rounded-xl p-4 bg-slate-50">
          <h3 className="font-semibold mb-3 text-slate-900">
            Single‑patient pipeline
          </h3>
          <ol className="list-decimal list-inside space-y-2 text-slate-600">
            <li>Assign index procedure year, age, sex, and type (TAVR/SAVR).</li>
            <li>Draw risk category (low / intermediate / high risk).</li>
            <li>
              Sample survival time and valve durability (bimodal for TAVR, age‑
              split for SAVR).
            </li>
            <li>
              Determine failure year vs death year; candidate if failure occurs
              before death and within the horizon.
            </li>
          </ol>
          <p className="mt-3 text-xs text-slate-500">
            Durability and survival distributions are inherited from the
            Genereux/Ohno setup; we mainly change how many index procedures
            enter the engine each year.
          </p>
        </div>

        <div className="border border-slate-200 rounded-xl p-4 bg-white shadow-sm">
          <h3 className="font-semibold mb-3 text-slate-900">
            From failures to realised ViV
          </h3>
          <ul className="space-y-2 text-slate-600">
            <li>All valve failures in a year.</li>
            <li>→ Subset where patient is alive at failure.</li>
            <li>→ ViV candidates (alive + within forecast horizon).</li>
            <li>→ Subtract redo‑SAVR targets (risk‑based).</li>
            <li>→ ViV‑eligible pool.</li>
            <li>→ Apply chosen penetration scenario → realised ViV.</li>
          </ul>
          <p className="mt-3 text-xs text-slate-500">
            Key idea: candidates and eligible counts are data‑driven; penetration
            is explicitly a scenario, not embedded in the core forecast.
          </p>
        </div>
      </div>
    </div>
  ),

  // 10. CORE KOREAN RESULT: VIV CANDIDATES VS REALISED ----------------------
  Slide10: () => (
    <div className="flex flex-col h-full bg-white text-slate-900 p-10 md:p-14 animate-slide-enter">
      <div className="flex items-center gap-3 mb-4">
        <div className="p-2 rounded-full bg-emerald-100 text-emerald-600">
          <Icons.BarChart3 className="w-5 h-5" />
        </div>
        <h2 className="text-2xl md:text-3xl font-semibold text-slate-900">
          Korea: ViV Candidates vs Realised ViV
        </h2>
      </div>

      <div className="flex flex-col flex-1 min-h-0 gap-4">
        <div className="flex-1 min-h-0 rounded-lg overflow-hidden border border-slate-200 bg-slate-50 flex items-center justify-center">
          <img
            src={V6_DIAGRAMS.koreaVivCandidatesVsRealisedImageD}
            alt="Available ViV candidates vs realised ViV, 2022–2050"
            className="max-h-full max-w-full object-contain"
          />
        </div>
        <p className="text-xs text-slate-500 text-right">
          Grey bars: total candidates · Solid lines: TAVR‑in‑SAVR / TAVR‑in‑TAVR
          candidates · Dashed: realised ViV under US‑style penetration.
        </p>

        <div className="grid md:grid-cols-3 gap-4 text-sm">
          <div className="border border-slate-200 rounded-xl p-3 bg-slate-50">
            <h3 className="font-semibold mb-1 text-slate-900">
              2.5–3×, not 8×
            </h3>
            <p className="text-slate-600">
              Total ViV candidates rise from ~790 in 2024 to just under 2,000 in
              2035 — roughly a 2.5–3× increase driven by ageing, not explosive
              extrapolation.
            </p>
          </div>
          <div className="border border-slate-200 rounded-xl p-3 bg-white shadow-sm">
            <h3 className="font-semibold mb-1 text-slate-900">
              Split by ViV type
            </h3>
            <p className="text-slate-600">
              Both TAVR‑in‑SAVR and TAVR‑in‑TAVR increase, with TAVR‑in‑TAVR
              gradually taking a larger share of candidates and realised
              procedures.
            </p>
          </div>
          <div className="border border-slate-200 rounded-xl p-3 bg-slate-50">
            <h3 className="font-semibold mb-1 text-slate-900">
              Planning signal
            </h3>
            <p className="text-slate-600">
              Within 2025–2035, the demography‑driven increase is moderate but
              clinically significant — a more realistic planning baseline for
              Korean structural heart programmes.
            </p>
          </div>
        </div>
      </div>
    </div>
  ),

  // 11a. SUPPORTING: INDEX VOLUMES ------------------------------------------
  Slide11a: () => (
    <div className="flex flex-col h-full bg-white text-slate-900 p-10 md:p-14 animate-slide-enter">
      <div className="flex items-center gap-3 mb-4">
        <div className="p-2 rounded-full bg-slate-100 text-slate-600">
          <Icons.BarChart3 className="w-5 h-5" />
        </div>
        <h2 className="text-2xl md:text-3xl font-semibold text-slate-900">
          Supporting: Index Volumes (2015–2050)
        </h2>
      </div>

      <div className="flex flex-col flex-1 min-h-0 gap-4">
        <div className="flex-1 min-h-0 rounded-lg overflow-hidden border border-slate-200 bg-slate-50 flex items-center justify-center">
          <img
            src={V6_DIAGRAMS.koreaIndexProjectionFigure4}
            alt="Index TAVR and SAVR observed vs projected"
            className="max-h-full max-w-full object-contain"
          />
        </div>
        <ul className="space-y-1 text-slate-600 text-sm">
          <li>
            2023 spike is preserved but not used as a straight‑line
            extrapolation.
          </li>
          <li>
            Post‑2025, volumes evolve with demography; SAVR can plateau or
            decline naturally.
          </li>
          <li>
            Addresses editorial concerns about aggressive extrapolation on top
            of flat SAVR.
          </li>
        </ul>
      </div>
    </div>
  ),

  // 11b. SUPPORTING: RISK PROFILES ------------------------------------------
  Slide11b: () => (
    <div className="flex flex-col h-full bg-white text-slate-900 p-10 md:p-14 animate-slide-enter">
      <div className="flex items-center gap-3 mb-4">
        <div className="p-2 rounded-full bg-slate-100 text-slate-600">
          <Icons.BarChart3 className="w-5 h-5" />
        </div>
        <h2 className="text-2xl md:text-3xl font-semibold text-slate-900">
          Supporting: Per‑capita Risk Profiles
        </h2>
      </div>

      <div className="flex flex-col flex-1 min-h-0 gap-4">
        <div className="flex-1 min-h-0 rounded-lg overflow-hidden border border-slate-200 bg-slate-50 flex items-center justify-center">
          <img
            src={V6_DIAGRAMS.koreaPerCapitaRiskByAgeSex}
            alt="Per-capita TAVR/SAVR/redo-SAVR risk by age band & sex"
            className="max-h-full max-w-full object-contain"
          />
        </div>
        <ul className="space-y-1 text-slate-600 text-sm">
          <li>
            Risks concentrated in the oldest age bands (≥75, ≥80), with sex
            differences in some bands.
          </li>
          <li>
            Same framework used for redo‑SAVR to derive absolute yearly
            redo‑SAVR targets.
          </li>
          <li>
            These rates are exactly what we propagate into the future when
            computing index volumes.
          </li>
        </ul>
      </div>
    </div>
  ),

  // 12. EXTENSION TO SINGAPORE & UN PROJECTIONS -----------------------------
  Slide12: () => (
    <div className="flex flex-col h-full bg-white text-slate-900 p-10 md:p-14 animate-slide-enter">
      <div className="flex items-center gap-3 mb-4">
        <div className="p-2 rounded-full bg-blue-100 text-blue-600">
          <Icons.Globe2 className="w-5 h-5" />
        </div>
        <h2 className="text-2xl md:text-3xl font-semibold text-slate-900">
          Extension to Singapore &amp; UN Population Projections
        </h2>
      </div>

      <div className="grid md:grid-cols-2 gap-6 flex-1 min-h-0 pb-4">
        <div className="border border-slate-200 rounded-xl p-4 bg-slate-50 flex flex-col">
          <h3 className="font-semibold mb-3 text-slate-900">Korea (2022–2050)</h3>
          <div className="flex-1 rounded-lg overflow-hidden border border-slate-200 bg-slate-50 flex items-center justify-center">
            <img 
              src={V6_DIAGRAMS.koreaPopProjection} 
              alt="Korea Population Projection (Men)" 
              className="max-h-full max-w-full object-contain" 
            />
          </div>
        </div>

        <div className="border border-slate-200 rounded-xl p-4 bg-slate-50 flex flex-col">
          <h3 className="font-semibold mb-3 text-slate-900">Singapore (2022–2050)</h3>
          <div className="flex-1 rounded-lg overflow-hidden border border-slate-200 bg-slate-50 flex items-center justify-center">
            <img 
              src={V6_DIAGRAMS.singaporePopProjection} 
              alt="Singapore Population Projection (Men)" 
              className="max-h-full max-w-full object-contain" 
            />
          </div>
        </div>
      </div>
      
      <p className="text-center text-slate-500 text-sm mt-2">
        Comparing the "SAVR Generation" (70-74) vs "TAVI Generation" (80+) across both nations.
      </p>
    </div>
  ),

  // 13. PUBLISHING VALUE & LIMITATIONS --------------------------------------
  Slide13: () => (
    <div className="flex flex-col h-full bg-white text-slate-900 p-10 md:p-14 animate-slide-enter">
      <div className="flex items-center gap-3 mb-6">
        <div className="p-2 rounded-full bg-slate-100 text-slate-600">
          <Icons.FileText className="w-5 h-5" />
        </div>
        <h2 className="text-2xl md:text-3xl font-semibold text-slate-900">
          What This Model Offers · Publishing Value &amp; Limitations
        </h2>
      </div>

      <div className="grid md:grid-cols-2 gap-6 text-sm">
        <div className="border border-slate-200 rounded-xl p-4 bg-slate-50">
          <h3 className="font-semibold mb-2 text-slate-900">
            Contribution &amp; publication angle
          </h3>
          <ul className="space-y-2 text-slate-600">
            <li>
              <span className="font-semibold">
                Clinical planning signal (Korea):
              </span>{" "}
              demography‑driven ~3× rise in ViV candidates over a decade, not
              8×, with explicit TAVR‑in‑SAVR vs TAVR‑in‑TAVR splits.
            </li>
            <li>
              <span className="font-semibold">Methodological advance:</span>{" "}
              demography‑anchored risk × population design; candidates vs
              penetration clearly separated; redo‑SAVR handled consistently.
            </li>
            <li>
              <span className="font-semibold">
                Comparative insight (with Singapore):
              </span>{" "}
              same engine applied to two Asian countries with distinct
              demographies, as a constructive extension/critique of the US–Japan
              models.
            </li>
          </ul>
        </div>

        <div className="border border-slate-200 rounded-xl p-4 bg-white shadow-sm">
          <h3 className="font-semibold mb-2 text-slate-900">
            Key limitations &amp; uncertainty
          </h3>
          <ul className="space-y-2 text-slate-600">
            <li>Per‑capita risks assumed stationary at 2023–24 levels.</li>
            <li>
              Survival/durability curves taken from international literature,
              not re‑estimated in a Korean cohort.
            </li>
            <li>
              Device evolution, guideline shifts, and policy shocks not
              explicitly modelled.
            </li>
            <li>
              ViV penetration scenarios remain speculative, even though now
              clearly labelled as such.
            </li>
          </ul>
          <p className="mt-3 text-xs text-slate-500">
            The goal is transparency: make the mapping from assumptions → output
            explicit so that assumptions can be updated and the model rerun.
          </p>
        </div>
      </div>
    </div>
  ),

  // 14. FUTURE WORK: ADOPTION CURVES & WRAP-UP ------------------------------
  Slide14: () => (
    <div className="flex flex-col h-full bg-white text-slate-900 p-10 md:p-14 animate-slide-enter">
      <div className="flex items-center gap-3 mb-6">
        <div className="p-2 rounded-full bg-emerald-100 text-emerald-600">
          <Icons.TrendingUp className="w-5 h-5" />
        </div>
        <h2 className="text-2xl md:text-3xl font-semibold text-slate-900">
          Future Direction: Explicit Adoption Curves &amp; Wrap‑Up
        </h2>
      </div>

      <div className="grid md:grid-cols-[1.1fr_1fr] gap-6 text-sm mb-6">
        <div className="border border-slate-200 rounded-xl p-4 bg-slate-50">
          <h3 className="font-semibold mb-3 text-slate-900">
            From baseline volumes to adoption dynamics
          </h3>
          <ul className="space-y-2 text-slate-600">
            <li>
              Once baseline TAVR/SAVR volumes are anchored to demography, we can
              model adoption as explicit logistic (sigmoid) curves.
            </li>
            <li>
              Slow initial uptake → rapid growth → plateau, calibrated against
              historical Korean data and cross‑country patterns.
            </li>
            <li>
              Allows us to move beyond &quot;frozen&quot; per‑capita risk while
              keeping core demography logic intact.
            </li>
          </ul>
          <div className="mt-4 rounded-lg overflow-hidden border border-slate-200 bg-white h-32 flex items-center justify-center">
            <img
              src={V6_DIAGRAMS.sigmoidAdoptionCurveSchematic}
              alt="Sigmoid adoption curve schematic for TAVR adoption"
              className="max-h-full max-w-full object-contain"
            />
          </div>
        </div>

        <div className="border border-slate-200 rounded-xl p-4 bg-white shadow-sm">
          <h3 className="font-semibold mb-3 text-slate-900">
            Summary &amp; feedback we’re seeking
          </h3>
          <ul className="space-y-2 text-slate-600">
            <li>
              Does the demography‑anchored framing feel clinically intuitive and
              credible?
            </li>
            <li>
              Which scenarios (policy, device, adoption) should we prioritise in
              sensitivity analyses?
            </li>
            <li>
              Thoughts on target journals, conferences, and authorship structure
              for the Korea+Singapore paper.
            </li>
          </ul>
          <p className="mt-3 text-xs text-slate-500">
            This closes the current progress update; the next milestone is to
            complete the Singapore run and assemble a comparative manuscript.
          </p>
        </div>
      </div>
    </div>
  ),

  // 15. BASELINE ADOPTION ---------------------------------------------------
  Slide15: () => (
    <div className="flex flex-col h-full bg-white text-slate-900 p-10 md:p-14 animate-slide-enter">
      <div className="flex items-center gap-3 mb-6">
        <div className="p-2 rounded-full bg-blue-100 text-blue-600">
          <Icons.TrendingUp className="w-5 h-5" />
        </div>
        <h2 className="text-2xl md:text-3xl font-semibold text-slate-900">
          Baseline Adoption: Historical &amp; Projected
        </h2>
      </div>
      <div className="grid md:grid-cols-2 gap-6 h-full pb-10">
        <div className="border border-slate-200 rounded-xl p-4 bg-slate-50 flex flex-col">
          <h3 className="font-semibold mb-3 text-slate-900">Historical Adoption</h3>
          <div className="flex-1 rounded-lg overflow-hidden border border-slate-200 bg-slate-50 flex items-center justify-center">
            <img src={V6_DIAGRAMS.baselineAdoptionHistorical} alt="Historical Adoption" className="max-h-full max-w-full object-contain" />
          </div>
        </div>
        <div className="border border-slate-200 rounded-xl p-4 bg-slate-50 flex flex-col">
          <h3 className="font-semibold mb-3 text-slate-900">Projected Adoption</h3>
          <div className="flex-1 rounded-lg overflow-hidden border border-slate-200 bg-slate-50 flex items-center justify-center">
            <img src={V6_DIAGRAMS.baselineAdoptionProjected} alt="Projected Adoption" className="max-h-full max-w-full object-contain" />
          </div>
        </div>
      </div>
    </div>
  ),

  // 16. POPULATION PROJECTIONS ----------------------------------------------
  Slide16: () => (
    <div className="flex flex-col h-full bg-white text-slate-900 p-10 md:p-14 animate-slide-enter">
      <div className="flex items-center gap-3 mb-6">
        <div className="p-2 rounded-full bg-emerald-100 text-emerald-600">
          <Icons.Users className="w-5 h-5" />
        </div>
        <h2 className="text-2xl md:text-3xl font-semibold text-slate-900">
          Population Projections (2022–2070)
        </h2>
      </div>
      <div className="grid md:grid-cols-3 gap-4 h-full pb-10">
        <div className="border border-slate-200 rounded-xl p-3 bg-slate-50 flex flex-col">
          <h3 className="font-semibold mb-2 text-slate-900 text-sm">Men</h3>
          <div className="flex-1 rounded-lg overflow-hidden border border-slate-200 bg-slate-50 flex items-center justify-center">
            <img src={V6_DIAGRAMS.popProjMen} alt="Population Projection Men" className="max-h-full max-w-full object-contain" />
          </div>
        </div>
        <div className="border border-slate-200 rounded-xl p-3 bg-slate-50 flex flex-col">
          <h3 className="font-semibold mb-2 text-slate-900 text-sm">Women</h3>
          <div className="flex-1 rounded-lg overflow-hidden border border-slate-200 bg-slate-50 flex items-center justify-center">
            <img src={V6_DIAGRAMS.popProjWomen} alt="Population Projection Women" className="max-h-full max-w-full object-contain" />
          </div>
        </div>
        <div className="border border-slate-200 rounded-xl p-3 bg-slate-50 flex flex-col">
          <h3 className="font-semibold mb-2 text-slate-900 text-sm">Summary Trends</h3>
          <div className="flex-1 rounded-lg overflow-hidden border border-slate-200 bg-slate-50 flex items-center justify-center">
            <img src={V6_DIAGRAMS.popProjTrends} alt="Demography Summary Trends" className="max-h-full max-w-full object-contain" />
          </div>
        </div>
      </div>
    </div>
  ),

  // 17. REALISED VIV SCENARIOS ----------------------------------------------
  Slide17: () => (
    <div className="flex flex-col h-full bg-white text-slate-900 p-10 md:p-14 animate-slide-enter">
      <div className="flex items-center gap-3 mb-6">
        <div className="p-2 rounded-full bg-purple-100 text-purple-600">
          <Icons.BarChart3 className="w-5 h-5" />
        </div>
        <h2 className="text-2xl md:text-3xl font-semibold text-slate-900">
          Realised ViV Scenarios
        </h2>
      </div>
      <div className="grid md:grid-cols-2 gap-6 h-full pb-10">
        <div className="border border-slate-200 rounded-xl p-4 bg-slate-50 flex flex-col">
          <h3 className="font-semibold mb-3 text-slate-900">ViV Candidates vs Realised</h3>
          <div className="flex-1 rounded-lg overflow-hidden border border-slate-200 bg-slate-50 flex items-center justify-center">
            <img src={V6_DIAGRAMS.realisedVivCandidates} alt="ViV Candidates vs Realised" className="max-h-full max-w-full object-contain" />
          </div>
        </div>
        <div className="border border-slate-200 rounded-xl p-4 bg-slate-50 flex flex-col">
          <h3 className="font-semibold mb-3 text-slate-900">Realised ViV (Pretty)</h3>
          <div className="flex-1 rounded-lg overflow-hidden border border-slate-200 bg-slate-50 flex items-center justify-center">
            <img src={V6_DIAGRAMS.realisedVivPretty} alt="Realised ViV Pretty" className="max-h-full max-w-full object-contain" />
          </div>
        </div>
      </div>
    </div>
  ),

  // 18. WATERFALL PATHWAYS --------------------------------------------------
  Slide18: () => (
    <div className="flex flex-col h-full bg-white text-slate-900 p-10 md:p-14 animate-slide-enter">
      <div className="flex items-center gap-3 mb-6">
        <div className="p-2 rounded-full bg-orange-100 text-orange-600">
          <Icons.GitBranch className="w-5 h-5" />
        </div>
        <h2 className="text-2xl md:text-3xl font-semibold text-slate-900">
          Waterfall Pathways (TAVR-in-SAVR)
        </h2>
      </div>
      <div className="grid md:grid-cols-3 gap-4 h-full pb-10">
        <div className="border border-slate-200 rounded-xl p-3 bg-slate-50 flex flex-col">
          <h3 className="font-semibold mb-2 text-slate-900 text-sm">2025</h3>
          <div className="flex-1 rounded-lg overflow-hidden border border-slate-200 bg-slate-50 flex items-center justify-center">
            <img src={V6_DIAGRAMS.waterfall2025} alt="Waterfall 2025" className="max-h-full max-w-full object-contain" />
          </div>
        </div>
        <div className="border border-slate-200 rounded-xl p-3 bg-slate-50 flex flex-col">
          <h3 className="font-semibold mb-2 text-slate-900 text-sm">2030</h3>
          <div className="flex-1 rounded-lg overflow-hidden border border-slate-200 bg-slate-50 flex items-center justify-center">
            <img src={V6_DIAGRAMS.waterfall2030} alt="Waterfall 2030" className="max-h-full max-w-full object-contain" />
          </div>
        </div>
        <div className="border border-slate-200 rounded-xl p-3 bg-slate-50 flex flex-col">
          <h3 className="font-semibold mb-2 text-slate-900 text-sm">2035</h3>
          <div className="flex-1 rounded-lg overflow-hidden border border-slate-200 bg-slate-50 flex items-center justify-center">
            <img src={V6_DIAGRAMS.waterfall2035} alt="Waterfall 2035" className="max-h-full max-w-full object-contain" />
          </div>
        </div>
      </div>
    </div>
  ),

  // 19. SAVR EXPLANATION: RISK PROFILES -------------------------------------
  SAVR_Explanation_Risk: () => (
    <div className="flex flex-col h-full bg-white text-slate-900 p-10 md:p-14 animate-slide-enter">
      <div className="flex items-center gap-3 mb-4">
        <div className="p-2 rounded-full bg-rose-100 text-rose-600">
          <Icons.BarChart3 className="w-5 h-5" />
        </div>
        <h2 className="text-2xl md:text-3xl font-semibold text-slate-900">
          Why SAVR Baseline Decreases: 1. Distinct Risk Profiles
        </h2>
      </div>

      <div className="grid md:grid-cols-2 gap-6 flex-1 min-h-0 pb-4">
        <div className="border border-slate-200 rounded-xl p-4 bg-slate-50 flex flex-col">
          <h3 className="font-semibold mb-3 text-slate-900">SAVR Risk Profile</h3>
          <div className="flex-1 rounded-lg overflow-hidden border border-slate-200 bg-slate-50 flex items-center justify-center">
            <img 
              src="images/savr_bar_avg_risks.png" 
              alt="SAVR Risk by Age" 
              className="max-h-full max-w-full object-contain" 
            />
          </div>
          <p className="mt-3 text-sm text-slate-600">
            SAVR risk peaks in the <strong>70-74</strong> and <strong>75-79</strong> age bands.
            It drops significantly for patients 80+.
          </p>
        </div>

        <div className="border border-slate-200 rounded-xl p-4 bg-slate-50 flex flex-col">
          <h3 className="font-semibold mb-3 text-slate-900">TAVI Risk Profile</h3>
          <div className="flex-1 rounded-lg overflow-hidden border border-slate-200 bg-slate-50 flex items-center justify-center">
            <img 
              src="images/tavi_bar_avg_risks.png" 
              alt="TAVI Risk by Age" 
              className="max-h-full max-w-full object-contain" 
            />
          </div>
          <p className="mt-3 text-sm text-slate-600">
            TAVI risk is highest in the <strong>80+</strong> age bands (3-4x higher than SAVR).
            This age-dependence is critical.
          </p>
        </div>
      </div>
    </div>
  ),

  // 20. SAVR EXPLANATION: DEMOGRAPHIC SHIFT ---------------------------------
  SAVR_Explanation_Demog: () => (
    <div className="flex flex-col h-full bg-white text-slate-900 p-10 md:p-14 animate-slide-enter">
      <div className="flex items-center gap-3 mb-4">
        <div className="p-2 rounded-full bg-emerald-100 text-emerald-600">
          <Icons.Users className="w-5 h-5" />
        </div>
        <h2 className="text-2xl md:text-3xl font-semibold text-slate-900">
          Why SAVR Baseline Decreases: 2. Demographic Shift
        </h2>
      </div>

      <div className="flex flex-col flex-1 min-h-0 gap-4">
        <div className="flex-1 min-h-0 rounded-lg overflow-hidden border border-slate-200 bg-slate-50 flex items-center justify-center">
          <img 
            src="images/population_projections_korea/2022-2050/age_projection_lines_Combined.png" 
            alt="Population Projection Lines Men" 
            className="max-h-full max-w-full object-contain" 
          />
        </div>
        
        <div className="border border-slate-200 rounded-xl p-4 bg-slate-50">
          <h3 className="font-semibold mb-2 text-slate-900 text-lg">The "SAVR Generation" Shrinks</h3>
          <ul className="space-y-2 text-slate-600 text-sm">
            <li>
              <strong className="text-emerald-600">70-74 Age Cohort:</strong> This group drives SAVR volume. It peaks around <strong>2042</strong> and then <strong>decreases</strong> significantly by 2050.
            </li>
            <li>
              <strong className="text-blue-600">80+ Age Cohort:</strong> This group drives TAVI volume. It continues to <strong>grow rapidly</strong> through 2050.
            </li>
            <li className="pt-2 border-t border-slate-200">
              <strong>Result:</strong> Even though the <em>average</em> age increases, the specific bucket for SAVR shrinks, causing the baseline volume decline.
            </li>
          </ul>
        </div>
      </div>
    </div>
  ),
};



// --- VERSION 6 SLIDES ---

// import { V6_DIAGRAMS } from "./v6-diagramConfig";
// import { Icons } from "./Icons"; // adjust to your actual Icons import


const V6_Slides = [
  // 1. Introduction & Plan
  { component: V6.Slide1, label: "Title", section: "1. Intro" },
  { component: V6.ContentsSlide, label: "Contents", section: "1. Intro" },
  
  { component: V6.Section_Intro, label: "1. Intro", section: "1. Intro" },
  { component: V6.Slide3, label: "Objective", section: "1. Intro" },
  { component: V6.Slide2, label: "Where We Are", section: "1. Intro" },
  { component: V6.Slide4a, label: "Genereux (US)", section: "1. Intro" },
  { component: V6.Slide4b, label: "Ohno (Japan)", section: "1. Intro" },

  // 2. Initial Findings
  { component: V6.Section_Findings, label: "2. Findings", section: "2. Findings" },
  { component: V6.Slide6a, label: "Korea Data", section: "2. Findings" },
  { component: V6.Slide6b, label: "Old Forecast", section: "2. Findings" },

  // 3. The Pivot
  { component: V6.Section_Pivot, label: "3. Pivot", section: "3. Pivot" },
  { component: V6.Slide_Problem, label: "Why It Failed", section: "3. Pivot" },
  { component: V6.Slide5, label: "Critique", section: "3. Pivot" },
  { component: V6.Slide_Concept, label: "The Pivot", section: "3. Pivot" },

  // 4. New Approach
  { component: V6.Section_Approach, label: "4. Approach", section: "4. Approach" },
  { component: V6.Slide7, label: "Design Goals", section: "4. Approach" },
  { component: V6.Slide_UN_Data, label: "UN Data Source", section: "4. Approach" },
  { component: V6.Slide8a, label: "Demography Formula", section: "4. Approach" },
  { component: V6.Slide8b, label: "Demography Trends", section: "4. Approach" },
  { component: V6.Slide_Methodology, label: "Architecture", section: "4. Approach" },
  { component: V6.Slide_Mechanism, label: "The Race", section: "4. Approach" },
  { component: V6.Slide_Logic, label: "Distributions", section: "4. Approach" },
  { component: V6.Slide9, label: "Monte Carlo & Flow", section: "4. Approach" },

  // 5. Results
  { component: V6.Section_Results, label: "5. Results", section: "5. Results" },
  { component: V6.Slide10, label: "Korea Results", section: "5. Results" },
  { component: V6.Slide_Results, label: "Comparison", section: "5. Results" },
  { component: V6.Slide11a, label: "Index Volumes", section: "5. Results" },
  { component: V6.SAVR_Explanation_Risk, label: "SAVR Risk Profile", section: "5. Results" },
  { component: V6.SAVR_Explanation_Demog, label: "Demographic Shift", section: "5. Results" },
  { component: V6.Slide12, label: "Singapore & UN", section: "5. Results" },
  { component: V6.Slide13, label: "Value & Limitations", section: "5. Results" },
  { component: V6.Slide14, label: "Adoption & Wrap-Up", section: "5. Results" },
  
  // Appendix / Extras
  { component: V6.Slide15, label: "Baseline Adoption", section: "Appendix" },
  { component: V6.Slide16, label: "Pop Projections", section: "Appendix" },
  { component: V6.Slide17, label: "Realised ViV", section: "Appendix" },
  { component: V6.Slide18, label: "Waterfalls", section: "Appendix" },
];



// --- VERSION 5 SLIDES ---
const V5 = {
    Slide1: () => (
    <div className="flex flex-col items-center justify-center h-full bg-slate-900 text-white p-12 text-center animate-slide-enter">
        <div className="mb-6 p-4 bg-blue-600 rounded-full shadow-lg shadow-blue-800/40">
        <Icons.Activity />
        </div>

        <div className="text-xs font-mono text-blue-300 mb-3 tracking-[0.25em]">
        VIV–TAVR FORECASTING · PROGRESS UPDATE
        </div>

        <h1 className="text-4xl md:text-5xl font-bold mb-4">
        Demography‑Anchored Forecasting of ViV‑TAVR
        </h1>

        <p className="text-lg md:text-xl text-slate-300 mb-8 max-w-3xl">
        Korea today, Singapore next: a refreshed Monte Carlo framework grounded in
        registry data and national demography.
        </p>

        <div className="border-t border-slate-700 pt-6 w-full max-w-3xl flex flex-col md:flex-row items-center md:justify-between gap-2 text-sm text-slate-400">
        <span>Hyunjin Ahn · Charles Yap</span>
        <span>Department of Cardiology / Cardiac Surgery</span>
        <span className="text-emerald-400 font-medium">
            Progress update &amp; refreshed modelling strategy
        </span>
        </div>
    </div>
    ),
    
    Slide2: () => (
    <div className="p-12 h-full flex flex-col animate-slide-enter">
        <h2 className="text-3xl font-bold text-slate-800 mb-6">
        Clinical &amp; Modelling Objective
        </h2>

        <div className="grid md:grid-cols-2 gap-8 flex-1">
        {/* Text side */}
        <div className="flex flex-col">
            <div className="bg-slate-50 p-5 rounded-xl border border-slate-200 mb-5">
            <h3 className="text-base font-semibold text-blue-700 mb-2">
                Goal:
            </h3>
            <p className="text-slate-700 text-sm">
                Estimate future volumes of valve‑in‑valve (ViV) TAVR in Korea (and
                later Singapore), split into TAVR‑in‑SAVR and TAVR‑in‑TAVR, using
                registry data plus a patient‑level Monte Carlo model.
            </p>
            </div>

            <h4 className="font-semibold text-slate-700 mb-2 text-sm uppercase tracking-wide">
            What the Model Must Deliver
            </h4>
            <ul className="list-disc pl-6 space-y-2 text-sm text-slate-700">
            <li>
                <span className="font-semibold">
                Future annual ViV‑TAVR volume in Korea,
                </span>{" "}
                with extension to Singapore using the same framework.
            </li>
            <li>
                Clear separation of{" "}
                <span className="font-semibold">TAVR‑in‑SAVR</span> vs{" "}
                <span className="font-semibold">TAVR‑in‑TAVR</span> trajectories.
            </li>
            <li>
                Inputs: registry‑level TAVI / SAVR / redo‑SAVR counts and
                demography; Outputs: ViV candidates &amp; realised ViV per year.
            </li>
            </ul>
        </div>

        {/* Schematic side */}
        <div className="flex items-center justify-center">
            <div className="bg-slate-50 border border-slate-200 rounded-2xl p-6 w-full max-w-md">
            <h3 className="text-sm font-semibold text-slate-700 mb-4 uppercase tracking-wide">
                Schematic: Two Paths to ViV
            </h3>

            <div className="space-y-4 text-xs text-slate-700">
                {/* SAVR path */}
                <div className="flex items-center">
                <div className="flex-1">
                    <div className="inline-flex items-center px-2 py-1 rounded-md bg-slate-900 text-slate-50 text-[11px] font-semibold">
                    SAVR (Bioprosthetic)
                    </div>
                </div>
                <div className="w-10 text-center text-slate-400 text-lg">
                    →
                </div>
                <div className="flex-1 flex flex-col items-center">
                    <div className="w-full bg-amber-50 border border-amber-200 rounded-md px-2 py-1 text-[11px]">
                    Structural valve degeneration
                    </div>
                    <div className="text-[10px] text-slate-400 mt-1">
                    durability curve
                    </div>
                </div>
                <div className="w-10 text-center text-slate-400 text-lg">
                    →
                </div>
                <div className="flex-1">
                    <div className="inline-flex items-center px-2 py-1 rounded-md bg-emerald-100 text-emerald-900 text-[11px] font-semibold">
                    ViV TAVI (TAVR‑in‑SAVR)
                    </div>
                </div>
                </div>

                {/* TAVR path */}
                <div className="flex items-center">
                <div className="flex-1">
                    <div className="inline-flex items-center px-2 py-1 rounded-md bg-slate-800 text-slate-50 text-[11px] font-semibold">
                    Index TAVI
                    </div>
                </div>
                <div className="w-10 text-center text-slate-400 text-lg">
                    →
                </div>
                <div className="flex-1 flex flex-col items-center">
                    <div className="w-full bg-amber-50 border border-amber-200 rounded-md px-2 py-1 text-[11px]">
                    TAVR valve failure
                    </div>
                    <div className="text-[10px] text-slate-400 mt-1">
                    bimodal durability
                    </div>
                </div>
                <div className="w-10 text-center text-slate-400 text-lg">
                    →
                </div>
                <div className="flex-1">
                    <div className="inline-flex items-center px-2 py-1 rounded-md bg-indigo-100 text-indigo-900 text-[11px] font-semibold">
                    ViV TAVI (TAVR‑in‑TAVR)
                    </div>
                </div>
                </div>

                <p className="mt-4 text-[11px] text-slate-500">
                A Monte Carlo engine combines registry volumes, survival curves,
                and durability distributions to estimate how many patients reach
                these ViV stages each year.
                </p>
            </div>
            </div>
        </div>
        </div>
    </div>
    ),
    
    Slide3: () => (
    <div className="p-12 h-full flex flex-col animate-slide-enter">
        <h2 className="text-3xl font-bold text-slate-800 mb-6">
        Prior Work: Genereux &amp; Ohno Models
        </h2>

        <div className="grid md:grid-cols-2 gap-8 flex-1">
        {/* Thumbnails */}
        <div className="space-y-6">
            <div className="bg-slate-50 rounded-xl border border-slate-200 p-4">
            <h3 className="text-sm font-semibold text-slate-700 mb-2">
                Genereux et al. – US TVT Registry
            </h3>
            <div className="bg-white border border-slate-200 rounded-lg h-44 flex items-center justify-center overflow-hidden">
                <img
                src="images/genereux_us_viv_forecast.png"
                alt="Genereux US ViV forecast plot"
                className="w-full h-full object-contain"
                />
            </div>
            <p className="mt-2 text-[11px] text-slate-500">
                Forecast of ViV volume in the US based on TVT registry data, volume
                extrapolation, and penetration curves.
            </p>
            </div>

            <div className="bg-slate-50 rounded-xl border border-slate-200 p-4">
            <h3 className="text-sm font-semibold text-slate-700 mb-2">
                Ohno &amp; Genereux – Japan vs USA
            </h3>
            <div className="bg-white border border-slate-200 rounded-lg h-44 flex items-center justify-center overflow-hidden">
                <img
                src="images/ohno_us_japan_comparison.png"
                alt="Ohno JACC Asia comparison plot"
                className="w-full h-full object-contain"
                />
            </div>
            <p className="mt-2 text-[11px] text-slate-500">
                JACC Asia conference paper comparing projected ViV volumes in Japan
                and the US.
            </p>
            </div>
        </div>

        {/* Bullet summary */}
        <div className="flex flex-col">
            <h3 className="text-sm font-semibold text-slate-700 mb-3 uppercase tracking-wide">
            Shared Core Methodology
            </h3>
            <ul className="list-disc pl-6 space-y-2 text-sm text-slate-700 mb-4">
            <li>Use historical index volumes (TAVR and SAVR) by age.</li>
            <li>Project those index volumes forward by extrapolation.</li>
            <li>Run a Monte Carlo engine for durability, survival, and failure.</li>
            <li>
                Apply assumed ViV penetration curves (e.g. 10%→60% uptake) to
                failures to obtain future ViV volume.
            </li>
            </ul>

            <div className="mt-auto bg-amber-50 border border-amber-100 rounded-lg p-4 text-xs text-amber-900">
            <p>
                These models provided our starting blueprint: we reuse the
                patient‑level Monte Carlo structure, but will later change how index
                volumes and penetration are handled for Korea.
            </p>
            </div>
        </div>
        </div>
    </div>
    ),
    
    Slide4: () => (
    <div className="p-12 h-full flex flex-col animate-slide-enter">
        <h2 className="text-3xl font-bold text-slate-800 mb-4">
        Replicating the Original Approach for Korea
        </h2>

        <p className="text-sm text-slate-700 mb-4 max-w-3xl">
        Our first iteration (roughly <span className="font-mono">model_v7.py</span>)
        ported the Genereux / Ohno methodology directly to Korean registry data up
        to 2023.
        </p>

        <div className="grid md:grid-cols-2 gap-8 flex-1">
        <div className="bg-slate-50 rounded-xl border border-slate-200 p-6 flex flex-col">
            <h3 className="text-sm font-semibold text-slate-700 mb-3 uppercase tracking-wide">
            Design Choices in the First Pass
            </h3>
            <ul className="list-disc pl-6 space-y-2 text-sm text-slate-700">
            <li>
                <span className="font-semibold">Index SAVR</span>: held fixed after
                2023 at a constant volume.
            </li>
            <li>
                <span className="font-semibold">Index TAVR</span>: extrapolated
                upward based on recent annual growth (linear / constant options).
            </li>
            <li>
                <span className="font-semibold">ViV penetration</span>: assumed to
                ramp from ~10% in 2022 to ~60% in 2035, borrowing US patterns.
            </li>
            <li>Inputs restricted to HIRA registry data up to 2023.</li>
            </ul>
            <p className="mt-4 text-xs text-slate-500">
            At first glance, this produced ViV forecasts with similar shape to the
            published US/Japan curves, though absolute numbers were initially
            understated due to an input reporting bug.
            </p>
        </div>

        <div className="bg-slate-50 rounded-xl border border-slate-200 p-6 flex flex-col">
            <h3 className="text-sm font-semibold text-slate-700 mb-3 uppercase tracking-wide">
            Korean Replication Plot (First Pass)
            </h3>
            <div className="flex-1 bg-white border border-slate-200 rounded-lg flex items-center justify-center overflow-hidden">
            <img
                src="images/korea_viv_forecast_v7_replication.png"
                alt="First-pass Korean ViV forecast replicating Genereux/Ohno"
                className="w-full h-64 object-contain"
            />
            </div>
            <p className="mt-2 text-[11px] text-slate-500">
            First-pass ViV forecast for Korea using extrapolated index volumes and
            a 10%→60% penetration curve, pre‑bug‑fix.
            </p>
        </div>
        </div>
    </div>
    ),
    
    Slide5: () => (
    <div className="p-12 h-full flex flex-col animate-slide-enter">
        <h2 className="text-3xl font-bold text-slate-800 mb-6">
        External Critique of the Standard Approach
        </h2>

        <div className="grid md:grid-cols-2 gap-8 flex-1">
        {/* Quote box */}
        <div className="bg-slate-900 text-slate-50 rounded-2xl p-6 flex flex-col">
            <h3 className="text-sm font-semibold text-amber-300 mb-3 uppercase tracking-wide">
            JACC Asia Editorial on Ohno et al.
            </h3>
            <div className="bg-slate-800/60 rounded-xl p-5 flex-1 flex items-center">
            <p className="text-sm leading-relaxed">
                <span className="text-amber-300">“</span>
                Assumptions at various levels and different magnitude were made…
                extrapolations were used to project potential procedural volumes…
                such adjustments and extrapolations may potentially misrepresent the
                true picture.
                <span className="text-amber-300">”</span>
            </p>
            </div>
            <p className="mt-3 text-[11px] text-slate-400">
            (Valve-in-Valve Transcatheter Aortic Valve Replacement (ViV‑TAVR): The
            Past, Present, and Future – editorial comment on Ohno et al.)
            </p>
        </div>

        {/* Bullet summary */}
        <div className="bg-slate-50 rounded-2xl border border-slate-200 p-6 flex flex-col">
            <h3 className="text-sm font-semibold text-slate-700 mb-3 uppercase tracking-wide">
            Key Concerns Highlighted
            </h3>
            <ul className="list-disc pl-6 space-y-2 text-sm text-slate-700 mb-4">
            <li>
                Heavy reliance on <span className="font-semibold">extrapolated</span>{" "}
                index TAVR / SAVR volumes.
            </li>
            <li>
                SAVR projected to be{" "}
                <span className="font-semibold">plateauing / flat</span> instead of
                allowing decline or other trajectories.
            </li>
            <li>
                Limited handling of{" "}
                <span className="font-semibold">
                clinical factors, patient characteristics, and demography
                </span>
                .
            </li>
            </ul>

            <div className="mt-auto bg-red-50 border border-red-100 rounded-lg p-4 text-xs text-red-900">
            <p>
                These criticisms mirror the issues we started to see in our Korean
                replication: extrapolation layered on top of atypical recent years
                (e.g. 2023 spike) risks giving a misleading picture of future ViV
                demand.
            </p>
            </div>
        </div>
        </div>
    </div>
    ),
    
    Slide6: () => (
    <div className="p-12 h-full flex flex-col animate-slide-enter">
        <h2 className="text-3xl font-bold text-slate-800 mb-4">
        Our Discomfort with the First Pass (Korean Data)
        </h2>

        <div className="grid md:grid-cols-2 gap-8 flex-1">
        {/* Plot of index TAVR/SAVR */}
        <div className="bg-slate-50 rounded-xl border border-slate-200 p-6 flex flex-col">
            <h3 className="text-sm font-semibold text-slate-700 mb-3 uppercase tracking-wide">
            Index TAVR / SAVR Volumes 2015–2024
            </h3>
            <div className="flex-1 bg-white border border-slate-200 rounded-lg flex items-center justify-center overflow-hidden">
            <img
                src="images/korea_tavi_savr_index_2015_2024.png"
                alt="Index TAVR and SAVR volumes 2015–2024 with 2023 spike"
                className="w-full h-64 object-contain"
            />
            </div>
            <p className="mt-2 text-[11px] text-slate-500">
            Note the sharp spike in 2023 and partial regression in 2024.
            </p>
        </div>

        {/* Bullet explanations */}
        <div className="flex flex-col">
            <h3 className="text-sm font-semibold text-slate-700 mb-3 uppercase tracking-wide">
            Korea‑Specific Issues
            </h3>
            <ul className="list-disc pl-6 space-y-2 text-sm text-slate-700 mb-4">
            <li>
                <span className="font-semibold">2023 spike</span> in TAVR and SAVR
                volumes, followed by a drop in 2024 (still above 2022).
            </li>
            <li>
                Likely blend of{" "}
                <span className="font-semibold">post‑COVID backlog</span> and{" "}
                <span className="font-semibold">insurance policy changes</span>{" "}
                around 2024 influencing procedure timing.
            </li>
            <li>
                Korea has one of the{" "}
                <span className="font-semibold">fastest ageing populations</span> in
                the world – a crucial driver for a therapy concentrated in the
                oldest age bands.
            </li>
            </ul>

            <div className="mt-auto bg-blue-50 border border-blue-100 rounded-lg p-4 text-xs text-blue-900">
            <p>
                Conclusion: simply fitting a straight line through historical
                volumes and extending it – as in the US/Japan models – is
                inappropriate for Korea. We need a framework that starts from
                demography and per‑capita risk, not raw volume extrapolation.
            </p>
            </div>
        </div>
        </div>
    </div>
    ),
    
    Slide7: () => (
    <div className="p-12 h-full flex flex-col animate-slide-enter">
        <h2 className="text-3xl font-bold text-slate-800 mb-6">
        Design Goals for the New Model
        </h2>

        <div className="grid md:grid-cols-3 gap-6 mb-8">
        <div className="bg-slate-50 rounded-xl border border-slate-200 p-5">
            <div className="text-xs font-semibold text-blue-700 mb-2 uppercase tracking-wide">
            Goal 1
            </div>
            <h3 className="text-sm font-semibold text-slate-800 mb-2">
            Anchor to Demography
            </h3>
            <p className="text-sm text-slate-700">
            Make age and sex structure explicit. Drive index TAVR/SAVR volumes via{" "}
            <span className="font-semibold">per‑capita risk × population</span>,
            not arbitrary regression lines.
            </p>
        </div>

        <div className="bg-slate-50 rounded-xl border border-slate-200 p-5">
            <div className="text-xs font-semibold text-blue-700 mb-2 uppercase tracking-wide">
            Goal 2
            </div>
            <h3 className="text-sm font-semibold text-slate-800 mb-2">
            Minimise Speculative Inputs
            </h3>
            <p className="text-sm text-slate-700">
            Especially for ViV penetration curves that cannot be calibrated in
            Korea. Report <span className="font-semibold">candidates</span> first;
            treat penetration as an optional scenario layer.
            </p>
        </div>

        <div className="bg-slate-50 rounded-xl border border-slate-200 p-5">
            <div className="text-xs font-semibold text-blue-700 mb-2 uppercase tracking-wide">
            Goal 3
            </div>
            <h3 className="text-sm font-semibold text-slate-800 mb-2">
            Preserve Patient‑Level Monte Carlo
            </h3>
            <p className="text-sm text-slate-700">
            Retain the Genereux / Ohno style Monte Carlo engine so that durability
            and survival remain modelled at the individual patient level.
            </p>
        </div>
        </div>

        {/* Old vs new schematic */}
        <div className="flex-1 flex flex-col md:flex-row gap-6">
        <div className="flex-1 bg-slate-900 text-slate-50 rounded-xl p-5">
            <h3 className="text-xs font-semibold text-amber-300 mb-3 uppercase tracking-wide">
            &quot;Old&quot; Approach
            </h3>
            <ol className="list-decimal pl-5 space-y-2 text-sm">
            <li>Take historical TAVR / SAVR volumes.</li>
            <li>Fit linear trend / hold SAVR flat.</li>
            <li>Project volumes forward indefinitely.</li>
            <li>Apply ViV penetration curves to failures.</li>
            </ol>
        </div>
        <div className="flex items-center justify-center text-slate-500">
            <span className="text-3xl">⇨</span>
        </div>
        <div className="flex-1 bg-emerald-50 text-slate-900 rounded-xl p-5">
            <h3 className="text-xs font-semibold text-emerald-800 mb-3 uppercase tracking-wide">
            &quot;New&quot; Demography‑Anchored Approach
            </h3>
            <ol className="list-decimal pl-5 space-y-2 text-sm">
            <li>
                Estimate age‑ and sex‑specific{" "}
                <span className="font-semibold">per‑capita risks</span> (TAVI,
                SAVR, redo‑SAVR) from 2023–24 registry data.
            </li>
            <li>
                Combine those with national{" "}
                <span className="font-semibold">population projections</span> to get
                index volumes by year.
            </li>
            <li>
                Run patient‑level Monte Carlo to obtain{" "}
                <span className="font-semibold">ViV candidates</span>.
            </li>
            <li>
                Optionally overlay ViV penetration scenarios to obtain{" "}
                <span className="font-semibold">realised</span> ViV.
            </li>
            </ol>
        </div>
        </div>
    </div>
    ),
    
    Slide8: () => (
    <div className="p-12 h-full flex flex-col animate-slide-enter">
        <h2 className="text-3xl font-bold text-slate-800 mb-6">
        Pipeline Overview – <span className="font-mono">model_v9</span>
        </h2>

        <div className="bg-slate-50 border border-slate-200 rounded-2xl p-6 flex-1 flex flex-col">
        {/* High-level pipeline diagram */}
        <div className="flex flex-col lg:flex-row items-stretch gap-4 mb-6">
            {/* Precompute */}
            <div className="flex-1 bg-blue-50 border border-blue-200 rounded-xl p-4">
            <h3 className="text-xs font-semibold text-blue-800 mb-2 uppercase tracking-wide">
                Stage 1 · Precompute
            </h3>
            <ul className="list-disc pl-5 space-y-2 text-xs text-slate-800">
                <li>
                Read Korean population projections and rebuild{" "}
                <span className="font-semibold">
                    year × sex × age‑band / age
                </span>{" "}
                tables.
                </li>
                <li>
                Compute 2023–24{" "}
                <span className="font-semibold">
                    per‑capita risks of TAVI, SAVR, and redo‑SAVR
                </span>{" "}
                for each age–sex band.
                </li>
                <li>
                Project absolute{" "}
                <span className="font-semibold">redo‑SAVR targets</span> by
                applying those risks to future demography.
                </li>
            </ul>
            </div>

            {/* Simulation */}
            <div className="flex-1 bg-purple-50 border border-purple-200 rounded-xl p-4">
            <h3 className="text-xs font-semibold text-purple-800 mb-2 uppercase tracking-wide">
                Stage 2 · Monte Carlo Simulation
            </h3>
            <ul className="list-disc pl-5 space-y-2 text-xs text-slate-800">
                <li>
                Generate index TAVR/SAVR counts each year as{" "}
                <span className="font-semibold">risk × population</span>,
                stratified by age and sex.
                </li>
                <li>
                For each index: draw{" "}
                <span className="font-semibold">durability</span> and{" "}
                <span className="font-semibold">survival</span>, compute failure
                year and death year.
                </li>
                <li>
                Identify <span className="font-semibold">ViV candidates</span>{" "}
                (failed valves in living patients within the horizon).
                </li>
                <li>
                Subtract projected redo‑SAVR; optionally apply{" "}
                <span className="font-semibold">ViV penetration</span> scenarios
                to obtain realised ViV.
                </li>
            </ul>
            </div>

            {/* Outputs */}
            <div className="flex-1 bg-emerald-50 border border-emerald-200 rounded-xl p-4">
            <h3 className="text-xs font-semibold text-emerald-800 mb-2 uppercase tracking-wide">
                Stage 3 · Outputs &amp; QC
            </h3>
            <ul className="list-disc pl-5 space-y-2 text-xs text-slate-800">
                <li>
                Tables: index projections, per‑run ViV candidates &amp; realised
                counts, patient flow summaries.
                </li>
                <li>
                Figures: index series, ViV line charts (pre/post redo), population
                and risk heatmaps, waterfall plots.
                </li>
                <li>
                Image C &amp; Image D: headline ViV forecasts used in the slides
                later in this talk.
                </li>
            </ul>
            </div>
        </div>

        <p className="text-[11px] text-slate-500">
            Implementation lives in{" "}
            <span className="font-mono">model_v9.py</span> with outputs organised
            under <span className="font-mono">runs/model_v9_forecasting/&lt;timestamp&gt;/</span>{" "}
            (subfolders for <span className="font-mono">tables/</span>,{" "}
            <span className="font-mono">figures/</span>,{" "}
            <span className="font-mono">qc/</span>, etc.).
        </p>
        </div>
    </div>
    ),

    Slide9: () => (
    <div className="p-12 h-full flex flex-col animate-slide-enter">
        <h2 className="text-3xl font-bold text-slate-800 mb-6">
        Demography Precompute · Korean Population Reconstruction
        </h2>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 flex-1">
        {/* Left: process bullets */}
        <div className="bg-slate-50 border border-slate-200 rounded-xl p-6 flex flex-col">
            <h3 className="text-sm font-semibold text-slate-700 mb-3">
            Input: Ministry Population Table
            </h3>
            <ul className="list-disc pl-6 space-y-2 text-sm text-slate-700">
            <li>
                Source file:{" "}
                <span className="font-mono text-xs">
                korea_population_combined.csv
                </span>
                .
            </li>
            <li>
                Broad age buckets:{" "}
                <span className="font-semibold">50–64, ≥65, ≥70, ≥75, ≥85</span>{" "}
                plus sex ratios.
            </li>
            <li>
                Projection years: 2025, 2030, 2040, 2050 with linear interpolation
                in between.
            </li>
            </ul>

            <h3 className="text-sm font-semibold text-slate-700 mt-6 mb-2">
            How we refine it
            </h3>
            <ul className="list-disc pl-6 space-y-2 text-sm text-slate-700">
            <li>
                Split <span className="font-semibold">50–64</span> into{" "}
                <span className="font-mono text-xs">50–54, 55–59, 60–64</span>{" "}
                using configurable shares.
            </li>
            <li>
                Split <span className="font-semibold">75–84</span> into{" "}
                <span className="font-mono text-xs">75–79</span> and{" "}
                <span className="font-mono text-xs">80–84</span>.
            </li>
            <li>
                Expand <span className="font-semibold">≥85</span> into an open-ended
                band for very old patients.
            </li>
            <li>
                Build:
                <ul className="list-disc pl-5 mt-1 text-xs text-slate-600 space-y-1">
                <li>
                    <span className="font-mono">year × sex × age_band</span> table
                    (5-year bands, 50–54…≥85).
                </li>
                <li>
                    <span className="font-mono">year × sex × age</span> table
                    (single-year ages from 50+).
                </li>
                </ul>
            </li>
            </ul>

            <div className="mt-4 text-xs text-slate-600">
            Result: for each year (≈2022–2050), we know exactly how many men and
            women are in each age band and at each single-year age. This is the
            backbone for all risk × population calculations.
            </div>
        </div>

        {/* Right: Figure 3 population plots */}
        <div className="bg-white border border-slate-200 rounded-xl p-6 flex flex-col">
            <h3 className="text-sm font-semibold text-slate-700 mb-3 text-center">
            Figure 3 · Projected Population Trends in Korea
            </h3>
            <div className="flex-1 flex flex-col lg:flex-row items-center justify-center gap-4">
            <div className="w-full lg:w-1/2">
                <img
                src="images/age_projection_lines_Men.png"
                alt="Projected Korean male population by age band"
                className="w-full h-40 md:h-56 object-contain"
                />
                <p className="mt-1 text-[11px] text-slate-500 text-center">
                Men: 50–54 … ≥85 (lines interpolated between 2025, 2030, 2040,
                2050).
                </p>
            </div>
            <div className="w-full lg:w-1/2">
                <img
                src="images/age_projection_lines_Women.png"
                alt="Projected Korean female population by age band"
                className="w-full h-40 md:h-56 object-contain"
                />
                <p className="mt-1 text-[11px] text-slate-500 text-center">
                Women: similar expansion, highlighting rapid growth in ≥75 and
                ≥85.
                </p>
            </div>
            </div>
            <div className="mt-3 text-[11px] text-slate-500 text-center">
            Underneath, we also maintain a{" "}
            <span className="font-mono">year × age</span> table (men+women) for
            single-year modelling and heatmaps.
            </div>
        </div>
        </div>
    </div>
    ),

    Slide10: () => (
    <div className="p-12 h-full flex flex-col animate-slide-enter">
        <h2 className="text-3xl font-bold text-slate-800 mb-6">
        Baseline Per-Capita Risk of Procedures
        </h2>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 flex-1">
        {/* Left: method */}
        <div className="bg-slate-50 border border-slate-200 rounded-xl p-6 flex flex-col">
            <h3 className="text-sm font-semibold text-slate-700 mb-3">
            Step 1 · Join Registry Counts with Population
            </h3>
            <ul className="list-disc pl-6 space-y-2 text-sm text-slate-700">
            <li>
                Take HIRA registry for <span className="font-semibold">2023</span>{" "}
                and <span className="font-semibold">2024</span>.
            </li>
            <li>
                For each of TAVI, SAVR, redo-SAVR, we use{" "}
                <span className="font-mono text-xs">
                year × sex × age_band × count
                </span>{" "}
                tables.
            </li>
            <li>
                Match these to our <span className="font-mono">year × sex × age_band</span>{" "}
                population table.
            </li>
            </ul>

            <h3 className="text-sm font-semibold text-slate-700 mt-6 mb-2">
            Step 2 · Compute Per-Capita Risk
            </h3>
            <div className="bg-white border border-slate-200 rounded-lg p-4 text-xs text-slate-700">
            <p className="mb-2">
                For each procedure type, year, sex, and age band:
            </p>
            <p className="font-mono text-[11px] bg-slate-900 text-emerald-200 px-3 py-2 rounded inline-block">
                risk = procedures / population
            </p>
            <p className="mt-2">
                i.e. the probability that a person in that age–sex band has TAVI /
                SAVR / redo-SAVR in that calendar year.
            </p>
            </div>

            <div className="mt-4 text-xs text-slate-600">
            We deliberately restrict to{" "}
            <span className="font-semibold">2023–2024</span> risks, as these years
            best reflect post-COVID backlog resolution, current reimbursement, and
            mature TAVR adoption in Korea.
            </div>
        </div>

        {/* Right: Figure 5 risk heatmaps / bars */}
        <div className="bg-white border border-slate-200 rounded-xl p-6 flex flex-col">
            <h3 className="text-sm font-semibold text-slate-700 mb-3 text-center">
            Figure 5 · Per-Capita Risks by Age Band &amp; Sex
            </h3>
            <div className="flex-1 flex flex-col gap-3">
            <div className="flex flex-col md:flex-row gap-3">
                <div className="w-full md:w-1/2">
                <img
                    src="images/tavi_bar_avg_risks.png"
                    alt="Average TAVI risk by age band and sex"
                    className="w-full h-32 md:h-40 object-contain"
                />
                <p className="mt-1 text-[11px] text-slate-500 text-center">
                    TAVI: risk concentrated in the oldest age bands, with modest sex
                    differences.
                </p>
                </div>
                <div className="w-full md:w-1/2">
                <img
                    src="images/savr_bar_avg_risks.png"
                    alt="Average SAVR risk by age band and sex"
                    className="w-full h-32 md:h-40 object-contain"
                />
                <p className="mt-1 text-[11px] text-slate-500 text-center">
                    SAVR: similar pattern, but with different relative weighting by
                    age.
                </p>
                </div>
            </div>
            <div className="w-full">
                <img
                src="images/redo_savr_bar_avg_risks.png"
                alt="Average redo-SAVR risk by age band and sex"
                className="w-full h-32 md:h-36 object-contain"
                />
                <p className="mt-1 text-[11px] text-slate-500 text-center">
                Redo-SAVR: lower absolute risk, but used later as an external
                target when subtracting from ViV-eligible failures.
                </p>
            </div>
            </div>
        </div>
        </div>
    </div>
    ),

    Slide11: () => (
    <div className="p-12 h-full flex flex-col animate-slide-enter">
        <h2 className="text-3xl font-bold text-slate-800 mb-6">
        Projecting Index Volumes · Risk × Population
        </h2>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 flex-1">
        {/* Left: equation + explanation */}
        <div className="bg-slate-50 border border-slate-200 rounded-xl p-6">
            <h3 className="text-sm font-semibold text-slate-700 mb-3">
            Core Projection Formula (for y ≥ 2025)
            </h3>
            <div className="bg-slate-900 text-emerald-200 rounded-lg p-4 text-xs">
            <p className="font-mono">
                IndexVolume<sub>y, age, sex</sub> = risk<sub>2023–24, age, sex</sub>{" "}
                × population<sub>y, age, sex</sub>
            </p>
            </div>
            <ul className="list-disc pl-6 space-y-2 text-sm text-slate-700 mt-4">
            <li>
                For each year y ≥ 2025, and for every age–sex band, we multiply the{" "}
                <span className="font-semibold">fixed risk</span> (anchored to
                2023–2024) by the forecast population in that band.
            </li>
            <li>
                Summing over age and sex gives total TAVR and SAVR index volumes per
                year.
            </li>
            <li>
                This automatically incorporates{" "}
                <span className="font-semibold">ageing</span> and{" "}
                <span className="font-semibold">sex shifts</span> in the Korean
                population without needing arbitrary volume extrapolations.
            </li>
            </ul>

            <div className="mt-4 text-xs text-slate-600">
            Intuition: instead of saying “TAVR grows linearly by X% per year”, we
            ask “if each age–sex group keeps being treated at the 2023–2024 rate,
            what happens as the population pyramid changes?”
            </div>
        </div>

        {/* Right: Figure 4 index volumes */}
        <div className="bg-white border border-slate-200 rounded-xl p-6 flex flex-col">
            <h3 className="text-sm font-semibold text-slate-700 mb-3 text-center">
            Figure 4 · Index TAVI &amp; SAVR · Observed vs Projected
            </h3>
            <div className="flex-1 flex items-center justify-center mb-2">
            <img
                src="images/index_volume_overlay.png"
                alt="Index TAVI and SAVR volumes, observed 2015–2024 and projected 2025–2050"
                className="w-full h-64 object-contain"
            />
            </div>
            <p className="text-[11px] text-slate-500 text-center">
            Left side: observed volumes 2015–2024 (including the 2023 spike).{" "}
            Right side: demography-driven projections from 2025 onwards based on
            risk × population.
            </p>
        </div>
        </div>
    </div>
    ),

    Slide12: () => (
    <div className="p-12 h-full flex flex-col animate-slide-enter">
        <h2 className="text-3xl font-bold text-slate-800 mb-6">
        Redo-SAVR Targets · Risk-Based Instead of Ad-Hoc
        </h2>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 flex-1">
        {/* Left: bullet flow */}
        <div className="bg-slate-50 border border-slate-200 rounded-xl p-6">
            <h3 className="text-sm font-semibold text-slate-700 mb-3">
            From Risks to Absolute Redo-SAVR Counts
            </h3>
            <ol className="list-decimal pl-6 space-y-3 text-sm text-slate-700">
            <li>
                Use the same per-capita risk framework as for TAVR/SAVR, but now for{" "}
                <span className="font-semibold">redo-SAVR</span>, derived from
                2023–2024 registry counts.
            </li>
            <li>
                Multiply redo-SAVR risk by{" "}
                <span className="font-semibold">projected population</span> in each
                age–sex band to obtain yearly{" "}
                <span className="font-semibold">absolute redo-SAVR targets</span>{" "}
                (e.g. 2022–2050).
            </li>
            <li>
                Save these as an external CSV and feed them into the simulation via{" "}
                <span className="font-mono text-xs">redo_savr_numbers</span>.
            </li>
            <li>
                Configure{" "}
                <span className="font-mono text-xs">mode = "replace_rates"</span>{" "}
                so the Monte Carlo does{" "}
                <span className="font-semibold">not</span> use per-event redo
                probabilities on top.
            </li>
            </ol>

            <div className="mt-4 bg-slate-900 text-slate-100 rounded-lg p-4 text-xs">
            <p className="font-semibold mb-1">
                What &quot;replace_rates&quot; means in practice:
            </p>
            <ul className="list-disc pl-5 space-y-1">
                <li>
                Inside the Monte Carlo,{" "}
                <span className="font-mono text-[11px]">redo_rates</span> are set
                effectively to 0.
                </li>
                <li>
                After simulation, we subtract the{" "}
                <span className="font-semibold">absolute redo-SAVR targets</span>{" "}
                from the TAVR-in-SAVR pool in post-processing.
                </li>
            </ul>
            <p className="mt-2">
                This avoids double-counting and keeps redo-SAVR aligned with
                external epidemiology.
            </p>
            </div>
        </div>

        {/* Right: mini timeline diagram */}
        <div className="bg-white border border-slate-200 rounded-xl p-6 flex flex-col">
            <h3 className="text-sm font-semibold text-slate-700 mb-3 text-center">
            Conceptual Timeline · How Redo-SAVR Interacts with ViV
            </h3>
            <div className="flex-1 flex items-center justify-center">
            <div className="w-full max-w-md">
                <div className="flex items-center justify-between text-xs text-slate-700 mb-6">
                <div className="flex flex-col items-center">
                    <div className="px-3 py-2 bg-slate-800 text-white rounded-full">
                    Age–Sex Risks
                    </div>
                    <span className="mt-1">2023–2024</span>
                </div>
                <span className="text-2xl">×</span>
                <div className="flex flex-col items-center">
                    <div className="px-3 py-2 bg-sky-500 text-white rounded-full">
                    Pop Forecast
                    </div>
                    <span className="mt-1">2022–2050</span>
                </div>
                <span className="text-2xl">=</span>
                <div className="flex flex-col items-center">
                    <div className="px-3 py-2 bg-emerald-500 text-white rounded-full">
                    Redo-SAVR Targets
                    </div>
                    <span className="mt-1">per year</span>
                </div>
                </div>

                <div className="border-t border-dashed border-slate-300 my-4" />

                <div className="text-xs text-slate-700 space-y-1">
                <p>
                    These targets then sit alongside ViV candidates in each year:
                </p>
                <ul className="list-disc pl-5">
                    <li>
                    <span className="font-semibold">Candidates</span> from failed
                    SAVR while alive.
                    </li>
                    <li>
                    <span className="font-semibold">Redo-SAVR targets</span>{" "}
                    removed from this pool.
                    </li>
                    <li>
                    Remaining are truly{" "}
                    <span className="font-semibold">ViV-eligible</span> cases.
                    </li>
                </ul>
                </div>
            </div>
            </div>
            <p className="mt-2 text-[11px] text-slate-500 text-center">
            All done before any ViV penetration assumptions are applied.
            </p>
        </div>
        </div>
    </div>
    ),

    Slide13: () => (
    <div className="p-12 h-full flex flex-col animate-slide-enter">
        <h2 className="text-3xl font-bold text-slate-800 mb-6">
        Monte Carlo Engine Breakdown
        </h2>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 flex-1">
        {/* Patient timeline */}
        <div className="lg:col-span-2 bg-white border border-slate-200 rounded-xl p-6 flex flex-col">
            <h3 className="text-sm font-semibold text-slate-700 mb-3">
            Pipeline for a Single Patient
            </h3>
            <div className="flex-1 flex items-center justify-center">
            <div className="w-full max-w-2xl">
                <div className="flex flex-col md:flex-row md:items-center md:justify-between text-xs text-slate-700 gap-4">
                <div className="flex flex-col items-center">
                    <div className="px-3 py-2 bg-slate-800 text-white rounded-full">
                    1. Index Procedure
                    </div>
                    <p className="mt-2 text-center">
                    Year y, age drawn uniformly within registry age band; procedure
                    type TAVR or SAVR.
                    </p>
                </div>
                <span className="hidden md:inline text-2xl">→</span>
                <div className="flex flex-col items-center">
                    <div className="px-3 py-2 bg-sky-600 text-white rounded-full">
                    2. Risk Category
                    </div>
                    <p className="mt-2 text-center">
                    Sampled via year-specific risk mix (low / intermediate / high
                    risk).
                    </p>
                </div>
                <span className="hidden md:inline text-2xl">→</span>
                <div className="flex flex-col items-center">
                    <div className="px-3 py-2 bg-amber-500 text-white rounded-full">
                    3. Survival &amp; Durability
                    </div>
                    <p className="mt-2 text-center">
                    Draw survival time (risk-specific Normal, with age hazard) and
                    valve durability (bimodal for TAVR; age-split for SAVR).
                    </p>
                </div>
                <span className="hidden md:inline text-2xl">→</span>
                <div className="flex flex-col items-center">
                    <div className="px-3 py-2 bg-emerald-600 text-white rounded-full">
                    4. Failure vs Death
                    </div>
                    <p className="mt-2 text-center">
                    Floor both to calendar years to get failure year and death
                    year; candidate if failure ≤ death and within horizon.
                    </p>
                </div>
                </div>
            </div>
            </div>
        </div>

        {/* Distributions box */}
        <div className="bg-slate-50 border border-slate-200 rounded-xl p-6 flex flex-col">
            <h3 className="text-sm font-semibold text-slate-700 mb-3">
            Key Distributions (from config)
            </h3>
            <ul className="list-disc pl-6 space-y-2 text-xs text-slate-700">
            <li>
                <span className="font-semibold">Durability</span>:
                <ul className="list-disc pl-4 mt-1 space-y-1">
                <li>
                    TAVR: mixture of early (≈4y) and late (≈11.5y) modes.
                </li>
                <li>
                    SAVR: different means for &lt;70 vs ≥70 years at index.
                </li>
                <li>Enforced minimum durability (e.g. ≥1 year).</li>
                </ul>
            </li>
            <li>
                <span className="font-semibold">Survival</span>:
                <ul className="list-disc pl-4 mt-1 space-y-1">
                <li>
                    Normal distributions by risk category (low / intermediate /
                    high).
                </li>
                <li>
                    Age-hazard scaling around reference age (e.g. 75y) via hazard
                    ratios per 5 years.
                </li>
                </ul>
            </li>
            <li>
                <span className="font-semibold">Risk mix</span>:
                <ul className="list-disc pl-4 mt-1 space-y-1">
                <li>Piecewise time-varying shares for TAVR.</li>
                <li>Fixed shares for SAVR.</li>
                </ul>
            </li>
            </ul>

            <div className="mt-4 text-xs text-slate-600">
            Note: these distributions are largely unchanged from our v7 model and
            the original Genereux–Ohno setup; what has changed is how{" "}
            <span className="font-semibold">index volumes</span> and{" "}
            <span className="font-semibold">redo-SAVR</span> are generated.
            </div>
        </div>
        </div>
    </div>
    ),

    Slide14: () => (
    <div className="p-12 h-full flex flex-col animate-slide-enter">
        <h2 className="text-3xl font-bold text-slate-800 mb-6">
        Defining ViV Candidates vs Realised ViV
        </h2>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 flex-1">
        {/* Left: step flow diagram */}
        <div className="bg-white border border-slate-200 rounded-xl p-6 flex flex-col">
            <h3 className="text-sm font-semibold text-slate-700 mb-3">
            Conceptual Flow (Per Year)
            </h3>
            <div className="flex-1 flex items-center justify-center">
            <div className="w-full max-w-xl">
                <div className="flex flex-col gap-3 text-xs text-center">
                {/* Row 1 */}
                <div className="flex items-center justify-center gap-2">
                    <div className="flex-1 bg-slate-100 border border-slate-300 rounded-lg px-3 py-2">
                    <div className="font-semibold text-slate-700">
                        All valve failures
                    </div>
                    </div>
                    <div className="text-slate-400 text-lg font-bold">→</div>
                    <div className="flex-1 bg-blue-50 border border-blue-300 rounded-lg px-3 py-2">
                    <div className="font-semibold text-blue-700">
                        Alive at failure
                    </div>
                    </div>
                </div>

                {/* Row 2 */}
                <div className="flex items-center justify-center gap-2">
                    <div className="flex-1 bg-emerald-50 border border-emerald-300 rounded-lg px-3 py-2">
                    <div className="font-semibold text-emerald-700">
                        ViV candidates
                    </div>
                    <div className="text-[11px] text-emerald-800 mt-1">
                        Alive at failure, within horizon
                    </div>
                    </div>
                    <div className="text-slate-400 text-lg font-bold">→</div>
                    <div className="flex-1 bg-amber-50 border border-amber-300 rounded-lg px-3 py-2">
                    <div className="font-semibold text-amber-800">
                        ViV‑eligible
                    </div>
                    <div className="text-[11px] text-amber-900 mt-1">
                        After redo‑SAVR subtraction
                    </div>
                    </div>
                </div>

                {/* Row 3 */}
                <div className="flex items-center justify-center gap-2">
                    <div className="flex-1" />
                    <div className="text-slate-400 text-lg font-bold">↓</div>
                    <div className="flex-1 bg-purple-50 border border-purple-400 rounded-lg px-3 py-2">
                    <div className="font-semibold text-purple-700">
                        Realised ViV
                    </div>
                    <div className="text-[11px] text-purple-800 mt-1">
                        ViV‑eligible × penetration
                    </div>
                    </div>
                    <div className="flex-1" />
                </div>
                </div>
            </div>
            </div>
        </div>

        {/* Right: bullets matching the diagram labels */}
        <div className="bg-slate-50 border border-slate-200 rounded-xl p-6 flex flex-col">
            <h3 className="text-sm font-semibold text-slate-700 mb-3">
            What We Track Each Year
            </h3>
            <ol className="list-decimal pl-6 space-y-2 text-sm text-slate-700">
            <li>
                <span className="font-semibold">All valve failures:</span> every
                failed TAVR or SAVR valve in that year.
            </li>
            <li>
                <span className="font-semibold">Alive at failure:</span> failures
                where the patient is still alive.
            </li>
            <li>
                <span className="font-semibold">ViV candidates:</span> alive‑at‑failure
                cases whose failure year lies inside the simulation window.
            </li>
            <li>
                <span className="font-semibold">ViV‑eligible:</span> ViV candidates
                after subtracting redo‑SAVR for that year.
            </li>
            <li>
                <span className="font-semibold">Realised ViV:</span> ViV‑eligible
                patients who actually receive ViV under the chosen penetration
                scenario.
            </li>
            </ol>

            <div className="mt-4 bg-emerald-50 border border-emerald-100 rounded-lg p-3 text-xs text-emerald-900">
            Key idea:{" "}
            <span className="font-semibold">ViV candidates and ViV‑eligible</span>{" "}
            are fully data‑driven;{" "}
            <span className="font-semibold">penetration</span> is a separate,
            optional layer on top.
            </div>
        </div>
        </div>
    </div>
    ),


    Slide15: () => (
    <div className="p-12 h-full flex flex-col animate-slide-enter">
        <h2 className="text-3xl font-bold text-slate-800 mb-6">
        Penetration · From “Baked In” to “Scenario on Top”
        </h2>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 flex-1">
        {/* Old vs new contrast */}
        <div className="bg-red-50 border border-red-100 rounded-xl p-6 flex flex-col">
            <h3 className="text-sm font-semibold text-red-800 mb-3">
            Old Approach (US/Japan-style)
            </h3>
            <ul className="list-disc pl-6 space-y-2 text-sm text-red-900">
            <li>
                ViV penetration curves (e.g.{" "}
                <span className="font-semibold">10% → 60%</span> uptake) are{" "}
                <span className="font-semibold">baked into</span> the primary
                output.
            </li>
            <li>
                Only reported series is &quot;realised ViV&quot; under one specific
                penetration curve.
            </li>
            <li>
                Difficult to see how many patients are ViV-eligible versus how many
                are actually treated.
            </li>
            </ul>
        </div>

        <div className="bg-emerald-50 border border-emerald-100 rounded-xl p-6 flex flex-col">
            <h3 className="text-sm font-semibold text-emerald-800 mb-3">
            New Approach (model_v9)
            </h3>
            <ul className="list-disc pl-6 space-y-2 text-sm text-emerald-900">
            <li>
                Primary output:{" "}
                <span className="font-semibold">ViV candidates</span> (after
                redo-SAVR subtraction).
            </li>
            <li>
                Penetration is added only as a{" "}
                <span className="font-semibold">scenario on top</span> – e.g.{" "}
                &quot;what if Korea followed a 10%→60% ramp similar to the US?&quot;
            </li>
            <li>
                We can report both:
                <ul className="list-disc pl-5 mt-1 text-xs">
                <li>Candidate trajectories (agnostic to penetration).</li>
                <li>
                    Realised ViV trajectories under chosen penetration scenarios
                    (for comparison to Genereux/Ohno).
                </li>
                </ul>
            </li>
            </ul>

            <div className="mt-4 text-xs text-emerald-900">
            In the results, you&apos;ll see{" "}
            <span className="font-semibold">
                candidates vs realised under a US-like 10%→60% ramp
            </span>
            , making it easy to isolate the impact of demography versus
            assumptions about ViV adoption.
            </div>
        </div>
        </div>
    </div>
    ),

    Slide16: () => (
    <div className="p-12 h-full flex flex-col animate-slide-enter">
        <h2 className="text-3xl font-bold text-slate-800 mb-6">
        Quality Control · The 2024→2025 Dip Bug
        </h2>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 flex-1">
        {/* Left: screenshot of odd pattern */}
        <div className="bg-white border border-slate-200 rounded-xl p-6 flex flex-col">
            <h3 className="text-sm font-semibold text-slate-700 mb-3 text-center">
            Before Fix · Candidates Flat, Realised TAVR-in-SAVR Dips
            </h3>
            <div className="flex-1 flex items-center justify-center">
            <img
                src="images/bug_tavi_in_savr_dip_2024_2025.png"
                alt="Old plot showing 2024-2025 dip in realised TAVR-in-SAVR despite flat candidates"
                className="w-full h-64 object-contain"
            />
            </div>
            <p className="mt-2 text-[11px] text-slate-500 text-center">
            In an early run, the realised TAVR-in-SAVR line dropped sharply from
            2024 to 2025 even though candidates remained roughly flat.
            </p>
        </div>

        {/* Right: explanation of cause & fix */}
        <div className="bg-slate-50 border border-slate-200 rounded-xl p-6 flex flex-col">
            <h3 className="text-sm font-semibold text-slate-700 mb-3">
            Root Cause &amp; Fix
            </h3>
            <ul className="list-disc pl-6 space-y-3 text-sm text-slate-700">
            <li>
                <span className="font-semibold">Cause:</span> our precompute only
                projected redo-SAVR targets from{" "}
                <span className="font-semibold">2025 onwards</span>. Years before
                2025 had{" "}
                <span className="font-semibold">
                no redo-SAVR subtraction at all
                </span>
                .
            </li>
            <li>
                This meant:
                <ul className="list-disc pl-5 mt-1 text-xs">
                <li>
                    2024: realised TAVR-in-SAVR ≈ candidates (no subtraction).
                </li>
                <li>
                    2025: realised TAVR-in-SAVR = candidates − redo-SAVR target,
                    creating an artificial step down.
                </li>
                </ul>
            </li>
            <li>
                <span className="font-semibold">Fix:</span>
                <ul className="list-disc pl-5 mt-1 text-xs">
                <li>
                    Extend <span className="font-mono text-[11px]">
                    precompute.project_years
                    </span>{" "}
                    so redo-SAVR targets cover the full simulation window.
                </li>
                <li>
                    Smooth missing years (linear / forward fill) so subtraction is
                    continuous in time.
                </li>
                </ul>
            </li>
            </ul>

            <div className="mt-4 bg-emerald-50 border border-emerald-100 rounded-lg p-3 text-xs text-emerald-900">
            After this fix, the candidates and realised TAVR-in-SAVR series line
            up logically with no artificial step at 2025; the remaining shape is
            driven by demographics and penetration only.
            </div>
        </div>
        </div>
    </div>
    ),

    Slide17: () => (
    <div className="p-12 h-full flex flex-col animate-slide-enter">
        <h2 className="text-3xl font-bold text-slate-800 mb-6">
        Results · ViV Candidates vs Realised (Image D)
        </h2>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 flex-1">
        {/* Left: Image D */}
        <div className="bg-white border border-slate-200 rounded-xl p-6 flex flex-col">
            <h3 className="text-sm font-semibold text-slate-700 mb-3 text-center">
            Image D · Available ViV Candidates vs Realised ViV
            </h3>
            <div className="flex-1 flex items-center justify-center">
            <img
                src="images/image_D_viv_candidates_vs_realized_2022_2050.png"
                alt="Available ViV candidates vs realised ViV, 2022-2050"
                className="w-full h-64 object-contain"
            />
            </div>
            <p className="mt-2 text-[11px] text-slate-500 text-center">
            Grey bars: total ViV candidates. Solid lines: TAVR-in-SAVR and
            TAVR-in-TAVR candidates. Dashed lines: realised ViV under the
            penetration scenario, after redo-SAVR subtraction. Yellow band: focus
            years 2025–2035.
            </p>
        </div>

        {/* Right: verbal summary */}
        <div className="bg-slate-50 border border-slate-200 rounded-xl p-6 flex flex-col">
            <h3 className="text-sm font-semibold text-slate-700 mb-3">
            Interpretation of the Core Output
            </h3>
            <ul className="list-disc pl-6 space-y-3 text-sm text-slate-700">
            <li>
                <span className="font-semibold">Total candidates:</span> increase
                steadily from about{" "}
                <span className="font-semibold">~790 in 2024</span> to just under{" "}
                <span className="font-semibold">2,000 in 2035</span>, roughly a{" "}
                <span className="font-semibold">2.5-fold</span> rise.
            </li>
            <li>
                The growth is largely driven by{" "}
                <span className="font-semibold">population ageing</span> and stable
                per-capita treatment risks, not by arbitrary volume extrapolation.
            </li>
            <li>
                <span className="font-semibold">Split by type:</span> both
                TAVR-in-SAVR and TAVR-in-TAVR streams grow, with TAVR-in-TAVR
                gradually increasing its share over time.
            </li>
            <li>
                <span className="font-semibold">Realised ViV (dashed lines):</span>{" "}
                reflect the same candidate dynamics plus the penetration scenario
                and redo-SAVR subtraction. The shapes closely track candidates after
                the QC fix for redo-SAVR.
            </li>
            </ul>

            <div className="mt-4 bg-slate-900 text-slate-100 rounded-lg p-3 text-xs">
            Within the 2025–2035 focus window (highlighted in yellow in the
            figure), we see a demography-driven ViV demand increase that is{" "}
            <span className="font-semibold">moderate but clinically significant</span>{" "}
            — a key planning signal for Korean structural heart programs.
            </div>
        </div>
        </div>
    </div>
    ),

    Slide18: () => (
    <div className="p-12 h-full flex flex-col animate-slide-enter">
        <h2 className="text-3xl font-bold text-slate-800 mb-4">
        Results: Realised ViV Under US‑Style Penetration (Figure 2)
        </h2>

        <div className="flex-1 bg-slate-50 border border-slate-200 rounded-xl p-5 flex flex-col">
        <h3 className="text-sm font-semibold text-slate-700 mb-2 uppercase">
            Figure 2. Predicted Realised ViV Volume (2023–2035)
        </h3>
        <div className="flex-1 bg-white border border-slate-200 rounded-lg flex items-center justify-center">
            <img
            src="images/image_C_viv_pretty_2025_2035.png"
            alt="Predicted realised ViV patient volume, TAVR-in-SAVR, TAVR-in-TAVR, and total"
            className="w-full h-full object-contain"
            />
        </div>

        <div className="mt-4 grid grid-cols-3 gap-4 text-xs">
            <div className="bg-slate-900 text-slate-50 rounded-lg p-3">
            <div className="font-semibold text-[11px] mb-1">Series Shown</div>
            <ul className="list-disc pl-4 space-y-1">
                <li>
                <span className="font-semibold">Red line:</span> TAVR‑in‑SAVR.
                </li>
                <li>
                <span className="font-semibold">Blue line:</span> TAVR‑in‑TAVR.
                </li>
                <li>
                <span className="font-semibold">Grey bars:</span> total realised
                ViV (TAVR‑in‑SAVR + TAVR‑in‑TAVR).
                </li>
            </ul>
            </div>
            <div className="bg-slate-50 border border-slate-200 rounded-lg p-3">
            <div className="font-semibold text-slate-800 text-[11px] mb-1">
                Magnitude
            </div>
            <p className="text-[11px] text-slate-700">
                Under the US‑style penetration ramp (≈10% → 60%), Korea’s realised
                ViV volume increases from about{" "}
                <span className="font-semibold">325 cases in 2023</span> to about{" "}
                <span className="font-semibold">1,080 cases in 2035</span> — roughly
                a <span className="font-semibold">3‑fold rise</span>.
            </p>
            </div>
            <div className="bg-emerald-50 border border-emerald-100 rounded-lg p-3">
            <div className="font-semibold text-emerald-900 text-[11px] mb-1">
                Comparison to US/Japan
            </div>
            <p className="text-[11px] text-emerald-900">
                Ohno &amp; Genereux report ~7‑fold (US) and ~9‑fold (Japan)
                increases over similar horizons. Our more modest 3‑fold rise occurs
                because index volumes are{" "}
                <span className="font-semibold">demography‑anchored</span> and do not
                grow indefinitely.
            </p>
            </div>
        </div>
        </div>
    </div>
    ),


    // --- V5 SLIDE 19: SUPPORTING RESULT – POPULATION TRENDS ----------

    Slide19: () => (
    <div className="p-12 h-full flex flex-col animate-slide-enter">
        <h2 className="text-3xl font-bold text-slate-800 mb-4">
        Supporting Result: Population Trends (Figure 3)
        </h2>

        <div className="grid grid-cols-5 gap-8 flex-1">
        {/* Population plots reused */}
        <div className="col-span-3 bg-slate-50 border border-slate-200 rounded-xl p-4 flex flex-col">
            <h3 className="text-sm font-semibold text-slate-700 mb-2 uppercase">
            Figure 3. Population by Age &amp; Sex
            </h3>
            <div className="grid grid-cols-2 gap-4 h-56">
            <div className="bg-white border border-slate-200 rounded-lg flex items-center justify-center">
                <img
                src="images/age_projection_lines_Men.png"
                alt="Projected male population by age band"
                className="w-full h-full object-contain"
                />
            </div>
            <div className="bg-white border border-slate-200 rounded-lg flex items-center justify-center">
                <img
                src="images/age_projection_lines_Women.png"
                alt="Projected female population by age band"
                className="w-full h-full object-contain"
                />
            </div>
            </div>
            <div className="mt-3 bg-white border border-slate-200 rounded-lg h-32 flex items-center justify-center">
            <img
                src="images/age_heatmap_allsex.png"
                alt="Population heatmap by year and age"
                className="w-full h-full object-contain"
            />
            </div>
        </div>

        {/* Bullet callouts */}
        <div className="col-span-2 bg-white border border-slate-200 rounded-xl p-5 flex flex-col">
            <h3 className="text-sm font-semibold text-slate-800 mb-3 uppercase">
            What the Demography Tells Us
            </h3>
            <ul className="text-xs text-slate-700 space-y-2">
            <li>
                <span className="font-semibold">Younger cohorts</span> are stable or
                shrinking over time.
            </li>
            <li>
                Older bands, especially{" "}
                <span className="font-semibold">75–79, 80–84, and ≥85</span>, grow
                markedly, reflecting Korea’s rapid ageing.
            </li>
            <li>
                At the highest ages,{" "}
                <span className="font-semibold">women dominate</span> numerically,
                which also affects the sex mix of ViV candidates.
            </li>
            <li>
                Even with per‑capita risks held constant, this demographic shift
                alone is enough to produce a{" "}
                <span className="font-semibold">2.5× increase</span> in ViV
                candidates between 2024 and 2035.
            </li>
            </ul>
            <div className="mt-4 text-[11px] text-slate-600 bg-slate-50 border border-slate-200 rounded-lg p-3">
            These population curves are the backbone of the new model: they drive
            TAVI/SAVR index volumes and, through durability &amp; survival, the
            future burden of ViV‑eligible patients.
            </div>
        </div>
        </div>
    </div>
    ),


    // --- V5 SLIDE 20: SUPPORTING RESULT – INDEX PROJECTIONS ----------

    Slide20: () => (
    <div className="p-12 h-full flex flex-col animate-slide-enter">
        <h2 className="text-3xl font-bold text-slate-800 mb-4">
        Supporting Result: Index TAVI/SAVR Projection (Figure 4)
        </h2>

        <div className="grid grid-cols-5 gap-8 flex-1">
        {/* Index overlay figure */}
        <div className="col-span-3 bg-slate-50 border border-slate-200 rounded-xl p-4 flex flex-col">
            <h3 className="text-sm font-semibold text-slate-700 mb-2 uppercase">
            Figure 4. Index TAVI and SAVR (2015–2050)
            </h3>
            <div className="flex-1 bg-white border border-slate-200 rounded-lg flex items-center justify-center">
            <img
                src="images/index_volume_overlay.png"
                alt="Observed vs projected index TAVI and SAVR volumes"
                className="w-full h-full object-contain"
            />
            </div>
            <p className="mt-2 text-xs text-slate-500">
            Solid segments: observed registry counts through 2024. Dashed segments:
            risk × population projections from 2025 onwards.
            </p>
        </div>

        {/* Interpretation bullets */}
        <div className="col-span-2 bg-white border border-slate-200 rounded-xl p-5 flex flex-col">
            <h3 className="text-sm font-semibold text-slate-800 mb-3 uppercase">
            Interpreting the Shape
            </h3>
            <ul className="text-xs text-slate-700 space-y-2">
            <li>
                The <span className="font-semibold">2023 spike</span> in TAVI/SAVR
                is preserved as a real feature of the data, but we do{" "}
                <span className="font-semibold">not</span> extrapolate a straight
                line from it.
            </li>
            <li>
                Post‑2025 trajectories are determined by{" "}
                <span className="font-semibold">risk × population</span>, so volumes
                can plateau or even decline if the underlying population does.
            </li>
            <li>
                This automatically allows SAVR to{" "}
                <span className="font-semibold">decline or stabilise</span> with
                changing age structure, rather than being forced flat.
            </li>
            <li>
                It also avoids one of the key editorial criticisms of the Ohno
                model: aggressive extrapolations on top of a flat SAVR assumption.
            </li>
            </ul>
            <div className="mt-4 text-[11px] text-slate-600 bg-slate-50 border border-slate-200 rounded-lg p-3">
            In short, index volumes are no longer a free‑hand curve; they are a
            direct consequence of demography plus fixed per‑capita treatment risk.
            </div>
        </div>
        </div>
    </div>
    ),


    // --- V5 SLIDE 21: SUPPORTING RESULT – RISK PROFILES -------------- 

    Slide21: () => (
    <div className="p-12 h-full flex flex-col animate-slide-enter">
        <h2 className="text-3xl font-bold text-slate-800 mb-4">
        Supporting Result: Risk Profiles by Age &amp; Sex (Figure 5)
        </h2>

        <div className="grid grid-cols-5 gap-8 flex-1">
        {/* Risk bar charts */}
        <div className="col-span-3 bg-slate-50 border border-slate-200 rounded-xl p-4 flex flex-col">
            <h3 className="text-sm font-semibold text-slate-700 mb-2 uppercase">
            Figure 5. Average Per‑Capita Risk (2023–2024)
            </h3>
            <div className="grid grid-cols-3 gap-3 flex-1">
            <div className="bg-white border border-slate-200 rounded-lg flex flex-col">
                <div className="text-[11px] font-semibold text-center mt-2">
                TAVI
                </div>
                <div className="flex-1 flex items-center justify-center p-1">
                <img
                    src="images/tavi_bar_avg_risks.png"
                    alt="Average per-capita TAVI risk by age band and sex"
                    className="w-full h-full object-contain"
                />
                </div>
            </div>
            <div className="bg-white border border-slate-200 rounded-lg flex flex-col">
                <div className="text-[11px] font-semibold text-center mt-2">
                SAVR
                </div>
                <div className="flex-1 flex items-center justify-center p-1">
                <img
                    src="images/savr_bar_avg_risks.png"
                    alt="Average per-capita SAVR risk by age band and sex"
                    className="w-full h-full object-contain"
                />
                </div>
            </div>
            <div className="bg-white border border-slate-200 rounded-lg flex flex-col">
                <div className="text-[11px] font-semibold text-center mt-2">
                Redo‑SAVR
                </div>
                <div className="flex-1 flex items-center justify-center p-1">
                <img
                    src="images/redo_savr_bar_avg_risks.png"
                    alt="Average per-capita redo-SAVR risk by age band and sex"
                    className="w-full h-full object-contain"
                />
                </div>
            </div>
            </div>
        </div>

        {/* Bullet commentary */}
        <div className="col-span-2 bg-white border border-slate-200 rounded-xl p-5 flex flex-col">
            <h3 className="text-sm font-semibold text-slate-800 mb-3 uppercase">
            Key Patterns
            </h3>
            <ul className="text-xs text-slate-700 space-y-2">
            <li>
                Risk is strongly concentrated in the{" "}
                <span className="font-semibold">oldest age bands</span>, especially
                ≥75 and ≥80, which aligns with clinical expectations.
            </li>
            <li>
                There are visible <span className="font-semibold">sex differences</span>{" "}
                in some bands, with higher rates in men for certain ages and in women
                at the extreme elderly tail.
            </li>
            <li>
                The same risk framework is used for{" "}
                <span className="font-semibold">redo‑SAVR</span>, ensuring internal
                consistency when we subtract redo cases from the ViV pool.
            </li>
            </ul>
            <div className="mt-4 text-[11px] text-slate-600 bg-slate-50 border border-slate-200 rounded-lg p-3">
            These profiles are not just descriptive; they are the exact rates we
            propagate into the future when we compute index volumes and
            redo‑SAVR targets.
            </div>
        </div>
        </div>
    </div>
    ),


    // --- V5 SLIDE 22: FLOW / WATERFALL DIAGRAMS ----------------------

    Slide22: () => (
    <div className="p-12 h-full flex flex-col animate-slide-enter">
        <h2 className="text-3xl font-bold text-slate-800 mb-4">
        Flow Decomposition: Waterfall Diagrams (Figure 6)
        </h2>

        <div className="grid grid-cols-5 gap-8 flex-1">
        {/* Example waterfalls */}
        <div className="col-span-3 bg-slate-50 border border-slate-200 rounded-xl p-4 flex flex-col">
            <h3 className="text-sm font-semibold text-slate-700 mb-2 uppercase">
            Figure 6. Example Waterfalls for Selected Years
            </h3>
            <div className="grid grid-cols-3 gap-3 flex-1">
            <div className="bg-white border border-slate-200 rounded-lg flex items-center justify-center p-1">
                <img
                src="images/waterfall_TAVR-in-SAVR_2025.png"
                alt="Waterfall for TAVR-in-SAVR 2025"
                className="w-full h-full object-contain"
                />
            </div>
            <div className="bg-white border border-slate-200 rounded-lg flex items-center justify-center p-1">
                <img
                src="images/waterfall_TAVR-in-SAVR_2030.png"
                alt="Waterfall for TAVR-in-SAVR 2030"
                className="w-full h-full object-contain"
                />
            </div>
            <div className="bg-white border border-slate-200 rounded-lg flex items-center justify-center p-1">
                <img
                src="images/waterfall_TAVR-in-SAVR_2035.png"
                alt="Waterfall for TAVR-in-SAVR 2035"
                className="w-full h-full object-contain"
                />
            </div>
            </div>
            <p className="mt-2 text-xs text-slate-500">
            Each panel decomposes one year’s events into failures, ViV‑eligible
            patients, redo‑SAVR, realised ViV, and residual eligible patients with
            no ViV.
            </p>
        </div>

        {/* Stage explanation */}
        <div className="col-span-2 bg-white border border-slate-200 rounded-xl p-5 flex flex-col">
            <h3 className="text-sm font-semibold text-slate-800 mb-3 uppercase">
            Stages Tracked in Each Waterfall
            </h3>
            <ol className="list-decimal pl-4 text-xs text-slate-700 space-y-2">
            <li>
                <span className="font-semibold">Valve failures</span>: all
                durability events in that calendar year.
            </li>
            <li>
                <span className="font-semibold">ViV‑eligible (alive)</span>: subset
                where the patient is alive at failure and inside the forecast
                horizon.
            </li>
            <li>
                <span className="font-semibold">Redo‑SAVR</span>: cases diverted to
                redo open‑heart surgery based on the risk‑derived yearly targets.
            </li>
            <li>
                <span className="font-semibold">Realised ViV</span>: ViV
                procedures under the penetration scenario.
            </li>
            <li>
                <span className="font-semibold">Eligible but no ViV</span>: residual
                candidates who are neither redo‑SAVR nor ViV in that year.
            </li>
            </ol>
            <div className="mt-4 text-[11px] text-slate-600 bg-slate-50 border border-slate-200 rounded-lg p-3">
            These waterfalls are mainly for internal QA: they confirm that the
            Monte Carlo accounting matches our conceptual flow of patients through
            the system.
            </div>
        </div>
        </div>
    </div>
    ),


    // --- V5 SLIDE 23: COMPARISON VS GENEREUX / OHNO ------------------

    Slide23: () => (
    <div className="p-12 h-full flex flex-col animate-slide-enter">
        <h2 className="text-3xl font-bold text-slate-800 mb-4">
        How Our Model Differs from Genereux / Ohno
        </h2>

        <div className="flex-1 bg-slate-50 border border-slate-200 rounded-xl p-6 grid grid-cols-2 gap-6">
        <div className="bg-white border border-slate-200 rounded-lg p-4 text-xs flex flex-col">
            <h3 className="text-sm font-semibold text-slate-800 mb-2">
            US / Japan Models (Genereux, Ohno)
            </h3>
            <ul className="list-disc pl-4 space-y-2 text-slate-700">
            <li>
                <span className="font-semibold">Index volumes:</span> extrapolated
                volume trend (linear / assumed plateau) from historical TAVR/SAVR.
            </li>
            <li>
                <span className="font-semibold">Demography:</span> mostly implicit;
                age structure not explicitly tied to national projections.
            </li>
            <li>
                <span className="font-semibold">Penetration:</span> calibrated
                curves; model reports only realised ViV after penetration.
            </li>
            <li>
                <span className="font-semibold">Redo‑SAVR:</span> adjustments more
                ad hoc, not explicitly risk × population derived.
            </li>
            </ul>
        </div>
        <div className="bg-emerald-50 border border-emerald-100 rounded-lg p-4 text-xs flex flex-col">
            <h3 className="text-sm font-semibold text-emerald-900 mb-2">
            Our Korean Model
            </h3>
            <ul className="list-disc pl-4 space-y-2 text-emerald-900">
            <li>
                <span className="font-semibold">Index volumes:</span>{" "}
                risk × population, anchored to 2023–2024 per‑capita rates and
                official demographic projections.
            </li>
            <li>
                <span className="font-semibold">Demography:</span>{" "}
                explicit &amp; central; ageing drives the volume evolution.
            </li>
            <li>
                <span className="font-semibold">Penetration:</span>{" "}
                scenario on top; we report ViV candidates independently of any
                penetration assumption.
            </li>
            <li>
                <span className="font-semibold">Redo‑SAVR:</span> generated from the
                same risk × population framework and fed in as absolute yearly
                targets.
            </li>
            </ul>
        </div>
        </div>

        <div className="mt-4 text-center text-xs text-slate-700">
        Net effect: a model that is{" "}
        <span className="font-semibold">
            more robust to short‑term anomalies
        </span>{" "}
        and better aligned with the demographic realities that will drive ViV
        demand in Korea.
        </div>
    </div>
    ),


    // --- V5 SLIDE 24: LIMITATIONS & OPEN QUESTIONS -------------------

    Slide24: () => (
    <div className="p-12 h-full flex flex-col animate-slide-enter">
        <h2 className="text-3xl font-bold text-slate-800 mb-4">
        Limitations &amp; Open Questions
        </h2>

        <div className="flex-1 bg-slate-50 border border-slate-200 rounded-xl p-6 grid grid-cols-2 gap-6">
        <div className="bg-white border border-slate-200 rounded-lg p-4 text-xs">
            <h3 className="text-sm font-semibold text-slate-800 mb-2">
            Modelling Assumptions
            </h3>
            <ul className="list-disc pl-4 space-y-2 text-slate-700">
            <li>
                Per‑capita risks are{" "}
                <span className="font-semibold">assumed stationary</span> at
                2023–2024 levels; real practice will evolve.
            </li>
            <li>
                Survival &amp; durability curves are derived from{" "}
                <span className="font-semibold">international literature</span>, not
                re‑estimated from a Korean cohort.
            </li>
            <li>
                No explicit modelling of device evolution, programme expansion, or
                guideline shifts beyond what is baked into 2023–2024 behaviour.
            </li>
            </ul>
        </div>
        <div className="bg-white border border-slate-200 rounded-lg p-4 text-xs">
            <h3 className="text-sm font-semibold text-slate-800 mb-2">
            External Shocks &amp; Penetration
            </h3>
            <ul className="list-disc pl-4 space-y-2 text-slate-700">
            <li>
                Socioeconomic or policy shocks (new reimbursement rules, pandemics,
                etc.) are{" "}
                <span className="font-semibold">not explicitly simulated</span>.
            </li>
            <li>
                ViV penetration scenarios remain{" "}
                <span className="font-semibold">speculative</span>, although they are
                now clearly separated from the candidate forecast.
            </li>
            </ul>
            <div className="mt-3 text-[11px] text-slate-600 bg-slate-50 border border-slate-200 rounded-lg p-3">
            The goal is not to claim a single “true” forecast, but to make the
            mapping from assumptions → results{" "}
            <span className="font-semibold">transparent and modifiable</span>.
            </div>
        </div>
        </div>
    </div>
    ),


    // --- V5 SLIDE 25: EXTENSION TO SINGAPORE -------------------------

    Slide25: () => (
    <div className="p-12 h-full flex flex-col animate-slide-enter">
        <h2 className="text-3xl font-bold text-slate-800 mb-4">
        Next Step: Extension to Singapore &amp; Comparative Analysis
        </h2>

        <div className="grid grid-cols-5 gap-8 flex-1">
        {/* Korea vs Singapore cartoon */}
        <div className="col-span-3 bg-slate-50 border border-slate-200 rounded-xl p-5 flex items-center justify-center">
            <div className="grid grid-cols-2 gap-6 text-xs w-full">
            <div className="bg-white border border-slate-200 rounded-lg p-3">
                <div className="font-semibold text-slate-800 mb-2">
                Korea Pipeline (Implemented)
                </div>
                <ul className="list-disc pl-4 space-y-1 text-slate-700">
                <li>National age–sex projections reconstructed.</li>
                <li>HIRA risks for 2023–2024 computed.</li>
                <li>Monte Carlo engine calibrated and run.</li>
                </ul>
            </div>
            <div className="bg-blue-50 border border-blue-100 rounded-lg p-3">
                <div className="font-semibold text-blue-900 mb-2">
                Singapore Pipeline (Next)
                </div>
                <ul className="list-disc pl-4 space-y-1 text-blue-900">
                <li>Ingest Singapore age–sex population projections.</li>
                <li>
                    Compute per‑capita risks for TAVI/SAVR/redo‑SAVR (2023–2024).
                </li>
                <li>Run the same Monte Carlo engine without code changes.</li>
                </ul>
            </div>
            </div>
        </div>

        {/* Comparative outputs */}
        <div className="col-span-2 bg-white border border-slate-200 rounded-xl p-5 text-xs flex flex-col">
            <h3 className="text-sm font-semibold text-slate-800 mb-3 uppercase">
            Planned Comparative Outputs
            </h3>
            <ul className="list-disc pl-4 space-y-2 text-slate-700">
            <li>
                ViV <span className="font-semibold">candidate trajectories</span> in
                Korea vs Singapore under a common modelling framework.
            </li>
            <li>
                Realised ViV under shared penetration scenarios (e.g. US‑style ramp)
                to highlight structural differences rather than assumptions.
            </li>
            <li>
                Role of differing{" "}
                <span className="font-semibold">demographic structures</span> and
                risk profiles in shaping future ViV burden.
            </li>
            </ul>
            <div className="mt-4 text-[11px] text-slate-600 bg-slate-50 border border-slate-200 rounded-lg p-3">
            This should allow a cleaner, side‑by‑side comparison than the original
            US–Japan paper, with both countries analysed using the same
            demography‑anchored engine.
            </div>
        </div>
        </div>
    </div>
    ),


    // --- V5 SLIDE 26: PUBLICATION PATH & FEEDBACK --------------------

    Slide26: () => (
    <div className="p-12 h-full flex flex-col animate-slide-enter">
        <h2 className="text-3xl font-bold text-slate-800 mb-4">
        Publication Path &amp; Feedback We&apos;re Seeking
        </h2>

        <div className="flex-1 bg-slate-50 border border-slate-200 rounded-xl p-6 grid grid-cols-2 gap-6">
        <div className="bg-white border border-slate-200 rounded-lg p-4 text-xs flex flex-col">
            <h3 className="text-sm font-semibold text-slate-800 mb-2">
            Target &amp; Paper Structure
            </h3>
            <ul className="list-disc pl-4 space-y-2 text-slate-700">
            <li>
                Potential outlets:{" "}
                <span className="font-semibold">
                JACC Asia, EHJ, or regional cardiology journals
                </span>{" "}
                (to be discussed).
            </li>
            <li>
                <span className="font-semibold">Methods:</span> emphasise the
                demography‑anchored risk × population design and Monte Carlo engine.
            </li>
            <li>
                <span className="font-semibold">Results:</span> focus on ViV
                candidates, with realised ViV presented as scenario‑based overlays.
            </li>
            <li>
                <span className="font-semibold">Comparison:</span> position as a
                constructive critique / extension of Genereux &amp; Ohno for Asian
                populations.
            </li>
            </ul>
        </div>

        <div className="bg-white border border-slate-200 rounded-lg p-4 text-xs flex flex-col">
            <h3 className="text-sm font-semibold text-slate-800 mb-2">
            To‑Do &amp; Feedback Needed
            </h3>
            <ul className="list-disc pl-4 space-y-2 text-slate-700">
            <li>Finalize the Singapore analysis and comparative figures.</li>
            <li>
                Run sensitivity analyses:
                <ul className="list-disc pl-4 mt-1 space-y-1">
                <li>Choice of risk years (2023 vs 2024 vs average).</li>
                <li>Alternative survival/durability curves.</li>
                <li>Different penetration scenarios.</li>
                </ul>
            </li>
            <li>
                Clean and document the code repository for{" "}
                <span className="font-semibold">reproducibility</span> in peer
                review.
            </li>
            <li>
                Feedback requested on:
                <ul className="list-disc pl-4 mt-1 space-y-1">
                <li>
                    How convincing the demography‑anchored framing is for clinicians.
                </li>
                <li>
                    Which comparison metrics between Korea and Singapore would be
                    most impactful.
                </li>
                <li>Preferred journals / conferences and authorship structure.</li>
                </ul>
            </li>
            </ul>
        </div>
        </div>
    </div>
    ),


}


// --- VERSION 4 SLIDES ---

const V4 = {



  Slide6: () => (
  <div className="p-12 h-full flex flex-col animate-slide-enter">
    <h2 className="text-3xl font-bold text-slate-800 mb-6">
      Our New Demography-Anchored Framework
    </h2>

    <div className="grid grid-cols-3 gap-6 mb-6">
      <div className="bg-slate-50 p-4 rounded-xl border border-slate-200">
        <h3 className="text-sm font-bold text-blue-700 mb-2 uppercase">
          Step 1 · Data
        </h3>
        <ul className="text-sm text-slate-600 space-y-1">
          <li>HIRA registry (TAVI / SAVR / redo-SAVR).</li>
          <li>Counts by year × sex × 5-year age band.</li>
          <li>National population projections (age + sex).</li>
        </ul>
      </div>

      <div className="bg-slate-50 p-4 rounded-xl border border-slate-200">
        <h3 className="text-sm font-bold text-blue-700 mb-2 uppercase">
          Step 2 · Risk × Population
        </h3>
        <ul className="text-sm text-slate-600 space-y-1">
          <li>Estimate per-capita risk in 2023–2024 only.</li>
          <li>Risk for TAVI, SAVR, and redo-SAVR separately.</li>
          <li>Apply fixed risks to future age–sex structure.</li>
        </ul>
      </div>

      <div className="bg-slate-50 p-4 rounded-xl border border-slate-200">
        <h3 className="text-sm font-bold text-blue-700 mb-2 uppercase">
          Step 3 · Monte Carlo
        </h3>
        <ul className="text-sm text-slate-600 space-y-1">
          <li>Simulate durability + survival per patient.</li>
          <li>Track failures, deaths, redo-SAVR, ViV candidates.</li>
          <li>
            Penetration is now a <span className="font-semibold">scenario</span>,
            not a built-in assumption.
          </li>
        </ul>
      </div>
    </div>

    <div className="mt-4 bg-blue-50 border border-blue-100 rounded-xl p-6">
      <p className="text-sm text-blue-900">
        <span className="font-bold">Key idea:</span> instead of extrapolating raw
        TAVI/SAVR volumes, we hold age–sex specific risks at their 2023–2024
        levels and let the <span className="font-semibold">ageing Korean
        population</span> drive future procedure volumes.
      </p>
    </div>
  </div>
  ),

  Slide7: () => (
  <div className="p-12 h-full flex flex-col animate-slide-enter">
    <h2 className="text-3xl font-bold text-slate-800 mb-4">
      Demography & Baseline Risk (2023–2024)
    </h2>
    <div className="grid grid-cols-2 gap-8 flex-1">
      <div className="bg-slate-50 rounded-xl border border-slate-200 p-6 flex flex-col">
        <h3 className="text-lg font-semibold text-slate-700 mb-2">
          1. Korea&apos;s Ageing Population
        </h3>
        <p className="text-sm text-slate-600 mb-4">
          We rebuild Ministry projections into yearly population counts by
          year × sex × 5-year age band (50–54, 55–59, …, ≥85), then down to
          single-year ages.
        </p>
        <div className="flex-1 bg-white rounded-lg border border-slate-200 overflow-hidden flex items-center justify-center">
          <img
            src="images/korea_population_trends.png"
            alt="Projected Korean population by age band and sex"
            className="w-full h-64 object-contain"
          />
        </div>
        <p className="mt-3 text-xs text-slate-500">
          Strong growth in ≥75 and ≥85 age bands is the primary driver of
          future ViV demand.
        </p>
      </div>

      <div className="bg-slate-50 rounded-xl border border-slate-200 p-6 flex flex-col">
        <h3 className="text-lg font-semibold text-slate-700 mb-2">
          2. Per-Capita Risk of TAVI / SAVR / Redo-SAVR
        </h3>
        <p className="text-sm text-slate-600 mb-4">
          For 2023–2024, we compute procedure risk as:
        </p>
        <p className="text-sm font-mono bg-slate-900 text-slate-100 px-3 py-2 rounded mb-4">
          risk(age, sex) = procedures / population
        </p>
        <p className="text-sm text-slate-600 mb-4">
          Risks are derived separately for TAVI, SAVR, and redo-SAVR and then
          averaged across the two years to reflect current practice and
          reimbursement.
        </p>
        <div className="flex-1 bg-white rounded-lg border border-slate-200 overflow-hidden flex items-center justify-center">
          <img
            src="images/baseline_risk_scores_by_sex.png"
            alt="Baseline procedure risk by age-band and sex"
            className="w-full h-64 object-contain"
          />
        </div>
        <p className="mt-3 text-xs text-slate-500">
          Risk is sharply concentrated in the oldest age bands and differs
          modestly by sex.
        </p>
      </div>
    </div>
  </div>
  ),

  Slide8: () => (
  <div className="p-12 h-full flex flex-col animate-slide-enter">
    <h2 className="text-3xl font-bold text-slate-800 mb-4">
      Key Result: ViV Candidates vs Realised ViV
    </h2>

    <div className="bg-slate-50 rounded-xl border border-slate-200 p-6 mb-6">
      <div className="flex flex-col md:flex-row gap-6">
        <div className="flex-1 flex items-center justify-center">
          <div className="bg-white rounded-lg border border-slate-200 p-4 w-full">
            <img
              src="images/image_D_viv_candidates_vs_realized_2022_2050.png"
              alt="Available ViV candidates vs realised procedures, 2022–2050"
              className="w-full h-64 object-contain"
            />
            <p className="mt-2 text-xs text-slate-500 text-center">
              Bars: total ViV candidates (TAVR-in-SAVR + TAVR-in-TAVR).
              Solid lines: candidates by type. Dashed lines: realised ViV under
              a penetration scenario.
            </p>
          </div>
        </div>

        <div className="flex-1 flex flex-col justify-between">
          <div>
            <h3 className="text-lg font-semibold text-slate-700 mb-2">
              What the simulation shows (2025–2035 focus)
            </h3>
            <ul className="text-sm text-slate-600 space-y-2">
              <li>
                Total ViV candidates increase from{" "}
                <span className="font-semibold">≈791</span> in 2024 to{" "}
                <span className="font-semibold">≈1,973</span> in 2035
                &nbsp;(<span className="font-semibold">~2.5×</span>).
              </li>
              <li>
                Growth is gradual and driven by the ageing population, not by
                infinite linear growth of index TAVI.
              </li>
              <li>
                Under a US-style penetration scenario, realised ViV rises from{" "}
                <span className="font-semibold">≈325</span> (2023) to{" "}
                <span className="font-semibold">≈1,078</span> (2035) — around{" "}
                <span className="font-semibold">3×</span>, far less explosive
                than the 7–9× increases reported for US/Japan.
              </li>
            </ul>
          </div>
          <div className="mt-4 bg-emerald-50 border border-emerald-100 rounded-lg p-3 text-xs text-emerald-900">
            <span className="font-semibold">Interpretation:</span> anchoring the
            model to age–sex specific risks and Korean demography produces a
            substantial but realistic ViV wave, suitable for capacity planning
            and for comparison with other countries.
          </div>
        </div>
      </div>
    </div>

    <p className="text-xs text-slate-500">
      Note: years before 2022 are omitted here because we lack consistent
      age–sex population detail for those years; the forecast window is aligned
      with the demography precompute.
    </p>
  </div>
  ),


  Slide9: () => (
  <div className="p-12 h-full flex flex-col animate-slide-enter">
    <h2 className="text-3xl font-bold text-slate-800 mb-6">
      From Index Procedures to ViV Candidates
    </h2>

    <div className="bg-slate-50 border border-slate-200 rounded-xl p-6 mb-6">
      <p className="text-sm text-slate-700 mb-4">
        The core of <span className="font-mono">model_v9</span> is a
        patient-level pipeline: every index TAVI or SAVR contributes potential
        ViV candidates depending on durability and survival.
      </p>

      {/* Pipeline diagram */}
      <div className="flex flex-col md:flex-row items-stretch justify-between gap-4 text-xs text-slate-700">
        <div className="flex-1 flex flex-col items-center">
          <div className="w-full max-w-xs bg-blue-50 border border-blue-200 rounded-lg p-3">
            <p className="font-semibold text-blue-700 mb-1">Index TAVI / SAVR</p>
            <p>Generated by risk × population (age &amp; sex specific).</p>
          </div>
          <div className="my-2 text-2xl text-slate-400">↓</div>
          <p className="text-[11px] text-slate-500">Year × age × sex × risk</p>
        </div>

        <div className="flex-1 flex flex-col items-center">
          <div className="w-full max-w-xs bg-purple-50 border border-purple-200 rounded-lg p-3">
            <p className="font-semibold text-purple-700 mb-1">Monte Carlo</p>
            <p>
              Draw durability (valve) and survival (patient) for each index
              procedure.
            </p>
          </div>
          <div className="my-2 text-2xl text-slate-400">↓</div>
          <p className="text-[11px] text-slate-500">
            Failure year &amp; death year for each patient
          </p>
        </div>

        <div className="flex-1 flex flex-col items-center">
          <div className="w-full max-w-xs bg-emerald-50 border border-emerald-200 rounded-lg p-3">
            <p className="font-semibold text-emerald-700 mb-1">
              ViV-Eligible Failures
            </p>
            <p>
              Failures where the patient is still alive and within the forecast
              window.
            </p>
          </div>
          <div className="my-2 text-2xl text-slate-400">↓</div>
          <p className="text-[11px] text-slate-500">
            TAVR-in-SAVR &amp; TAVR-in-TAVR candidates
          </p>
        </div>

        <div className="flex-1 flex flex-col items-center">
          <div className="w-full max-w-xs bg-rose-50 border border-rose-200 rounded-lg p-3">
            <p className="font-semibold text-rose-700 mb-1">
              Redo-SAVR &amp; ViV
            </p>
            <p>
              Subtract redo-SAVR targets. Optionally apply ViV penetration
              scenarios to estimate realised ViV.
            </p>
          </div>
          <div className="my-2 text-2xl text-slate-400">↓</div>
          <p className="text-[11px] text-slate-500">
            Final outputs: candidates &amp; realised ViV
          </p>
        </div>
      </div>
    </div>

    <p className="text-xs text-slate-500">
      This patient-level pipeline is repeated for many Monte Carlo runs; the
      model reports means and variability (SD) across runs.
    </p>
  </div>
  ),

  Slide10: () => (
  <div className="p-12 h-full flex flex-col animate-slide-enter">
    <h2 className="text-3xl font-bold text-slate-800 mb-4">
      Index Volumes: Observed vs Forecast (Risk × Population)
    </h2>

    <div className="grid md:grid-cols-2 gap-8 flex-1">
      <div className="bg-slate-50 rounded-xl border border-slate-200 p-6 flex flex-col">
        <h3 className="text-sm font-semibold text-slate-700 mb-2 uppercase tracking-wide">
          A. TAVI Index — Observed &amp; Projected
        </h3>
        <p className="text-sm text-slate-700 mb-3">
          Observed 2015–2024 volumes are preserved, including the 2023 spike;
          from 2025 onwards, volumes arise from risk × population instead of
          straight-line extrapolation.
        </p>
        <div className="flex-1 bg-white rounded-lg border border-slate-200 flex items-center justify-center overflow-hidden">
          <img
            src="images/tavi_index_projection.png"
            alt="Observed vs projected TAVI index volumes"
            className="w-full h-64 object-contain"
          />
        </div>
        <p className="mt-2 text-xs text-slate-500">
          Solid = observed; dashed / shaded = population-rate projection.
        </p>
      </div>

      <div className="bg-slate-50 rounded-xl border border-slate-200 p-6 flex flex-col">
        <h3 className="text-sm font-semibold text-slate-700 mb-2 uppercase tracking-wide">
          B. SAVR Index — Observed &amp; Projected
        </h3>
        <p className="text-sm text-slate-700 mb-3">
          SAVR projections reflect underlying age structure without forcing a
          flat or arbitrarily increasing trajectory.
        </p>
        <div className="flex-1 bg-white rounded-lg border border-slate-200 flex items-center justify-center overflow-hidden">
          <img
            src="images/savr_index_projection.png"
            alt="Observed vs projected SAVR index volumes"
            className="w-full h-64 object-contain"
          />
        </div>
        <p className="mt-2 text-xs text-slate-500">
          Projection respects demographic shifts; no hard-coded &quot;flat
          SAVR&quot; assumption.
        </p>
      </div>
    </div>

    <div className="mt-4 text-xs text-slate-500">
      (Figures exported from <span className="font-mono">figures/index</span> in
      the run folder and copied into{" "}
      <span className="font-mono">html_slides/2nd_dec/images</span>.)
    </div>
  </div>
  ),

  Slide11: () => (
  <div className="p-12 h-full flex flex-col animate-slide-enter">
    <h2 className="text-3xl font-bold text-slate-800 mb-4">
      ViV Candidates Over Time (TAVR-in-SAVR &amp; TAVR-in-TAVR)
    </h2>

    <div className="bg-slate-50 rounded-xl border border-slate-200 p-6 flex-1 flex flex-col md:flex-row gap-6">
      <div className="flex-1 flex items-center justify-center">
        <div className="w-full bg-white rounded-lg border border-slate-200 p-4">
          <img
            src="images/image_D_viv_candidates_vs_realized_2022_2050.png"
            alt="ViV candidates vs realised procedures (2022–2050)"
            className="w-full h-72 object-contain"
          />
          <p className="mt-2 text-xs text-slate-500 text-center">
            Image D: Available ViV candidates vs realised procedures
            (post–redo-SAVR), 2022–2050.
          </p>
        </div>
      </div>

      <div className="flex-1 flex flex-col justify-between">
        <div>
          <h3 className="text-sm font-semibold text-slate-700 mb-3 uppercase tracking-wide">
            Main Findings (Candidates)
          </h3>
          <ul className="text-sm text-slate-700 space-y-2 mb-3">
            <li>
              Total ViV candidates rise from{" "}
              <span className="font-semibold">≈791</span> in 2024 to{" "}
              <span className="font-semibold">≈1,973</span> in 2035 (~2.5×).
            </li>
            <li>
              Both TAVR-in-SAVR and TAVR-in-TAVR candidates increase, with
              TAVR-in-TAVR contributing a growing share in later years.
            </li>
            <li>
              The trend is smooth after extending redo-SAVR targets back to the
              simulation start year (no artificial 2024→2025 dip).
            </li>
          </ul>
        </div>

        <div className="bg-emerald-50 border border-emerald-100 rounded-lg p-4 text-xs text-emerald-900">
          <p>
            <span className="font-semibold">Interpretation:</span> even with
            stable per-capita treatment risks, the ageing population alone is
            sufficient to produce a substantial ViV wave over the next decade.
          </p>
        </div>
      </div>
    </div>
  </div>
  ),

  Slide12: () => (
  <div className="p-12 h-full flex flex-col animate-slide-enter">
    <h2 className="text-3xl font-bold text-slate-800 mb-4">
      Realised ViV Under a US‑Style Penetration Scenario
    </h2>

    <div className="bg-slate-50 rounded-xl border border-slate-200 p-6 flex-1 flex flex-col md:flex-row gap-6">
      <div className="flex-1 flex items-center justify-center">
        <div className="w-full bg-white rounded-lg border border-slate-200 p-4">
          <img
            src="images/image_C_viv_pretty_2025_2035.png"
            alt="Realised ViV volumes 2025–2035 under penetration scenario"
            className="w-full h-72 object-contain"
          />
          <p className="mt-2 text-xs text-slate-500 text-center">
            Image C: realised ViV volumes (TAVR-in-SAVR, TAVR-in-TAVR, and
            total) for 2025–2035 under a 10%→60% penetration ramp.
          </p>
        </div>
      </div>

      <div className="flex-1 flex flex-col justify-between">
        <div>
          <h3 className="text-sm font-semibold text-slate-700 mb-3 uppercase tracking-wide">
            Comparison with US/Japan Forecasts
          </h3>
          <ul className="text-sm text-slate-700 space-y-2 mb-3">
            <li>
              Total realised ViV increases from{" "}
              <span className="font-semibold">≈325</span> (2023) to{" "}
              <span className="font-semibold">≈1,078</span> (2035) — roughly{" "}
              <span className="font-semibold">3‑fold</span>.
            </li>
            <li>
              Ohno / Genereux report ~7× growth in the US and ~9× in Japan over
              a similar horizon under extrapolated index volumes.
            </li>
            <li>
              Our smaller multiple reflects demography‑anchored index volumes
              instead of aggressive linear growth.
            </li>
          </ul>
        </div>

        <div className="bg-amber-50 border border-amber-100 rounded-lg p-4 text-xs text-amber-900">
          <p>
            <span className="font-semibold">Key point:</span> even with the{" "}
            <span className="italic">same</span> penetration assumptions, a
            demography‑based index model yields a more moderate but still
            clinically significant ViV surge.
          </p>
        </div>
      </div>
    </div>
  </div>
  ),

  Slide13: () => (
  <div className="p-12 h-full flex flex-col animate-slide-enter">
    <h2 className="text-3xl font-bold text-slate-800 mb-4">
      Patient Flow Anatomy: Failures, Redo-SAVR, and ViV
    </h2>

    <div className="grid md:grid-cols-2 gap-8 flex-1">
      {/* Conceptual stacked bar diagram */}
      <div className="bg-slate-50 border border-slate-200 rounded-xl p-6 flex flex-col">
        <h3 className="text-sm font-semibold text-slate-700 mb-3 uppercase tracking-wide">
          A. Conceptual Flow for a Single Year
        </h3>
        <p className="text-sm text-slate-700 mb-4">
          For each year we can decompose TAVR-in-SAVR and TAVR-in-TAVR pathways
          into a sequence of steps:
        </p>

        <div className="flex items-end justify-around flex-1 mb-4">
          <div className="flex flex-col items-center w-16">
            <div className="w-full h-28 bg-slate-300 rounded-t" />
            <span className="mt-2 text-[11px] text-slate-600 text-center">
              Failures
            </span>
          </div>
          <div className="flex flex-col items-center w-16">
            <div className="w-full h-24 bg-emerald-300 rounded-t" />
            <span className="mt-2 text-[11px] text-slate-600 text-center">
              ViV‑eligible
            </span>
          </div>
          <div className="flex flex-col items-center w-16">
            <div className="w-full h-10 bg-rose-400 rounded-t" />
            <span className="mt-2 text-[11px] text-slate-600 text-center">
              Redo‑SAVR
            </span>
          </div>
          <div className="flex flex-col items-center w-16">
            <div className="w-full h-14 bg-blue-400 rounded-t" />
            <span className="mt-2 text-[11px] text-slate-600 text-center">
              ViV
            </span>
          </div>
          <div className="flex flex-col items-center w-16">
            <div className="w-full h-10 bg-slate-200 rounded-t" />
            <span className="mt-2 text-[11px] text-slate-600 text-center">
              Remaining
            </span>
          </div>
        </div>

        <p className="text-xs text-slate-500">
          In the code this breakdown is captured in{" "}
          <span className="font-mono">patient_flow_mean.csv</span> and the
          waterfall helper functions.
        </p>
      </div>

      {/* Placeholder for real waterfall figures */}
      <div className="bg-slate-50 border border-slate-200 rounded-xl p-6 flex flex-col">
        <h3 className="text-sm font-semibold text-slate-700 mb-3 uppercase tracking-wide">
          B. Waterfall Plots (2025, 2030, 2035)
        </h3>
        <p className="text-sm text-slate-700 mb-4">
          We also generate explicit waterfall plots from the simulation outputs
          to double-check accounting for key years.
        </p>

        <div className="flex-1 grid grid-cols-1 gap-3">
          <div className="h-24 rounded-lg border border-dashed border-slate-300 flex items-center justify-center text-[11px] text-slate-500">
            (Insert <span className="font-mono">waterfall_TAVR-in-SAVR_2025.png</span>)
          </div>
          <div className="h-24 rounded-lg border border-dashed border-slate-300 flex items-center justify-center text-[11px] text-slate-500">
            (Insert <span className="font-mono">waterfall_TAVR-in-SAVR_2030.png</span>)
          </div>
          <div className="h-24 rounded-lg border border-dashed border-slate-300 flex items-center justify-center text-[11px] text-slate-500">
            (Insert <span className="font-mono">waterfall_TAVR-in-TAVR_2035.png</span>)
          </div>
        </div>

        <p className="mt-3 text-xs text-slate-500">
          These figures can be exported from{" "}
          <span className="font-mono">figures/flow</span> and dropped into{" "}
          <span className="font-mono">images/</span>.
        </p>
      </div>
    </div>
  </div>
  ),

  Slide14: () => (
  <div className="p-12 h-full flex flex-col animate-slide-enter">
    <h2 className="text-3xl font-bold text-slate-800 mb-6">
      How Our Approach Differs from Genereux / Ohno
    </h2>

    <div className="grid md:grid-cols-2 gap-6 flex-1">
      <div className="bg-slate-900 text-slate-50 rounded-xl p-6 flex flex-col">
        <h3 className="text-sm font-semibold text-amber-300 mb-3 uppercase tracking-wide">
          &quot;Standard&quot; US / Japan Models
        </h3>
        <ul className="text-sm space-y-2 mb-4">
          <li>Index TAVI / SAVR volumes extrapolated linearly.</li>
          <li>SAVR often held flat; TAVI grows indefinitely.</li>
          <li>
            ViV penetration curves tuned to match observed ViV counts (where
            available).
          </li>
          <li>Demography and redo-SAVR only partially addressed.</li>
        </ul>
        <div className="mt-auto text-xs text-slate-400">
          Editorial critique: heavy reliance on extrapolation and limited
          treatment of demographic / clinical factors.
        </div>
      </div>

      <div className="bg-emerald-50 text-slate-900 rounded-xl p-6 flex flex-col">
        <h3 className="text-sm font-semibold text-emerald-800 mb-3 uppercase tracking-wide">
          Our Demography-Anchored Korean Model
        </h3>
        <ul className="text-sm space-y-2 mb-4">
          <li>
            Index volumes built from{" "}
            <span className="font-semibold">risk × population</span>, anchored
            to 2023–24.
          </li>
          <li>
            ViV <span className="font-semibold">candidates</span> are the
            primary output; penetration is an explicit scenario layer.
          </li>
          <li>
            Redo-SAVR volumes projected via the same framework and fed in as
            external targets.
          </li>
          <li>
            Ageing and sex structure made explicit via precompute demography
            tables.
          </li>
        </ul>
        <div className="mt-auto text-xs text-emerald-900">
          Result: a more realistic, demography-aware forecast that still uses
          the familiar Monte Carlo engine but answers a slightly different (and
          more clinically relevant) question.
        </div>
      </div>
    </div>
  </div>
  ),

  Slide15: () => (
  <div className="p-12 h-full flex flex-col animate-slide-enter">
    <h2 className="text-3xl font-bold text-slate-800 mb-4">
      Next Steps: Validation, Singapore, and Publication
    </h2>

    <div className="grid md:grid-cols-3 gap-6 flex-1 mb-4">
      <div className="bg-slate-50 border border-slate-200 rounded-xl p-6 flex flex-col">
        <h3 className="text-sm font-semibold text-slate-700 mb-2 uppercase tracking-wide">
          1. Technical Validation
        </h3>
        <ul className="text-sm text-slate-700 space-y-2">
          <li>Additional QC on index projections and candidate counts.</li>
          <li>
            Sensitivity runs:
            <ul className="list-disc pl-4 text-xs text-slate-600 space-y-1 mt-1">
              <li>Risk years (2023 only vs 2023–24).</li>
              <li>Alternative durability / survival curves.</li>
              <li>Different ViV penetration scenarios.</li>
            </ul>
          </li>
        </ul>
      </div>

      <div className="bg-slate-50 border border-slate-200 rounded-xl p-6 flex flex-col">
        <h3 className="text-sm font-semibold text-slate-700 mb-2 uppercase tracking-wide">
          2. Singapore Extension
        </h3>
        <ul className="text-sm text-slate-700 space-y-2">
          <li>Replicate the same pipeline with Singapore registry data.</li>
          <li>
            Build Singapore-specific demography &amp; risk tables, then reuse
            the same Monte Carlo engine.
          </li>
          <li>
            Compare ViV candidate &amp; realised trajectories between Korea and
            Singapore under shared scenarios.
          </li>
        </ul>
      </div>

      <div className="bg-slate-50 border border-slate-200 rounded-xl p-6 flex flex-col">
        <h3 className="text-sm font-semibold text-slate-700 mb-2 uppercase tracking-wide">
          3. Manuscript &amp; Sharing
        </h3>
        <ul className="text-sm text-slate-700 space-y-2">
          <li>Decide on target journal / conference (e.g. JACC Asia).</li>
          <li>Prepare literature review &amp; methods section emphasising the demography-based design.</li>
          <li>
            Clean up <span className="font-mono">model_v9</span> repo and run
            configs for reproducibility (for reviewers and collaborators).
          </li>
        </ul>
      </div>
    </div>

    <div className="bg-blue-50 border border-blue-100 rounded-xl p-4 text-xs text-blue-900">
      <span className="font-semibold">Discussion prompt:</span> Are there any
      additional clinical factors, local policy changes, or device innovations
      that we should explicitly encode as future scenario knobs before we move
      to the Singapore analysis and manuscript drafting?
    </div>
  </div>
  ),


};

// --- VERSION 1 SLIDES ---

const V1 = {

  Slide1: () => (
  <div className="flex flex-col items-center justify-center h-full bg-slate-900 text-white p-12 text-center animate-slide-enter">
    <div className="mb-6 p-4 bg-blue-600 rounded-full">
      <Icons.Activity />
    </div>
    <div className="text-sm font-mono text-blue-300 mb-2">
      SIMULATION REPORT
    </div>
    <h1 className="text-4xl font-bold mb-4">
      Forecasting ViV TAVI in Korea (2025-2035)
    </h1>
    <p className="text-lg text-slate-300 mb-8 max-w-2xl">
      A Demography-Anchored Monte Carlo Simulation to estimate clinical demand,
      correcting for post-COVID anomalies.
    </p>
    <div className="border-t border-slate-700 pt-6 w-full max-w-md flex justify-between text-sm text-slate-400">
      <span>HYUNJIN AHN, CHARLES YAP</span>
      <span className="text-green-400">Status: Validated (Nov 2024)</span>
    </div>
  </div>
  ),

  Slide2: () => (
  <div className="p-12 h-full flex flex-col animate-slide-enter">
    <h2 className="text-3xl font-bold text-slate-800 mb-8">
      Background & Original Goal
    </h2>
    <div className="bg-slate-50 p-6 rounded-xl border border-slate-200 mb-6">
      <h3 className="text-xl font-bold text-blue-700 mb-2">Original Goal</h3>
      <p className="text-slate-700">
        Replicate the methodology proposed by Genereux et al. and Ohno et al. to
        predict ViV surges.
      </p>
    </div>
    <div className="space-y-4">
      <h4 className="font-bold text-slate-700">
        Original Methodology (Genereux/Ohno)
      </h4>
      <ul className="list-disc pl-6 space-y-2 text-slate-600">
        <li>Linear extrapolation of TAVI/SAVR rates.</li>
        <li>Speculative "Penetration Curves" (assuming 60-80% uptake).</li>
        <li>Applied to US and Japanese registries.</li>
      </ul>
    </div>
  </div>
  ),

  Slide3: () => (
  <div className="p-12 h-full flex flex-col items-center justify-center animate-slide-enter">
    <h2 className="text-2xl font-bold text-slate-800 mb-4 self-start">
      Context: The Original Goal
    </h2>
    <div className="bg-blue-50 p-8 rounded-xl w-full max-w-3xl border border-blue-100 text-center">
      <div className="flex items-end justify-center space-x-4 h-48 mb-4">
        <div className="w-12 bg-blue-300 h-12"></div>
        <div className="w-12 bg-blue-400 h-24"></div>
        <div className="w-12 bg-blue-500 h-36"></div>
        <div className="w-12 bg-blue-600 h-48"></div>
        <div className="w-12 bg-red-500 h-64 shadow-lg relative">
          <span className="absolute -top-6 left-2 text-xs text-red-600 font-bold">
            Exponential
          </span>
        </div>
      </div>
      <p className="text-sm text-slate-500 italic">
        Representative Chart: Aggressive Linear Extrapolation
      </p>
    </div>
    <div className="mt-8 text-slate-600 text-sm max-w-2xl">
      This standard methodology assumes linear growth and applies a speculative
      penetration curve (e.g. 60% capture), often predicting massive/exponential
      growth.
    </div>
  </div>
  ),

  Slide4: () => (
  <div className="p-12 h-full flex flex-col animate-slide-enter">
    <h2 className="text-3xl font-bold text-slate-800 mb-6">
      Why the Original Approach Failed
    </h2>
    <h3 className="text-xl font-bold text-red-600 mb-2">
      1. The 2023 Data Spike
    </h3>
    <p className="text-slate-600 mb-4">
      Korean registry data showed a massive, anomalous spike in 2023 (Post-COVID
      Backlog?).
    </p>

    <div className="flex-1 bg-slate-50 rounded-xl border border-slate-200 p-8 relative flex items-end justify-around">
      <div className="w-16 bg-slate-300 h-1/4 rounded-t"></div>
      <div className="w-16 bg-slate-300 h-1/4 rounded-t"></div>
      <div className="w-16 bg-slate-300 h-1/3 rounded-t"></div>
      <div className="w-16 bg-red-500 h-full rounded-t relative group">
        <div className="absolute -top-8 left-1/2 -translate-x-1/2 bg-red-100 text-red-800 text-xs font-bold px-2 py-1 rounded">
          Backlog?
        </div>
      </div>
      <div className="w-16 bg-slate-300 h-1/2 rounded-t opacity-50"></div>
    </div>
    <p className="text-center text-red-500 font-bold mt-4 text-sm">
      Linear extrapolation from 2023 creates infinite unrealistic growth.
    </p>
  </div>
  ),

  Slide5: () => (
  <div className="p-12 h-full flex flex-col animate-slide-enter">
    <h2 className="text-3xl font-bold text-slate-800 mb-4">
      Research Context & The Pivot
    </h2>
    <div className="grid grid-cols-2 gap-8 h-full">
      <div className="bg-slate-100 p-6 rounded-xl">
        <h3 className="font-bold text-slate-600 mb-4">
          A. The "Standard" Model
        </h3>
        <ul className="text-sm space-y-2 text-slate-500">
          <li>Based on Genereux et al. (USA) / Ohno et al. (Japan).</li>
          <li>Input: Historical TAVI/SAVR volumes.</li>
          <li>Projection: Linear regression ($y=mx+c$).</li>
          <li>Predicts 7-9 fold increase.</li>
        </ul>
      </div>
      <div className="bg-red-50 p-6 rounded-xl border border-red-100">
        <h3 className="font-bold text-red-700 mb-4">
          B. Why it failed in Korea
        </h3>
        <div className="mb-4">
          <strong className="text-red-900 text-sm">
            1. The 2023 "Backlog Spike"
          </strong>
          <p className="text-xs text-red-800 mt-1">
            Massive outlier makes linear regression impossible.
          </p>
        </div>
        <div>
          <strong className="text-red-900 text-sm">
            2. The Demographic Disconnect
          </strong>
          <p className="text-xs text-red-800 mt-1">
            Standard models fail to account for Asian-specific aging velocity
            (JACC Critique).
          </p>
        </div>
      </div>
    </div>
    <div className="mt-4 text-center font-bold text-blue-800">
      Conclusion: We need a model anchored to Risk & Demography, not just
      historical lines.
    </div>
  </div>
  ),

  Slide6: () => (
  <div className="flex items-center justify-center h-full bg-blue-600 text-white p-12 animate-slide-enter">
    <div className="text-center">
      <div className="mb-6 flex justify-center">
        <div className="p-6 bg-white/20 rounded-full">
          <Icons.Target />
        </div>
      </div>
      <h2 className="text-4xl font-bold mb-4">New Model (v9)</h2>
      <h3 className="text-2xl font-bold text-blue-200 mb-8">
        Risk × Demography
      </h3>
      <p className="text-xl italic max-w-2xl mx-auto text-blue-100 border-l-4 border-white pl-6 text-left">
        "Given the stable risk profile of a Korean patient, how many failures
        occur as the population ages?"
      </p>
    </div>
  </div>
  ),

  Slide7: () => (
  <div className="p-12 h-full flex flex-col animate-slide-enter">
    <h2 className="text-3xl font-bold text-slate-800 mb-8">
      Simulation Architecture (model_v9)
    </h2>
    <div className="flex items-center justify-between gap-4 h-64">
      <div className="bg-slate-800 text-white p-6 rounded-lg flex-1 h-full flex flex-col">
        <div className="bg-blue-500 w-8 h-8 rounded-full flex items-center justify-center font-bold mb-4">
          1
        </div>
        <h3 className="font-bold mb-2">Pre-Compute</h3>
        <p className="text-sm text-slate-400">
          Registry Data + Pop Projections
        </p>
      </div>
      <div className="text-slate-400">→</div>
      <div className="bg-slate-800 text-white p-6 rounded-lg flex-1 h-full flex flex-col">
        <div className="bg-blue-500 w-8 h-8 rounded-full flex items-center justify-center font-bold mb-4">
          2
        </div>
        <h3 className="font-bold mb-2">Risk Calculation</h3>
        <p className="text-sm text-slate-400">Risk = Obs / Population</p>
      </div>
      <div className="text-slate-400">→</div>
      <div className="bg-slate-800 text-white p-6 rounded-lg flex-1 h-full flex flex-col">
        <div className="bg-blue-500 w-8 h-8 rounded-full flex items-center justify-center font-bold mb-4">
          3
        </div>
        <h3 className="font-bold mb-2">Monte Carlo</h3>
        <p className="text-sm text-slate-400">
          100 runs, Durability vs Survival
        </p>
      </div>
    </div>
    <div className="mt-8 bg-slate-100 p-4 rounded border border-slate-200">
      <p className="text-sm font-bold text-slate-700">
        Logic: Extrapolate Pop, hold Risk constant. NO speculative penetration.
      </p>
    </div>
  </div>
  ),

  Slide8: () => (
  <div className="p-12 h-full flex flex-col animate-slide-enter">
    <h2 className="text-3xl font-bold text-slate-800 mb-6">
      The "Demography Anchor"
    </h2>
    <p className="text-slate-500 mb-8">
      Replacing linear trends with demographic probability.
    </p>

    <div className="grid grid-cols-2 gap-8">
      <div className="border-l-4 border-blue-500 pl-6 py-2">
        <h3 className="text-xl font-bold text-slate-800 mb-2">
          Step 1: Freeze the Risk
        </h3>
        <p className="text-slate-600 mb-4 text-sm">
          Calculate probability of procedure for every age/sex bracket based on
          2023-24 actuals.
        </p>
        <div className="bg-slate-100 p-3 rounded font-mono text-xs">
          Risk_Constant = Obs_Procedures(2024) / Total_Pop(2024)
        </div>
      </div>

      <div className="border-l-4 border-green-500 pl-6 py-2">
        <h3 className="text-xl font-bold text-slate-800 mb-2">
          Step 2: Apply to Future
        </h3>
        <p className="text-slate-600 mb-4 text-sm">
          Future volume is strictly a function of the aging population
          structure.
        </p>
        <div className="bg-slate-100 p-3 rounded font-mono text-xs">
          Future_Vol = Risk_Constant * Projected_Pop(Y)
        </div>
      </div>
    </div>
  </div>
  ),

  Slide9: () => (
  <div className="p-12 h-full flex flex-col animate-slide-enter">
    <h2 className="text-3xl font-bold text-slate-800 mb-4 flex items-center gap-3">
      <Icons.Activity /> 3. MC ENGINE
    </h2>
    <div className="flex-1 bg-white rounded-xl border border-slate-200 shadow-sm flex flex-col md:flex-row overflow-hidden">
      <div className="bg-slate-50 p-6 w-1/3 border-r border-slate-200">
        <h3 className="font-bold text-slate-700 mb-4">1. INPUTS</h3>
        <ul className="space-y-2 text-sm text-slate-600">
          <li>• Registry Data (2023-24)</li>
          <li>• Pop Stats (Ministry)</li>
          <li>• Age/Sex Bands</li>
        </ul>
      </div>
      <div className="p-6 w-1/3 border-r border-slate-200 bg-blue-50 flex flex-col justify-center text-center">
        <h3 className="font-bold text-blue-800 mb-2">ENGINE</h3>
        <div className="text-xs text-blue-600 space-y-1">
          <p>100 Runs</p>
          <p>Durability vs Survival</p>
          <p>Jitter/Uncertainty</p>
        </div>
      </div>
      <div className="bg-green-50 p-6 w-1/3">
        <h3 className="font-bold text-green-700 mb-4">4. OUTPUTS</h3>
        <ul className="space-y-2 text-sm text-green-800">
          <li>• ViV Candidates</li>
          <li>• Realized Volume</li>
          <li>• Visualization (PNG)</li>
        </ul>
      </div>
    </div>
  </div>
  ),

  Slide10: () => (
  <div className="p-12 h-full flex flex-col animate-slide-enter">
    <h2 className="text-3xl font-bold text-slate-800 mb-6">
      Step 1 & 2: Anchoring Logic
    </h2>
    <div className="flex gap-8 h-full">
      <div className="w-1/2">
        <p className="text-slate-600 mb-4">
          We assume risk per person remains stable, but the population structure
          shifts.
        </p>
        <div className="bg-slate-900 text-green-400 p-6 rounded-xl font-mono text-sm shadow-lg">
          <div className="text-slate-500"># Python Pseudocode</div>
          <div className="mt-2">def compute_risk_scores(obs, pop):</div>
          <div className="pl-4">for age_band in bands:</div>
          <div className="pl-8">risk = obs_count / total_pop</div>
          <div className="pl-8 text-slate-500"># e.g. Men, 75-79</div>
          <div className="pl-4">return risk_profile</div>
        </div>
      </div>
      <div className="w-1/2 flex flex-col items-center justify-center bg-slate-50 rounded-xl border border-slate-200">
        <div className="text-center space-y-4">
          <div>
            <div className="font-bold text-slate-400 text-xs">2024</div>
            <div className="font-bold">Small Elderly Pop x Risk</div>
          </div>
          <div className="text-2xl text-slate-400">↓</div>
          <div className="bg-blue-100 px-4 py-2 rounded text-blue-800 text-xs font-bold">
            Same Risk Multiplier
          </div>
          <div className="text-2xl text-slate-400">↓</div>
          <div>
            <div className="font-bold text-slate-400 text-xs">2035</div>
            <div className="font-bold text-blue-600">
              Massive Elderly Pop x Risk
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
  ),

  Slide11: () => (
  <div className="flex flex-col items-center justify-center h-full bg-slate-800 text-white p-12 animate-slide-enter">
    <h2 className="text-4xl font-bold mb-8">
      Step 3: The "Race" (Monte Carlo)
    </h2>
    <p className="text-xl text-slate-300 mb-12 text-center max-w-2xl">
      For every simulated patient in every run, we define "ViV Eligibility"
      based on a race between two events.
    </p>
    <div className="bg-white text-slate-900 p-8 rounded-xl shadow-2xl font-mono text-lg border-4 border-green-500">
      if (fail_year &le; death_year) AND (in_window):
      <br />
      <span className="text-green-600 font-bold ml-8">
        return True (ViV Candidate)
      </span>
    </div>
  </div>
  ),

  Slide12: () => (
  <div className="p-12 h-full flex flex-col animate-slide-enter">
    <h2 className="text-3xl font-bold text-slate-800 mb-6">
      Simulation Logic: The "Race"
    </h2>
    <div className="flex gap-8 items-center h-full">
      <div className="w-1/2 space-y-6">
        <div className="bg-yellow-50 p-4 rounded border-l-4 border-yellow-400">
          <h3 className="font-bold text-yellow-800">A. Durability Sampling</h3>
          <p className="text-sm text-slate-600">
            Sampled from bimodal distributions (TAVI vs SAVR).
          </p>
        </div>
        <div className="bg-slate-100 p-4 rounded border-l-4 border-slate-400">
          <h3 className="font-bold text-slate-800">B. Survival Sampling</h3>
          <p className="text-sm text-slate-600">
            Actuarial curves adjusted for risk category.
          </p>
        </div>
      </div>
      <div className="w-1/2 bg-white p-6 rounded-xl shadow-lg border border-slate-100 relative h-64">
        {/* Visualizing the race */}
        <div className="absolute top-1/3 left-4 right-4 h-2 bg-slate-200 rounded"></div>

        {/* Index */}
        <div className="absolute top-1/3 left-4 w-4 h-4 bg-blue-500 rounded-full -mt-1"></div>
        <div className="absolute top-1/4 left-4 text-xs font-bold text-blue-500">
          Index
        </div>

        {/* Failure */}
        <div className="absolute top-1/3 left-1/2 w-4 h-4 bg-yellow-500 rounded-full -mt-1 border-2 border-white z-10"></div>
        <div className="absolute top-1/4 left-1/2 -translate-x-1/2 text-xs font-bold text-yellow-600">
          Valve Fails
        </div>

        {/* Death */}
        <div className="absolute top-1/3 right-10 w-4 h-4 bg-slate-800 rounded-full -mt-1"></div>
        <div className="absolute top-1/4 right-10 text-xs font-bold text-slate-700">
          Death
        </div>

        <div className="absolute bottom-10 left-0 w-full text-center text-green-600 font-bold">
          Failure happens before Death = CANDIDATE
        </div>
      </div>
    </div>
  </div>
  ),

  Slide13: () => (
  <div className="p-12 h-full flex flex-col animate-slide-enter bg-slate-50">
    <h2 className="text-3xl font-bold text-slate-800 mb-2">
      Visualizing the Mechanism
    </h2>
    <h3 className="text-xl text-blue-600 font-bold mb-8">
      2035 DEMOGRAPHIC SHIFT
    </h3>

    <div className="space-y-8">
      <div className="flex items-center">
        <div className="w-32 font-bold text-right pr-4 text-slate-600">
          Pop (75+)
        </div>
        <div className="flex-1 h-16 bg-blue-600 rounded-r-xl text-white flex items-center px-4 font-bold shadow-lg">
          Massive Expansion
        </div>
      </div>

      <div className="flex items-center justify-center">
        <div className="bg-yellow-100 border-2 border-yellow-400 text-yellow-800 px-6 py-3 rounded-full font-mono font-bold flex items-center gap-2">
          <Icons.Alert /> Fixed Risk (0.5%)
        </div>
      </div>

      <div className="flex items-center">
        <div className="w-32 font-bold text-right pr-4 text-slate-600">
          Procedures
        </div>
        <div className="flex-1 h-12 bg-green-500 rounded-r-xl text-white flex items-center px-4 font-bold shadow-lg w-3/4">
          High Volume Output
        </div>
      </div>
    </div>

    <div className="mt-8 text-center italic text-slate-500">
      "We didn't change the rate. We just applied it to more people."
    </div>
  </div>
  ),

  Slide14: () => (
  <div className="p-12 h-full flex flex-col animate-slide-enter">
    <h2 className="text-3xl font-bold text-slate-800 mb-6">
      MC Engine: Setup & Uncertainty
    </h2>
    <div className="grid grid-cols-2 gap-8">
      <div className="bg-slate-800 text-green-400 p-6 rounded-xl font-mono text-sm">
        <div className="text-white border-b border-slate-600 pb-2 mb-4">
          sim_config.yaml
        </div>
        <div>n_runs: 100</div>
        <div>rng_seed: 2025</div>
        <div>window: 2022-2050</div>
      </div>

      <div className="bg-white border border-slate-200 p-6 rounded-xl shadow-sm">
        <h3 className="font-bold text-purple-700 mb-4">
          JITTER (Parameter Uncertainty)
        </h3>
        <p className="text-sm text-slate-600 mb-4">
          Multiplicative Gaussian noise applied per run.
        </p>
        <ul className="space-y-3 text-sm">
          <li className="flex justify-between border-b border-slate-100 pb-1">
            <span>Durability SD</span>
            <span className="font-mono font-bold">±7.5%</span>
          </li>
          <li className="flex justify-between border-b border-slate-100 pb-1">
            <span>Survival SD</span>
            <span className="font-mono font-bold">±5.0%</span>
          </li>
          <li className="flex justify-between border-b border-slate-100 pb-1">
            <span>Penetration SD</span>
            <span className="font-mono font-bold">±10.0%</span>
          </li>
        </ul>
      </div>
    </div>
  </div>
  ),

  Slide15: () => (
  <div className="flex flex-col items-center justify-center h-full bg-white p-12 animate-slide-enter">
    <h2 className="text-3xl font-bold text-slate-800 mb-12">
      The Core Logic: "The Race"
    </h2>

    <div className="flex items-center gap-4 text-2xl font-mono bg-slate-100 p-8 rounded-full shadow-inner mb-8">
      <span className="text-slate-500">if</span>
      <span className="bg-yellow-100 text-yellow-800 px-3 py-1 rounded border border-yellow-300">
        Fail_Year
      </span>
      <span className="text-slate-400">≤</span>
      <span className="bg-blue-100 text-blue-800 px-3 py-1 rounded border border-blue-300">
        Death_Year
      </span>
      <span className="text-slate-400">→</span>
      <span className="bg-green-500 text-white px-4 py-1 rounded font-bold shadow">
        CANDIDATE
      </span>
    </div>

    <div className="text-left w-full max-w-lg space-y-2 text-sm text-slate-500 font-mono">
      <div>Fail_Year = Index_Year + Sampled_Durability</div>
      <div>Death_Year = Index_Year + Sampled_Survival</div>
    </div>
  </div>
  ),

  Slide16: () => (
  <div className="p-12 h-full flex flex-col animate-slide-enter">
    <h2 className="text-3xl font-bold text-slate-800 mb-8">
      Simulation Scenarios
    </h2>

    {/* Scenario A */}
    <div className="mb-12">
      <div className="text-green-700 font-bold mb-2">A. ViV Candidate</div>
      <div className="relative h-12 bg-slate-50 rounded-full w-full flex items-center px-4">
        <div className="absolute h-2 bg-blue-300 w-[70%] top-4 rounded"></div>
        <div className="absolute h-2 bg-yellow-400 w-[50%] top-6 rounded z-10"></div>
        <div className="absolute left-[50%] top-0 h-full border-l-2 border-dashed border-yellow-600">
          <span className="bg-yellow-500 text-white text-[10px] px-1 rounded absolute -top-3 -left-4">
            Failure
          </span>
        </div>
        <div className="absolute left-[70%] top-0 h-full border-l-2 border-slate-400">
          <span className="text-slate-500 text-[10px] absolute top-12 -left-2">
            Death
          </span>
        </div>
        <div className="absolute left-[50%] w-[20%] h-full bg-green-100/50 flex items-center justify-center text-xs text-green-800 font-bold">
          Window
        </div>
      </div>
    </div>

    {/* Scenario B */}
    <div>
      <div className="text-red-700 font-bold mb-2">
        B. Non-Candidate (Died First)
      </div>
      <div className="relative h-12 bg-slate-50 rounded-full w-full flex items-center px-4 opacity-70">
        <div className="absolute h-2 bg-blue-300 w-[40%] top-4 rounded"></div>
        <div className="absolute h-2 bg-yellow-400 w-[60%] top-6 rounded z-10 opacity-50"></div>
        <div className="absolute left-[40%] top-0 h-full border-l-2 border-slate-400">
          <span className="bg-slate-600 text-white text-[10px] px-1 rounded absolute -top-3 -left-3">
            Death
          </span>
        </div>
        <div className="absolute left-[60%] top-0 h-full border-l-2 border-dashed border-yellow-600 opacity-50">
          <span className="bg-yellow-200 text-yellow-800 text-[10px] px-1 rounded absolute -top-3 -left-4">
            Failure
          </span>
        </div>
        <div className="absolute right-10 border-2 border-red-400 text-red-400 px-2 py-1 rounded -rotate-12 font-bold text-xs">
          DIED FIRST
        </div>
      </div>
    </div>
  </div>
  ),

  Slide17: () => (
  <div className="p-12 h-full flex flex-col animate-slide-enter">
    <h2 className="text-3xl font-bold text-slate-800 mb-6">
      Technical Architecture
    </h2>
    <div className="flex gap-4 h-full">
      <div className="flex-1 bg-white border border-slate-200 rounded p-4 shadow-sm flex flex-col">
        <h3 className="font-bold text-slate-400 text-sm mb-2">1. INGEST</h3>
        <div className="bg-slate-100 p-2 rounded mb-2 text-xs">
          tavi_counts.csv
        </div>
        <div className="bg-slate-100 p-2 rounded text-xs">korea_pop.csv</div>
      </div>
      <div className="flex-1 bg-blue-50 border border-blue-200 rounded p-4 shadow-sm flex flex-col">
        <h3 className="font-bold text-blue-400 text-sm mb-2">2. ANCHOR</h3>
        <div className="font-mono text-xs text-blue-800">compute_risk()</div>
        <p className="text-[10px] text-blue-600 mt-2">
          Calculates fixed risk profile per 5yr Age/Sex band.
        </p>
      </div>
      <div className="flex-1 bg-purple-50 border border-purple-200 rounded p-4 shadow-sm flex flex-col">
        <h3 className="font-bold text-purple-400 text-sm mb-2">3. ENGINE</h3>
        <div className="font-mono text-xs text-purple-800">
          ViVSimulator Class
        </div>
        <p className="text-[10px] text-purple-600 mt-2">
          n_runs=100, seed=2025
        </p>
      </div>
      <div className="flex-1 bg-green-50 border border-green-200 rounded p-4 shadow-sm flex flex-col">
        <h3 className="font-bold text-green-400 text-sm mb-2">4. ANALYTICS</h3>
        <div className="bg-green-100 p-2 rounded mb-2 text-xs text-green-800 font-bold">
          ViV Candidates
        </div>
        <div className="bg-green-100 p-2 rounded text-xs text-green-800">
          Realized Vol
        </div>
      </div>
    </div>
  </div>
  ),

  Slide18: () => (
  <div className="p-12 h-full flex flex-col animate-slide-enter">
    <h2 className="text-3xl font-bold text-slate-800 mb-2">
      Bimodal Mixture Model
    </h2>
    <p className="text-slate-500 mb-8">
      We assume valves fail in two distinct clusters: Early (Manufacturing) and
      Late (Wear).
    </p>

    <div className="flex-1 flex items-end justify-center pb-8 gap-1 relative border-b border-slate-300">
      {/* Early Cluster */}
      <div className="w-8 h-4 bg-yellow-300 rounded-t mx-1"></div>
      <div className="w-8 h-12 bg-yellow-400 rounded-t mx-1"></div>
      <div className="w-8 h-8 bg-yellow-300 rounded-t mx-1 mr-16"></div>

      {/* Late Cluster */}
      <div className="w-8 h-16 bg-yellow-300 rounded-t mx-1"></div>
      <div className="w-8 h-32 bg-yellow-400 rounded-t mx-1"></div>
      <div className="w-8 h-64 bg-yellow-500 rounded-t mx-1"></div>
      <div className="w-8 h-40 bg-yellow-400 rounded-t mx-1"></div>
      <div className="w-8 h-20 bg-yellow-300 rounded-t mx-1"></div>

      <div className="absolute bottom-0 left-20 text-xs font-bold text-slate-500">
        4 Years (20%)
      </div>
      <div className="absolute bottom-0 right-20 text-xs font-bold text-slate-500">
        11.5 Years (80%)
      </div>
    </div>
  </div>
  ),

  Slide19: () => (
  <div className="p-12 h-full flex flex-col animate-slide-enter">
    <h2 className="text-3xl font-bold text-slate-800 mb-6">
      Simulation Engine: Stochastic Sampling
    </h2>
    <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
      <div className="bg-yellow-50 p-6 rounded-xl border border-yellow-100">
        <h3 className="font-bold text-yellow-800 mb-2">1. Valve Durability</h3>
        <p className="text-sm text-slate-600">Bi-modal distribution.</p>
        <ul className="text-xs mt-2 space-y-1 font-mono">
          <li>Mean 4.0yr (20%)</li>
          <li>Mean 11.5yr (80%)</li>
        </ul>
      </div>
      <div className="bg-blue-50 p-6 rounded-xl border border-blue-100">
        <h3 className="font-bold text-blue-800 mb-2">2. Patient Survival</h3>
        <p className="text-sm text-slate-600">
          Actuarial curves + Risk modifiers.
        </p>
        <div className="text-xs mt-2 bg-white p-2 rounded italic text-blue-900">
          "Patients compete against their valve: will they die before it fails?"
        </div>
      </div>
      <div className="bg-purple-50 p-6 rounded-xl border border-purple-100">
        <h3 className="font-bold text-purple-800 mb-2">3. Jitter</h3>
        <p className="text-sm text-slate-600">
          Standard Deviation applied per run.
        </p>
        <ul className="text-xs mt-2 space-y-1 font-mono">
          <li>durability_sd: 7.5%</li>
          <li>survival_sd: 5.0%</li>
        </ul>
      </div>
    </div>
  </div>
  ),

  Slide20: () => (
  <div className="p-12 h-full flex flex-col animate-slide-enter">
    <h2 className="text-3xl font-bold text-slate-800 mb-8">
      Logic Flow: Candidate to Realized
    </h2>
    <div className="flex flex-col gap-4 max-w-3xl mx-auto w-full">
      <div className="bg-green-100 text-green-800 p-4 rounded-lg font-bold text-center border border-green-200">
        A. ViV Candidate
      </div>
      <div className="flex justify-center">
        <span className="text-slate-400">↓</span>
      </div>
      <div className="bg-orange-50 text-orange-800 p-4 rounded-lg border border-orange-200 relative">
        <span className="absolute top-2 left-2 text-[10px] text-orange-400 font-bold uppercase">
          Filter 1
        </span>
        <div className="text-center font-bold">Redo-SAVR Deduction</div>
        <div className="text-center text-xs mt-1 text-orange-600">
          Subtracts patients undergoing open-heart Redo instead of TAVI.
        </div>
      </div>
      <div className="flex justify-center">
        <span className="text-slate-400">↓</span>
      </div>
      <div className="bg-slate-50 text-slate-400 p-4 rounded-lg border border-slate-200 relative grayscale opacity-70">
        <span className="absolute top-2 left-2 text-[10px] font-bold uppercase">
          Filter 2
        </span>
        <div className="text-center font-bold">Market Penetration</div>
        <div className="text-center text-xs mt-1">
          Not applied in primary reporting (Conservative).
        </div>
      </div>
      <div className="flex justify-center">
        <span className="text-slate-400">↓</span>
      </div>
      <div className="bg-blue-600 text-white p-4 rounded-lg font-bold text-center shadow-lg">
        B. Realized ViV (Output)
      </div>
    </div>
  </div>
  ),

  Slide21: () => (
  <div className="p-12 h-full flex flex-col animate-slide-enter bg-white">
    <h2 className="text-3xl font-bold text-slate-800 mb-8">
      The Core Logic: "The Race"
    </h2>
    <div className="bg-slate-50 p-8 rounded-xl border border-slate-100">
      <div className="mb-2 font-bold text-slate-700">
        Scenario: Death before Failure
      </div>
      <div className="relative h-16 w-full flex items-center">
        <div className="h-1 bg-blue-200 w-full absolute"></div>

        {/* Life Segment */}
        <div className="h-2 bg-blue-500 w-1/2 absolute left-0 rounded-l"></div>
        <div className="absolute left-[50%] h-full border-l-2 border-slate-500 top-0">
          <span className="absolute -top-4 -left-2 text-xs font-bold text-slate-600">
            Death
          </span>
        </div>

        {/* Valve Segment */}
        <div className="h-1 bg-yellow-400 w-[80%] absolute left-0 top-10 opacity-40"></div>
        <div className="absolute left-[80%] h-full border-l-2 border-dashed border-yellow-500 top-6 opacity-40">
          <span className="absolute -top-4 -left-4 text-xs text-yellow-600">
            Valve Fails
          </span>
        </div>

        <div className="absolute right-0 text-red-500 font-bold border-2 border-red-200 p-1 px-4 rounded text-sm rotate-[-5deg]">
          NOT ELIGIBLE
        </div>
      </div>
    </div>
  </div>
  ),

  Slide22: () => (
  <div className="p-12 h-full flex flex-col animate-slide-enter">
    <h2 className="text-3xl font-bold text-slate-800 mb-4">
      Projected ViV Candidates (2025-2035)
    </h2>
    <div className="flex-1 bg-yellow-50 rounded-xl border border-yellow-100 relative flex items-end px-8 pb-8 gap-2">
      {/* Manually drawing the bars increasing */}
      {[...Array(11)].map((_, i) => (
        <div
          key={i}
          className="flex-1 bg-slate-300 hover:bg-blue-500 transition-colors rounded-t relative group"
          style={{ height: `${20 + i * 6}%` }}
        >
          <span className="absolute -bottom-6 left-1/2 -translate-x-1/2 text-[10px] text-slate-500 font-mono">
            {2025 + i}
          </span>
        </div>
      ))}
      <div className="absolute top-8 right-8 text-right">
        <div className="text-5xl font-bold text-yellow-600">2.5x</div>
        <div className="text-sm text-slate-600">Increase (2024-2035)</div>
      </div>
    </div>
  </div>
  ),

  Slide23: () => (
  <div className="p-12 h-full flex flex-col animate-slide-enter">
    <h2 className="text-3xl font-bold text-slate-800 mb-6">
      Comparison vs. Ohno et al.
    </h2>
    <div className="flex gap-12 h-full items-end pb-12">
      <div className="w-1/2 flex flex-col items-center">
        <div className="text-blue-600 font-bold mb-4 text-lg">
          Our Model (Realized)
        </div>
        <div className="w-full flex items-end justify-center gap-2 h-64 border-b-2 border-slate-200">
          <div className="w-12 bg-blue-500 h-1/3"></div>
          <div className="w-12 bg-blue-500 h-1/2"></div>
          <div className="w-12 bg-blue-600 h-2/3 relative">
            <span className="absolute -top-8 w-full text-center font-bold text-blue-800">
              ~3-fold
            </span>
          </div>
        </div>
      </div>

      <div className="w-1/2 flex flex-col items-center opacity-60">
        <div className="text-slate-500 font-bold mb-4 text-lg">
          Ohno / Genereux (US/Japan)
        </div>
        <div className="w-full flex items-end justify-center gap-2 h-64 border-b-2 border-slate-200">
          <div className="w-12 bg-slate-400 h-1/3"></div>
          <div className="w-12 bg-slate-400 h-2/3"></div>
          <div className="w-12 bg-red-400 h-full relative">
            <span className="absolute -top-8 w-full text-center font-bold text-red-600">
              7-9 fold
            </span>
          </div>
        </div>
      </div>
    </div>
    <p className="text-center text-xs text-slate-500 mt-4">
      *Our demographic anchor significantly tempers the "infinite growth" curve.
    </p>
  </div>
  ),

  Slide24: () => (
  <div className="p-12 h-full flex flex-col animate-slide-enter">
    <h2 className="text-3xl font-bold text-slate-800 mb-8">
      Conclusion & Next Steps
    </h2>
    <div className="mb-8 bg-green-50 p-6 rounded-xl border border-green-100">
      <h3 className="font-bold text-green-800 mb-2">Summary</h3>
      <ul className="space-y-2 text-green-900">
        <li>✓ Moved away from linear extrapolation.</li>
        <li>✓ Anchored prediction to 2023/24 risk profiles.</li>
        <li>✓ Established a conservative "demand floor".</li>
      </ul>
    </div>

    <h3 className="font-bold text-slate-700 mb-4">Immediate Action Items</h3>
    <div className="flex gap-4">
      <div className="flex-1 bg-white p-4 shadow rounded border-t-4 border-blue-500">
        <div className="font-bold text-slate-300 text-4xl">1</div>
        <div className="font-bold text-slate-800">
          Run pipeline on Singaporean Data
        </div>
      </div>
      <div className="flex-1 bg-white p-4 shadow rounded border-t-4 border-purple-500">
        <div className="font-bold text-slate-300 text-4xl">2</div>
        <div className="font-bold text-slate-800">
          Draft Manuscript (Targeting JACC)
        </div>
      </div>
      <div className="flex-1 bg-white p-4 shadow rounded border-t-4 border-slate-500">
        <div className="font-bold text-slate-300 text-4xl">3</div>
        <div className="font-bold text-slate-800">
          Clean & Publicize 'model_v9' Repo
        </div>
      </div>
    </div>
  </div>
  ),

  Slide25: () => (
  <div className="p-12 h-full flex flex-col animate-slide-enter bg-slate-50">
    <h2 className="text-3xl font-bold text-slate-800 mb-8">Definitions</h2>
    <div className="space-y-6">
      <div>
        <h3 className="font-bold text-blue-800">ViV Candidate</h3>
        <p className="text-slate-600">
          A patient who is still alive at the moment their bioprosthetic valve
          fails.
        </p>
      </div>
      <div>
        <h3 className="font-bold text-blue-800">Realized ViV</h3>
        <p className="text-slate-600">
          A Candidate who actually undergoes the TAVI procedure (after filtering
          for Redo-SAVR).
        </p>
      </div>
      <div>
        <h3 className="font-bold text-blue-800">Index Procedure</h3>
        <p className="text-slate-600">
          The very first TAVI or SAVR surgery a patient undergoes.
        </p>
      </div>
      <div>
        <h3 className="font-bold text-blue-800">Redo-SAVR</h3>
        <p className="text-slate-600">
          Open heart surgery to replace a valve (removed from ViV pool).
        </p>
      </div>
    </div>
  </div>
  ),

};

// --- VERSION 2 SLIDES ---

const V2 = {

  TitleSlide: () => (
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
  ),

  ProblemSlide: () => (
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
  ),

  ConceptSlide: () => (
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
  ),

  MethodologySlide: () => (
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
  ),

  MechanismSlide: () => (
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
  ),

  LogicSlide: () => (
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
  ),

  ResultsSlide: () => (
  <div className="p-10 h-full flex flex-col animate-slide-enter">
    <h2 className="text-3xl font-bold text-slate-800 mb-4">Model Comparison (2035)</h2>
    <div className="grid grid-cols-2 gap-12 flex-1 items-end pb-8">
      
      {/* Our Model */}
      <div className="flex flex-col items-center">
         <div className="text-lg font-bold text-blue-600 mb-2">Our Model (Realized)</div>
         <div className="flex items-end space-x-2 h-64 w-full justify-center border-b border-slate-300 pb-1">
            <div className="w-12 bg-blue-500 h-[30%] rounded-t animate-[pulse_3s_ease-in-out_infinite]"></div>
            <div className="w-12 bg-blue-500 h-[50%] rounded-t"></div>
            <div className="w-12 bg-blue-500 h-[75%] rounded-t"></div>
            <div className="w-12 bg-blue-600 h-[90%] rounded-t relative">
              <span className="absolute -top-8 left-1/2 -translate-x-1/2 font-bold text-blue-700 text-xl">~3x</span>
            </div>
         </div>
         <p className="text-center text-sm text-slate-600 mt-4 px-4">
           Anchored to demographic probability. Realistic growth curve.
         </p>
      </div>

      {/* Old Model */}
      <div className="flex flex-col items-center opacity-50 grayscale hover:grayscale-0 transition-all duration-500">
         <div className="text-lg font-bold text-slate-500 mb-2">Ohno / Genereux (Linear)</div>
         <div className="flex items-end space-x-2 h-64 w-full justify-center border-b border-slate-300 pb-1">
            <div className="w-12 bg-slate-400 h-[30%] rounded-t"></div>
            <div className="w-12 bg-slate-400 h-[60%] rounded-t"></div>
            <div className="w-12 bg-slate-400 h-[100%] rounded-t"></div>
            <div className="w-12 bg-red-400 h-[140%] rounded-t relative overflow-visible">
               <span className="absolute -top-8 left-1/2 -translate-x-1/2 font-bold text-red-500 text-xl w-32 text-center">7-9x</span>
            </div>
         </div>
         <p className="text-center text-sm text-slate-500 mt-4 px-4">
           Unchecked linear extrapolation creates improbable demand.
         </p>
      </div>

    </div>
  </div>
  ),

  NextStepsSlide: () => (
  <div className="p-10 h-full flex flex-col animate-slide-enter">
    <h2 className="text-3xl font-bold text-slate-800 mb-8">Conclusion & Next Steps</h2>
    
    <div className="bg-green-50 border border-green-200 p-6 rounded-xl mb-8">
      <h3 className="font-bold text-green-800 mb-2">Summary</h3>
      <ul className="space-y-2">
        <li className="flex items-center gap-2 text-green-900">
          <span className="text-green-500">✓</span> Moved away from linear extrapolation.
        </li>
        <li className="flex items-center gap-2 text-green-900">
          <span className="text-green-500">✓</span> Anchored prediction to 2023/24 risk profiles.
        </li>
        <li className="flex items-center gap-2 text-green-900">
          <span className="text-green-500">✓</span> Established a conservative "demand floor".
        </li>
      </ul>
    </div>

    <h3 className="font-bold text-slate-700 mb-4 uppercase tracking-wider text-sm">Immediate Action Items</h3>
    <div className="grid grid-cols-3 gap-4">
      <div className="bg-white p-4 rounded shadow border border-slate-200 border-t-4 border-t-blue-500">
        <div className="font-bold text-4xl text-slate-200 mb-2">1</div>
        <div className="font-bold text-slate-800">Run Pipeline on Singapore Data</div>
      </div>
      <div className="bg-white p-4 rounded shadow border border-slate-200 border-t-4 border-t-purple-500">
        <div className="font-bold text-4xl text-slate-200 mb-2">2</div>
        <div className="font-bold text-slate-800">Draft Manuscript (Target: JACC)</div>
      </div>
      <div className="bg-white p-4 rounded shadow border border-slate-200 border-t-4 border-t-slate-500">
        <div className="font-bold text-4xl text-slate-200 mb-2">3</div>
        <div className="font-bold text-slate-800">Clean & Publicize `model_v9` Repo</div>
      </div>
    </div>
  </div>
  ),

};

// --- EXPORT ---

const V1_Slides = [
    { component: V1.Slide1, label: 'Title' },
    { component: V1.Slide2, label: 'Background' },
    { component: V1.Slide3, label: 'Orig. Method' },
    { component: V1.Slide4, label: 'Failure Analysis' },
    { component: V1.Slide5, label: 'Research Context' },
    { component: V1.Slide6, label: 'New Model Concept' },
    { component: V1.Slide7, label: 'Architecture' },
    { component: V1.Slide8, label: 'Demo. Anchor' },
    { component: V1.Slide9, label: 'MC Engine' },
    { component: V1.Slide10, label: 'Anchor Logic' },
    { component: V1.Slide11, label: 'The Race Intro' },
    { component: V1.Slide12, label: 'Race Details' },
    { component: V1.Slide13, label: 'Visualizing 2035' },
    { component: V1.Slide14, label: 'Uncertainty' },
    { component: V1.Slide15, label: 'Core Math' },
    { component: V1.Slide16, label: 'Scenarios' },
    { component: V1.Slide17, label: 'Tech Arch' },
    { component: V1.Slide18, label: 'Mixture Model' },
    { component: V1.Slide19, label: 'Stochastic Sampling' },
    { component: V1.Slide20, label: 'Logic Flow' },
    { component: V1.Slide21, label: 'Race Timeline' },
    { component: V1.Slide22, label: 'Results (Cand)' },
    { component: V1.Slide23, label: 'Comparison' },
    { component: V1.Slide24, label: 'Conclusion' },
    { component: V1.Slide25, label: 'Definitions' },
];

const V2_Slides = [
    { component: V2.TitleSlide, label: 'Overview' },
    { component: V2.ProblemSlide, label: 'The Problem' },
    { component: V2.ConceptSlide, label: 'The Pivot' },
    { component: V2.MethodologySlide, label: 'Methodology' },
    { component: V2.MechanismSlide, label: 'Simulation Logic' },
    { component: V2.LogicSlide, label: 'Distributions' },
    { component: V2.ResultsSlide, label: 'Results' },
    { component: V2.NextStepsSlide, label: 'Next Steps' },
];

const V3_Slides = [
    { component: V2.TitleSlide, label: 'Overview' },
    { component: V2.ProblemSlide, label: 'The Problem' },
    { component: V1.Slide3, label: 'Context (Visual)' },
    { component: V2.ConceptSlide, label: 'The Pivot' },
    { component: V2.MethodologySlide, label: 'Architecture' },
    { component: V1.Slide8, label: 'Anchor Logic (Deep)' },
    { component: V1.Slide10, label: 'Anchor Logic (Vis)' },
    { component: V2.MechanismSlide, label: 'The Race' },
    { component: V2.LogicSlide, label: 'Distributions' },
    { component: V1.Slide14, label: 'Uncertainty' },
    { component: V1.Slide16, label: 'Scenarios' },
    { component: V1.Slide13, label: 'Visualizing 2035' },
    { component: V2.ResultsSlide, label: 'Results' },
    { component: V1.Slide20, label: 'Logic Flow' },
    { component: V2.NextStepsSlide, label: 'Conclusion' },
];

const V4_Slides = [
    { component: V1.Slide1, label: "Title" },
    { component: V1.Slide2, label: "Background" },
    { component: V1.Slide3, label: "Orig. Method" },
    { component: V1.Slide4, label: "Why It Fails" },
    { component: V1.Slide5, label: "Pivot" },
    { component: V4.Slide6, label: "New Philosophy" },
    { component: V4.Slide7, label: "Pipeline" },
    { component: V4.Slide8, label: "Demography & Risk" },
    { component: V4.Slide9, label: "Index → Candidates" },
    { component: V4.Slide10, label: "Index Volumes" },
    { component: V4.Slide11, label: "ViV Candidates" },
    { component: V4.Slide12, label: "Realised ViV" },
    { component: V4.Slide13, label: "Patient Flow" },
    { component: V4.Slide14, label: "Model Comparison" },
    { component: V4.Slide15, label: "Next Steps" },
];


const V5_Slides = [
    { component: V5.Slide1, label: "Title" },
    { component: V5.Slide2, label: "Objective" },
    { component: V5.Slide3, label: "Prior Work" },
    { component: V5.Slide4, label: "Korea First Pass" },
    { component: V5.Slide5, label: "External Critique" },
    { component: V5.Slide6, label: "Why It Failed" },
    { component: V5.Slide7, label: "Design Goals" },
    { component: V5.Slide8, label: "Pipeline Overview" },
    { component: V5.Slide9, label: "Demography Precompute" },
    { component: V5.Slide10, label: "Per-Capita Risks" },
    { component: V5.Slide11, label: "Risk × Population" },
    { component: V5.Slide12, label: "Redo-SAVR Targets" },
    { component: V5.Slide13, label: "Monte Carlo Engine" },
    { component: V5.Slide14, label: "Candidates vs Realised" },
    { component: V5.Slide15, label: "Penetration Scenarios" },
    { component: V5.Slide16, label: "QC · 2024→2025 Dip" },
    { component: V5.Slide17, label: "Image D · Core Result" },
    { component: V5.Slide18, label: "Image C Results" },
    { component: V5.Slide19, label: "Population Trends" },
    { component: V5.Slide20, label: "Index Projections" },
    { component: V5.Slide21, label: "Risk Profiles" },
    { component: V5.Slide22, label: "Waterfalls" },
    { component: V5.Slide23, label: "Model Comparison" },
    { component: V5.Slide24, label: "Limitations" },
    { component: V5.Slide25, label: "Singapore Plan" },
    { component: V5.Slide26, label: "Publication Path" },
];





window.SlideDeck = {
  Icons,
  V1: V1_Slides,
  V2: V2_Slides,
  V3: V3_Slides,
  V4: V4_Slides,
  V5: V5_Slides,
  V6: V6_Slides,
  get All() {
    return [...this.V1, ...this.V2];
  }
};
