import { useRef, useState } from 'react'
import {
  AlertCircle,
  FileText,
  Loader2,
  Upload,
  Sparkles,
  Stethoscope,
} from 'lucide-react'

const API_URL = 'http://127.0.0.1:8000/api/v1/analyze'
const FILE_API_URL = 'http://127.0.0.1:8000/api/v1/analyze-file'

const SAMPLE_REPORT = `Patient reports persistent fever, dry cough, and shortness of breath for the last 3 days.
History of asthma noted. Chest X-ray is suspicious for pneumonia.
Started on azithromycin and advised to return if symptoms worsen.`

function normalizeLabel(label = '') {
  return String(label)
    .replace(/^[BI]-/i, '')
    .replace(/[_-]+/g, ' ')
    .trim()
    .toUpperCase()
}

function getEntityLabel(entity = {}) {
  return (
    entity.frontend_label ??
    entity.label ??
    entity.entity_group ??
    entity.entity ??
    entity.tag ??
    'UNKNOWN'
  )
}

function getEntityTheme(label = '') {
  const normalized = normalizeLabel(label)

  const diseaseLabels = new Set([
    'DISEASE DISORDER',
    'DISEASE',
    'DISEASES',
    'DISORDER',
    'DISORDERS',
    'PROBLEM',
    'PROBLEMS',
    'PATHOLOGICAL CONDITION',
    'PATHOLOGICAL CONDITIONS',
  ])

  const symptomLabels = new Set([
    'SIGN SYMPTOM',
    'SIGN',
    'SIGNS',
    'SYMPTOM',
    'SYMPTOMS',
  ])

  const medicationLabels = new Set([
    'MEDICATION',
    'MEDICATIONS',
    'CHEMICAL',
    'CHEMICALS',
    'DRUG',
    'DRUGS',
    'MEDICAL_SUPPLY',
    'MEDICAL_SUPPLIES',
    'THERAPEUTIC OR PREVENTIVE PROCEDURE',
  ])

  const anatomyLabels = new Set([
    'BIOLOGICAL STRUCTURE',
    'BIOLOGICAL STRUCTURES',
    'ANATOMY',
    'ANATOMICAL STRUCTURE',
    'ANATOMICAL STRUCTURES',
  ])

  const diagnosticLabels = new Set([
    'DIAGNOSTIC PROCEDURE',
    'DIAGNOSTIC PROCEDURES',
    'TEST',
    'TESTS',
    'PROCEDURE',
    'PROCEDURES',
    'EXAM',
    'EXAMS',
    'LAB',
    'LABS',
    'IMAGING',
  ])

  if (
    diseaseLabels.has(normalized) ||
    normalized.includes('DISEASE') ||
    normalized.includes('CLINICAL OBSERVATION')
  ) {
    return {
      chip: 'bg-rose-100 text-rose-800 ring-rose-200',
      highlight: 'bg-rose-100 text-rose-950 ring-rose-200',
    }
  }

  if (symptomLabels.has(normalized) || normalized.includes('SYMPTOM')) {
    return {
      chip: 'bg-amber-100 text-amber-800 ring-amber-200',
      highlight: 'bg-amber-100 text-amber-950 ring-amber-200',
    }
  }

  if (
    medicationLabels.has(normalized) ||
    normalized.includes('MEDICATION') ||
    normalized.includes('CHEMICAL') ||
    normalized.includes('DRUG') ||
    normalized.includes('THERAPY')
  ) {
    return {
      chip: 'bg-indigo-100 text-indigo-800 ring-indigo-200',
      highlight: 'bg-indigo-100 text-indigo-950 ring-indigo-200',
    }
  }

  if (
    anatomyLabels.has(normalized) ||
    normalized.includes('BIOLOGICAL STRUCTURE') ||
    normalized.includes('ANATOMY') ||
    normalized.includes('STRUCTURE')
  ) {
    return {
      chip: 'bg-emerald-100 text-emerald-800 ring-emerald-200',
      highlight: 'bg-emerald-100 text-emerald-950 ring-emerald-200',
    }
  }

  if (
    diagnosticLabels.has(normalized) ||
    normalized.includes('DIAGNOSTIC') ||
    normalized.includes('EXAM') ||
    normalized.includes('TEST') ||
    normalized.includes('PROCEDURE')
  ) {
    return {
      chip: 'bg-cyan-100 text-cyan-800 ring-cyan-200',
      highlight: 'bg-cyan-100 text-cyan-950 ring-cyan-200',
    }
  }

  return {
    chip: 'bg-slate-100 text-slate-700 ring-slate-200',
    highlight: 'bg-slate-100 text-slate-900 ring-slate-200',
  }
}

function getEntityDisplayName(entity = {}) {
  const rawLabel = String(getEntityLabel(entity) ?? 'ENTITY').trim()
  return rawLabel || 'ENTITY'
}

function getEntitySummaryKey(label = '') {
  const normalized = normalizeLabel(label)

  if (normalized === 'SIGN_SYMPTOM' || normalized.includes('SYMPTOM') || normalized.includes('SIGN')) {
    return 'Sign_Symptom'
  }

  if (
    normalized === 'MEDICATION' ||
    normalized.includes('MEDICATION') ||
    normalized.includes('CHEMICAL') ||
    normalized.includes('DRUG')
  ) {
    return 'Medication'
  }

  if (
    normalized === 'DIAGNOSTIC_PROCEDURE' ||
    normalized.includes('DIAGNOSTIC') ||
    normalized.includes('EXAM') ||
    normalized.includes('TEST') ||
    normalized.includes('PROCEDURE')
  ) {
    return 'Diagnostic_Procedure'
  }

  // Everything else (Disease, Anatomy, Other, etc.) merges into one bucket
  return 'Clinical_Observations'
}

function buildSummaryMatrix(entities) {
  const buckets = {
    Sign_Symptom: [],
    Medication: [],
    Diagnostic_Procedure: [],
    Clinical_Observations: [],
  }

  for (const entity of entities ?? []) {
    const categoryKey = getEntitySummaryKey(getEntityLabel(entity))
    buckets[categoryKey].push(entity)
  }

  return buckets
}

function formatSummaryItems(items) {
  return items
    .map((entity) => entity.word || entity.text || getEntityDisplayName(entity))
    .filter(Boolean)
}

function renderHighlightedText(text, entities) {
  if (!text) {
    return null
  }

  const safeEntities = [...(entities ?? [])]
    .filter(
      (entity) =>
        Number.isFinite(entity?.start) &&
        Number.isFinite(entity?.end) &&
        entity.end > entity.start,
    )
    .sort((left, right) => left.start - right.start || left.end - right.end)

  const pieces = []
  let cursor = 0

  safeEntities.forEach((entity, index) => {
    const start = Math.max(0, Math.min(text.length, entity.start))
    const end = Math.max(start, Math.min(text.length, entity.end))

    if (start < cursor) {
      return
    }

    if (start > cursor) {
      pieces.push(
        <span key={`plain-${index}`}>{text.slice(cursor, start)}</span>,
      )
    }

    const rawLabel = getEntityLabel(entity)
    const theme = getEntityTheme(rawLabel)
    const label = text.slice(start, end) || entity.word || entity.text || ''

    pieces.push(
      <span
        key={`entity-${index}`}
        className={`inline-flex rounded-md px-1.5 py-0.5 ring-1 ring-inset ${theme.highlight}`}
        title={String(rawLabel)}
      >
        {label}
      </span>,
    )

    cursor = end
  })

  if (cursor < text.length) {
    pieces.push(<span key="plain-tail">{text.slice(cursor)}</span>)
  }

  return pieces.length > 0 ? pieces : text
}

function App() {
  const fileInputRef = useRef(null)
  const [reportText, setReportText] = useState(SAMPLE_REPORT)
  const [analysis, setAnalysis] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [outputTab, setOutputTab] = useState('visual')
  const [inputMode, setInputMode] = useState('manual')
  const [selectedFileName, setSelectedFileName] = useState('')
  const [isDraggingFile, setIsDraggingFile] = useState(false)

  async function handleAnalyze(event) {
    event.preventDefault()

    const trimmed = reportText.trim()
    if (!trimmed) {
      setError('Please paste a medical report before analyzing it.')
      return
    }

    setLoading(true)
    setError('')

    try {
      const response = await fetch(API_URL, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: trimmed }),
      })

      if (!response.ok) {
        const message = await response.text()
        throw new Error(message || 'The backend returned an error.')
      }

      const data = await response.json()
      setAnalysis(data)
      setOutputTab('visual')
    } catch (fetchError) {
      setError(
        fetchError instanceof Error
          ? fetchError.message
          : 'Something went wrong while contacting the backend.',
      )
      setAnalysis(null)
    } finally {
      setLoading(false)
    }
  }

  async function analyzeUploadedFile(file) {
    if (!file) {
      return
    }

    const filename = file.name || 'uploaded file'
    const extension = filename.split('.').pop()?.toLowerCase()

    if (!['txt', 'pdf'].includes(extension || '')) {
      setError('Only .txt and .pdf files are supported.')
      return
    }

    setLoading(true)
    setError('')
    setSelectedFileName(filename)
    setInputMode('file')

    try {
      const formData = new FormData()
      formData.append('file', file)

      const response = await fetch(FILE_API_URL, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        const message = await response.text()
        throw new Error(message || 'The backend returned an error while reading the file.')
      }

      const data = await response.json()
      setAnalysis(data)
      setReportText(data.text || '')
      setOutputTab('visual')
    } catch (uploadError) {
      setError(
        uploadError instanceof Error
          ? uploadError.message
          : 'Something went wrong while uploading the file.',
      )
      setAnalysis(null)
    } finally {
      setLoading(false)
    }
  }

  function handleFilePickerChange(event) {
    const file = event.target.files?.[0]
    if (file) {
      void analyzeUploadedFile(file)
    }
    event.target.value = ''
  }

  function handleDrop(event) {
    event.preventDefault()
    setIsDraggingFile(false)

    const file = event.dataTransfer.files?.[0]
    if (file) {
      void analyzeUploadedFile(file)
    }
  }

  const entities = analysis?.entities ?? []
  const reportSource = analysis?.text ?? reportText
  const highlightedText = renderHighlightedText(reportSource, entities)
  const summaryMatrix = buildSummaryMatrix(entities)

  const fileDropzoneClass = isDraggingFile
    ? 'border-cyan-400 bg-cyan-400/10 shadow-lg shadow-cyan-500/15'
    : 'border-white/10 bg-slate-950/50 hover:border-cyan-400/50 hover:bg-slate-950/70'

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <div className="absolute inset-0 -z-10 overflow-hidden">
        <div className="absolute left-[-8rem] top-[-8rem] h-80 w-80 rounded-full bg-cyan-500/10 blur-3xl" />
        <div className="absolute right-[-6rem] top-24 h-72 w-72 rounded-full bg-sky-500/10 blur-3xl" />
        <div className="absolute bottom-[-7rem] left-1/2 h-80 w-80 -translate-x-1/2 rounded-full bg-indigo-500/10 blur-3xl" />
      </div>

      <main className="mx-auto flex min-h-screen w-full max-w-7xl flex-col px-4 py-6 sm:px-6 lg:px-8 lg:py-8">
        <header className="mb-6 overflow-hidden rounded-3xl border border-white/10 bg-white/5 p-6 shadow-2xl shadow-cyan-950/20 backdrop-blur-xl sm:p-8">
          <div className="flex flex-col gap-6 lg:flex-row lg:items-end lg:justify-between">
            <div className="flex items-start gap-4">
              <div className="flex h-14 w-14 shrink-0 items-center justify-center rounded-2xl bg-cyan-400/15 text-cyan-300 ring-1 ring-cyan-300/20">
                <Stethoscope className="h-7 w-7" />
              </div>
              <div>
                <p className="text-xs font-semibold uppercase tracking-[0.32em] text-cyan-300/75">
                  Medical AI Workspace
                </p>
                <h1 className="mt-2 text-3xl font-semibold tracking-tight text-white sm:text-5xl">
                  SmartMed Portal
                </h1>
                <p className="mt-3 max-w-3xl text-sm leading-6 text-slate-300 sm:text-base">
                  Paste an unstructured clinical note, run BioBERT-powered NER,
                  and inspect the extracted entities in a clean, doctor-friendly
                  dashboard.
                </p>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-3 text-left sm:grid-cols-3">
              <div className="rounded-2xl border border-white/10 bg-slate-900/70 px-4 py-3">
                <p className="text-xs uppercase tracking-[0.24em] text-slate-400">
                  Backend
                </p>
                <p className="mt-1 text-sm font-medium text-slate-100">FastAPI</p>
              </div>
              <div className="rounded-2xl border border-white/10 bg-slate-900/70 px-4 py-3">
                <p className="text-xs uppercase tracking-[0.24em] text-slate-400">
                  Model
                </p>
                <p className="mt-1 text-sm font-medium text-slate-100">BioBERT</p>
              </div>
              <div className="col-span-2 rounded-2xl border border-cyan-400/20 bg-cyan-400/10 px-4 py-3 sm:col-span-1">
                <p className="text-xs uppercase tracking-[0.24em] text-cyan-200/80">
                  Status
                </p>
                <p className="mt-1 text-sm font-medium text-cyan-50">
                  Ready for analysis
                </p>
              </div>
            </div>
          </div>
        </header>

        <div className="grid flex-1 gap-6 lg:grid-cols-[1.15fr_0.95fr]">
          <section className="relative overflow-hidden rounded-3xl border border-white/10 bg-slate-900/80 p-5 shadow-2xl shadow-slate-950/20 backdrop-blur-xl sm:p-6">
            <div className="flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-2xl bg-sky-400/10 text-sky-300 ring-1 ring-sky-300/20">
                <FileText className="h-5 w-5" />
              </div>
              <div>
                <h2 className="text-lg font-semibold text-white">
                  Clinical Report Input
                </h2>
                <p className="text-sm text-slate-400">
                  Paste your note and let the model extract entities.
                </p>
              </div>
            </div>

            <div className="mt-5 w-full bg-amber-500/10 border border-amber-500/20 text-amber-300 text-xs rounded-xl p-3 mb-4 flex items-center gap-2">
              <AlertCircle className="h-4 w-4 shrink-0" />
              <p>
                <span className="font-semibold text-amber-100">Performance Note:</span>{' '}
                For accurate AI analysis, use formal clinical case reports.
                Conversational or casual formatting may yield incorrect entity
                extractions.
              </p>
            </div>

            <div className="mt-5 inline-flex rounded-2xl border border-white/10 bg-slate-950/70 p-1">
              <button
                type="button"
                onClick={() => setInputMode('manual')}
                className={`rounded-xl px-4 py-2 text-sm font-semibold transition ${
                  inputMode === 'manual'
                    ? 'bg-cyan-400 text-slate-950 shadow-lg shadow-cyan-500/20'
                    : 'text-slate-300 hover:text-white'
                }`}
              >
                Type Report
              </button>
              <button
                type="button"
                onClick={() => setInputMode('file')}
                className={`rounded-xl px-4 py-2 text-sm font-semibold transition ${
                  inputMode === 'file'
                    ? 'bg-cyan-400 text-slate-950 shadow-lg shadow-cyan-500/20'
                    : 'text-slate-300 hover:text-white'
                }`}
              >
                Upload File
              </button>
            </div>

            <form className="mt-5 space-y-4" onSubmit={handleAnalyze}>
              {inputMode === 'manual' ? (
                <>
                  <label className="block text-sm font-medium text-slate-200">
                    Unstructured medical report
                  </label>
                  <textarea
                    value={reportText}
                    onChange={(event) => setReportText(event.target.value)}
                    placeholder="Example: The patient has fever, cough, and chest pain. Pneumonia is suspected. Started on amoxicillin and advised follow-up in 1 week."
                    className="min-h-[320px] w-full rounded-2xl border border-white/10 bg-slate-950/80 px-4 py-4 text-sm leading-6 text-slate-100 outline-none transition placeholder:text-slate-500 focus:border-cyan-400/60 focus:ring-4 focus:ring-cyan-400/10"
                  />

                  <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                    <div className="flex flex-wrap gap-2 text-xs text-slate-400">
                      <span className="rounded-full border border-white/10 bg-white/5 px-3 py-1">
                        Local FastAPI backend
                      </span>
                      <span className="rounded-full border border-white/10 bg-white/5 px-3 py-1">
                        NER entity grouping
                      </span>
                      <span className="rounded-full border border-white/10 bg-white/5 px-3 py-1">
                        Tailwind UI
                      </span>
                    </div>

                    <button
                      type="submit"
                      disabled={loading}
                      className="inline-flex items-center justify-center gap-2 rounded-2xl bg-gradient-to-r from-cyan-400 to-sky-500 px-5 py-3 text-sm font-semibold text-slate-950 shadow-lg shadow-cyan-500/20 transition hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-70"
                    >
                      {loading ? (
                        <>
                          <Loader2 className="h-4 w-4 animate-spin" />
                          Analyzing...
                        </>
                      ) : (
                        <>
                          <Sparkles className="h-4 w-4" />
                          Analyze Report
                        </>
                      )}
                    </button>
                  </div>
                </>
              ) : (
                <div
                  role="button"
                  tabIndex={0}
                  onClick={() => fileInputRef.current?.click()}
                  onKeyDown={(event) => {
                    if (event.key === 'Enter' || event.key === ' ') {
                      event.preventDefault()
                      fileInputRef.current?.click()
                    }
                  }}
                  onDragOver={(event) => {
                    event.preventDefault()
                    setIsDraggingFile(true)
                  }}
                  onDragLeave={() => setIsDraggingFile(false)}
                  onDrop={handleDrop}
                  className={`flex min-h-[320px] cursor-pointer flex-col items-center justify-center rounded-2xl border-2 border-dashed px-6 py-8 text-center transition ${fileDropzoneClass}`}
                >
                  <div className="flex h-14 w-14 items-center justify-center rounded-full border border-cyan-400/20 bg-cyan-400/10 text-cyan-300 shadow-lg shadow-cyan-500/10">
                    <Upload className="h-6 w-6" />
                  </div>

                  <p className="mt-4 text-base font-semibold text-white">
                    Drag and drop a .txt or .pdf file here
                  </p>
                  <p className="mt-2 max-w-md text-sm leading-6 text-slate-400">
                    Or click to browse. The uploaded report will be parsed and
                    analyzed automatically.
                  </p>

                  <div className="mt-5 rounded-full border border-white/10 bg-white/5 px-4 py-2 text-xs text-slate-300">
                    Supported: plain text and PDF medical reports
                  </div>

                  {selectedFileName ? (
                    <div className="mt-4 rounded-2xl border border-white/10 bg-slate-950/70 px-4 py-3 text-sm text-slate-200">
                      Selected file: <span className="font-semibold text-cyan-300">{selectedFileName}</span>
                    </div>
                  ) : null}

                  <input
                    ref={fileInputRef}
                    type="file"
                    accept=".txt,.pdf,application/pdf,text/plain"
                    className="hidden"
                    onChange={handleFilePickerChange}
                  />
                </div>
              )}

              {error ? (
                <div className="flex items-start gap-3 rounded-2xl border border-rose-400/20 bg-rose-400/10 px-4 py-3 text-sm text-rose-100">
                  <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
                  <span>{error}</span>
                </div>
              ) : null}
            </form>
          </section>

          <section className="rounded-3xl border border-white/10 bg-slate-900/80 p-5 shadow-2xl shadow-slate-950/20 backdrop-blur-xl sm:p-6">
            <div className="flex items-center justify-between gap-4">
              <div>
                <h2 className="text-lg font-semibold text-white">
                  Analysis Output
                </h2>
                <p className="text-sm text-slate-400">
                  The backend response, highlighted report, and extracted entities.
                </p>
              </div>

              <div className="rounded-full border border-white/10 bg-white/5 px-3 py-1 text-xs font-medium text-slate-300">
                {analysis ? `${entities.length} entities` : 'Waiting for input'}
              </div>
            </div>

            <div className="mt-5 inline-flex rounded-2xl border border-white/10 bg-slate-950/70 p-1">
              <button
                type="button"
                onClick={() => setOutputTab('visual')}
                className={`rounded-xl px-4 py-2 text-sm font-semibold transition ${
                  outputTab === 'visual'
                    ? 'bg-cyan-400 text-slate-950 shadow-lg shadow-cyan-500/20'
                    : 'text-slate-300 hover:text-white'
                }`}
              >
                Visual Report
              </button>
              <button
                type="button"
                onClick={() => setOutputTab('json')}
                className={`rounded-xl px-4 py-2 text-sm font-semibold transition ${
                  outputTab === 'json'
                    ? 'bg-cyan-400 text-slate-950 shadow-lg shadow-cyan-500/20'
                    : 'text-slate-300 hover:text-white'
                }`}
              >
                Developer JSON View
              </button>
            </div>

            {loading ? (
              <div className="mt-6 flex min-h-[220px] flex-col items-center justify-center rounded-2xl border border-dashed border-cyan-400/20 bg-slate-950/40 px-6 py-10 text-center">
                <div className="flex h-14 w-14 items-center justify-center rounded-full bg-cyan-400/10 text-cyan-300">
                  <Loader2 className="h-7 w-7 animate-spin" />
                </div>
                <p className="mt-4 text-base font-medium text-white">
                  BioBERT is analyzing the report...
                </p>
                <p className="mt-2 text-sm text-slate-400">
                  Grouping named entities and preparing the response.
                </p>
              </div>
            ) : analysis && outputTab === 'visual' ? (
              <div className="mt-6 space-y-5">
                <div className="grid gap-3 sm:grid-cols-3">
                  <div className="rounded-2xl border border-white/10 bg-slate-950/60 px-4 py-3">
                    <p className="text-xs uppercase tracking-[0.24em] text-slate-400">
                      Entity count
                    </p>
                    <p className="mt-1 text-2xl font-semibold text-white">
                      {analysis.entity_count ?? entities.length}
                    </p>
                  </div>
                  <div className="rounded-2xl border border-white/10 bg-slate-950/60 px-4 py-3">
                    <p className="text-xs uppercase tracking-[0.24em] text-slate-400">
                      Model source
                    </p>
                    <p className="mt-1 break-words text-sm font-medium text-white">
                      {analysis.model_source ?? 'BioBERT'}
                    </p>
                  </div>
                  <div className="rounded-2xl border border-white/10 bg-slate-950/60 px-4 py-3">
                    <p className="text-xs uppercase tracking-[0.24em] text-slate-400">
                      Input length
                    </p>
                    <p className="mt-1 text-2xl font-semibold text-white">
                      {(analysis.text ?? '').length}
                    </p>
                  </div>
                </div>

                <div className="rounded-2xl border border-white/10 bg-slate-950/70 p-4">
                  <p className="text-xs font-semibold uppercase tracking-[0.28em] text-slate-400">
                    Highlighted report
                  </p>
                  <div className="mt-4 max-h-[500px] overflow-y-auto scrollbar-thin scrollbar-thumb-slate-700 scrollbar-track-transparent whitespace-pre-wrap break-words rounded-2xl border border-white/10 bg-slate-950 px-4 py-4 text-sm leading-7 text-slate-200">
                    {highlightedText}
                  </div>
                </div>

                <div className="rounded-2xl border border-white/10 bg-slate-950/70 p-4 shadow-inner shadow-slate-950/30">
                  <div className="flex items-center justify-between gap-3">
                    <div>
                      <p className="text-xs font-semibold uppercase tracking-[0.28em] text-slate-400">
                        Structured Summary Matrix
                      </p>
                      <h3 className="mt-1 text-base font-semibold text-white">
                        Parsed clinical entities, grouped by category
                      </h3>
                    </div>
                    <div className="rounded-full border border-white/10 bg-white/5 px-3 py-1 text-xs text-slate-300">
                      {entities.length} total findings
                    </div>
                  </div>

                  <div className="mt-4 grid gap-3 xl:grid-cols-3">
                    {[
                      {
                        key: 'Sign_Symptom',
                        title: '⚠️ Symptoms Identified',
                        theme: 'border-amber-300/30 bg-amber-400/10',
                        chipTheme: 'border-amber-300/30 bg-amber-100/15 text-amber-50',
                        items: formatSummaryItems(summaryMatrix.Sign_Symptom),
                        empty: 'No symptom entities were returned.',
                      },
                      {
                        key: 'Medication',
                        title: '💊 Prescribed Medications',
                        theme: 'border-indigo-300/30 bg-indigo-400/10',
                        chipTheme: 'border-indigo-300/30 bg-indigo-100/15 text-indigo-50',
                        items: formatSummaryItems(summaryMatrix.Medication),
                        empty: 'No medication entities were returned.',
                      },
                      {
                        key: 'Diagnostic_Procedure',
                        title: '🧪 Diagnostic Procedures',
                        theme: 'border-cyan-300/30 bg-cyan-400/10',
                        chipTheme: 'border-cyan-300/30 bg-cyan-100/15 text-cyan-50',
                        items: formatSummaryItems(summaryMatrix.Diagnostic_Procedure),
                        empty: 'No diagnostic procedure entities were returned.',
                      },
                    ].map((section) => (
                      <div
                        key={section.key}
                        className={`rounded-2xl border p-4 backdrop-blur-sm ${section.theme}`}
                      >
                        <h4 className="text-sm font-semibold text-white">
                          {section.title}
                        </h4>

                        {section.items.length > 0 ? (
                          <div className="mt-3 flex flex-wrap gap-2">
                            {section.items.map((item, index) => (
                              <span
                                key={`${section.key}-${index}-${item}`}
                                className={`inline-flex max-w-full items-center rounded-full border px-3 py-1 text-xs font-medium ${section.chipTheme}`}
                              >
                                <span className="truncate">{item}</span>
                              </span>
                            ))}
                          </div>
                        ) : (
                          <p className="mt-3 text-sm text-slate-300/80">
                            {section.empty}
                          </p>
                        )}
                      </div>
                    ))}
                  </div>

                  <div className="mt-3 rounded-2xl border border-rose-300/30 bg-rose-400/10 p-4 backdrop-blur-sm">
                    <h4 className="text-sm font-semibold text-white">
                      🩺 Clinical Observations & Findings
                    </h4>

                    {formatSummaryItems(summaryMatrix.Clinical_Observations).length > 0 ? (
                      <div className="mt-3 flex flex-wrap gap-2">
                        {formatSummaryItems(summaryMatrix.Clinical_Observations).map((item, index) => (
                          <span
                            key={`clinical-obs-${index}-${item}`}
                            className="inline-flex max-w-full items-center rounded-full border border-rose-300/30 bg-rose-100/15 px-3 py-1 text-xs font-medium text-rose-50"
                          >
                            <span className="truncate">{item}</span>
                          </span>
                        ))}
                      </div>
                    ) : (
                      <p className="mt-3 text-sm text-slate-300/80">
                        No clinical observations were returned.
                      </p>
                    )}
                  </div>
                </div>

                <div className="rounded-2xl border border-white/10 bg-slate-950/70 p-4">
                  <p className="text-xs font-semibold uppercase tracking-[0.28em] text-slate-400">
                    Extracted entities
                  </p>

                  {entities.length > 0 ? (
                    <div className="mt-4 flex flex-wrap gap-2">
                      {entities.map((entity, index) => {
                        const rawLabel = getEntityLabel(entity)
                        const theme = getEntityTheme(rawLabel)
                        const label = getEntityDisplayName(entity)

                        return (
                          <div
                            key={`${entity.start}-${entity.end}-${index}`}
                            className={`rounded-full px-3 py-1 text-xs font-semibold ring-1 ring-inset ${theme.chip}`}
                            title={String(rawLabel)}
                          >
                            <span>{label}</span>
                            <span className="ml-2 opacity-80">
                              {entity.word || entity.text || 'Entity'}
                            </span>
                          </div>
                        )
                      })}
                    </div>
                  ) : (
                    <p className="mt-4 text-sm text-slate-400">
                      No entities were returned for this report.
                    </p>
                  )}
                </div>
              </div>
            ) : analysis && outputTab === 'json' ? (
              <div className="mt-6 rounded-2xl border border-white/10 bg-slate-950/70 p-4">
                <p className="text-xs font-semibold uppercase tracking-[0.28em] text-slate-400">
                  Developer JSON View
                </p>
                <pre className="mt-4 overflow-auto rounded-2xl bg-slate-900 p-4 text-xs leading-6 text-slate-300 ring-1 ring-white/10">
                  {JSON.stringify(analysis, null, 2)}
                </pre>
              </div>
            ) : (
              <div className="mt-6 flex min-h-[220px] flex-col items-center justify-center rounded-2xl border border-dashed border-white/10 bg-slate-950/40 px-6 py-10 text-center">
                <div className="flex h-14 w-14 items-center justify-center rounded-full bg-slate-800 text-slate-300">
                  <FileText className="h-7 w-7" />
                </div>
                <p className="mt-4 text-base font-medium text-white">
                  Your analysis will appear here
                </p>
                <p className="mt-2 max-w-md text-sm text-slate-400">
                  Click <span className="font-semibold text-cyan-300">Analyze Report</span> to
                  send the text to the FastAPI backend and view the extracted
                  medical entities.
                </p>
              </div>
            )}
          </section>
        </div>
      </main>
    </div>
  )
}

export default App
