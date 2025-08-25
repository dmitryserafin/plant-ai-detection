import { useEffect, useRef, useState, type ChangeEvent } from 'react'
import { translations, type Language } from './translations'

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000'

export default function App() {
  const [file, setFile] = useState<File | null>(null)
  const [imgUrl, setImgUrl] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [language, setLanguage] = useState<Language>('en')
  const [result, setResult] = useState<null | {
    disease: string
    confidence: number
    suggestions: string[]
    inference_ms: number
    severity?: string
    plant_type?: string
    affected_parts?: string[]
    causative_agent?: string
    treatment_urgency?: string
  }>(null)
  const inputRef = useRef<HTMLInputElement | null>(null)

  const t = translations[language]

  useEffect(() => {
    return () => { if (imgUrl) URL.revokeObjectURL(imgUrl) }
  }, [imgUrl])

  function onFileChange(e: ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0]
    setError(null)
    setResult(null)
    if (f) {
      setFile(f)
      const url = URL.createObjectURL(f)
      setImgUrl(url)
    } else {
      setFile(null); setImgUrl(null)
    }
  }

  async function onPredict() {
    if (!file) { setError(t.selectImageError); return }
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const form = new FormData()
      form.append('image', file)
      form.append('language', language) // Send language preference
      const res = await fetch(`${API_BASE}/predict`, {
        method: 'POST',
        body: form,
      })
      if (!res.ok) {
        const txt = await res.text()
        throw new Error(`Error ${res.status}: ${txt}`)
      }
      const data = await res.json()
      setResult({
        disease: data.disease,
        confidence: data.confidence,
        suggestions: data.suggestions || [],
        inference_ms: data.inference_ms,
        severity: data.severity,
        plant_type: data.plant_type,
        affected_parts: data.affected_parts,
        causative_agent: data.causative_agent,
        treatment_urgency: data.treatment_urgency
      })
    } catch (e: any) {
      setError(e.message || t.predictionFailedError)
    } finally {
      setLoading(false)
    }
  }

  function onClear() {
    setFile(null); setImgUrl(null); setResult(null); setError(null)
    inputRef.current?.value && (inputRef.current.value = '')
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-emerald-50 via-green-50 to-teal-50 text-slate-800">
      {/* Animated background elements */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute -top-4 -right-4 w-72 h-72 bg-green-200 rounded-full mix-blend-multiply filter blur-xl opacity-30 animate-pulse"></div>
        <div className="absolute top-1/3 -left-4 w-72 h-72 bg-emerald-200 rounded-full mix-blend-multiply filter blur-xl opacity-30 animate-pulse animation-delay-2000"></div>
        <div className="absolute bottom-1/3 right-1/3 w-72 h-72 bg-teal-200 rounded-full mix-blend-multiply filter blur-xl opacity-30 animate-pulse animation-delay-4000"></div>
      </div>

      <header className="relative max-w-6xl mx-auto px-6 py-8">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="w-12 h-12 bg-gradient-to-br from-green-500 to-emerald-600 rounded-xl flex items-center justify-center shadow-lg">
              <span className="text-white text-2xl">🌱</span>
            </div>
            <div>
              <h1 className="text-3xl font-bold bg-gradient-to-r from-green-600 to-emerald-600 bg-clip-text text-transparent">
                {t.appTitle}
              </h1>
              <p className="text-sm text-slate-600">{t.appSubtitle}</p>
              <p className="text-xs text-slate-400 mt-1">{t.createdBy}</p>
            </div>
          </div>
          
          {/* Language Selector */}
          <div className="bg-white/50 backdrop-blur-sm border border-white/20 rounded-xl px-4 py-2">
            <label className="block text-xs text-slate-500 mb-1">{t.language}</label>
            <select 
              value={language} 
              onChange={(e) => setLanguage(e.target.value as Language)}
              className="bg-transparent text-sm font-medium text-slate-700 border-none outline-none cursor-pointer"
            >
              {Object.entries(t.languages).map(([code, name]) => (
                <option key={code} value={code}>{name}</option>
              ))}
            </select>
          </div>
        </div>
      </header>

      <main className="relative max-w-6xl mx-auto px-6 pb-16">
        <div className="grid lg:grid-cols-2 gap-8">
          {/* Upload Section */}
          <div className="space-y-6">
            <div className="bg-white/70 backdrop-blur-md border border-white/20 rounded-2xl shadow-xl p-8 transition-all duration-300 hover:shadow-2xl">
              <div className="space-y-6">
                <div className="text-center">
                  <h2 className="text-2xl font-bold text-slate-800 mb-2">{t.uploadTitle}</h2>
                  <p className="text-slate-600">{t.uploadSubtitle}</p>
                </div>
                
                <div 
                  className={`relative border-2 border-dashed rounded-xl p-8 transition-all duration-300 ${
                    file ? 'border-green-300 bg-green-50/50' : 'border-slate-300 hover:border-green-400 hover:bg-green-50/30'
                  }`}
                  onDragOver={(e) => e.preventDefault()}
                  onDrop={(e) => {
                    e.preventDefault();
                    const droppedFile = e.dataTransfer.files[0];
                    if (droppedFile && droppedFile.type.startsWith('image/')) {
                      setFile(droppedFile);
                      const url = URL.createObjectURL(droppedFile);
                      setImgUrl(url);
                      setError(null);
                      setResult(null);
                    }
                  }}
                >
                  <input 
                    ref={inputRef} 
                    type="file" 
                    accept="image/*" 
                    onChange={onFileChange} 
                    className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                  />
                  
                  {!imgUrl ? (
                    <div className="text-center">
                      <div className="w-16 h-16 mx-auto mb-4 bg-gradient-to-br from-green-100 to-emerald-100 rounded-full flex items-center justify-center">
                        <span className="text-2xl">📸</span>
                      </div>
                      <p className="text-slate-600">
                        {t.uploadSubtitle}
                      </p>
                      <p className="text-xs text-slate-500 mt-2">{t.supportedFormats}</p>
                    </div>
                  ) : (
                    <div className="relative">
                      <img 
                        src={imgUrl || ''} 
                        className="max-w-full max-h-64 mx-auto rounded-lg shadow-md object-cover" 
                        alt="Plant preview"
                      />
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          onClear();
                        }}
                        className="absolute top-2 right-2 w-8 h-8 bg-red-500 text-white rounded-full flex items-center justify-center hover:bg-red-600 transition-colors shadow-lg"
                      >
                        ×
                      </button>
                    </div>
                  )}
                </div>

                <div className="flex gap-4">
                  <button 
                    onClick={onPredict} 
                    disabled={loading || !file} 
                    className="flex-1 px-6 py-4 rounded-xl bg-gradient-to-r from-green-500 to-emerald-600 text-white font-medium disabled:opacity-50 disabled:cursor-not-allowed hover:from-green-600 hover:to-emerald-700 transition-all duration-200 shadow-lg hover:shadow-xl transform hover:-translate-y-0.5"
                  >
                    {loading ? (
                      <div className="flex items-center justify-center gap-2">
                        <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
                        {t.analyzingButton}
                      </div>
                    ) : (
                      <div className="flex items-center justify-center gap-2">
                        <span>🔍</span>
                        {t.analyzeButton}
                      </div>
                    )}
                  </button>
                  <button 
                    onClick={onClear} 
                    className="px-6 py-4 rounded-xl border border-slate-300 text-slate-700 hover:bg-slate-50 transition-all duration-200"
                  >
                    {t.clearButton}
                  </button>
                </div>

                {error && (
                  <div className="p-4 bg-red-50 border border-red-200 rounded-xl">
                    <p className="text-red-700 text-sm flex items-center gap-2">
                      <span>⚠️</span>
                      {error}
                    </p>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Results Section */}
          <div className="space-y-6">
            <div className="bg-white/70 backdrop-blur-md border border-white/20 rounded-2xl shadow-xl p-8">
              <div className="space-y-6">
                <div className="text-center">
                  <h2 className="text-2xl font-bold text-slate-800 mb-2">{t.resultsTitle}</h2>
                  <p className="text-slate-600">{t.resultsSubtitle}</p>
                </div>

                {!result ? (
                  <div className="text-center py-12">
                    <div className="w-20 h-20 mx-auto mb-6 bg-gradient-to-br from-slate-100 to-slate-200 rounded-full flex items-center justify-center">
                      <span className="text-3xl">🌿</span>
                    </div>
                    <p className="text-slate-500 text-lg">{t.uploadPrompt}</p>
                  </div>
                ) : (
                  <div className="space-y-6 animate-fadeIn">
                    {/* Disease Name */}
                    <div className="text-center">
                      <div className="inline-flex items-center gap-3 px-6 py-3 bg-gradient-to-r from-green-100 to-emerald-100 rounded-full">
                        <span className="text-2xl">🔬</span>
                        <span className="text-xl font-bold text-slate-800">{result?.disease || 'Unknown'}</span>
                      </div>
                    </div>

                    {/* Enhanced Disease Info */}
                    {(result?.plant_type || result?.severity || result?.causative_agent) && (
                      <div className="grid grid-cols-2 gap-4">
                        {result?.plant_type && (
                          <div className="bg-blue-50 border border-blue-100 rounded-lg p-3">
                            <span className="text-xs text-blue-600 font-medium">Plant Type</span>
                            <p className="text-sm text-blue-800">{result.plant_type}</p>
                          </div>
                        )}
                        {result?.severity && (
                          <div className={`border rounded-lg p-3 ${
                            result.severity === 'Critical' ? 'bg-red-50 border-red-100' :
                            result.severity === 'High' ? 'bg-orange-50 border-orange-100' :
                            result.severity === 'Moderate' ? 'bg-yellow-50 border-yellow-100' :
                            'bg-green-50 border-green-100'
                          }`}>
                            <span className="text-xs font-medium">Severity</span>
                            <p className="text-sm">{result.severity}</p>
                          </div>
                        )}
                        {result?.causative_agent && (
                          <div className="bg-purple-50 border border-purple-100 rounded-lg p-3">
                            <span className="text-xs text-purple-600 font-medium">Cause</span>
                            <p className="text-sm text-purple-800 capitalize">{result.causative_agent}</p>
                          </div>
                        )}
                        {result?.treatment_urgency && (
                          <div className="bg-indigo-50 border border-indigo-100 rounded-lg p-3">
                            <span className="text-xs text-indigo-600 font-medium">Urgency</span>
                            <p className="text-sm text-indigo-800 capitalize">{result.treatment_urgency.replace('_', ' ')}</p>
                          </div>
                        )}
                      </div>
                    )}

                    {/* Confidence Bar */}
                    <div className="space-y-3">
                      <div className="flex items-center justify-between">
                        <span className="text-sm font-medium text-slate-700">{t.confidenceLevel}</span>
                        <span className="text-lg font-bold text-green-600">{Math.round((result?.confidence || 0)*100)}%</span>
                      </div>
                      <div className="relative h-4 bg-slate-200 rounded-full overflow-hidden">
                        <div 
                          className="h-full bg-gradient-to-r from-green-400 to-emerald-500 rounded-full transition-all duration-1000 ease-out shadow-sm"
                          style={{ width: `${Math.round((result?.confidence || 0)*100)}%` }}
                        />
                        <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent animate-shimmer" />
                      </div>
                    </div>

                    {/* Analysis Time */}
                    <div className="flex items-center justify-center gap-2 text-sm text-slate-500">
                      <span>⚡</span>
                      <span>{t.analysisTime} {result?.inference_ms || 0}ms</span>
                    </div>

                    {/* Suggestions */}
                    {result?.suggestions?.length && result.suggestions.length > 0 && (
                      <div className="space-y-3">
                        <h3 className="text-lg font-semibold text-slate-800 flex items-center gap-2">
                          <span>💡</span>
                          {t.recommendations}
                        </h3>
                        <div className="space-y-2">
                          {result.suggestions.map((suggestion, i) => (
                            <div key={i} className="flex items-start gap-3 p-4 bg-blue-50 border border-blue-100 rounded-xl">
                              <div className="w-2 h-2 bg-blue-400 rounded-full mt-2 flex-shrink-0" />
                              <p className="text-slate-700 leading-relaxed">{suggestion}</p>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Action Buttons */}
                    <div className="pt-4 border-t border-slate-200">
                      <button
                        onClick={onClear}
                        className="w-full px-6 py-3 bg-gradient-to-r from-slate-100 to-slate-200 text-slate-700 rounded-xl hover:from-slate-200 hover:to-slate-300 transition-all duration-200 font-medium"
                      >
                        {t.analyzeAnotherButton}
                      </button>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </main>

      <footer className="relative text-center py-8">
        <div className="max-w-2xl mx-auto px-6 space-y-3">
          <p className="text-sm text-slate-500 bg-white/50 backdrop-blur-sm rounded-full px-6 py-3 inline-block border border-white/20">
            🌱 {t.poweredBy}
          </p>
          <p className="text-xs text-slate-400 bg-white/30 backdrop-blur-sm rounded-full px-4 py-2 inline-block border border-white/10">
            {t.credits}
          </p>
        </div>
      </footer>
    </div>
  )
}
