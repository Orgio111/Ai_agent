import { useEffect, useState } from 'react'
import axios from 'axios'

const MEMORY_URL = import.meta.env.VITE_MEMORY_URL || 'http://localhost:8003'

interface MemoryResult {
  memory_id: string
  content: string
  memory_type: string
  score: number
  confidence: number
  created_at: string
  age_hours: number
}

const TYPE_COLORS: Record<string, string> = {
  working: 'text-yellow-400 border-yellow-600',
  episodic: 'text-blue-400 border-blue-600',
  semantic: 'text-green-400 border-green-600',
  procedural: 'text-purple-400 border-purple-600',
}

export function MemoryInspector() {
  const [query, setQuery] = useState('')
  const [results, setResults] = useState<MemoryResult[]>([])
  const [stats, setStats] = useState<Record<string, unknown>>({})
  const [loading, setLoading] = useState(false)
  const [selectedTypes, setSelectedTypes] = useState(['episodic', 'semantic'])

  useEffect(() => {
    const fetchStats = async () => {
      try {
        const resp = await axios.get(`${MEMORY_URL}/stats`)
        setStats(resp.data)
      } catch { /* service may be offline */ }
    }
    fetchStats()
  }, [])

  const search = async () => {
    if (!query.trim()) return
    setLoading(true)
    try {
      const resp = await axios.post(`${MEMORY_URL}/query`, {
        query: query.trim(),
        memory_types: selectedTypes,
        limit: 20,
        min_score: 0.3,
        summarize: false,
      })
      setResults(resp.data.results || [])
    } catch (err) {
      alert(`Memory query failed: ${err}`)
    } finally {
      setLoading(false)
    }
  }

  const toggleType = (type: string) => {
    setSelectedTypes((prev) =>
      prev.includes(type) ? prev.filter((t) => t !== type) : [...prev, type]
    )
  }

  return (
    <div className="flex h-full">
      {/* Left: Query panel */}
      <div className="w-1/3 border-r border-gray-800 flex flex-col p-4">
        <h2 className="text-lg font-bold text-cyan-400 mb-4">🧠 Memory Inspector</h2>

        {/* Stats */}
        {stats.total_memories !== undefined && (
          <div className="bg-gray-800 rounded p-3 mb-4 text-xs space-y-1">
            <div className="text-gray-400 font-bold mb-1">Memory Statistics</div>
            <div className="flex justify-between">
              <span className="text-gray-500">Total</span>
              <span className="text-gray-200">{stats.total_memories as number}</span>
            </div>
            {stats.by_type && Object.entries(stats.by_type as Record<string, number>).map(([type, count]) => (
              <div key={type} className="flex justify-between">
                <span className="text-gray-500 capitalize">{type}</span>
                <span className="text-gray-200">{count}</span>
              </div>
            ))}
            <div className="flex justify-between">
              <span className="text-gray-500">Working keys</span>
              <span className="text-gray-200">{stats.working_memory_keys as number}</span>
            </div>
          </div>
        )}

        {/* Memory type filter */}
        <div className="mb-4">
          <div className="text-xs text-gray-500 mb-2">Memory Types</div>
          <div className="flex flex-wrap gap-2">
            {['working', 'episodic', 'semantic', 'procedural'].map((type) => (
              <button
                key={type}
                onClick={() => toggleType(type)}
                className={`px-3 py-1 text-xs rounded border transition-colors ${
                  selectedTypes.includes(type)
                    ? TYPE_COLORS[type] || 'text-gray-400 border-gray-600'
                    : 'text-gray-600 border-gray-700 hover:border-gray-500'
                }`}
              >
                {type}
              </button>
            ))}
          </div>
        </div>

        {/* Query input */}
        <div className="flex gap-2">
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && search()}
            placeholder="Search memories..."
            className="flex-1 bg-gray-800 border border-gray-700 rounded px-3 py-2 text-sm text-gray-100 placeholder-gray-500 focus:outline-none focus:border-cyan-500"
          />
          <button
            onClick={search}
            disabled={loading}
            className="px-4 py-2 bg-cyan-600 hover:bg-cyan-500 disabled:bg-gray-700 text-white rounded text-sm"
          >
            {loading ? '...' : '🔍'}
          </button>
        </div>
      </div>

      {/* Right: Results */}
      <div className="flex-1 p-4 overflow-y-auto">
        {results.length === 0 ? (
          <div className="flex items-center justify-center h-full text-gray-600 text-sm">
            Search memories above to explore stored knowledge.
          </div>
        ) : (
          <div className="space-y-3">
            {results.map((r) => (
              <div
                key={r.memory_id}
                className={`bg-gray-800 rounded p-3 border-l-4 ${
                  TYPE_COLORS[r.memory_type]?.split(' ')[1] || 'border-gray-600'
                }`}
              >
                <div className="flex items-center justify-between mb-1">
                  <div className="flex items-center gap-2">
                    <span className={`text-xs font-bold ${TYPE_COLORS[r.memory_type]?.split(' ')[0] || 'text-gray-400'}`}>
                      {r.memory_type.toUpperCase()}
                    </span>
                    <span className="text-xs text-gray-600">{r.memory_id.slice(0, 8)}...</span>
                  </div>
                  <div className="flex items-center gap-2 text-xs">
                    <span className="text-gray-500">score: {r.score.toFixed(2)}</span>
                    <span className="text-gray-500">conf: {r.confidence.toFixed(2)}</span>
                    <span className="text-gray-600">{r.age_hours.toFixed(1)}h ago</span>
                  </div>
                </div>
                <p className="text-sm text-gray-200">{r.content}</p>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
