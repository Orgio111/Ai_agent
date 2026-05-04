import { useJarvisStore } from '../stores/jarvisStore'
import { useEffect, useState } from 'react'
import axios from 'axios'

interface ServiceMetric {
  name: string
  url: string
  status: 'up' | 'down' | 'unknown'
  latency_ms: number
}

export function SystemDashboard() {
  const { systemHealth, goals, brokerEvents } = useJarvisStore()
  const [metrics, setMetrics] = useState<ServiceMetric[]>([])

  useEffect(() => {
    const services = [
      { name: 'Broker', url: 'http://localhost:8001/health' },
      { name: 'LLM Engine', url: 'http://localhost:8002/health' },
      { name: 'Memory', url: 'http://localhost:8003/health' },
      { name: 'Agent Core', url: 'http://localhost:8000/health' },
      { name: 'Tool System', url: 'http://localhost:8004/health' },
      { name: 'Voice', url: 'http://localhost:8005/health' },
    ]

    const checkServices = async () => {
      const results = await Promise.all(
        services.map(async (svc) => {
          const start = Date.now()
          try {
            await axios.get(svc.url, { timeout: 3000 })
            return { name: svc.name, url: svc.url, status: 'up' as const, latency_ms: Date.now() - start }
          } catch {
            return { name: svc.name, url: svc.url, status: 'down' as const, latency_ms: Date.now() - start }
          }
        })
      )
      setMetrics(results)
    }

    checkServices()
    const interval = setInterval(checkServices, 10000)
    return () => clearInterval(interval)
  }, [])

  return (
    <div className="p-6 space-y-6">
      <h2 className="text-xl font-bold text-cyan-400">📊 System Dashboard</h2>
      <div className="grid grid-cols-3 gap-4">
        {metrics.map((m) => (
          <div key={m.name} className="bg-gray-800 rounded p-4">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-medium text-gray-200">{m.name}</span>
              <div className={`w-3 h-3 rounded-full ${m.status === 'up' ? 'bg-green-400' : 'bg-red-500'}`} />
            </div>
            <div className="text-xs text-gray-500">{m.latency_ms}ms</div>
          </div>
        ))}
      </div>
    </div>
  )
}
