import { useEffect } from 'react'
import { useJarvisStore } from '../stores/jarvisStore'

const SERVICES = {
  broker: 'http://localhost:8001/health',
  llm_engine: 'http://localhost:8002/health',
  memory: 'http://localhost:8003/health',
  agent_core: 'http://localhost:8000/health',
  voice: 'http://localhost:8005/health',
  tool_system: 'http://localhost:8004/health',
} as const

export function useHealthPoller() {
  const updateHealth = useJarvisStore((s) => s.updateHealth)

  useEffect(() => {
    const checkAll = async () => {
      const checks = await Promise.all(
        Object.entries(SERVICES).map(async ([key, url]) => {
          try {
            const resp = await fetch(url, { signal: AbortSignal.timeout(3000) })
            return [key, resp.ok] as const
          } catch {
            return [key, false] as const
          }
        })
      )
      const health = Object.fromEntries(checks) as Record<string, boolean>
      updateHealth(health as Parameters<typeof updateHealth>[0])
    }

    checkAll()
    const interval = setInterval(checkAll, 15000)
    return () => clearInterval(interval)
  }, [updateHealth])
}
