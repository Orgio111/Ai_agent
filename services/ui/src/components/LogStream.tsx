import { useEffect, useRef } from 'react'
import { useJarvisStore, BrokerEvent } from '../stores/jarvisStore'

const TOPIC_COLORS: Record<string, string> = {
  'agent.task': 'text-blue-400',
  'agent.result': 'text-green-400',
  'agent.failure': 'text-red-400',
  'memory.store': 'text-purple-400',
  'goal.created': 'text-yellow-400',
  'goal.completed': 'text-green-400',
}

export function LogStream() {
  const brokerEvents = useJarvisStore((s) => s.brokerEvents)
  const endRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [brokerEvents])

  return (
    <div className="h-full flex flex-col">
      <div className="p-4 border-b border-gray-800 flex items-center justify-between">
        <h2 className="text-lg font-bold text-cyan-400">📋 Live Event Log</h2>
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
          <span className="text-xs text-gray-500">{brokerEvents.length} events</span>
          <button
            onClick={() => useJarvisStore.setState({ brokerEvents: [] })}
            className="text-xs text-gray-600 hover:text-gray-400"
          >
            Clear
          </button>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto font-mono text-xs p-4 space-y-1">
        {brokerEvents.length === 0 ? (
          <div className="text-gray-600 text-center mt-8">
            Waiting for broker events...
          </div>
        ) : (
          [...brokerEvents].reverse().map((event) => (
            <div key={event.id} className="flex gap-3 py-1 border-b border-gray-900">
              <span className="text-gray-600 shrink-0 w-32">
                {new Date(event.created_at).toLocaleTimeString()}
              </span>
              <span className={`shrink-0 w-32 ${TOPIC_COLORS[event.topic] || 'text-gray-400'}`}>
                {event.topic}
              </span>
              <span className="text-gray-400 truncate">
                {JSON.stringify(event.payload).slice(0, 120)}
              </span>
            </div>
          ))
        )}
        <div ref={endRef} />
      </div>
    </div>
  )
}
