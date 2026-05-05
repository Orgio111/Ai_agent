import { useEffect } from 'react'
import { useJarvisStore } from '../stores/jarvisStore'

const BROKER_WS_URL = import.meta.env.VITE_BROKER_WS_URL || 'ws://localhost:8001/ws'

export function useBrokerConnection() {
  const { setBrokerWs, addBrokerEvent } = useJarvisStore()

  useEffect(() => {
    let ws: WebSocket | null = null
    let reconnectTimeout: ReturnType<typeof setTimeout> | null = null
    let backoff = 1000

    const connect = () => {
      ws = new WebSocket(BROKER_WS_URL)

      ws.onopen = () => {
        setBrokerWs(ws)
        backoff = 1000
        // Subscribe to all events
        ws!.send(JSON.stringify({ action: 'subscribe', topic: '*', subscriber_id: 'ui' }))
      }

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data)
          if (data.topic && data.payload) {
            addBrokerEvent({
              id: data.id || `evt_${Date.now()}`,
              topic: data.topic,
              payload: data.payload,
              created_at: data.created_at || new Date().toISOString(),
            })
          }
        } catch { /* ignore parse errors */ }
      }

      ws.onclose = () => {
        setBrokerWs(null)
        reconnectTimeout = setTimeout(() => {
          backoff = Math.min(backoff * 2, 30000)
          connect()
        }, backoff)
      }

      ws.onerror = () => {
        ws?.close()
      }
    }

    connect()

    return () => {
      if (reconnectTimeout) clearTimeout(reconnectTimeout)
      ws?.close()
    }
  }, [setBrokerWs, addBrokerEvent])
}
