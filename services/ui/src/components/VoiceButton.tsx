import { useCallback, useEffect, useRef, useState } from 'react'
import { useJarvisStore } from '../stores/jarvisStore'

type VoiceState = 'idle' | 'recording' | 'processing'

export function VoiceButton() {
  const [state, setState] = useState<VoiceState>('idle')
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const chunksRef = useRef<Blob[]>([])
  const { settings, addMessage, sessionId } = useJarvisStore()

  const startRecording = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const recorder = new MediaRecorder(stream, { mimeType: 'audio/webm' })
      chunksRef.current = []

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data)
      }

      recorder.onstop = async () => {
        stream.getTracks().forEach((t) => t.stop())
        setState('processing')

        try {
          const blob = new Blob(chunksRef.current, { type: 'audio/webm' })
          const arrayBuffer = await blob.arrayBuffer()
          const base64 = btoa(String.fromCharCode(...new Uint8Array(arrayBuffer)))

          const VOICE_URL = import.meta.env.VITE_VOICE_URL || 'http://localhost:8005'
          const resp = await fetch(`${VOICE_URL}/stt`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ audio_b64: base64 }),
          })
          const data = await resp.json()

          if (data.text?.trim()) {
            // Inject transcribed text into chat
            const event = new CustomEvent('voice-transcript', { detail: data.text })
            window.dispatchEvent(event)
          }
        } catch (err) {
          console.error('STT error:', err)
        } finally {
          setState('idle')
        }
      }

      recorder.start()
      mediaRecorderRef.current = recorder
      setState('recording')
    } catch (err) {
      alert('Microphone access denied')
      setState('idle')
    }
  }, [])

  const stopRecording = useCallback(() => {
    mediaRecorderRef.current?.stop()
    mediaRecorderRef.current = null
  }, [])

  const handleClick = () => {
    if (state === 'idle') {
      startRecording()
    } else if (state === 'recording') {
      stopRecording()
    }
  }

  const colors = {
    idle: 'bg-gray-700 hover:bg-gray-600 text-gray-300',
    recording: 'bg-red-600 hover:bg-red-500 text-white animate-pulse',
    processing: 'bg-yellow-600 text-white',
  }

  const icons = { idle: '🎤', recording: '⏹️', processing: '⏳' }

  return (
    <button
      onClick={handleClick}
      disabled={state === 'processing'}
      title={state === 'idle' ? 'Start voice input' : state === 'recording' ? 'Stop recording' : 'Processing...'}
      className={`w-12 h-12 rounded-lg flex items-center justify-center text-lg transition-colors ${colors[state]}`}
    >
      {icons[state]}
    </button>
  )
}
