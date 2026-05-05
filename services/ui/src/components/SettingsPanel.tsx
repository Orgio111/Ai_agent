import { useJarvisStore } from '../stores/jarvisStore'

const MODELS = ['auto', 'meta/llama-3.1-70b-instruct', 'mistralai/mixtral-8x7b-instruct-v0.1', 'mistralai/mistral-7b-instruct-v0.3']
const VOICES = [
  { id: 'en-US-AriaNeural', label: 'Aria (US Female)' },
  { id: 'en-US-GuyNeural', label: 'Guy (US Male)' },
  { id: 'en-GB-SoniaNeural', label: 'Sonia (UK Female)' },
  { id: 'zh-CN-XiaoxiaoNeural', label: 'Xiaoxiao (CN Female)' },
]

export function SettingsPanel() {
  const { settings, updateSettings } = useJarvisStore()

  return (
    <div className="p-6 max-w-2xl">
      <h2 className="text-xl font-bold text-cyan-400 mb-6">⚙️ Settings</h2>

      <div className="space-y-6">
        {/* Model */}
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-2">LLM Model</label>
          <select
            value={settings.model}
            onChange={(e) => updateSettings({ model: e.target.value })}
            className="w-full bg-gray-800 border border-gray-700 text-gray-100 rounded px-3 py-2 text-sm focus:outline-none focus:border-cyan-500"
          >
            {MODELS.map((m) => (
              <option key={m} value={m}>{m}</option>
            ))}
          </select>
        </div>

        {/* Temperature */}
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-2">
            Temperature: {settings.temperature.toFixed(1)}
          </label>
          <input
            type="range"
            min="0"
            max="2"
            step="0.1"
            value={settings.temperature}
            onChange={(e) => updateSettings({ temperature: parseFloat(e.target.value) })}
            className="w-full accent-cyan-500"
          />
          <div className="flex justify-between text-xs text-gray-600 mt-1">
            <span>Precise (0)</span>
            <span>Balanced (1)</span>
            <span>Creative (2)</span>
          </div>
        </div>

        {/* Voice */}
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-2">TTS Voice</label>
          <select
            value={settings.voice}
            onChange={(e) => updateSettings({ voice: e.target.value })}
            className="w-full bg-gray-800 border border-gray-700 text-gray-100 rounded px-3 py-2 text-sm focus:outline-none focus:border-cyan-500"
          >
            {VOICES.map((v) => (
              <option key={v.id} value={v.id}>{v.label}</option>
            ))}
          </select>
        </div>

        {/* Toggles */}
        <div className="space-y-3">
          {[
            { key: 'ttsEnabled' as const, label: 'Enable Text-to-Speech', desc: 'Read assistant responses aloud' },
            { key: 'streamingEnabled' as const, label: 'Enable Streaming', desc: 'Stream responses in real-time' },
          ].map(({ key, label, desc }) => (
            <div key={key} className="flex items-center justify-between bg-gray-800 rounded p-3">
              <div>
                <div className="text-sm font-medium text-gray-200">{label}</div>
                <div className="text-xs text-gray-500">{desc}</div>
              </div>
              <button
                onClick={() => updateSettings({ [key]: !settings[key] })}
                className={`w-12 h-6 rounded-full transition-colors relative ${
                  settings[key] ? 'bg-cyan-600' : 'bg-gray-700'
                }`}
              >
                <div
                  className={`w-5 h-5 bg-white rounded-full absolute top-0.5 transition-transform ${
                    settings[key] ? 'translate-x-6' : 'translate-x-0.5'
                  }`}
                />
              </button>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
