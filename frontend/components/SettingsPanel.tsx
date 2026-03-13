import { useState, useEffect, useCallback } from 'react'
import { ChevronDown, Download } from 'lucide-react'
import { Select } from './ui/select'
import type { GenerationMode } from './ModeTabs'
import {
  FORCED_API_VIDEO_FPS,
  FORCED_API_VIDEO_RESOLUTIONS,
  getAllowedForcedApiDurations,
  sanitizeForcedApiVideoSettings,
} from '../lib/api-video-options'
import { backendFetch } from '../lib/backend'

export interface GenerationSettings {
  model: 'fast' | 'pro'
  duration: number
  videoResolution: string
  fps: number
  audio: boolean
  cameraMotion: string
  aspectRatio?: string
  // Image-specific settings
  imageResolution: string
  imageAspectRatio: string
  imageSteps: number
  variations?: number  // Number of image variations to generate
  negativePrompt?: string
  loraPath?: string | null
  loraStrength?: number
}

interface SettingsPanelProps {
  settings: GenerationSettings
  onSettingsChange: (settings: GenerationSettings) => void
  disabled?: boolean
  mode?: GenerationMode
  forceApiGenerations?: boolean
  hasAudio?: boolean
}

export function SettingsPanel({
  settings,
  onSettingsChange,
  disabled,
  mode = 'text-to-video',
  forceApiGenerations = false,
  hasAudio = false,
}: SettingsPanelProps) {
  const isImageMode = mode === 'text-to-image'
  const LOCAL_MAX_DURATION: Record<string, number> = { '540p': 20, '720p': 10, '1080p': 5 }

  // Pro model download state
  const [proModelsReady, setProModelsReady] = useState<boolean | null>(null)
  const [proDownloading, setProDownloading] = useState(false)

  const checkProModels = useCallback(() => {
    if (forceApiGenerations) return
    backendFetch('/api/models/status')
      .then(async (res) => {
        if (res.ok) {
          const data = await res.json()
          const models = data.models ?? []
          const fullCk = models.find((m: { id: string }) => m.id === 'full_checkpoint')
          const distLora = models.find((m: { id: string }) => m.id === 'distilled_lora')
          setProModelsReady((fullCk?.downloaded ?? false) && (distLora?.downloaded ?? false))
        }
      })
      .catch(() => { /* ignore */ })
  }, [forceApiGenerations])

  useEffect(() => {
    if (settings.model === 'pro') checkProModels()
  }, [settings.model, checkProModels])

  const handleProDownload = () => {
    setProDownloading(true)
    backendFetch('/api/models/download', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ modelTypes: ['full_checkpoint', 'distilled_lora'] }),
    })
      .then(() => {
        const poll = setInterval(async () => {
          try {
            const res = await backendFetch('/api/models/status')
            if (res.ok) {
              const data = await res.json()
              const models = data.models ?? []
              const fullCk = models.find((m: { id: string }) => m.id === 'full_checkpoint')
              const distLora = models.find((m: { id: string }) => m.id === 'distilled_lora')
              if ((fullCk?.downloaded ?? false) && (distLora?.downloaded ?? false)) {
                clearInterval(poll)
                setProModelsReady(true)
                setProDownloading(false)
              }
            }
          } catch { /* ignore */ }
        }, 3000)
      })
      .catch(() => setProDownloading(false))
  }

  const handleChange = (key: keyof GenerationSettings, value: string | number | boolean) => {
    const nextSettings = { ...settings, [key]: value } as GenerationSettings
    if (forceApiGenerations && !isImageMode) {
      onSettingsChange(sanitizeForcedApiVideoSettings(nextSettings, { hasAudio }))
      return
    }

    // Clamp duration when resolution changes for local generation
    if (key === 'videoResolution' && !forceApiGenerations) {
      const maxDur = LOCAL_MAX_DURATION[value as string] ?? 20
      if (nextSettings.duration > maxDur) {
        nextSettings.duration = maxDur
      }
    }

    onSettingsChange(nextSettings)
  }

  const localMaxDuration = LOCAL_MAX_DURATION[settings.videoResolution] ?? 20
  const durationOptions = forceApiGenerations
    ? [...getAllowedForcedApiDurations(settings.model, settings.videoResolution, settings.fps)]
    : [5, 6, 8, 10, 20].filter(d => d <= localMaxDuration)
  const resolutionOptions = forceApiGenerations
    ? (hasAudio ? ['1080p'] : [...FORCED_API_VIDEO_RESOLUTIONS])
    : ['1080p', '720p', '540p']
  const fpsOptions = forceApiGenerations ? [...FORCED_API_VIDEO_FPS] : [24, 25, 50]

  // Image mode settings
  if (isImageMode) {
    return (
      <div className="space-y-4">
        {/* Aspect Ratio and Quality side by side */}
        <div className="grid grid-cols-2 gap-3">
          <Select
            label="Aspect Ratio"
            value={settings.imageAspectRatio || '16:9'}
            onChange={(e) => handleChange('imageAspectRatio', e.target.value)}
            disabled={disabled}
          >
            <option value="1:1">1:1 (Square)</option>
            <option value="16:9">16:9 (Landscape)</option>
            <option value="9:16">9:16 (Portrait)</option>
            <option value="4:3">4:3 (Standard)</option>
            <option value="3:4">3:4 (Portrait Standard)</option>
            <option value="21:9">21:9 (Cinematic)</option>
          </Select>

          <Select
            label="Quality"
            value={settings.imageSteps || 4}
            onChange={(e) => handleChange('imageSteps', parseInt(e.target.value))}
            disabled={disabled}
          >
            <option value={4}>Fast</option>
            <option value={8}>Balanced</option>
            <option value={12}>High</option>
          </Select>
        </div>
      </div>
    )
  }

  // Video mode settings
  return (
    <div className="space-y-4">
      {/* Model Selection */}
      {!forceApiGenerations ? (
        <Select
          label="Model"
          value={settings.model}
          onChange={(e) => handleChange('model', e.target.value)}
          disabled={disabled}
        >
          <option value="fast">LTX 2.3 Fast</option>
          <option value="pro">LTX 2.3 Pro</option>
        </Select>
      ) : (
        <Select
          label="Model"
          value={settings.model}
          onChange={(e) => handleChange('model', e.target.value)}
          disabled={disabled}
        >
          <option value="fast" disabled={hasAudio}>LTX-2.3 Fast (API)</option>
          <option value="pro">LTX-2.3 Pro (API)</option>
        </Select>
      )}

      {/* Pro model download banner */}
      {!forceApiGenerations && settings.model === 'pro' && proModelsReady === false && (
        <div className="rounded-lg border border-amber-800/50 bg-amber-950/30 px-3 py-2.5 space-y-2">
          <p className="text-[11px] text-amber-400">
            Pro requires the full checkpoint (~43 GB) and distilled LoRA (~400 MB).
          </p>
          <button
            type="button"
            onClick={handleProDownload}
            disabled={proDownloading}
            className="flex items-center gap-1.5 rounded-md bg-amber-700 px-3 py-1.5 text-[11px] font-medium text-white hover:bg-amber-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            <Download className="h-3 w-3" />
            {proDownloading ? 'Downloading...' : 'Download Pro Models'}
          </button>
        </div>
      )}

      {/* Duration, Resolution, FPS Row */}
      <div className="grid grid-cols-3 gap-3">
        <Select
          label="Duration"
          value={settings.duration}
          onChange={(e) => handleChange('duration', parseInt(e.target.value))}
          disabled={disabled}
        >
          {durationOptions.map((duration) => (
            <option key={duration} value={duration}>
              {duration} sec
            </option>
          ))}
        </Select>

        <Select
          label="Resolution"
          value={settings.videoResolution}
          onChange={(e) => handleChange('videoResolution', e.target.value)}
          disabled={disabled}
        >
          {resolutionOptions.map((resolution) => (
            <option key={resolution} value={resolution}>
              {resolution}
            </option>
          ))}
        </Select>

        <Select
          label="FPS"
          value={settings.fps}
          onChange={(e) => handleChange('fps', parseInt(e.target.value))}
          disabled={disabled}
        >
          {fpsOptions.map((fps) => (
            <option key={fps} value={fps}>
              {fps}
            </option>
          ))}
        </Select>
      </div>

      {/* Aspect Ratio */}
      <Select
        label="Aspect Ratio"
        value={settings.aspectRatio || '16:9'}
        onChange={(e) => handleChange('aspectRatio', e.target.value)}
        disabled={disabled}
      >
        {hasAudio ? (
          <option value="16:9">16:9 Landscape</option>
        ) : (
          <>
            <option value="16:9">16:9 Landscape</option>
            <option value="9:16">9:16 Portrait</option>
          </>
        )}
      </Select>

      {/* Audio and Camera Motion Row */}
      <div className="flex gap-3">
        <div className="w-[140px] flex-shrink-0">
          <Select
            label="Audio"
            badge="PREVIEW"
            value={settings.audio ? 'on' : 'off'}
            onChange={(e) => handleChange('audio', e.target.value === 'on')}
            disabled={disabled}
          >
            <option value="on">On</option>
            <option value="off">Off</option>
          </Select>
        </div>

        <div className="flex-1">
          <Select
            label="Camera Motion"
            value={settings.cameraMotion}
            onChange={(e) => handleChange('cameraMotion', e.target.value)}
            disabled={disabled}
          >
            <option value="none">None</option>
            <option value="static">Static</option>
            <option value="focus_shift">Focus Shift</option>
            <option value="dolly_in">Dolly In</option>
            <option value="dolly_out">Dolly Out</option>
            <option value="dolly_left">Dolly Left</option>
            <option value="dolly_right">Dolly Right</option>
            <option value="jib_up">Jib Up</option>
            <option value="jib_down">Jib Down</option>
          </Select>
        </div>
      </div>

      {/* Advanced Section */}
      <AdvancedSection
        negativePrompt={settings.negativePrompt || ''}
        onNegativePromptChange={(value) => handleChange('negativePrompt', value)}
        loraPath={settings.loraPath ?? null}
        onLoraPathChange={(value) => handleChange('loraPath', value ?? '')}
        loraStrength={settings.loraStrength ?? 1.0}
        onLoraStrengthChange={(value) => handleChange('loraStrength', value)}
        disabled={disabled}
        forceApiGenerations={forceApiGenerations}
      />
    </div>
  )
}

function AdvancedSection({
  negativePrompt,
  onNegativePromptChange,
  loraPath,
  onLoraPathChange,
  loraStrength,
  onLoraStrengthChange,
  disabled,
  forceApiGenerations,
}: {
  negativePrompt: string
  onNegativePromptChange: (value: string) => void
  loraPath: string | null
  onLoraPathChange: (value: string | null) => void
  loraStrength: number
  onLoraStrengthChange: (value: number) => void
  disabled?: boolean
  forceApiGenerations?: boolean
}) {
  const [open, setOpen] = useState(false)
  const [availableLoras, setAvailableLoras] = useState<string[]>([])

  useEffect(() => {
    if (!open || forceApiGenerations) return
    backendFetch('/api/loras')
      .then(async (res) => {
        if (res.ok) {
          const data = await res.json()
          setAvailableLoras(data.loras ?? [])
        }
      })
      .catch(() => { /* ignore */ })
  }, [open, forceApiGenerations])

  return (
    <div className="border border-zinc-800 rounded-lg overflow-hidden">
      <button
        type="button"
        onClick={() => setOpen(!open)}
        className="flex w-full items-center justify-between px-3 py-2 text-xs font-medium text-zinc-400 hover:text-zinc-300 transition-colors"
      >
        Advanced
        <ChevronDown
          className={`h-3.5 w-3.5 transition-transform ${open ? 'rotate-180' : ''}`}
        />
      </button>
      {open && (
        <div className="px-3 pb-3 space-y-3">
          <div className="space-y-1.5">
            <label className="block text-[11px] font-medium text-zinc-500">
              Negative Prompt
            </label>
            <textarea
              value={negativePrompt}
              onChange={(e) => onNegativePromptChange(e.target.value)}
              placeholder="Describe what to avoid..."
              disabled={disabled}
              rows={2}
              className="w-full rounded-md border border-zinc-700 bg-zinc-800 px-2.5 py-2 text-xs text-white placeholder:text-zinc-500 focus:outline-none focus:ring-1 focus:ring-zinc-500 focus:border-zinc-500 disabled:cursor-not-allowed disabled:opacity-50 resize-y"
            />
            <p className="text-[10px] text-zinc-600">
              Works with Pro model, audio-to-video, and quality retake modes.
            </p>
          </div>

          {!forceApiGenerations && (
            <>
              <div className="space-y-1.5">
                <label className="block text-[11px] font-medium text-zinc-500">
                  LoRA
                </label>
                <select
                  value={loraPath || ''}
                  onChange={(e) => onLoraPathChange(e.target.value || null)}
                  disabled={disabled}
                  className="w-full rounded-md border border-zinc-700 bg-zinc-800 px-2.5 py-1.5 text-xs text-white focus:outline-none focus:ring-1 focus:ring-zinc-500 focus:border-zinc-500 disabled:cursor-not-allowed disabled:opacity-50"
                >
                  <option value="">None</option>
                  {availableLoras.map((lora) => (
                    <option key={lora} value={lora}>
                      {lora.split('/').pop() ?? lora}
                    </option>
                  ))}
                </select>
              </div>

              {loraPath && (
                <div className="space-y-1">
                  <div className="flex items-center justify-between">
                    <label className="text-[11px] font-medium text-zinc-500">LoRA Strength</label>
                    <span className="text-[11px] text-zinc-500">{loraStrength.toFixed(2)}</span>
                  </div>
                  <input
                    type="range"
                    min="0"
                    max="2"
                    step="0.05"
                    value={loraStrength}
                    onChange={(e) => onLoraStrengthChange(Number(e.target.value))}
                    disabled={disabled}
                    className="w-full accent-blue-500"
                  />
                </div>
              )}
            </>
          )}
        </div>
      )}
    </div>
  )
}
