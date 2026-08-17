import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
  type KeyboardEvent,
  type PointerEvent,
  type ReactNode,
} from 'react'
import { WakuIcon } from '@/components/waku-icon'
import { useI18n } from '@/lib/i18n'
import { cn } from '@/lib/utils'

type MermaidModule = typeof import('mermaid')

type SvgSize = { width: number; height: number }

type DragState = {
  pointerId: number
  startX: number
  startY: number
  scrollLeft: number
  scrollTop: number
}

const MIN_ZOOM = 0.25
const MAX_ZOOM = 8
const ZOOM_FACTOR = 1.1
const WHEEL_DELTA = 100
const KEYBOARD_SCROLL_STEP = 48

let mermaidModule: Promise<MermaidModule> | null = null
let nextDiagramId = 0

function loadMermaid(): Promise<MermaidModule> {
  if (!mermaidModule) {
    // Lazy-loaded so the renderer's parser pays for itself only on the first
    // diagram a session actually shows. A failed import is reset so a
    // transient bundle-load error can be retried on the next block.
    mermaidModule = import('mermaid').catch((error) => {
      mermaidModule = null
      throw error
    })
  }
  return mermaidModule
}

function prefersDark(): boolean {
  if (typeof document === 'undefined') return false
  if (document.documentElement.classList.contains('dark')) return true
  return typeof window.matchMedia === 'function'
    && window.matchMedia('(prefers-color-scheme: dark)').matches
}

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value))
}

/// The largest scale that fits the diagram inside the viewport's content box,
/// never upscaling. The viewport's padding belongs to the box, not the
/// diagram, so it is subtracted before comparing against the natural size.
function computeFitZoom(viewport: HTMLDivElement, naturalSize: SvgSize): number {
  const style = getComputedStyle(viewport)
  const padX = Number.parseFloat(style.paddingLeft) + Number.parseFloat(style.paddingRight)
  const padY = Number.parseFloat(style.paddingTop) + Number.parseFloat(style.paddingBottom)
  const boxWidth = Math.max(1, viewport.clientWidth - padX)
  const boxHeight = Math.max(1, viewport.clientHeight - padY)
  return clamp(
    Math.min(1, boxWidth / naturalSize.width, boxHeight / naturalSize.height),
    MIN_ZOOM,
    MAX_ZOOM,
  )
}

function parseDimension(value: string | null): number | null {
  if (!value) return null
  const normalized = value.trim()
  if (!normalized || normalized.endsWith('%')) return null
  const parsed = Number.parseFloat(normalized)
  return Number.isFinite(parsed) && parsed > 0 ? parsed : null
}

function parseSvgSize(svg: string): SvgSize | null {
  const openTag = svg.match(/<svg\b[^>]*>/i)?.[0]
  if (!openTag) return null
  const attribute = (name: string) => openTag.match(
    new RegExp(`${name}=["']([^"']+)["']`, 'i'),
  )?.[1] ?? null
  const width = parseDimension(attribute('width'))
  const height = parseDimension(attribute('height'))
  if (width !== null && height !== null) return { width, height }
  const viewBox = attribute('viewBox') ?? attribute('viewbox')
  if (viewBox) {
    const parts = viewBox.trim().split(/[\s,]+/).map(Number)
    if (parts.length === 4 && parts.every(Number.isFinite)) {
      return { width: parts[2]!, height: parts[3]! }
    }
  }
  return null
}

export function MermaidDiagram({
  code,
  fallback,
}: {
  code: string
  fallback: ReactNode
}) {
  const { t } = useI18n()
  const [id] = useState(() => {
    nextDiagramId += 1
    return `mermaid-diagram-${nextDiagramId}`
  })
  const [svg, setSvg] = useState<string | null>(null)
  const [failed, setFailed] = useState(false)
  const [naturalSize, setNaturalSize] = useState<SvgSize | null>(null)
  const [zoom, setZoom] = useState(1)
  const [dragging, setDragging] = useState(false)
  const [canReset, setCanReset] = useState(false)
  const requestRef = useRef(0)
  const viewportRef = useRef<HTMLDivElement>(null)
  const dragState = useRef<DragState | null>(null)
  const zoomRef = useRef(1)
  /// The fitted default for the current diagram, measured once the viewport
  /// exists. `null` until then; reset and the "zoomed away" indicator compare
  /// against it instead of a fixed 100%.
  const fitZoomRef = useRef<number | null>(null)
  const naturalSizeRef = useRef<SvgSize | null>(null)
  const pendingScroll = useRef<{ left: number; top: number } | null>(null)

  useEffect(() => {
    let cancelled = false
    const request = ++requestRef.current
    setFailed(false)
    setSvg(null)
    setNaturalSize(null)
    naturalSizeRef.current = null
    zoomRef.current = 1
    setZoom(1)
    fitZoomRef.current = null
    pendingScroll.current = null
    dragState.current = null
    setDragging(false)
    setCanReset(false)

    void loadMermaid().then((module) => {
      if (cancelled) return
      const mermaid = module.default
      mermaid.initialize({
        startOnLoad: false,
        securityLevel: 'strict',
        theme: prefersDark() ? 'dark' : 'default',
        // The surrounding markdown surface already paints the background; a
        // full-canvas rect would otherwise show a hard box in both themes.
        themeVariables: { background: 'transparent' },
        fontFamily: 'var(--font-sans), -apple-system, BlinkMacSystemFont, sans-serif',
      })
      mermaid.render(id, code).then(
        (result) => {
          if (cancelled || requestRef.current !== request) return
          const size = parseSvgSize(result.svg)
          naturalSizeRef.current = size
          setNaturalSize(size)
          setSvg(result.svg)
        },
        () => {
          if (cancelled || requestRef.current === request) setFailed(true)
        },
      )
    }, () => {
      if (cancelled || requestRef.current === request) setFailed(true)
    })

    return () => {
      cancelled = true
      if (requestRef.current === request) requestRef.current += 1
    }
  }, [code, id])

  const updateCanReset = useCallback(() => {
    const viewport = viewportRef.current
    if (!viewport) return
    // Without a fitted default (unparseable natural size) only scrolling can
    // move the view, so that alone controls the affordance.
    const zoomed = fitZoomRef.current !== null && zoomRef.current !== fitZoomRef.current
    const active = viewport.scrollLeft > 0 || viewport.scrollTop > 0 || zoomed
    setCanReset((current) => current === active ? current : active)
  }, [])

  const zoomBy = useCallback((factor: number, cursorX: number, cursorY: number) => {
    const viewport = viewportRef.current
    const size = naturalSizeRef.current
    if (!viewport || !size) return
    const current = zoomRef.current
    const next = clamp(current * factor, MIN_ZOOM, MAX_ZOOM)
    if (next === current) return
    const anchorX = viewport.scrollLeft + cursorX
    const anchorY = viewport.scrollTop + cursorY
    pendingScroll.current = {
      left: (anchorX * next) / current - cursorX,
      top: (anchorY * next) / current - cursorY,
    }
    zoomRef.current = next
    setZoom(next)
  }, [])

  const reset = useCallback(() => {
    const viewport = viewportRef.current
    const size = naturalSizeRef.current
    if (viewport && size) {
      // Re-measure so reset restores a fresh fit even after a resize while
      // the diagram was zoomed away from it.
      const fit = computeFitZoom(viewport, size)
      fitZoomRef.current = fit
      zoomRef.current = fit
      setZoom(fit)
    } else {
      zoomRef.current = 1
      setZoom(1)
    }
    pendingScroll.current = null
    if (viewport) {
      viewport.scrollLeft = 0
      viewport.scrollTop = 0
    }
    setCanReset(false)
  }, [])

  // The wheel listener must be non-passive so Ctrl/Cmd+wheel can suppress the
  // browser's page zoom and drive the diagram zoom instead.
  useEffect(() => {
    if (!svg) return
    const viewport = viewportRef.current
    if (!viewport) return
    const onWheel = (event: WheelEvent) => {
      if (!event.ctrlKey && !event.metaKey) return
      event.preventDefault()
      const rect = viewport.getBoundingClientRect()
      zoomBy(
        Math.pow(ZOOM_FACTOR, -event.deltaY / WHEEL_DELTA),
        event.clientX - rect.left,
        event.clientY - rect.top,
      )
    }
    viewport.addEventListener('wheel', onWheel, { passive: false })
    return () => viewport.removeEventListener('wheel', onWheel)
  }, [svg, zoomBy])

  useLayoutEffect(() => {
    const viewport = viewportRef.current
    if (!viewport) return
    // The default view fits the diagram to its render box. This runs before
    // paint with the viewport already measured, so the fitted scale lands on
    // the first visible frame; the `setZoom` below re-runs this effect with
    // the fit recorded, making the computation one-shot per diagram.
    if (naturalSize && fitZoomRef.current === null) {
      const fit = computeFitZoom(viewport, naturalSize)
      fitZoomRef.current = fit
      zoomRef.current = fit
      setZoom(fit)
    }
    if (pendingScroll.current) {
      viewport.scrollLeft = pendingScroll.current.left
      viewport.scrollTop = pendingScroll.current.top
      pendingScroll.current = null
    }
    updateCanReset()
  }, [naturalSize, zoom, updateCanReset])

  // A resize while the view is still at the fitted default re-fits; a manual
  // zoom keeps its scale across resizes.
  useEffect(() => {
    const viewport = viewportRef.current
    if (!viewport || !naturalSize) return
    const observer = new ResizeObserver(() => {
      if (fitZoomRef.current === null) return
      if (Math.abs(zoomRef.current - fitZoomRef.current) > 0.001) return
      const fit = computeFitZoom(viewport, naturalSize)
      if (Math.abs(fit - fitZoomRef.current) > 0.001) {
        fitZoomRef.current = fit
        zoomRef.current = fit
        setZoom(fit)
      }
    })
    observer.observe(viewport)
    return () => observer.disconnect()
  }, [naturalSize])

  function onPointerDown(event: PointerEvent<HTMLDivElement>) {
    // Keep touch drags on the native scroll gesture; only mouse/trackpad get
    // the explicit drag-to-pan behavior.
    if (event.pointerType !== 'mouse' || event.button !== 0) return
    const viewport = viewportRef.current
    if (!viewport) return
    event.currentTarget.setPointerCapture(event.pointerId)
    dragState.current = {
      pointerId: event.pointerId,
      startX: event.clientX,
      startY: event.clientY,
      scrollLeft: viewport.scrollLeft,
      scrollTop: viewport.scrollTop,
    }
    setDragging(true)
  }

  function onPointerMove(event: PointerEvent<HTMLDivElement>) {
    const state = dragState.current
    const viewport = viewportRef.current
    if (!state || !viewport || state.pointerId !== event.pointerId) return
    viewport.scrollLeft = state.scrollLeft - (event.clientX - state.startX)
    viewport.scrollTop = state.scrollTop - (event.clientY - state.startY)
    updateCanReset()
  }

  function onPointerUp(event: PointerEvent<HTMLDivElement>) {
    if (dragState.current?.pointerId !== event.pointerId) return
    dragState.current = null
    setDragging(false)
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId)
    }
  }

  function onKeyDown(event: KeyboardEvent<HTMLDivElement>) {
    const viewport = event.currentTarget
    let handled = true
    switch (event.key) {
      case 'ArrowLeft':
        viewport.scrollLeft -= KEYBOARD_SCROLL_STEP
        break
      case 'ArrowRight':
        viewport.scrollLeft += KEYBOARD_SCROLL_STEP
        break
      case 'ArrowUp':
        viewport.scrollTop -= KEYBOARD_SCROLL_STEP
        break
      case 'ArrowDown':
        viewport.scrollTop += KEYBOARD_SCROLL_STEP
        break
      case '+':
      case '=':
        zoomBy(ZOOM_FACTOR, viewport.clientWidth / 2, viewport.clientHeight / 2)
        break
      case '-':
      case '_':
        zoomBy(1 / ZOOM_FACTOR, viewport.clientWidth / 2, viewport.clientHeight / 2)
        break
      case '0':
        reset()
        break
      default:
        handled = false
    }
    if (handled) {
      event.preventDefault()
      updateCanReset()
    }
  }

  if (failed) return <>{fallback}</>
  if (!svg) {
    return (
      <div
        aria-hidden="true"
        className="mermaid-diagram mermaid-diagram--loading"
      >
        <span className="size-4 animate-pulse rounded-full bg-[var(--muted)] motion-reduce:animate-none" />
      </div>
    )
  }
  const contentWidth = naturalSize ? Math.round(naturalSize.width * zoom) : undefined
  const contentHeight = naturalSize ? Math.round(naturalSize.height * zoom) : undefined
  return (
    <div className="relative">
      <div
        aria-label={t('mermaid.diagram')}
        className={cn(
          'mermaid-diagram',
          naturalSize ? 'mermaid-diagram--zoomable' : 'mermaid-diagram--fallback',
          dragging && 'mermaid-diagram--dragging',
        )}
        ref={viewportRef}
        role="region"
        tabIndex={0}
        onKeyDown={onKeyDown}
        onScroll={updateCanReset}
      >
        <div
          className="mermaid-diagram__content"
          draggable={false}
          style={contentWidth !== undefined ? { width: contentWidth, height: contentHeight } : undefined}
          onPointerCancel={onPointerUp}
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={onPointerUp}
        >
          <div dangerouslySetInnerHTML={{ __html: svg }} />
        </div>
      </div>
      {canReset && (
        <button
          aria-label={t('common.reset')}
          className="mermaid-diagram__reset"
          type="button"
          onClick={reset}
        >
          <WakuIcon className="size-3" name="rotateCw" />
          {t('common.reset')}
        </button>
      )}
    </div>
  )
}
