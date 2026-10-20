import React, { useEffect, useRef, useState } from 'react'
import appIcon from '../assets/app_icon_64.png?inline'
import { createRoot } from 'react-dom/client'
import './style.css'

type Rect = { x: number; y: number; w: number; h: number }
type Selection = { mode: 'template' | 'regions'; regions?: Rect[]; templateId?: string }
type ImageItem = {
  id: string
  name: string
  path: string
  width: number
  height: number
  bytes: number
  preview: string
  maskPreview: string | null
  defaultMode: 'template' | 'regions'
  defaultTemplate: string
}
type ImportPayload = { items: ImageItem[]; errors: string[] }
type Result = { state: 'queued' | 'processing' | 'success' | 'failed' | 'cancelled'; path?: string; name?: string; preview?: string; error?: string }
type Job = { id: string; state: 'running' | 'finished' | 'cancelled'; total: number; completed: number; results: Record<string, Result> }
type Api = {
  bootstrap(): Promise<{ outputDir: string }>
  choose_images(): Promise<ImportPayload>
  import_dropped_file(name: string, encoded: string): Promise<ImportPayload>
  choose_output_dir(): Promise<string | null>
  remove_item(id: string): Promise<boolean>
  get_editor_preview(id: string): Promise<string>
  get_mask_preview(id: string, templateId: string): Promise<string | null>
  start_job(requests: { id: string; selection: Selection }[], outputDir: string): Promise<string>
  get_job_status(id: string): Promise<Job>
  cancel_job(id: string): Promise<boolean>
  open_result(jobId: string, imageId: string, reveal: boolean): Promise<boolean>
  open_output_dir(): Promise<boolean>
}

declare global {
  interface Window {
    pywebview?: { api: Api }
  }
}

function getApi(): Api {
  if (!window.pywebview?.api) throw new Error('桌面服务尚未就绪，请稍等片刻')
  return window.pywebview.api
}

function Icon({ name, size = 20 }: { name: 'plus' | 'image' | 'folder' | 'spark' | 'trash' | 'arrow' | 'check' | 'close' | 'edit' | 'open' | 'stop'; size?: number }) {
  const shared = { width: size, height: size, viewBox: '0 0 24 24', fill: 'none', stroke: 'currentColor', strokeWidth: 1.8, strokeLinecap: 'round' as const, strokeLinejoin: 'round' as const }
  const shapes: Record<typeof name, React.ReactNode> = {
    plus: <path d="M12 5v14M5 12h14" />,
    image: <><rect x="3" y="3" width="18" height="18" rx="3" /><circle cx="8.5" cy="8.5" r="1.5" /><path d="m21 16-5-5L5 21" /></>,
    folder: <path d="M3 7a2 2 0 0 1 2-2h5l2 2h7a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z" />,
    spark: <><path d="m12 3 1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8z" /><path d="m19 17 .7 1.3L21 19l-1.3.7L19 21l-.7-1.3L17 19l1.3-.7z" /></>,
    trash: <><path d="M4 7h16M9 7V4h6v3m-9 0 1 13h10l1-13M10 11v5m4-5v5" /></>,
    arrow: <path d="m8 5 7 7-7 7" />,
    check: <path d="m5 12 4 4L19 6" />,
    close: <path d="M5 5 19 19M19 5 5 19" />,
    edit: <><path d="m4 16 11-11 4 4L8 20H4z" /><path d="m13 7 4 4" /></>,
    open: <><path d="M13 5h6v6M19 5l-9 9" /><path d="M19 14v4a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V7a2 2 0 0 1 2-2h4" /></>,
    stop: <rect x="6" y="6" width="12" height="12" rx="2" />,
  }
  return <svg {...shared} aria-hidden="true">{shapes[name]}</svg>
}

function formatBytes(bytes: number) {
  return bytes < 1024 * 1024 ? `${(bytes / 1024).toFixed(0)} KB` : `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

function Editor({ item, initial, onSave, onClose }: { item: ImageItem; initial: Selection; onSave: (selection: Selection) => void; onClose: () => void }) {
  const [mode, setMode] = useState<Selection['mode']>(initial.mode)
  const [regions, setRegions] = useState<Rect[]>(initial.regions || [])
  const [live, setLive] = useState<Rect | null>(null)
  const [preview, setPreview] = useState(item.preview)
  const [templateId, setTemplateId] = useState(initial.templateId || item.defaultTemplate)
  const [maskPreview, setMaskPreview] = useState(item.maskPreview)
  const [maskReady, setMaskReady] = useState(false)
  const stage = useRef<HTMLDivElement>(null)
  const origin = useRef<{ x: number; y: number } | null>(null)

  useEffect(() => {
    getApi().get_editor_preview(item.id).then(setPreview).catch(() => undefined)
  }, [item.id])

  useEffect(() => {
    let active = true
    setMaskReady(false)
    setMaskPreview(null)
    getApi().get_mask_preview(item.id, templateId).then(value => {
      if (active) { setMaskPreview(value); setMaskReady(Boolean(value)) }
    }).catch(() => { if (active) setMaskReady(false) })
    return () => { active = false }
  }, [item.id, templateId])

  const coordinates = (event: React.PointerEvent<HTMLDivElement>) => {
    const box = stage.current!.getBoundingClientRect()
    return { x: Math.max(0, Math.min(1, (event.clientX - box.left) / box.width)), y: Math.max(0, Math.min(1, (event.clientY - box.top) / box.height)) }
  }
  const normalize = (a: { x: number; y: number }, b: { x: number; y: number }): Rect => ({ x: Math.min(a.x, b.x), y: Math.min(a.y, b.y), w: Math.abs(a.x - b.x), h: Math.abs(a.y - b.y) })
  const start = (event: React.PointerEvent<HTMLDivElement>) => {
    if (mode !== 'regions') return
    event.currentTarget.setPointerCapture(event.pointerId)
    origin.current = coordinates(event)
    setLive(null)
  }
  const move = (event: React.PointerEvent<HTMLDivElement>) => {
    if (origin.current) setLive(normalize(origin.current, coordinates(event)))
  }
  const end = (event: React.PointerEvent<HTMLDivElement>) => {
    if (!origin.current) return
    const rect = normalize(origin.current, coordinates(event))
    if (rect.w > 0.005 && rect.h > 0.005) setRegions(current => [...current, rect].slice(0, 12))
    origin.current = null
    setLive(null)
  }
  const styleFor = (rect: Rect): React.CSSProperties => ({ left: `${rect.x * 100}%`, top: `${rect.y * 100}%`, width: `${rect.w * 100}%`, height: `${rect.h * 100}%` })

  return <div className="modal-backdrop" role="presentation" onMouseDown={onClose}>
    <div className="editor-modal" role="dialog" aria-modal="true" aria-label="调整去除区域" onMouseDown={e => e.stopPropagation()}>
      <div className="modal-heading"><div><span className="eyebrow">区域设置</span><h2>{item.name}</h2></div><button className="icon-button" onClick={onClose} aria-label="关闭"><Icon name="close" /></button></div>
      <div className="editor-layout">
        <div className="editor-stage-wrap">
          <div className={`editor-stage ${mode === 'regions' ? 'is-drawing' : ''}`} ref={stage} onPointerDown={start} onPointerMove={move} onPointerUp={end} onPointerCancel={end}>
            <img src={preview} alt={item.name} draggable={false} />
            {mode === 'template' && maskPreview && <img className="mask-preview" src={maskPreview} alt="内置掩膜覆盖区域" draggable={false} />}
            {mode === 'regions' && regions.map((rect, index) => <div className="drawn-region" style={styleFor(rect)} key={index}><span>{index + 1}</span></div>)}
            {mode === 'regions' && live && <div className="drawn-region live" style={styleFor(live)} />}
          </div>
        </div>
        <div className="editor-controls">
          <h3>选择修复范围</h3>
          <label className={`choice-card ${mode === 'template' ? 'chosen' : ''} ${!item.maskPreview ? 'disabled' : ''}`}><input type="radio" checked={mode === 'template'} disabled={!item.maskPreview} onChange={() => setMode('template')} /><div><strong>使用内置掩膜</strong><p>适合版式匹配的 3:4 竖图。红色区域会被修复。</p></div></label>
          {mode === 'template' && <label>水印模板<select aria-label="水印模板" value={templateId} onChange={e => setTemplateId(e.target.value)}><option value="account-11578066187">账号 11578066187</option><option value="account-267777923">账号 267777923</option><option value="legacy">旧版图片模板</option></select><p className="editor-note">仅适合对应账号及右下角版式。处理前请核对红色覆盖范围；其他账号请手动圈选。</p>{!maskReady && <p>模板预览尚未就绪，请稍候或改用手动圈选。</p>}</label>}
          <label className={`choice-card ${mode === 'regions' ? 'chosen' : ''}`}><input type="radio" checked={mode === 'regions'} onChange={() => setMode('regions')} /><div><strong>手动圈定区域</strong><p>在左侧图片上拖动，逐个框住水印。最多 12 个区域。</p></div></label>
          {mode === 'regions' && <div className="region-actions"><span>已选 {regions.length} 个区域</span><button onClick={() => setRegions(current => current.slice(0, -1))} disabled={!regions.length}>撤销上一步</button><button onClick={() => setRegions([])} disabled={!regions.length}>清空</button></div>}
          <p className="editor-note">修复会估算被遮挡的画面。请让选区尽量贴合水印，避免覆盖不相关内容。</p>
        </div>
      </div>
      <div className="modal-footer"><button className="button subtle" onClick={onClose}>取消</button><button className="button primary" onClick={() => onSave({ mode, ...(mode === 'regions' ? { regions } : { templateId }) })} disabled={mode === 'regions' ? !regions.length : !maskReady}>应用设置</button></div>
    </div>
  </div>
}

function App() {
  const [ready, setReady] = useState(Boolean(window.pywebview?.api))
  const [items, setItems] = useState<ImageItem[]>([])
  const [selections, setSelections] = useState<Record<string, Selection>>({})
  const [outputDir, setOutputDir] = useState('')
  const [editor, setEditor] = useState<ImageItem | null>(null)
  const [jobId, setJobId] = useState<string | null>(null)
  const [job, setJob] = useState<Job | null>(null)
  const [notice, setNotice] = useState('')
  const [dragging, setDragging] = useState(false)
  const fileCount = items.length
  const running = job?.state === 'running'
  const importedIds = useRef(new Set<string>())

  const importPayload = (payload: ImportPayload) => {
    const fresh = payload.items.filter(item => !importedIds.current.has(item.id))
    fresh.forEach(item => importedIds.current.add(item.id))
    setItems(current => [...current, ...fresh])
    setSelections(current => ({ ...current, ...Object.fromEntries(fresh.map(item => [item.id, { mode: item.defaultMode, templateId: item.defaultTemplate, regions: [] }])) }))
    if (payload.errors.length) setNotice(payload.errors.slice(0, 3).join('；'))
  }

  useEffect(() => {
    const onReady = () => { setReady(true); getApi().bootstrap().then(data => setOutputDir(data.outputDir)).catch(err => setNotice(String(err))) }
    const onImport = (event: Event) => { importPayload((event as CustomEvent<ImportPayload>).detail); setDragging(false) }
    window.addEventListener('pywebviewready', onReady)
    window.addEventListener('corner-cleaner:import', onImport)
    if (window.pywebview?.api) onReady()
    return () => { window.removeEventListener('pywebviewready', onReady); window.removeEventListener('corner-cleaner:import', onImport) }
  }, [])

  useEffect(() => {
    if (!jobId) return
    let disposed = false
    const poll = async () => {
      try {
        const current = await getApi().get_job_status(jobId)
        if (!disposed) {
          setJob(current)
          if (current.state !== 'running') window.clearInterval(timer)
        }
      } catch (err) { if (!disposed) setNotice(String(err)) }
    }
    void poll()
    const timer = window.setInterval(poll, 500)
    return () => { disposed = true; window.clearInterval(timer) }
  }, [jobId])

  const choose = async () => {
    try { importPayload(await getApi().choose_images()) } catch (err) { setNotice(String(err)) }
  }
  const dropFiles = async (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault()
    setDragging(false)
    if (running) { setNotice('请等待当前批次处理完成后再添加图片'); return }
    const files = Array.from(event.dataTransfer.files)
    if (!files.length) return
    for (const file of files) {
      try {
        if (file.size > 35 * 1024 * 1024) throw new Error(`${file.name} 超过拖入上限 35 MB，请点击加号选择`)
        const encoded = await new Promise<string>((resolve, reject) => {
          const reader = new FileReader()
          reader.onerror = () => reject(new Error(`${file.name} 读取失败`))
          reader.onload = () => resolve(String(reader.result).split(',')[1])
          reader.readAsDataURL(file)
        })
        importPayload(await getApi().import_dropped_file(file.name, encoded))
      } catch (err) { setNotice(String(err)) }
    }
  }
  const chooseDir = async () => {
    try { const directory = await getApi().choose_output_dir(); if (directory) setOutputDir(directory) } catch (err) { setNotice(String(err)) }
  }
  const remove = async (id: string) => {
    try { await getApi().remove_item(id); setItems(current => current.filter(item => item.id !== id)); setSelections(current => { const next = { ...current }; delete next[id]; return next }) } catch (err) { setNotice(String(err)) }
  }
  const startJob = async () => {
    const invalid = items.find(item => selections[item.id]?.mode === 'regions' && !selections[item.id]?.regions?.length)
    if (invalid) { setNotice(`请先为「${invalid.name}」圈定水印区域`); setEditor(invalid); return }
    try {
      const id = await getApi().start_job(items.map(item => ({ id: item.id, selection: selections[item.id] })), outputDir)
      setJobId(id)
      setJob({ id, state: 'running', total: items.length, completed: 0, results: {} })
      setNotice('')
    } catch (err) { setNotice(String(err)) }
  }
  const resultEntries = job && job.state !== 'running' ? items.filter(item => job.results[item.id]) : []

  return <div className="app-shell" onDragEnter={e => { e.preventDefault(); setDragging(true) }} onDragOver={e => e.preventDefault()} onDragLeave={e => { if (!e.currentTarget.contains(e.relatedTarget as Node)) setDragging(false) }} onDrop={dropFiles}>
    <header className="topbar"><div className="brand-mark"><img src={appIcon} alt="" /></div><div className="brand-title">水印消除<span>本地图片处理</span></div><div className="topbar-status"><span className="status-dot" />离线处理 · 原图保留</div></header>
    <main className={fileCount ? 'workspace' : 'workspace empty-workspace'}>
      {fileCount === 0 ? <section className={`empty-state ${dragging ? 'drag-active' : ''}`}>
        <button className="empty-drop" onClick={choose} disabled={!ready} aria-label="选择图片"><div className="empty-icon"><Icon name="plus" size={34} /></div><h1>从这里开始</h1><p>点击添加图片，或把图片拖到窗口中</p><span>支持 JPG、PNG、WebP · 可一次添加多张</span></button>
        <div className="empty-footnote">图片只在你的电脑上处理，不会上传</div>
      </section> : <>
        <div className="page-heading"><div><span className="eyebrow">图片工作台</span><h1>去除右下角水印</h1><p>检查图片与修复区域，选好保存位置后开始处理。</p></div><div className="count-pill">{fileCount} 张待处理</div></div>
        <section className="panel input-panel"><div className="section-heading"><div><span className="section-index">01</span><div><h2>待处理图片</h2><p>支持批量处理；不匹配内置掩膜的图片请先手动圈选。</p></div></div><button className="button add-button" onClick={choose} disabled={running}><Icon name="plus" size={18} />继续添加</button></div>
          <div className="image-grid">{items.map((item, index) => {
            const selection = selections[item.id]
            const needsRegion = selection?.mode === 'regions' && !selection?.regions?.length
            return <article className="image-card" key={item.id}><div className="card-thumb"><img src={item.preview} alt={item.name} /><span className="card-number">{String(index + 1).padStart(2, '0')}</span></div><div className="card-body"><div className="card-name" title={item.name}>{item.name}</div><div className="card-meta">{item.width} × {item.height}<span>·</span>{formatBytes(item.bytes)}</div><div className={`mask-status ${needsRegion ? 'needs-region' : ''}`}>{needsRegion ? '需要圈定区域' : selection?.mode === 'template' ? `匹配模板 · ${selection.templateId?.replace('account-', '') || '请检查'}` : `手动区域 · ${selection?.regions?.length || 0}`}</div><div className="card-actions"><button onClick={() => setEditor(item)}><Icon name="edit" size={16} />调整区域</button><button className="icon-button danger" onClick={() => remove(item.id)} disabled={running} aria-label={`移除 ${item.name}`}><Icon name="trash" size={16} /></button></div></div></article>
          })}</div>
        </section>
        <section className="action-bar"><div className="action-left"><button className="button primary start-button" onClick={startJob} disabled={running}><Icon name="spark" size={19} />{running ? `处理中 ${job?.completed || 0}/${job?.total || fileCount}` : '开始去水印'}</button>{running && <button className="button subtle" onClick={() => jobId && getApi().cancel_job(jobId).catch(err => setNotice(String(err)))}><Icon name="stop" size={16} />取消后续</button>}</div><div className="destination"><span>保存到</span><button className="destination-picker" onClick={chooseDir} disabled={running} title={outputDir}><Icon name="folder" size={18} /><span>{outputDir || '选择保存位置'}</span><Icon name="arrow" size={15} /></button></div></section>
        {running && <div className="progress-track"><div style={{ width: `${((job?.completed || 0) / Math.max(1, job?.total || 1)) * 100}%` }} /></div>}
        {resultEntries.length > 0 && <section className="panel results-panel"><div className="section-heading"><div><span className="section-index">02</span><div><h2>处理结果</h2><p>成功 {resultEntries.filter(item => job?.results[item.id]?.state === 'success').length} 张 · 失败 {resultEntries.filter(item => job?.results[item.id]?.state === 'failed').length} 张。请放大检查水印位置，复杂背景可能留痕。</p></div></div><button className="button add-button" onClick={() => getApi().open_output_dir().catch(err => setNotice(String(err)))}><Icon name="folder" size={17} />打开文件夹</button></div><div className="results-list">{resultEntries.map(item => { const result = job!.results[item.id]; return <div className="result-row" key={item.id}><img src={result.preview || item.preview} alt="处理结果" /><div className="result-info"><strong>{result.name || item.name}</strong><span className={result.state === 'success' ? 'result-good' : 'result-bad'}>{result.state === 'success' ? '已处理 · 请检查局部细节' : result.state === 'failed' ? result.error : '已取消'}</span></div>{result.state === 'success' && <div className="result-actions"><button onClick={() => getApi().open_result(job!.id, item.id, false).catch(err => setNotice(String(err)))}>查看图片</button><button onClick={() => getApi().open_result(job!.id, item.id, true).catch(err => setNotice(String(err)))}><Icon name="open" size={16} /></button></div>}</div> })}</div></section>}
      </>}
    </main>
    {notice && <div className="toast" role="alert"><span>{notice}</span><button onClick={() => setNotice('')} aria-label="关闭提示"><Icon name="close" size={16} /></button></div>}
    {editor && <Editor item={editor} initial={selections[editor.id]} onClose={() => setEditor(null)} onSave={selection => { setSelections(current => ({ ...current, [editor.id]: selection })); setEditor(null) }} />}
  </div>
}

createRoot(document.getElementById('root')!).render(<App />)
