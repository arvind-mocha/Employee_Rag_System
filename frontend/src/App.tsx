import { lazy, Suspense, useEffect, useState } from 'react'
import { BarChart3, DatabaseZap, MessageSquareText, Settings } from 'lucide-react'
import {
  NavLink,
  Navigate,
  Route,
  Routes,
  useLocation,
  useNavigate,
} from 'react-router-dom'
import { BrandMark } from './components/BrandMark'
import { ModeSwitch } from './components/ModeSwitch'
import { useIngestionStatus } from './hooks/useIngestionStatus'
import { AskPage } from './pages/AskPage'

const DataPage = lazy(() =>
  import('./pages/DataPage').then((module) => ({
    default: module.DataPage,
  })),
)

const EvaluationPage = lazy(() =>
  import('./pages/EvaluationPage').then((module) => ({
    default: module.EvaluationPage,
  })),
)

const SettingsPage = lazy(() =>
  import('./pages/SettingsPage').then((module) => ({
    default: module.SettingsPage,
  })),
)

function readInitialMode() {
  return window.localStorage.getItem('employee-rag-mode') === 'developer'
}

function dataPageFallback(message: string) {
  return (
    <div className="route-loading">
      <i className="spinner dark" /> {message}
    </div>
  )
}

function ChatRoute({ developerMode }: { developerMode: boolean }) {
  const ingestionQuery = useIngestionStatus()

  if (ingestionQuery.isLoading) {
    return dataPageFallback('Checking knowledge base…')
  }

  if (ingestionQuery.data?.status !== 'completed') {
    return (
      <Suspense fallback={dataPageFallback('Loading data setup…')}>
        <DataPage />
      </Suspense>
    )
  }

  return <AskPage developerMode={developerMode} />
}

export default function App() {
  const [developerMode, setDeveloperMode] = useState(readInitialMode)
  const location = useLocation()
  const navigate = useNavigate()

  useEffect(() => {
    window.localStorage.setItem(
      'employee-rag-mode',
      developerMode ? 'developer' : 'user',
    )
    if (!developerMode && location.pathname.startsWith('/evaluation')) {
      navigate('/', { replace: true })
    }
  }, [developerMode, location.pathname, navigate])

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="topbar-inner">
          <NavLink className="brand" to="/" aria-label="Krag home">
            <BrandMark />
            <span>
              <strong>Krag</strong>
              <small>Employee knowledge</small>
            </span>
          </NavLink>

          <nav className="main-nav" aria-label="Primary navigation">
            <NavLink to="/" end>
              <MessageSquareText size={17} />
              Ask
            </NavLink>
            {developerMode && (
              <>
                <NavLink to="/data">
                  <DatabaseZap size={17} />
                  Data
                </NavLink>
                <NavLink to="/evaluation">
                  <BarChart3 size={17} />
                  Evaluation
                </NavLink>
              </>
            )}
            <NavLink to="/settings">
              <Settings size={17} />
              Settings
            </NavLink>
          </nav>

          <ModeSwitch
            developerMode={developerMode}
            onChange={setDeveloperMode}
          />
        </div>
      </header>

      <main>
        <Routes>
          <Route path="/" element={<ChatRoute developerMode={developerMode} />} />
          <Route
            path="/data"
            element={
              <Suspense fallback={dataPageFallback('Loading data setup…')}>
                <DataPage />
              </Suspense>
            }
          />
          <Route
            path="/evaluation"
            element={
              developerMode ? (
                <Suspense
                  fallback={
                    <div className="route-loading">
                      <i className="spinner dark" /> Loading quality lab…
                    </div>
                  }
                >
                  <EvaluationPage />
                </Suspense>
              ) : (
                <Navigate to="/" replace />
              )
            }
          />
          <Route
            path="/settings"
            element={
              <Suspense
                fallback={
                  <div className="route-loading">
                    <i className="spinner dark" /> Loading settings…
                  </div>
                }
              >
                <SettingsPage />
              </Suspense>
            }
          />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>
    </div>
  )
}
