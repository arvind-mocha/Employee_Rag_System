import { Code2, UserRound } from 'lucide-react'

interface ModeSwitchProps {
  developerMode: boolean
  onChange: (value: boolean) => void
}

export function ModeSwitch({ developerMode, onChange }: ModeSwitchProps) {
  return (
    <div className="mode-switch" aria-label="Interface mode">
      <button
        className={!developerMode ? 'active' : ''}
        type="button"
        onClick={() => onChange(false)}
        aria-pressed={!developerMode}
      >
        <UserRound size={15} />
        User
      </button>
      <button
        className={developerMode ? 'active' : ''}
        type="button"
        onClick={() => onChange(true)}
        aria-pressed={developerMode}
      >
        <Code2 size={15} />
        Developer
      </button>
    </div>
  )
}
