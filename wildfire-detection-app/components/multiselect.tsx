"use client"

interface Option {
  value: string
  label: string
}

interface MultiSelectProps {
  label: string
  options: Option[]
  selected: string[]
  onChange: (values: string[]) => void
}

export default function MultiSelect({ label, options, selected, onChange }: MultiSelectProps) {
  const handleToggle = (value: string) => {
    if (selected.includes(value)) {
      onChange(selected.filter((item) => item !== value))
    } else {
      onChange([...selected, value])
    }
  }

  return (
    <div className="flex flex-col gap-3">
      <label className="text-sm font-semibold text-foreground">{label}</label>
      <div className="flex flex-wrap gap-2">
        {options.map((option) => (
          <button
            key={option.value}
            type="button"
            onClick={() => handleToggle(option.value)}
            className={`px-4 py-2 rounded-lg font-medium transition-all ${
              selected.includes(option.value)
                ? "bg-primary text-primary-foreground shadow-smooth"
                : "bg-muted text-foreground border border-input hover:border-primary"
            }`}
          >
            {option.label}
          </button>
        ))}
      </div>
    </div>
  )
}
