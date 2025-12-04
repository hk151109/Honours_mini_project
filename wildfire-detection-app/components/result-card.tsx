"use client"

interface PredictionResult {
  emissionScore: number
  status: "Low" | "Moderate" | "High"
  recommendations: string[]
}

interface ResultCardProps {
  result: PredictionResult
  onRecalculate: () => void
}

const getStatusColor = (status: string) => {
  switch (status) {
    case "Low":
      return "bg-green-100 text-green-800 border-green-300"
    case "Moderate":
      return "bg-yellow-100 text-yellow-800 border-yellow-300"
    case "High":
      return "bg-red-100 text-red-800 border-red-300"
    default:
      return "bg-gray-100 text-gray-800 border-gray-300"
  }
}

const getStatusIcon = (status: string) => {
  switch (status) {
    case "Low":
      return "🌱"
    case "Moderate":
      return "⚠️"
    case "High":
      return "🔴"
    default:
      return "❓"
  }
}

export default function ResultCard({ result, onRecalculate }: ResultCardProps) {
  return (
    <div className="bg-card rounded-xl p-6 md:p-12 border border-border shadow-smooth-lg">
      <div className="text-center mb-12">
        <h2 className="text-3xl md:text-4xl font-bold text-foreground mb-4">Your Carbon Footprint Analysis</h2>
        <p className="text-muted-foreground">Based on your lifestyle inputs</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-12">
        {/* Emission Score */}
        <div className="bg-gradient-to-br from-primary/5 to-accent/5 rounded-lg p-8 text-center border border-primary/20">
          <p className="text-muted-foreground text-sm mb-2 font-medium">Carbon Emission Score</p>
          <p className="text-5xl md:text-6xl font-bold text-primary mb-2">{result.emissionScore.toFixed(1)}</p>
          <p className="text-xs text-muted-foreground">kg CO₂e/month</p>
        </div>

        {/* Status Badge */}
        <div className="flex items-center justify-center">
          <div className={`px-6 py-4 rounded-lg border text-center ${getStatusColor(result.status)}`}>
            <p className="text-2xl mb-2">{getStatusIcon(result.status)}</p>
            <p className="text-sm font-semibold mb-1">Sustainability Status</p>
            <p className="text-2xl font-bold">{result.status} Footprint</p>
          </div>
        </div>

        {/* Impact Info */}
        <div className="bg-gradient-to-br from-accent/5 to-primary/5 rounded-lg p-8 text-center border border-accent/20">
          <p className="text-muted-foreground text-sm mb-2 font-medium">Annual Impact</p>
          <p className="text-5xl md:text-6xl font-bold text-accent mb-2">{(result.emissionScore * 12).toFixed(0)}</p>
          <p className="text-xs text-muted-foreground">kg CO₂e/year</p>
        </div>
      </div>

      {/* Recommendations */}
      <div className="bg-muted/30 rounded-lg p-8 mb-8 border border-border">
        <h3 className="text-xl font-bold text-foreground mb-6">Personalized Recommendations</h3>
        <ul className="space-y-4">
          {result.recommendations.map((recommendation, index) => (
            <li key={index} className="flex gap-4 items-start">
              <span className="flex-shrink-0 w-6 h-6 rounded-full bg-primary/20 text-primary flex items-center justify-center text-sm font-bold">
                {index + 1}
              </span>
              <span className="text-foreground leading-relaxed">{recommendation}</span>
            </li>
          ))}
        </ul>
      </div>

      <button
        onClick={onRecalculate}
        className="w-full bg-primary hover:bg-primary/90 text-primary-foreground font-semibold py-3 rounded-lg transition-all duration-300 shadow-smooth hover:shadow-smooth-lg"
      >
        Recalculate
      </button>
    </div>
  )
}
