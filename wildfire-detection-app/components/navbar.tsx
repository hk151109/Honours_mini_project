export default function Navbar() {
  return (
    <nav className="fixed top-0 w-full z-50 backdrop-blur-md bg-white/80 border-b border-border">
      <div className="max-w-7xl mx-auto px-4 md:px-8 py-4 flex items-center justify-between">

        {/* Logo */}
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 bg-gradient-to-br from-red-500 to-orange-500 rounded-lg flex items-center justify-center">
            <span className="text-white font-bold text-sm">W</span>
          </div>
          <h1 className="text-xl md:text-2xl font-bold text-foreground">
            Wildfire Detection
          </h1>
        </div>

        {/* Subtitle */}
        <div className="text-sm text-muted-foreground">
          Satellite-Based Fire Risk Analysis
        </div>

      </div>
    </nav>
  )
}
