"use client"

import { useState } from "react"
import Navbar from "@/components/navbar"
import Footer from "@/components/footer"
import EarthLoader from "@/components/loading-spinner"

// ShadCN UI
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card"
import { Label } from "@/components/ui/label"
import { Input } from "@/components/ui/input"
import { MapPin } from "lucide-react"

interface WildfireResult {
  image_used: string
  checkpoint: string
  probability: number
  prediction: number
  label: string
  threshold: number
  error?: string
}

export default function Home() {
  const [result, setResult] = useState<WildfireResult | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [trueColorUrl, setTrueColorUrl] = useState<string | null>(null)
  const [fetchingImage, setFetchingImage] = useState(false)

  // Coordinates
  const [lat1, setLat1] = useState("")
  const [lon1, setLon1] = useState("")
  const [lat2, setLat2] = useState("")
  const [lon2, setLon2] = useState("")

  // Fetch Sentinel Image
  const fetchSentinelImage = async () => {
    if (!lat1 || !lon1 || !lat2 || !lon2) {
      alert("Please enter all coordinates!")
      return
    }

    setFetchingImage(true)
    setTrueColorUrl(null)
    setResult(null)

    try {
      const res = await fetch("/api/sentinel-image", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ lat1, lon1, lat2, lon2 }),
      })

      const data = await res.json()
      if (data.error) {
        alert(data.error)
      } else {
        setTrueColorUrl(data.trueColorUrl) // show image immediately
      }
    } catch (err) {
      console.error(err)
      alert("Failed to fetch satellite image")
    } finally {
      setFetchingImage(false)
    }
  }

  // Run Wildfire Prediction
  const runWildfireDetection = async () => {
    if (!trueColorUrl) {
      alert("Please fetch the satellite image first!")
      return
    }

    setIsLoading(true)
    setResult(null)

    try {
      const res = await fetch("http://localhost:5000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image_url: trueColorUrl }),
      })

      const data = await res.json()
      setResult(data)
      window.scrollTo({ top: 0, behavior: "smooth" })
    } catch (err) {
      console.error(err)
      setResult({ error: "Failed to connect to prediction server" } as any)
    } finally {
      setIsLoading(false)
    }
  }

  const handleRecalculate = () => {
    setResult(null)
    setTrueColorUrl(null)
    window.scrollTo({ top: 0, behavior: "smooth" })
  }

  return (
    <div className="min-h-screen flex flex-col bg-gradient-to-br from-background via-background to-primary/5">
      <Navbar />
      {isLoading && <EarthLoader />}

      <section className="flex-1 pt-20 pb-12 px-4 md:px-8">
        <div className="max-w-6xl mx-auto">
          <h1 className="text-5xl md:text-6xl font-bold mb-4 text-center">
            Wildfire Detection <span className="text-primary">AI</span>
          </h1>
          <p className="text-lg md:text-xl text-muted-foreground mb-8 text-center max-w-3xl mx-auto">
            Analyze the most recent satellite image and detect wildfire risk instantly.
          </p>

          {/* Input Card */}
          <div className="max-w-xl mx-auto mb-4">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <MapPin className="w-5 h-5" />
                  Enter Forest Coordinates
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid md:grid-cols-2 gap-6">
                  <div>
                    <Label htmlFor="lat1">Latitude 1</Label>
                    <Input id="lat1" value={lat1} onChange={e => setLat1(e.target.value)} placeholder="Enter latitude 1" className="mt-2"/>
                  </div>
                  <div>
                    <Label htmlFor="lon1">Longitude 1</Label>
                    <Input id="lon1" value={lon1} onChange={e => setLon1(e.target.value)} placeholder="Enter longitude 1" className="mt-2"/>
                  </div>
                  <div>
                    <Label htmlFor="lat2">Latitude 2</Label>
                    <Input id="lat2" value={lat2} onChange={e => setLat2(e.target.value)} placeholder="Enter latitude 2" className="mt-2"/>
                  </div>
                  <div>
                    <Label htmlFor="lon2">Longitude 2</Label>
                    <Input id="lon2" value={lon2} onChange={e => setLon2(e.target.value)} placeholder="Enter longitude 2" className="mt-2"/>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Fetch Satellite Button */}
          <div className="text-center mb-8">
            <button
              onClick={fetchSentinelImage}
              disabled={fetchingImage}
              className={`bg-secondary text-white font-semibold px-6 py-2 rounded-lg shadow-md transition-all ${
                fetchingImage ? "opacity-50 cursor-not-allowed" : "hover:bg-secondary/90"
              }`}
            >
              {fetchingImage ? "Fetching..." : "Fetch Satellite Image"}
            </button>
          </div>

          {/* Two-column layout */}
          <div className="flex flex-col md:flex-row gap-8">
            {/* Left: Satellite Image */}
            <div className="md:w-1/2 flex justify-center">
              {trueColorUrl && (
                <img src={trueColorUrl} alt="Satellite True Color" className="rounded-lg shadow-md max-w-full" />
              )}
            </div>

            {/* Right: Wildfire Prediction */}
            <div className="md:w-1/2 flex flex-col gap-4">
              {trueColorUrl && (
                <>
                  <button
                    onClick={runWildfireDetection}
                    disabled={isLoading}
                    className={`bg-primary text-primary-foreground font-semibold px-6 py-3 rounded-lg shadow-md transition-all ${
                      isLoading ? "opacity-50 cursor-not-allowed" : "hover:bg-primary/90"
                    }`}
                  >
                    Run Wildfire Analysis
                  </button>

                  {isLoading && <EarthLoader />}
                </>
              )}

              {result && !result.error && (
                <Card className="shadow-lg rounded-xl">
                  <CardHeader>
                    <CardTitle>Wildfire Analysis Result</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-2">
                    <p className="text-xl font-semibold">
                      <strong>Status:</strong>{" "}
                      {result.label === "Wildfire" ? (
                        <span className="text-red-600">🔥 Wildfire Detected</span>
                      ) : (
                        <span className="text-green-600">✔ No Wildfire</span>
                      )}
                    </p>
                    <p><strong>Probability:</strong> {(result.probability * 100).toFixed(2)}%</p>
                    <p><strong>Prediction:</strong> {result.prediction}</p>
                    <button
                      onClick={handleRecalculate}
                      className="mt-4 w-full bg-primary text-primary-foreground py-2 rounded-lg hover:bg-primary/90 transition-all"
                    >
                      Analyze Again
                    </button>
                  </CardContent>
                </Card>
              )}

              {result && result.error && (
                <Card className="bg-red-200 text-red-900 shadow-md rounded-lg">
                  <CardContent>
                    <p><strong>Error:</strong> {result.error}</p>
                  </CardContent>
                </Card>
              )}
            </div>
          </div>
        </div>
      </section>

      <Footer />
    </div>
  )
}
