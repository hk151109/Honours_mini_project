"use client"

import type React from "react"

import { useState } from "react"
import InputField from "./input-field"
import MultiSelect from "./multiselect"

interface FormData {
  bodyType: string
  sex: string
  diet: string
  showerFrequency: string
  heatingEnergy: string
  transport: string
  vehicleType: string
  socialActivity: string
  monthlyGroceryBill: number
  airTravelFrequency: string
  vehicleMonthlyKm: number
  wasteBagSize: string
  wasteBagWeeklyCount: number
  tvPcHoursDaily: number
  newClothesMonthly: number
  internetHoursDaily: number
  energyEfficiency: string
  recycling: string[]
  cookingWith: string[]
}

interface CarbonFormProps {
  onSubmit: (data: FormData) => void
  isLoading: boolean
}

export default function CarbonForm({ onSubmit, isLoading }: CarbonFormProps) {
  const [formData, setFormData] = useState<FormData>({
    bodyType: "",
    sex: "",
    diet: "",
    showerFrequency: "",
    heatingEnergy: "",
    transport: "",
    vehicleType: "",
    socialActivity: "",
    monthlyGroceryBill: 0,
    airTravelFrequency: "",
    vehicleMonthlyKm: 0,
    wasteBagSize: "",
    wasteBagWeeklyCount: 0,
    tvPcHoursDaily: 0,
    newClothesMonthly: 0,
    internetHoursDaily: 0,
    energyEfficiency: "",
    recycling: [],
    cookingWith: [],
  })

  const handleChange = (field: keyof FormData, value: any) => {
    setFormData((prev) => ({ ...prev, [field]: value }))
  }

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    onSubmit(formData)
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-8">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 bg-card rounded-xl p-6 md:p-8 border border-border shadow-smooth">
        <h3 className="col-span-full text-xl font-semibold text-foreground mb-4">Personal Information</h3>

        <InputField
          label="Body Type"
          type="select"
          value={formData.bodyType}
          onChange={(value) => handleChange("bodyType", value)}
          options={[
            { value: "", label: "Select body type" },
            { value: "overweight", label: "Overweight" },
            { value: "obese", label: "Obese" },
            { value: "normal", label: "Normal" },
          ]}
        />

        <InputField
          label="Sex"
          type="select"
          value={formData.sex}
          onChange={(value) => handleChange("sex", value)}
          options={[
            { value: "", label: "Select sex" },
            { value: "male", label: "Male" },
            { value: "female", label: "Female" },
          ]}
        />

        <InputField
          label="Diet"
          type="select"
          value={formData.diet}
          onChange={(value) => handleChange("diet", value)}
          options={[
            { value: "", label: "Select diet type" },
            { value: "vegetarian", label: "Vegetarian" },
            { value: "pescatarian", label: "Pescatarian" },
            { value: "omnivore", label: "Omnivore" },
          ]}
        />

        <InputField
          label="How Often Shower"
          type="select"
          value={formData.showerFrequency}
          onChange={(value) => handleChange("showerFrequency", value)}
          options={[
            { value: "", label: "Select frequency" },
            { value: "rarely", label: "Rarely" },
            { value: "sometimes", label: "Sometimes" },
            { value: "often", label: "Often" },
            { value: "very_often", label: "Very Often" },
          ]}
        />

        <InputField
          label="Heating Energy Source"
          type="select"
          value={formData.heatingEnergy}
          onChange={(value) => handleChange("heatingEnergy", value)}
          options={[
            { value: "", label: "Select energy source" },
            { value: "coal", label: "Coal" },
            { value: "wood", label: "Wood" },
            { value: "natural_gas", label: "Natural Gas" },
            { value: "electricity", label: "Electricity" },
          ]}
        />

        <InputField
          label="Energy Efficiency"
          type="select"
          value={formData.energyEfficiency}
          onChange={(value) => handleChange("energyEfficiency", value)}
          options={[
            { value: "", label: "Select option" },
            { value: "yes", label: "Yes" },
            { value: "no", label: "No" },
            { value: "sometimes", label: "Sometimes" },
          ]}
        />
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 bg-card rounded-xl p-6 md:p-8 border border-border shadow-smooth">
        <h3 className="col-span-full text-xl font-semibold text-foreground mb-4">Transportation & Lifestyle</h3>

        <InputField
          label="Transport Mode"
          type="select"
          value={formData.transport}
          onChange={(value) => handleChange("transport", value)}
          options={[
            { value: "", label: "Select transport" },
            { value: "public", label: "Public Transportation" },
            { value: "private", label: "Private Vehicle" },
            { value: "walk_bicycle", label: "Walk/Bicycle" },
          ]}
        />

        <InputField
          label="Vehicle Type"
          type="text"
          placeholder="e.g., Sedan, SUV, Truck"
          value={formData.vehicleType}
          onChange={(value) => handleChange("vehicleType", value)}
        />

        <InputField
          label="Vehicle Monthly Distance (km)"
          type="number"
          value={formData.vehicleMonthlyKm}
          onChange={(value) => handleChange("vehicleMonthlyKm", Number.parseFloat(value) || 0)}
        />

        <InputField
          label="Air Travel Frequency"
          type="select"
          value={formData.airTravelFrequency}
          onChange={(value) => handleChange("airTravelFrequency", value)}
          options={[
            { value: "", label: "Select frequency" },
            { value: "never", label: "Never" },
            { value: "rarely", label: "Rarely" },
            { value: "sometimes", label: "Sometimes" },
            { value: "often", label: "Often" },
          ]}
        />

        <InputField
          label="Social Activity"
          type="select"
          value={formData.socialActivity}
          onChange={(value) => handleChange("socialActivity", value)}
          options={[
            { value: "", label: "Select frequency" },
            { value: "never", label: "Never" },
            { value: "sometimes", label: "Sometimes" },
            { value: "often", label: "Often" },
          ]}
        />
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 bg-card rounded-xl p-6 md:p-8 border border-border shadow-smooth">
        <h3 className="col-span-full text-xl font-semibold text-foreground mb-4">Consumption & Waste</h3>

        <InputField
          label="Monthly Grocery Bill (₹)"
          type="number"
          value={formData.monthlyGroceryBill}
          onChange={(value) => handleChange("monthlyGroceryBill", Number.parseFloat(value) || 0)}
        />

        <InputField
          label="Waste Bag Size"
          type="select"
          value={formData.wasteBagSize}
          onChange={(value) => handleChange("wasteBagSize", value)}
          options={[
            { value: "", label: "Select size" },
            { value: "small", label: "Small" },
            { value: "medium", label: "Medium" },
            { value: "large", label: "Large" },
            { value: "extra_large", label: "Extra Large" },
          ]}
        />

        <InputField
          label="Waste Bag Weekly Count"
          type="number"
          value={formData.wasteBagWeeklyCount}
          onChange={(value) => handleChange("wasteBagWeeklyCount", Number.parseFloat(value) || 0)}
        />

        <InputField
          label="New Clothes Monthly"
          type="number"
          value={formData.newClothesMonthly}
          onChange={(value) => handleChange("newClothesMonthly", Number.parseFloat(value) || 0)}
        />

        <InputField
          label="TV/PC Daily Hours"
          type="number"
          value={formData.tvPcHoursDaily}
          onChange={(value) => handleChange("tvPcHoursDaily", Number.parseFloat(value) || 0)}
        />

        <InputField
          label="Internet Daily Hours"
          type="number"
          value={formData.internetHoursDaily}
          onChange={(value) => handleChange("internetHoursDaily", Number.parseFloat(value) || 0)}
        />
      </div>

      <div className="bg-card rounded-xl p-6 md:p-8 border border-border shadow-smooth">
        <h3 className="text-xl font-semibold text-foreground mb-6">Preferences</h3>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <MultiSelect
            label="Recycling"
            options={[
              { value: "plastic", label: "Plastic" },
              { value: "metal", label: "Metal" },
              { value: "paper", label: "Paper" },
              { value: "glass", label: "Glass" },
            ]}
            selected={formData.recycling}
            onChange={(value) => handleChange("recycling", value)}
          />

          <MultiSelect
            label="Cooking With"
            options={[
              { value: "gas", label: "Gas" },
              { value: "electric", label: "Electric" },
              { value: "induction", label: "Induction" },
            ]}
            selected={formData.cookingWith}
            onChange={(value) => handleChange("cookingWith", value)}
          />
        </div>
      </div>

      <button
        type="submit"
        disabled={isLoading}
        className="w-full bg-primary hover:bg-primary/90 disabled:bg-muted disabled:cursor-not-allowed text-primary-foreground font-semibold py-3 rounded-lg transition-all duration-300 shadow-smooth hover:shadow-smooth-lg"
      >
        {isLoading ? "Analyzing Your Data..." : "Predict My Carbon Footprint"}
      </button>
    </form>
  )
}
