module.exports = [
"[externals]/next/dist/compiled/next-server/app-route-turbo.runtime.dev.js [external] (next/dist/compiled/next-server/app-route-turbo.runtime.dev.js, cjs)", ((__turbopack_context__, module, exports) => {

const mod = __turbopack_context__.x("next/dist/compiled/next-server/app-route-turbo.runtime.dev.js", () => require("next/dist/compiled/next-server/app-route-turbo.runtime.dev.js"));

module.exports = mod;
}),
"[externals]/next/dist/compiled/next-server/app-page-turbo.runtime.dev.js [external] (next/dist/compiled/next-server/app-page-turbo.runtime.dev.js, cjs)", ((__turbopack_context__, module, exports) => {

const mod = __turbopack_context__.x("next/dist/compiled/next-server/app-page-turbo.runtime.dev.js", () => require("next/dist/compiled/next-server/app-page-turbo.runtime.dev.js"));

module.exports = mod;
}),
"[externals]/next/dist/server/app-render/work-unit-async-storage.external.js [external] (next/dist/server/app-render/work-unit-async-storage.external.js, cjs)", ((__turbopack_context__, module, exports) => {

const mod = __turbopack_context__.x("next/dist/server/app-render/work-unit-async-storage.external.js", () => require("next/dist/server/app-render/work-unit-async-storage.external.js"));

module.exports = mod;
}),
"[externals]/next/dist/server/app-render/work-async-storage.external.js [external] (next/dist/server/app-render/work-async-storage.external.js, cjs)", ((__turbopack_context__, module, exports) => {

const mod = __turbopack_context__.x("next/dist/server/app-render/work-async-storage.external.js", () => require("next/dist/server/app-render/work-async-storage.external.js"));

module.exports = mod;
}),
"[externals]/next/dist/shared/lib/no-fallback-error.external.js [external] (next/dist/shared/lib/no-fallback-error.external.js, cjs)", ((__turbopack_context__, module, exports) => {

const mod = __turbopack_context__.x("next/dist/shared/lib/no-fallback-error.external.js", () => require("next/dist/shared/lib/no-fallback-error.external.js"));

module.exports = mod;
}),
"[project]/Downloads/enviro-meter/app/api/predict/route.ts [app-route] (ecmascript)", ((__turbopack_context__) => {
"use strict";

__turbopack_context__.s([
    "POST",
    ()=>POST
]);
function generateRecommendations(formData, emissionScore) {
    const recommendations = [];
    // Energy usage
    if (formData.heatingEnergy !== "electricity" || formData.energyEfficiency === "no") {
        recommendations.push("Improve energy efficiency at home (insulation, LED bulbs, etc.)");
    }
    // Transport
    if (formData.transport === "private" || formData.vehicleMonthlyKm > 200) {
        recommendations.push("Use public transport, cycle, or walk more often");
    }
    // Air travel
    if (formData.airTravelFrequency !== "never" && formData.airTravelFrequency !== "rarely") {
        recommendations.push("Limit air travel when possible");
    }
    // Diet
    if (formData.diet === "omnivore") {
        recommendations.push("Consider a more plant-based diet to reduce carbon footprint");
    }
    // Waste and recycling
    if (!formData.recycling || formData.recycling.length === 0) {
        recommendations.push("Recycle consistently to reduce waste");
    }
    if (formData.wasteBagWeeklyCount > 3 || formData.newClothesMonthly > 5) {
        recommendations.push("Reduce unnecessary consumption and purchases");
    }
    // Digital usage
    if (formData.tvPcHoursDaily > 6 || formData.internetHoursDaily > 8) {
        recommendations.push("Limit screen time or use energy-efficient devices");
    }
    // Fallback if no recommendations
    if (recommendations.length === 0) {
        recommendations.push("Maintain your sustainable lifestyle!");
    }
    // Optional: add a note if emission is high
    if (emissionScore > 1000) {
        recommendations.push("Your carbon footprint is high — take immediate action to reduce emissions!");
    }
    return recommendations;
}
async function POST(request) {
    try {
        const formData = await request.json();
        // Map frontend values to dataset-compatible values
        const showerMap = {
            "rarely": "less frequently",
            "sometimes": "frequently",
            "often": "often",
            "very_often": "daily"
        };
        const socialActivityMap = {
            "never": "rarely",
            "sometimes": "sometimes",
            "often": "often"
        };
        const airTravelMap = {
            "never": "never",
            "rarely": "rarely",
            "sometimes": "frequently",
            "often": "frequently"
        };
        const heatingEnergyMap = {
            "coal": "coal",
            "wood": "wood",
            "natural_gas": "natural gas",
            "electricity": "electricity"
        };
        const INR_TO_USD = 0.011;
        const payload = {
            "Body Type": formData.bodyType,
            "Sex": formData.sex,
            "Diet": formData.diet,
            "How Often Shower": showerMap[formData.showerFrequency] || "daily",
            "Heating Energy Source": heatingEnergyMap[formData.heatingEnergy] || "electricity",
            "Transport": formData.transport,
            "Vehicle Type": formData.vehicleType || "None",
            "Social Activity": socialActivityMap[formData.socialActivity] || "often",
            "Monthly Grocery Bill": Number(formData.monthlyGroceryBill) * INR_TO_USD,
            "Frequency of Traveling by Air": airTravelMap[formData.airTravelFrequency] || "rarely",
            "Vehicle Monthly Distance Km": Number(formData.vehicleMonthlyKm),
            "Waste Bag Size": formData.wasteBagSize,
            "Waste Bag Weekly Count": Number(formData.wasteBagWeeklyCount),
            "How Long TV PC Daily Hour": Number(formData.tvPcHoursDaily),
            "How Many New Clothes Monthly": Number(formData.newClothesMonthly),
            "How Long Internet Daily Hour": Number(formData.internetHoursDaily),
            "Energy efficiency": formData.energyEfficiency,
            "Recycling": JSON.stringify(formData.recycling.length ? formData.recycling.map((v)=>v.charAt(0).toUpperCase() + v.slice(1)) : [
                "Metal"
            ]),
            "Cooking_With": JSON.stringify(formData.cookingWith.length ? formData.cookingWith.map((v)=>v.charAt(0).toUpperCase() + v.slice(1)) : [
                "Stove"
            ])
        };
        const response = await fetch("http://127.0.0.1:5000/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(payload)
        });
        const result = await response.json();
        if (!response.ok) {
            console.error("Flask error:", result);
            throw new Error("Flask API returned an error");
        }
        const recommendations = generateRecommendations(formData, result.predicted_carbon_emission);
        return Response.json({
            emissionScore: result.predicted_carbon_emission,
            status: result.predicted_carbon_emission < 50 ? "Low" : result.predicted_carbon_emission < 100 ? "Moderate" : "High",
            recommendations
        });
    } catch (error) {
        console.error("Prediction error:", error);
        return Response.json({
            error: "Failed to fetch prediction from backend"
        }, {
            status: 500
        });
    }
}
}),
];

//# sourceMappingURL=%5Broot-of-the-server%5D__94d4a4d2._.js.map