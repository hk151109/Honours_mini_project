"use client"

import * as THREE from "three"
import { Canvas, useLoader } from "@react-three/fiber"
import { OrbitControls, Stars } from "@react-three/drei"
import { Suspense } from "react"

// Ultra-realistic Earth component
function Earth() {
  // Load textures
  const [colorMap, normalMap, specularMap] = useLoader(THREE.TextureLoader, [
    "/textures/earth_daymap.jpg",
    "/textures/earth_normalmap.jpg",   // converted normal map
    "/textures/earth_specularmap.jpg", // converted specular map
  ])

  return (
    <mesh rotation={[0, 0, 0]}>
      <sphereGeometry args={[1, 64, 64]} />
      <meshPhongMaterial
        map={colorMap}
        normalMap={normalMap}
        specularMap={specularMap}
        specular={0x555555}
      />
    </mesh>
  )
}

export default function EarthLoader() {
  return (
    <div className="fixed inset-0 flex items-center justify-center z-50 bg-black/30 backdrop-blur-sm">
      <Canvas camera={{ position: [0, 0, 3] }}>
        <ambientLight intensity={0.5} />
        <directionalLight position={[5, 3, 5]} intensity={1} />
        <Stars radius={100} depth={50} count={5000} factor={4} saturation={0} fade />
        <Suspense fallback={null}>
          <Earth />
        </Suspense>
        <OrbitControls enableZoom={false} enablePan={false} autoRotate autoRotateSpeed={0.5} />
      </Canvas>
    </div>
  )
}
