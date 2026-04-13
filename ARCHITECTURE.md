# 🏛️ ARCHITECTURE - Detalles Técnicos Profundos

## Índice
1. [Fundamentos Teóricos](#fundamentos-teóricos)
2. [Fase 1: CSI Virtual](#fase-1-csi-virtual)
3. [Fase 2: Motor Neuromórfico](#fase-2-motor-neuromórfico)
4. [Fase 3: SLAM Topológico](#fase-3-slam-topológico)
5. [Fase 4: Visualización](#fase-4-visualización)
6. [Flujos de Datos Detallados](#flujos-de-datos-detallados)
7. [Optimizaciones GPU](#optimizaciones-gpu)

---

## Fundamentos Teóricos

### ¿Por qué RSSI Varianza = CSI Virtual?

**Problem:** No tenemos acceso a Channel State Information (CSI) con hardware estándar.

**Solution:** Usamos varianza temporal de RSSI como proxy estadístico.

#### Matemática:
```
CSI_virtual(t) ≈ σ²(RSSI[t-w:t])

Donde:
- σ² = varianza
- RSSI[t-w:t] = ventana móvil de muestras RSSI
- w = tamaño de ventana (típicamente 100-500ms)
```

#### Por qué funciona:

1. **Aire tranquilo** = ruido blanco gaussiano N(μ, σ₀²)
   - Varianza estable, ~2-3 dB

2. **Presencia de biomasa** = interferencia constructiva/destructiva
   - El objeto acuático actúa como obstáculo dieléctrico
   - Crea "sombra de radio" que modula el RSSI
   - Varianza aumenta a ~5-10 dB

3. **Diferencia de bandas** (2.4GHz vs 5GHz)
   - Agua absorbe 5GHz más que 2.4GHz
   - Ratio = indicador de profundidad/tamaño

---

## FASE 1: CSI Virtual

### Arquitectura de Datos

```
┌─────────────────────────────────────────┐
│  Router ZTE - Escaneo WiFi              │
│  (ADB Polling - 100+ samples/sec)       │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│  Buffer Circular RSSI                   │
│  [RSSI_1, RSSI_2, ..., RSSI_N]         │
│  Tamaño: 5000 muestras (~50ms @ 100Hz) │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│  Ventanas Móviles (Sliding Windows)     │
│  ┌──────┐                               │
│  │ w1   │  ← overlap = 50%              │
│  │  σ²  │                               │
│  └──────┐                               │
│     └──────┐                            │
│     │ w2   │  ← siguiente ventana        │
│     │  σ²  │                            │
│     └──────┘                            │
│  Output: [σ²_1, σ²_2, σ²_3, ...]       │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│  Análisis Espectral (FFT)               │
│  Power = |FFT(σ²(t))|²                 │
│  → Detecta oscilaciones ~1-10 Hz        │
│     (movimiento respiratorio, etc)      │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│  CSI Virtual Matrix                     │
│  Shape: (num_routers, num_bands, T)    │
│  Ejemplo: (153, 2, 1000)                │
└─────────────────────────────────────────┘
```

### Algoritmo de Varianza Adaptativa

```python
def adaptive_rssi_variance(rssi_window, baseline_variance):
    """
    Calcula varianza normalizada vs línea base
    
    Args:
        rssi_window: array de RSSI[t-w:t]
        baseline_variance: σ₀² de referencia
    
    Returns:
        normalized_variance: (σ²_actual - σ₀²) / σ₀²
    """
    
    # Estimación robusta con MAD (Median Absolute Deviation)
    median = np.median(rssi_window)
    mad = np.median(np.abs(rssi_window - median))
    
    # Varianza clásica
    variance = np.var(rssi_window)
    
    # Detección de outliers (espectro = 1.4826 * MAD)
    sigma_estimated = 1.4826 * mad
    
    # Normalización contra baseline
    if baseline_variance > 0:
        return (variance - baseline_variance) / baseline_variance
    else:
        return variance
```

### Fusión de Bandas

```
2.4GHz y 5GHz detectan el mismo objeto
pero con atenuaciones diferentes:

Attenuation_ratio = (P_2.4GHz - P_5GHz) / P_2.4GHz

Si ratio alto → objeto grande (agua profunda / humano)
Si ratio bajo → objeto pequeño (perro / pez)
```

---

## FASE 2: Motor Neuromórfico

### Red E-SKAN (Event-driven Spiking Kernel Architecture Network)

#### Arquitectura de Capas:

```
Input Layer (153 sensores WiFi × 2 bandas)
    │
    ├─→ [Neurona-1] → spike si Σ varianza > θ₁
    ├─→ [Neurona-2] → spike si movimento detectado
    ├─→ [Neurona-N]
    │
    ▼
Hidden Layer (Integración Temporal)
    │
    ├─→ [Integrador-1] ─┐
    ├─→ [Integrador-2] ─┼─→ Coincidence Detector
    └─→ [Integrador-N] ─┘
    │
    ▼
Output Layer (Decisión)
    │
    ├─→ SPIKE = Presencia Detectada + Ubicación Aproximada
    └─→ NO SPIKE = Ruido / Falsa Alarma
```

#### Ecuación de Neurona con Spike:

```
V(t) = Σᵢ wᵢ × xᵢ(t) + Σⱼ bⱼ(t)

Donde:
- wᵢ = peso sináptico (aprendible)
- xᵢ(t) = entrada i en tiempo t
- bⱼ(t) = bias adaptativo
- b(t) decae exponencialmente si no hay spikes

Si V(t) > θ (threshold):
    → spike(t) = 1
    → V(t+1) = -Vrest (período refractario)
Sino:
    → spike(t) = 0
```

#### Filtro de Partículas Bayesiano

```
Estado: x(t) = [x_pos, y_pos, vx, vy, tamaño]

Predict:
    x̂⁻(t) = f(x(t-1), u_motion_model)
    P⁻(t) = F·P(t-1)·Fᵀ + Q  (covarianza predicción)

Update (cuando hay spike):
    K(t) = P⁻(t)·Hᵀ / (H·P⁻(t)·Hᵀ + R)  (Kalman gain)
    x̂(t) = x̂⁻(t) + K(t)·(z(t) - H·x̂⁻(t))
    P(t) = (I - K(t)·H)·P⁻(t)

Donde:
- Q = incertidumbre modelo (browniano si Q alto)
- R = incertidumbre medición (basada en RSSI SNR)
- H = matriz de observación (mapea estado a sensores)
```

#### Clustering Dinámico

**Algoritmo: Agglomerative Clustering adaptativo**

```python
# Distancia entre masas basada en "huella de radio"
def radio_signature_distance(mass_a, mass_b):
    """
    Calcula diferencia de patrones de ocupación
    
    signature = [router_1_detections, router_2_detections, ...]
    
    distance = euclidean(normalize(sig_a), normalize(sig_b))
    
    Si masa A bloquea routers {1,2,3,4}
    y masa B bloquea routers {1,2}
    → son probablemente el mismo objeto
    
    Si masa A bloquea {1,2,3} y masa B bloquea {10,11,12}
    → probablemente objetos diferentes
    """
    sig_a = mass_a.router_occupancy_vector
    sig_b = mass_b.router_occupancy_vector
    
    # Normalizar (L2)
    sig_a = sig_a / (np.linalg.norm(sig_a) + 1e-6)
    sig_b = sig_b / (np.linalg.norm(sig_b) + 1e-6)
    
    return np.linalg.norm(sig_a - sig_b)
```

---

## FASE 3: SLAM Topológico

### Mapa Emergente: Principios

```
1. Inicialmente: Mapa totalmente negro (ocupancia desconocida)

2. A cada spike (detección):
   - Pintar píxel caliente en ubicación (x,y)
   - Aumentar confianza (ocupancia ↑)

3. Si spike continúa en mismo lugar:
   - Movimiento lento → marcar como "sólido" (pared)
   - Movimiento rápido → marcar como "tránsito" (pasillo)

4. Dinámicamente:
   - Desvanecimiento exponencial: ocupancia(t) *= decay^(Δt)
   - Si router desaparece: mapa estira matemáticamente
```

### Ecuación de Ocupancia (Inverse Sensor Model)

```
log-odds representation:

L(x,y,t) = L(x,y,t-1) + log(P(z|occupied) / P(z|free))

Donde z = spike / no-spike observado en (x,y)

Conversión a probabilidad:
P(occupied | z) = 1 / (1 + exp(-L))

Visualización:
color = temperature_map[P(occupied)]
```

### Detección Automática de Obstáculos

```
Para cada píxel (x,y):

Estado = empty    si movimiento frecuente
Estado = solid    si señal débil pero nunca movimiento
Estado = transit  si movimiento moderado en trayectorias

Heurística:
- Ocupancia ↑ pero movimiento ↓ → SOLID (pared)
- Ocupancia ↑ y movimiento ↑ → TRANSIT (pasillo)
- Ocupancia baja → EMPTY (espacio libre)
```

---

## FASE 4: Visualización

### Pipeline de Rendering

```
Datos de SLAM + Clusters
    │
    ├─→ Proyección 2D (UMAP reduction si es necesario)
    │
    ├─→ Rasterización del Heatmap
    │   ├─ Ocupancia → color HSV (H=temperatura)
    │   └─ Aplicar Gaussian blur para suavidad
    │
    ├─→ Renderizado de Clusters
    │   ├─ Posición actual → círculo brillante
    │   ├─ Histórico → rastro (trail)
    │   └─ Glow effect = intensidad de incertidumbre
    │
    └─→ Canvas 2D WebGL
        ├─ Motion blur (acumular fotogramas)
        └─ Enviar frame a 30 FPS vía WebSocket
```

### Motion Blur Temporal

```javascript
// GPU-based temporal accumulation
canvas_t = 0.7 * canvas_{t-1} + 0.3 * current_frame

// Resultado: rastro visual de movimiento
// sin necesidad de guardar histórico
```

---

## Flujos de Datos Detallados

### Flujo Completo de Una Detección

```
T=0ms   → Router ZTE detecta cambio en RSSI
T=10ms  → ADB polling captura muestras

T=100ms → Ventana móvil calcula σ²
T=105ms → FFT detecta componente oscilatoria
T=110ms → Paso a JAX pipeline

T=120ms → E-SKAN genera spike (evento)
T=125ms → Filtro Bayesiano actualiza posición

T=130ms → SLAM proyecta en mapa
T=135ms → Canvas renderiza con motion blur

T=140ms → WebSocket envía frame
T=150ms → Navegador muestra actualización

Latencia total: ~150ms (tolerable para vigilancia)
```

### Matriz de Datos: Forma y Dimensiones

```
RSSI_raw: (num_routers=153, num_bands=2, timesteps=5000)
    Tipo: float32
    Actualización: 100 Hz → 50ms de buffer

CSI_virtual: (num_routers=153, num_bands=2, timesteps=100)
    Tipo: float32
    Cálculo: media móvil cada 500ms

Spike_events: (num_neurons=256, timesteps=100)
    Tipo: bool
    Latencia: ~10ms

Clusters: [num_clusters ≤ 10]
    Tipo: {x: float, y: float, vx: float, vy: float, 
           size: float, confidence: float}

Map_occupancy: (grid_x=500, grid_y=500)
    Tipo: float32, rango [0, 1]
    Actualización: 30 Hz
```

---

## Optimizaciones GPU

### JAX JIT Compilation

```python
# Sin JIT: JAX traduce a XLA en cada llamada (~100ms)
# Con JIT: primera llamada es lenta, luego ~1ms

@jax.jit
def neuromorphic_step(rssi_signal, particles, weights):
    """
    Compilado a código GPU optimizado
    """
    # Actualizar partículas en paralelo
    new_particles = vmap(particle_update)(particles)
    
    # Calcular likelihood en paralelo
    likelihoods = vmap(compute_likelihood)(rssi_signal)
    
    # Resamplear solo si necesario
    new_weights = weights * likelihoods
    new_weights = new_weights / jnp.sum(new_weights)
    
    return new_particles, new_weights
```

### Vectorización Masiva

```python
# Lugar de loops Python (lento):
for i in range(153):  # 153 routers
    for j in range(256):  # 256 neuronas
        compute_spike(...)  # ~lento

# Usar vmap (rápido en GPU):
compute_all_spikes = vmap(
    vmap(compute_spike, axis=0),  # sobre routers
    axis=0  # sobre neuronas
)
result = compute_all_spikes(rssi_matrix)  # paralelizado
```

### Gestión de Memoria

```
Asignación típica:

RSSI buffers:        50 MB (5000 muestras × 153 routers × 2 bandas)
Partículas:          20 MB (1000 partículas × 256 dim)
Mapa SLAM:           100 MB (500×500 × float32)
Modelos JAX:         ~10 MB (pesos de red)

Total GPU: ~200 MB (cabe en cualquier GPU moderna)

Monitoreo:
>>> import jax
>>> jax.devices()  # listar GPUs
>>> gpu_memory = jax.device_memory_usage()
```

---

## Benchmarks Esperados

| Módulo | Input | Output | Latencia | GPU Utilización |
|--------|-------|--------|----------|-----------------|
| CSI Virtual | 153×100 RSSI | 153×100 σ² | 5ms | 10% |
| E-SKAN | 153×100 σ² | 256 spikes | 15ms | 20% |
| Particle Filter | 256 spikes | 10 clusters | 20ms | 30% |
| SLAM | 10 clusters | 500×500 map | 25ms | 15% |
| **Total** | **RSSI stream** | **Frame** | **~100ms** | **~50%** |

*(en RTX 3060)*

---

## Troubleshooting Arquitectura

### Problema: Muchos falsos positivos
```
→ Aumentar spike_threshold en config.yaml
→ Reducir learning_rate para que la red sea más conservadora
→ Verificar que baseline_variance está bien calibrada
```

### Problema: Detecciones lentas
```
→ Aumentar max_samples_per_second
→ Reducir window_size_ms
→ Verificar GPU no está throttling (overheating)
```

### Problema: Mapa distorsionado
```
→ Aumentar num_particles (más precisión)
→ Reducir motion_model uncertainty (Q)
→ Verificar que topología de routers no cambió
```

---

**Última actualización:** Abril 2026
