# 🌊 VESTIGIUM: Radar de Biomasa Acuática

**Zero Budget Edition** - Un sistema de detección de movimiento basado en análisis avanzado de señal WiFi, sin hardware adicional.

> *"Ver lo invisible: detectar presencia de biomasa acuática usando solo la varianza de señal WiFi"*

---

## 📋 Visión General

VESTIGIUM es un sistema de radar inteligente que transforma tu ZTE (u otro router WiFi) en un sensor de presencia de movimiento. Mediante procesamiento avanzado de señal en software, creamos una "visión" sin cámaras, puramente electromagnética.

**Stack Tecnológico:**
- 🖥️ **Hardware:** ZTE Router + PC con GPU (RTX 3060+)
- 🧠 **AI:** JAX + Redes Neuromórficas
- 📡 **Procesamiento:** Análisis de RSSI + Varianza de Señal
- 🗺️ **Visualización:** Frontend JS con Motion Blur

---

## 🎯 ¿Por qué VESTIGIUM?

| Tradicional | VESTIGIUM |
|---|---|
| CSI por hardware ❌ | CSI virtual por análisis estadístico ✅ |
| Un sensor = un punto | 153+ routers = una red de sensores |
| Alto costo | **$0 en hardware extra** |
| Rígido | Auto-calibración emergente |

---

## 🏗️ Arquitectura: 4 Fases

### **FASE 1: Extracción de "CSI Virtual"** 🔍
**Objetivo:** Crear información de fase sin hardware CSI

#### Componentes:
1. **Saturación de Polling ADB**
   - Fuerza el escaneo WiFi al límite del kernel
   - Máximas muestras RSSI por segundo posibles
   - Permite detectar micro-fluctuaciones

2. **Análisis de Micro-Fluctuaciones (Scintillation)**
   - Ignora el valor promedio del RSSI
   - Enfocarse en el "ruido" de alta frecuencia
   - El aire limpio = ruido blanco gaussiano
   - Biomasa acuática = patrón de interferencia único
   - **JAX identifica la firma sin ver la fase**

3. **Fusión de Bandas (2.4GHz + 5GHz)**
   - Diferencia de atenuación = sensor de profundidad
   - Datos complementarios para triangulación

**Input:** Stream de RSSI en tiempo real  
**Output:** "CSI Virtual" - matriz de varianza temporal

---

### **FASE 2: Motor Neuromórfico de Spikes** 🧠
**Objetivo:** Procesamiento sensible a cambios mínimos mediante redes neuromórficas

#### Componentes:

1. **Filtro de Partículas Bayesianas**
   - No genera coordenadas fijas
   - Mantiene una "nube de probabilidad" por objeto detectado
   - Fluctuación del 2% en señal → nube se reposiciona
   - Manejo robusto de incertidumbre

2. **Red E-SKAN de Contraste**
   ```
   153 redes WiFi 
       ↓ (extraer variaciones)
   Neurona Integradora (suma ponderada)
       ↓ (comparar con umbral)
   Spike (cuando supera ruido base)
       ↓
   "Objeto Detectado"
   ```

3. **Clustering por Huella de Radio**
   - Diferencia masas por área de impacto
   - Humano = bloquea más routers simultáneamente
   - Perro = área más pequeña
   - **Aprendizaje automático de patrones de tamaño**

**Input:** CSI Virtual  
**Output:** Eventos de detección + clusters de biomasa

---

### **FASE 3: SLAM Topológico Emergente** 🗺️
**Objetivo:** Construir mapa dinámico sin lidar ni cámaras

#### Componentes:

1. **Heatmap de Probabilidad**
   - Mapa inicia totalmente negro
   - Cada detección = píxel de calor en mapa 2D
   - Reducción de dimensiones vía Dynamic UMAP
   - Patrón de tránsito visible en tiempo real

2. **Definición Automática de Obstáculos**
   - Señal pero nunca hay movimiento → **Sólido** (pared/mueble)
   - Movimiento frecuente → **Vía de Tránsito**
   - Sin mapeo manual

3. **Auto-Calibración Estocástica**
   - Router del vecino se apaga → mapa se "estira" matemáticamente
   - Ajustes sin perder historicales
   - Sistema resiliente a cambios de topología

**Input:** Eventos de detección + ubicaciones  
**Output:** Mapa dinámico + tabla de espacios

---

### **FASE 4: Visualización de Bajo Consumo** 🎨
**Objetivo:** Frontend eficiente con UX intuitivo

#### Componentes:

1. **Canvas 2D con Motion Blur**
   - Clústeres como orbes de luz
   - Rastros que permiten ver trayectorias
   - Señal intermitente → trayectoria clara

2. **WebSocket Stream**
   - Un solo canal de datos
   - Envía solo: `(x, y, biomass_id, timestamp)`
   - Mínimo ancho de banda en red local

3. **Estética "Marauder's Map"**
   - Interfaz intuitiva tipo mapa del merodeador
   - Optimizada para actualizaciones rápidas

**Input:** Stream de clústeres en tiempo real  
**Output:** Visualización interactiva en navegador

---

## ⚡ Requisitos Técnicos

### Hardware
- ✅ Router ZTE (cualquier modelo con RSSI accesible)
- ✅ PC con GPU: NVIDIA RTX 3060 o superior (o CPU potente)
- ✅ Red WiFi local

### Software
```
Python 3.9+
JAX (GPU support)
NumPy, SciPy
Flask/FastAPI (backend)
WebSocket (comunicación)
Matplotlib/Plotly (visualización)
```

### Red
- Acceso a ADB para escaneo WiFi
- WebSocket en puerto local (ej: 5000)
- Mínimo 100 MB/s entre router y PC

---

## 🚀 Instalación Rápida

### 1. Clonar el repositorio
```bash
git clone https://github.com/XIA01/Vestigium.git
cd Vestigium
```

### 2. Entorno virtual
```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# o
venv\Scripts\activate  # Windows
```

### 3. Dependencias
```bash
pip install -r requirements.txt
```

### 4. Configurar conexión al router
```bash
# Editar config.yaml con IP del ZTE y credenciales
cp config.example.yaml config.yaml
nano config.yaml
```

### 5. Ejecutar sistema
```bash
# Terminal 1: Backend (procesamiento IA)
python src/backend.py

# Terminal 2: Frontend (visualización)
python src/frontend.py

# Abrir navegador en http://localhost:5000
```

---

## 📊 Flujo de Datos

```
Router ZTE (RSSI Stream)
    ↓
[Fase 1] CSI Virtual Processor
    ↓ (varianza temporal + fusión de bandas)
[Fase 2] Motor Neuromórfico (JAX)
    ↓ (filtro bayesiano + clustering)
[Fase 3] SLAM Topológico
    ↓ (heatmap + mapa emergente)
[Fase 4] WebSocket → Frontend
    ↓
👤 Visualización en Tiempo Real
```

---

## 🎮 Uso

### Comando principal
```bash
vestigium start --router-ip 192.168.1.1
```

### Opciones
```
--router-ip        IP del router ZTE
--mode             [normal|debug|training]
--log-level        [INFO|DEBUG|TRACE]
--fps              Fotogramas de visualización (default: 30)
--sensitivity      Umbral de detección 0-1 (default: 0.5)
```

### Ejemplo avanzado
```bash
vestigium start \
  --router-ip 192.168.1.1 \
  --mode debug \
  --sensitivity 0.3 \
  --fps 60
```

---

## 📈 Roadmap

### ✅ Completado
- [ ] Core JAX (red neuromórfica)
- [ ] Polling ADB saturado
- [ ] Parser RSSI básico

### 🔄 En Desarrollo
- [ ] Fusión 2.4GHz + 5GHz
- [ ] Filtro de partículas bayesiano
- [ ] SLAM topológico
- [ ] Frontend WebSocket

### 🚀 Próximas Fases
- [ ] Entrenamiento con Dataset de movimiento
- [ ] Multi-router orchestration
- [ ] API REST para integración externa
- [ ] Docker containers para deployment
- [ ] Benchmarks de latencia en tiempo real

---

## 🔧 Arquitectura de Carpetas

```
Vestigium/
├── src/
│   ├── backend/
│   │   ├── signal_processor.py      # Fase 1: CSI Virtual
│   │   ├── neuromorphic_engine.py   # Fase 2: JAX Core
│   │   ├── slam_topological.py      # Fase 3: SLAM
│   │   └── websocket_server.py      # Comunicación
│   ├── frontend/
│   │   ├── app.py                    # Visualización
│   │   ├── static/
│   │   │   ├── canvas-renderer.js    # Motion blur
│   │   │   └── style.css
│   │   └── templates/
│   │       └── index.html
│   └── utils/
│       ├── config.py
│       ├── logger.py
│       └── device_interface.py       # Comunicación ZTE
├── tests/
├── docs/
├── requirements.txt
├── config.example.yaml
└── README.md
```

---

## 🧪 Testing

```bash
# Tests unitarios
pytest tests/ -v

# Simulación sin hardware
python tests/sim_rssi_stream.py

# Benchmark de rendimiento
python benchmarks/latency_test.py
```

---

## 📚 Documentación

- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** - Detalles técnicos profundos
- **[PHASE_1.md](docs/PHASE_1.md)** - CSI Virtual: guía de implementación
- **[PHASE_2.md](docs/PHASE_2.md)** - Motor neuromórfico con JAX
- **[PHASE_3.md](docs/PHASE_3.md)** - SLAM topológico emergente
- **[API.md](docs/API.md)** - Especificación de endpoints WebSocket

---

## 🤝 Contribuir

Las contribuciones son bienvenidas. Para cambios importantes:

1. Fork el repositorio
2. Crea una rama de feature (`git checkout -b feature/AmazingFeature`)
3. Commit cambios (`git commit -m 'Add AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

**Áreas de oportunidad:**
- Optimización de JAX en GPU
- Implementación de más backends de router
- Visualizaciones avanzadas
- Datasets de entrenamiento

---

## ⚠️ Limitaciones Conocidas

- 📶 Requiere mínimo 5 routers WiFi cercanos
- 🕐 Latencia de ~500ms en detección inicial
- 🌧️ Interferencia electromagnética afecta precisión
- 💻 GPU recomendada para tiempo real (CPU funciona pero lento)

---

## 📄 Licencia

Este proyecto está bajo la licencia MIT. Ver [LICENSE](LICENSE) para detalles.

---

## 📞 Contacto & Soporte

- 🐛 **Issues:** [GitHub Issues](https://github.com/XIA01/Vestigium/issues)
- 💬 **Discusiones:** [GitHub Discussions](https://github.com/XIA01/Vestigium/discussions)
- 📧 **Email:** [Tu email aquí]

---

## 🙏 Agradecimientos

- Inspirado en el análisis WiFi sensing de MIT
- Neuromorfismo basado en investigación de SNNs (Spiking Neural Networks)
- SLAM emergente inspirado en comportamiento animal

---

**Última actualización:** Abril 2026  
**Estado:** 🔧 En desarrollo activo

